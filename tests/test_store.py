"""
Tests for repolens/store.py.

OpenAI calls are mocked — tests never hit the network.
ChromaDB uses tmp_path for isolation between tests.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from repolens.chunker import Chunk
from repolens.store import (
    hash_file,
    build_embed_text,
    chunk_to_metadata,
    index_chunks,
    query_chunks,
    EMBEDDING_MODEL,
)


# ── Fixtures ────────────────────────────────────────────────────────────────

def make_chunk(**kwargs) -> Chunk:
    """Return a Chunk with sensible defaults, overridable via kwargs."""
    defaults = dict(
        file_path="/repo/auth.py",
        node_type="function_definition",
        name="authenticate_user",
        source="def authenticate_user(token):\n    return True",
        start_line=1,
        end_line=2,
        token_count=12,
        calls=["validate_token"],
        docstring="Validates user credentials.",
        parent_class=None,
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def mock_openai_client(embedding_dim: int = 8) -> MagicMock:
    """
    Return a MagicMock that behaves like an OpenAI client.
    Embeddings are zero vectors of length embedding_dim.
    """
    client = MagicMock()
    embedding = MagicMock()
    embedding.embedding = [0.0] * embedding_dim
    client.embeddings.create.return_value.data = [embedding]
    return client


# ── hash_file ────────────────────────────────────────────────────────────────

class TestHashFile:

    def test_returns_64_char_hex_string(self, tmp_path):
        f = tmp_path / "f.py"
        f.write_text("x = 1")
        result = hash_file(f)
        assert len(result) == 64
        assert all(c in "0123456789abcdef" for c in result)

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1")
        f2.write_text("x = 1")
        assert hash_file(f1) == hash_file(f2)

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("x = 1")
        f2.write_text("x = 2")
        assert hash_file(f1) != hash_file(f2)


# ── build_embed_text ──────────────────────────────────────────────────────────

class TestBuildEmbedText:

    def test_includes_function_name(self):
        chunk = make_chunk()
        text = build_embed_text(chunk)
        assert "authenticate_user" in text

    def test_includes_docstring_when_present(self):
        chunk = make_chunk(docstring="Validates user credentials.")
        text = build_embed_text(chunk)
        assert "Validates user credentials." in text

    def test_includes_source(self):
        chunk = make_chunk()
        text = build_embed_text(chunk)
        assert chunk.source in text

    def test_no_docstring_still_includes_name_and_source(self):
        chunk = make_chunk(docstring=None)
        text = build_embed_text(chunk)
        assert "authenticate_user" in text
        assert chunk.source in text


# ── chunk_to_metadata ─────────────────────────────────────────────────────────

class TestChunkToMetadata:

    def test_calls_serialized_as_string(self):
        chunk = make_chunk(calls=["foo", "bar"])
        meta = chunk_to_metadata(chunk)
        assert meta["calls"] == "bar,foo" or meta["calls"] == "foo,bar"

    def test_empty_calls_is_empty_string(self):
        chunk = make_chunk(calls=[])
        meta = chunk_to_metadata(chunk)
        assert meta["calls"] == ""

    def test_none_docstring_becomes_empty_string(self):
        chunk = make_chunk(docstring=None)
        meta = chunk_to_metadata(chunk)
        assert meta["docstring"] == ""

    def test_all_values_are_chromadb_compatible_types(self):
        chunk = make_chunk()
        meta = chunk_to_metadata(chunk)
        for v in meta.values():
            assert isinstance(v, (str, int, float, bool))


# ── index_chunks ──────────────────────────────────────────────────────────────

class TestIndexChunks:

    def test_indexes_chunks_successfully(self, tmp_path):
        src = tmp_path / "auth.py"
        src.write_text("def foo(): pass")
        chunk = make_chunk(file_path=str(src))
        client = mock_openai_client()

        result = index_chunks(
            chunks=[chunk],
            file_path=src,
            store_path=tmp_path / "db",
            openai_client=client,
        )

        assert result["indexed"] == 1
        assert result["skipped"] is False

    def test_skips_unchanged_file(self, tmp_path):
        src = tmp_path / "auth.py"
        src.write_text("def foo(): pass")
        chunk = make_chunk(file_path=str(src))
        client = mock_openai_client()
        db_path = tmp_path / "db"

        # First index
        index_chunks([chunk], src, db_path, client)
        # Second index — same file, should skip
        result = index_chunks([chunk], src, db_path, client)

        assert result["skipped"] is True
        assert result["indexed"] == 0

    def test_force_reindexes_unchanged_file(self, tmp_path):
        src = tmp_path / "auth.py"
        src.write_text("def foo(): pass")
        chunk = make_chunk(file_path=str(src))
        client = mock_openai_client()
        db_path = tmp_path / "db"

        index_chunks([chunk], src, db_path, client)
        result = index_chunks([chunk], src, db_path, client, force=True)

        assert result["skipped"] is False
        assert result["indexed"] == 1

    def test_reindexes_when_file_changes(self, tmp_path):
        src = tmp_path / "auth.py"
        src.write_text("def foo(): pass")
        chunk = make_chunk(file_path=str(src))
        client = mock_openai_client()
        db_path = tmp_path / "db"

        index_chunks([chunk], src, db_path, client)

        src.write_text("def foo(): return 1")
        result = index_chunks([chunk], src, db_path, client)

        assert result["skipped"] is False

    def test_empty_chunks_updates_hash_but_indexes_zero(self, tmp_path):
        src = tmp_path / "empty.py"
        src.write_text("")
        client = mock_openai_client()
        db_path = tmp_path / "db"

        result = index_chunks([], src, db_path, client)

        assert result["indexed"] == 0
        assert result["skipped"] is False
