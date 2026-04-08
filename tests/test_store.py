"""
Tests for repolix/store.py.

OpenAI calls are mocked — tests never hit the network.
ChromaDB uses tmp_path for isolation between tests.
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from repolix.chunker import Chunk
from repolix.store import (
    hash_file,
    build_embed_text,
    chunk_to_metadata,
    index_chunks,
    index_repo,
    keyword_search,
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
        is_truncated=False,
    )
    defaults.update(kwargs)
    return Chunk(**defaults)


def mock_openai_client(embedding_dim: int = 8) -> MagicMock:
    """
    Return a MagicMock that behaves like an OpenAI client.

    Returns one zero-vector embedding per input text so the mock
    works correctly for any batch size. Using a fixed return_value
    would always produce one embedding regardless of input count,
    causing ChromaDB to reject the length mismatch on multi-chunk
    index calls.
    """
    client = MagicMock()

    def _create_embeddings(**kwargs):
        inputs = kwargs.get("input", [])
        response = MagicMock()
        response.data = [
            MagicMock(embedding=[0.0] * embedding_dim)
            for _ in inputs
        ]
        return response

    client.embeddings.create.side_effect = _create_embeddings
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

    def test_is_truncated_false_stored_as_bool(self):
        chunk = make_chunk(is_truncated=False)
        meta = chunk_to_metadata(chunk)
        assert meta["is_truncated"] is False

    def test_is_truncated_true_stored_as_bool(self):
        chunk = make_chunk(is_truncated=True)
        meta = chunk_to_metadata(chunk)
        assert meta["is_truncated"] is True


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


# ── index_repo orphan cleanup ────────────────────────────────────────────────

class TestIndexRepoOrphanCleanup:
    """
    Tests that index_repo removes chunks and hash entries for files
    that were previously indexed but no longer exist in the repo.
    """

    def _make_repo_with_file(self, tmp_path: Path, filename: str, source: str) -> tuple[Path, Path]:
        """Create a minimal repo with one Python file. Returns (repo_path, db_path)."""
        repo = tmp_path / "repo"
        repo.mkdir()
        (repo / filename).write_text(source)
        db_path = tmp_path / "db"
        return repo, db_path

    def test_deleted_file_chunks_are_removed(self, tmp_path):
        repo, db_path = self._make_repo_with_file(tmp_path, "auth.py", "def foo(): pass")
        client = mock_openai_client()

        # chunk_file is imported locally inside index_repo from repolix.chunker,
        # so we patch it there, not on repolix.store.
        with patch("repolix.chunker.chunk_file") as mock_chunk:
            mock_chunk.return_value = [make_chunk(file_path=str(repo / "auth.py"))]
            index_repo(repo, db_path, client)

        # Delete the file from disk so next run treats it as orphaned
        (repo / "auth.py").unlink()

        with patch("repolix.chunker.chunk_file") as mock_chunk:
            mock_chunk.return_value = []
            stats = index_repo(repo, db_path, client)

        assert stats["cleaned"] == 1

    def test_deleted_file_hash_is_removed(self, tmp_path):
        repo, db_path = self._make_repo_with_file(tmp_path, "auth.py", "def foo(): pass")
        client = mock_openai_client()

        with patch("repolix.chunker.chunk_file") as mock_chunk:
            mock_chunk.return_value = [make_chunk(file_path=str(repo / "auth.py"))]
            index_repo(repo, db_path, client)

        (repo / "auth.py").unlink()

        with patch("repolix.chunker.chunk_file") as mock_chunk:
            mock_chunk.return_value = []
            index_repo(repo, db_path, client)

        # Hash entry should be gone — next run of the same deleted path
        # would not find a stored hash.
        import chromadb
        from chromadb.config import Settings
        from repolix.store import HASHES_COLLECTION
        db = chromadb.PersistentClient(path=str(db_path), settings=Settings(anonymized_telemetry=False))
        hashes_col = db.get_or_create_collection(HASHES_COLLECTION, embedding_function=None)
        stored = hashes_col.get(include=[])
        assert str(repo / "auth.py") not in stored["ids"]

    def test_existing_files_are_not_cleaned(self, tmp_path):
        repo, db_path = self._make_repo_with_file(tmp_path, "auth.py", "def foo(): pass")
        client = mock_openai_client()

        with patch("repolix.chunker.chunk_file") as mock_chunk:
            mock_chunk.return_value = [make_chunk(file_path=str(repo / "auth.py"))]
            index_repo(repo, db_path, client)

        # Run again with the file still present — nothing should be cleaned
        with patch("repolix.chunker.chunk_file") as mock_chunk:
            mock_chunk.return_value = [make_chunk(file_path=str(repo / "auth.py"))]
            stats = index_repo(repo, db_path, client)

        assert stats["cleaned"] == 0

    def test_stats_cleaned_key_always_present(self, tmp_path):
        repo, db_path = self._make_repo_with_file(tmp_path, "auth.py", "def foo(): pass")
        client = mock_openai_client()

        with patch("repolix.chunker.chunk_file") as mock_chunk:
            mock_chunk.return_value = []
            stats = index_repo(repo, db_path, client)

        assert "cleaned" in stats


# ── keyword_search ────────────────────────────────────────────────────────────

def _index_chunk(chunk: Chunk, tmp_path: Path) -> Path:
    """Index a single chunk into a fresh ChromaDB store and return the store path."""
    src = tmp_path / "src.py"
    src.write_text("placeholder")
    db_path = tmp_path / "db"
    client = mock_openai_client()
    index_chunks([chunk], src, db_path, client)
    return db_path


class TestKeywordSearch:

    def test_finds_chunk_by_three_char_token(self, tmp_path):
        chunk = make_chunk(source="def walk_repo(path): pass")
        db = _index_chunk(chunk, tmp_path)
        results = keyword_search("walk", db)
        assert any(r["name"] == "authenticate_user" or "walk" in r["source"]
                   for r in results)

    def test_finds_chunk_by_two_char_token(self, tmp_path):
        chunk = make_chunk(source="import os\ndef list_files(path):\n    return os.listdir(path)")
        db = _index_chunk(chunk, tmp_path)
        results = keyword_search("os", db)
        assert len(results) == 1

    def test_two_char_token_not_dropped(self, tmp_path):
        # Regression: tokens of length 2 were previously filtered out by
        # the > 2 guard, meaning queries containing "os", "db", "id" etc.
        # never reached ChromaDB at all.
        chunk = make_chunk(source="def connect_db(db): return db.cursor()")
        db = _index_chunk(chunk, tmp_path)
        results = keyword_search("db connection", db)
        assert len(results) == 1

    def test_single_char_token_still_dropped(self, tmp_path):
        # Single characters are intentionally excluded — they match too
        # broadly via $contains and would pollute results.
        chunk = make_chunk(source="def foo(a, b): return a + b")
        db = _index_chunk(chunk, tmp_path)
        results = keyword_search("a b", db)
        # Both tokens are length 1 — nothing reaches ChromaDB, empty result.
        assert results == []

    def test_returns_empty_when_no_match(self, tmp_path):
        chunk = make_chunk(source="def foo(): pass")
        db = _index_chunk(chunk, tmp_path)
        results = keyword_search("zzznomatch", db)
        assert results == []

    def test_more_token_matches_ranks_higher(self, tmp_path):
        src = tmp_path / "src.py"
        src.write_text("placeholder")
        db_path = tmp_path / "db"
        client = mock_openai_client()

        chunk_both = make_chunk(
            file_path=str(tmp_path / "src.py"),
            name="walk_repo",
            source="def walk_repo(): walk(); repo()",
            start_line=1, end_line=2,
        )
        chunk_one = make_chunk(
            file_path=str(tmp_path / "src.py"),
            name="just_walk",
            source="def just_walk(): walk()",
            start_line=3, end_line=4,
        )
        index_chunks([chunk_both, chunk_one], src, db_path, client)

        results = keyword_search("walk repo", db_path)
        assert results[0]["name"] == "walk_repo"
