"""
store.py

Manages the ChromaDB vector store and OpenAI embedding pipeline.

Responsibilities:
- Computing SHA-256 hashes of source files for incremental indexing
- Building enriched text for embedding (name + docstring + source)
- Calling OpenAI to embed chunks in batches
- Storing and retrieving chunks from ChromaDB
- Skipping files whose hash has not changed since last index
"""

import hashlib
import os
from pathlib import Path

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from repolens.chunker import Chunk

# Embedding model. text-embedding-3-small is cheap, fast, and high
# quality. 1536-dimensional output vectors. Cost is ~$0.02 per
# 1 million tokens — negligible for any reasonably sized repo.
EMBEDDING_MODEL = "text-embedding-3-small"

# How many chunks we embed in a single OpenAI API call.
# OpenAI accepts up to 2048 inputs per batch. We use 100 as a
# conservative default that keeps individual requests fast and
# avoids hitting rate limits during initial indexing.
EMBED_BATCH_SIZE = 100

# ChromaDB collection names. We use two collections:
# - "chunks" stores the actual code chunks with their embeddings
# - "file_hashes" stores one document per file recording its hash,
#   used to detect which files changed since last index run.
CHUNKS_COLLECTION = "repolens_chunks"
HASHES_COLLECTION = "repolens_hashes"


def _get_client(store_path: str | Path) -> chromadb.ClientAPI:
    """
    Create a persistent ChromaDB client rooted at store_path.

    ChromaDB persists its data to disk at this path. The same path
    across runs gives you the same data — this is how incremental
    indexing works.
    """
    return chromadb.PersistentClient(
        path=str(store_path),
        settings=Settings(anonymized_telemetry=False),
    )


def hash_file(file_path: str | Path) -> str:
    """
    Compute a SHA-256 hash of a file's raw bytes.

    SHA-256 produces a 64-character hex string that is unique to the
    file's content. If even one byte changes, the hash changes. We use
    raw bytes (not text) so encoding differences don't cause false cache
    hits.

    Returns:
        64-character lowercase hex string.
    """
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()


def build_embed_text(chunk: Chunk) -> str:
    """
    Build the text we send to OpenAI for embedding.

    We enrich beyond raw source code by prepending the function or
    class name and docstring when available. This improves retrieval
    because natural language queries map better to natural language
    descriptions than to raw syntax alone.

    Example output for a documented function:
        function: authenticate_user
        Validates user credentials and returns a session token.

        def authenticate_user(token: str) -> Session:
            ...
    """
    parts: list[str] = []
    parts.append(f"{chunk.node_type.replace('_definition', '')}: {chunk.name}")
    if chunk.docstring:
        parts.append(chunk.docstring)
    parts.append(chunk.source)
    return "\n\n".join(parts)


def _embed_texts(texts: list[str], client: OpenAI) -> list[list[float]]:
    """
    Embed a list of texts using OpenAI in batches.

    Returns a list of embedding vectors in the same order as the input.
    Each vector is a list of 1536 floats.
    """
    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        # OpenAI returns embeddings in the same order as input.
        all_embeddings.extend([e.embedding for e in response.data])

    return all_embeddings


def chunk_to_metadata(chunk: Chunk) -> dict:
    """
    Convert a Chunk's metadata fields to a ChromaDB-compatible dict.

    ChromaDB metadata values must be str, int, float, or bool.
    Lists (like calls) must be serialized to a string.
    """
    return {
        "file_path": chunk.file_path,
        "node_type": chunk.node_type,
        "name": chunk.name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "token_count": chunk.token_count,
        "calls": ",".join(chunk.calls),
        "docstring": chunk.docstring or "",
    }


def index_chunks(
    chunks: list[Chunk],
    file_path: str | Path,
    store_path: str | Path,
    openai_client: OpenAI,
    force: bool = False,
) -> dict:
    """
    Embed and store chunks for a single file in ChromaDB.

    Checks the file's current hash against the stored hash. If they
    match and force=False, skips the file entirely — no API calls,
    no writes. If the hash changed or force=True, deletes the old
    chunks for this file and writes the new ones.

    Args:
        chunks: List of Chunk objects from chunk_file().
        file_path: Path to the source file these chunks came from.
        store_path: Directory where ChromaDB persists its data.
        openai_client: Initialized OpenAI client.
        force: If True, re-index even if the hash matches.

    Returns:
        Dict with keys:
            "skipped": True if file was skipped due to hash match.
            "indexed": Number of chunks indexed (0 if skipped).
    """
    file_path = str(Path(file_path).resolve())
    db = _get_client(store_path)
    chunks_col = db.get_or_create_collection(CHUNKS_COLLECTION)
    # embedding_function=None: we only look up hashes by id, never by
    # similarity, so we don't want ChromaDB to auto-embed the documents.
    hashes_col = db.get_or_create_collection(
        HASHES_COLLECTION, embedding_function=None
    )

    current_hash = hash_file(file_path)

    # Check stored hash
    if not force:
        existing = hashes_col.get(ids=[file_path])
        if existing["ids"] and existing["documents"][0] == current_hash:
            return {"skipped": True, "indexed": 0}

    # Delete old chunks for this file before writing new ones.
    # ChromaDB does not support upsert by metadata filter in all versions,
    # so we delete-then-insert to keep the index consistent.
    existing_chunks = chunks_col.get(where={"file_path": file_path})
    if existing_chunks["ids"]:
        chunks_col.delete(ids=existing_chunks["ids"])

    if not chunks:
        # File has no indexable chunks (e.g. only module-level code).
        # Still update the hash so we don't reprocess it next run.
        hashes_col.upsert(
            ids=[file_path],
            documents=[current_hash],
            embeddings=[[0.0]],
        )
        return {"skipped": False, "indexed": 0}

    # Build enriched embed texts and get embeddings from OpenAI.
    texts = [build_embed_text(c) for c in chunks]
    embeddings = _embed_texts(texts, openai_client)

    # Generate stable IDs. We use file_path + start_line so IDs are
    # deterministic across runs for the same chunk position.
    ids = [f"{file_path}:{c.start_line}" for c in chunks]

    chunks_col.add(
        ids=ids,
        documents=[c.source for c in chunks],
        embeddings=embeddings,
        metadatas=[chunk_to_metadata(c) for c in chunks],
    )

    # Update stored hash after successful indexing.
    hashes_col.upsert(
        ids=[file_path],
        documents=[current_hash],
        embeddings=[[0.0]],
    )

    return {"skipped": False, "indexed": len(chunks)}


def query_chunks(
    query_text: str,
    store_path: str | Path,
    openai_client: OpenAI,
    n_results: int = 10,
) -> list[dict]:
    """
    Embed a query and retrieve the top n_results similar chunks.

    Returns a list of result dicts, each containing the chunk's
    source text and all metadata fields. Results are ordered by
    cosine similarity descending (most similar first).

    Args:
        query_text: The user's plain English question.
        store_path: Directory where ChromaDB data lives.
        openai_client: Initialized OpenAI client.
        n_results: How many chunks to retrieve before re-ranking.

    Returns:
        List of dicts with keys: source, file_path, name,
        node_type, start_line, end_line, calls, docstring, distance.
    """
    db = _get_client(store_path)
    chunks_col = db.get_or_create_collection(CHUNKS_COLLECTION)

    # Embed the query using the same model used during indexing.
    # Using a different model would produce vectors in a different
    # space — similarity scores would be meaningless.
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query_text],
    )
    query_vector = response.data[0].embedding

    results = chunks_col.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten ChromaDB's nested response structure into a clean list.
    output: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "source": doc,
            "file_path": meta["file_path"],
            "name": meta["name"],
            "node_type": meta["node_type"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "calls": meta["calls"].split(",") if meta["calls"] else [],
            "docstring": meta["docstring"] or None,
            "distance": dist,
        })

    return output
