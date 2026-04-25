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

from repolix.chunker import Chunk

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
CHUNKS_COLLECTION = "repolix_chunks"
HASHES_COLLECTION = "repolix_hashes"


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


def build_embed_text(chunk: Chunk, file_rel_path: str = "") -> str:
    """
    Build the text we send to OpenAI for embedding and store as the
    searchable document in ChromaDB.

    We enrich beyond raw source code by prepending the file path,
    function or class name, and docstring when available. Storing this
    enriched text as the ChromaDB document means keyword search can
    find chunks by file name (e.g. "chunker") or function name even
    when those strings don't appear in the raw source body.

    Example output for a documented function in repolix/chunker.py:
        file: repolix/chunker.py
        function: chunk_file
        Parse a Python file and return a list of Chunk objects.

        def chunk_file(file_path: str | Path) -> list[Chunk]:
            ...
    """
    parts: list[str] = []
    if file_rel_path:
        parts.append(f"file: {file_rel_path}")
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


def chunk_to_metadata(chunk: Chunk, repo_root: str | None = None) -> dict:
    """
    Convert a Chunk's metadata to a ChromaDB-compatible dict.

    All values must be str, int, float, or bool.
    Lists are comma-joined. None becomes empty string.

    repo_root: if provided, stores relative path from repo root
    as file_rel_path for better disambiguation in re-ranking.
    """
    if repo_root:
        try:
            file_rel_path = str(
                Path(chunk.file_path).relative_to(repo_root)
            )
        except ValueError:
            file_rel_path = chunk.file_path
    else:
        file_rel_path = chunk.file_path

    return {
        "file_path": chunk.file_path,
        "file_rel_path": file_rel_path,
        "node_type": chunk.node_type,
        "name": chunk.name,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "token_count": chunk.token_count,
        "calls": ",".join(chunk.calls),
        "docstring": chunk.docstring or "",
        "parent_class": chunk.parent_class or "",
        "is_truncated": chunk.is_truncated,
        # Raw source stored separately so query_chunks and keyword_search
        # can return clean source for display while ChromaDB documents
        # hold the enriched text (file path + name + docstring + source)
        # that enables keyword search to find chunks by file/function name.
        "source_text": chunk.source,
    }


def index_chunks(
    chunks: list[Chunk],
    file_path: str | Path,
    store_path: str | Path,
    openai_client: OpenAI,
    force: bool = False,
    repo_root: str | None = None,
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

    # Compute the relative path once for all chunks in this file.
    # Used to prefix the enriched document text so keyword search can
    # find chunks by file name (e.g. "chunker" finds chunker.py functions).
    if repo_root:
        try:
            embed_rel_path = str(Path(file_path).relative_to(repo_root))
        except ValueError:
            embed_rel_path = file_path
    else:
        embed_rel_path = file_path

    # Build enriched embed texts and get embeddings from OpenAI.
    # The same enriched text is stored as the ChromaDB document so that
    # keyword search (which uses $contains on the document) benefits from
    # the file path, function name, and docstring — not just raw source.
    texts = [build_embed_text(c, file_rel_path=embed_rel_path) for c in chunks]
    embeddings = _embed_texts(texts, openai_client)

    # Generate stable IDs. We use file_path + start_line so IDs are
    # deterministic across runs for the same chunk position.
    ids = [f"{file_path}:{c.start_line}" for c in chunks]

    chunks_col.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=[chunk_to_metadata(c, repo_root=repo_root) for c in chunks],
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
    # Use source_text from metadata (raw source) rather than the document
    # (enriched text with file path/name prefix) so callers display and
    # send clean code to the LLM. Fall back to doc for old index entries.
    output: list[dict] = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "source": meta.get("source_text") or doc,
            "file_path": meta["file_path"],
            "file_rel_path": meta.get("file_rel_path", meta["file_path"]),
            "name": meta["name"],
            "node_type": meta["node_type"],
            "start_line": meta["start_line"],
            "end_line": meta["end_line"],
            "token_count": meta.get("token_count", 0),
            "calls": meta["calls"].split(",") if meta["calls"] else [],
            "docstring": meta["docstring"] or None,
            "parent_class": meta.get("parent_class") or None,
            "is_truncated": meta.get("is_truncated", False),
            "distance": dist,
        })

    return output


def keyword_search(
    query: str,
    store_path: str | Path,
    n_results: int = 10,
) -> list[dict]:
    """
    Search stored chunks using keyword matching on source text,
    function names, and docstrings.

    Splits the query into tokens and searches ChromaDB for chunks
    whose stored document contains any token as a substring.
    Results are deduplicated and returned in match-count order —
    chunks matching more tokens rank higher.

    Unlike vector search, this requires no OpenAI API call.
    It catches exact identifier matches that vector search misses.

    Args:
        query: Plain English query string.
        store_path: Path to the ChromaDB persistence directory.
        n_results: Maximum results to return.

    Returns:
        List of result dicts in the same format as query_chunks,
        with distance set to 0.0 (not applicable for keyword search).
    """
    db = _get_client(store_path)
    chunks_col = db.get_or_create_collection(CHUNKS_COLLECTION)

    tokens = [
        t.strip(".,?!:;\"'()[]{}").lower()
        for t in query.split()
        if t.strip(".,?!:;\"'()[]{}").lower()
        and len(t.strip(".,?!:;\"'()[]{}")) >= 2
    ]

    if not tokens:
        return []

    # Search for each token separately and track how many tokens
    # each chunk matches. More token matches = more relevant.
    match_counts: dict[str, int] = {}
    result_map: dict[str, dict] = {}

    for token in tokens:
        try:
            results = chunks_col.get(
                where_document={"$contains": token},
                include=["documents", "metadatas"],
            )
        except Exception:
            continue

        for doc, meta in zip(results["documents"], results["metadatas"]):
            key = f"{meta['file_path']}:{meta['start_line']}"
            match_counts[key] = match_counts.get(key, 0) + 1
            result_map[key] = {
                "source": meta.get("source_text") or doc,
                "file_path": meta["file_path"],
                "name": meta["name"],
                "node_type": meta["node_type"],
                "start_line": meta["start_line"],
                "end_line": meta["end_line"],
                "calls": meta["calls"].split(",") if meta["calls"] else [],
                "docstring": meta["docstring"] or None,
                "is_truncated": meta.get("is_truncated", False),
                "distance": 0.0,
            }

    # Sort by match count descending — more token matches rank higher.
    sorted_keys = sorted(
        match_counts.keys(),
        key=lambda k: match_counts[k],
        reverse=True,
    )

    return [result_map[k] for k in sorted_keys[:n_results]]


def index_repo(
    repo_path: str | Path,
    store_path: str | Path,
    openai_client: OpenAI,
    force: bool = False,
    progress_callback=None,
    exclude_tests: bool = True,
) -> dict:
    """
    Index an entire repository end to end.

    Walks the repo, chunks every Python file, embeds the chunks,
    and stores everything in ChromaDB. Skips files whose hash has
    not changed since the last index run unless force=True.

    This is the primary entry point for the CLI and FastAPI backend.

    Args:
        repo_path: Path to the repository root.
        store_path: Path to the ChromaDB persistence directory.
        openai_client: Initialized OpenAI client.
        force: If True, re-index all files regardless of hash.
        progress_callback: Optional callable(current, total, file_path)
            called after each file is processed. Used for progress bars.

    Returns:
        Dict with keys:
            total_files: int — files found by walker
            indexed: int — files actually re-embedded
            skipped: int — files skipped due to hash match
            total_chunks: int — chunks stored across all indexed files
            errors: list[str] — files that failed with error messages
    """
    from repolix.walker import walk_repo
    from repolix.chunker import chunk_file

    repo_path = Path(repo_path).resolve()
    store_path = Path(store_path).resolve()
    repo_root = str(repo_path)

    files = walk_repo(repo_path, exclude_tests=exclude_tests)
    total = len(files)

    stats = {
        "total_files": total,
        "indexed": 0,
        "skipped": 0,
        "total_chunks": 0,
        "cleaned": 0,
        "errors": [],
    }

    for i, file_path in enumerate(files):
        try:
            chunks = chunk_file(file_path)
            result = index_chunks(
                chunks=chunks,
                file_path=file_path,
                store_path=store_path,
                openai_client=openai_client,
                force=force,
                repo_root=repo_root,
            )

            if result["skipped"]:
                stats["skipped"] += 1
            else:
                stats["indexed"] += 1
                stats["total_chunks"] += result["indexed"]

        except Exception as e:
            import traceback
            stats["errors"].append(
                f"{file_path}: {type(e).__name__}: {e}\n"
                f"{traceback.format_exc()}"
            )

        if progress_callback:
            progress_callback(i + 1, total, str(file_path))

    # Garbage-collect orphaned entries — chunks and hashes for files
    # that no longer exist in the repo (deleted, renamed, moved).
    # walk_repo only returns currently existing files, so any stored
    # hash whose ID is not in the current file set is stale.
    current_paths = {str(Path(f).resolve()) for f in files}
    db = _get_client(store_path)
    chunks_col = db.get_or_create_collection(CHUNKS_COLLECTION)
    hashes_col = db.get_or_create_collection(
        HASHES_COLLECTION, embedding_function=None
    )

    stored = hashes_col.get(include=[])  # IDs only — no documents needed
    orphaned = [p for p in stored["ids"] if p not in current_paths]

    for orphaned_path in orphaned:
        existing = chunks_col.get(where={"file_path": orphaned_path})
        if existing["ids"]:
            chunks_col.delete(ids=existing["ids"])
        hashes_col.delete(ids=[orphaned_path])
        stats["cleaned"] += 1

    return stats
