"""
tour.py

Proactive orientation briefing for an indexed repository.

Works in two phases:
  Phase 1 — Local computation, zero API calls:
    Scan all chunk metadata from ChromaDB, count inbound references,
    detect entry points, select the top 8 structurally significant
    chunks, and build a context summary.

  Phase 2 — Single LLM call:
    Send the context summary to a tour-specific system prompt and
    parse the fixed-structure response into sections.

Tour never embeds a query. It operates entirely on metadata
already stored in ChromaDB.
"""

from pathlib import Path

from openai import OpenAI

from repolix.store import CHUNKS_COLLECTION, _get_client

ENTRY_POINT_FILES = frozenset({
    "main.py", "app.py", "__main__.py", "index.py",
    "server.py", "cli.py", "run.py", "manage.py",
    "index.ts", "index.js", "server.ts", "server.js",
    "app.ts", "app.js", "main.ts", "main.js",
})

ENTRY_POINT_FUNCTIONS = frozenset({
    "main", "run", "start", "create_app", "init",
    "setup", "bootstrap", "serve", "launch",
})

BUILTIN_NAMES: frozenset[str] = frozenset({
    # Python builtins and common stdlib calls
    "get", "append", "extend", "update", "pop", "remove",
    "len", "str", "int", "float", "bool", "list", "dict",
    "set", "tuple", "print", "range", "enumerate", "zip",
    "map", "filter", "sorted", "reversed", "isinstance",
    "hasattr", "getattr", "setattr", "open", "read", "write",
    "join", "split", "strip", "lstrip", "rstrip", "format",
    "replace", "encode", "decode", "items", "keys", "values",
    "Path", "resolve", "exists", "mkdir", "unlink", "stat",
    "any", "all", "next", "iter", "sum", "min", "max",
    "super", "vars", "dir", "type", "id", "hash", "repr",
    "lower", "upper", "startswith", "endswith", "count",
    # JS/TS builtins and common patterns
    "push", "forEach", "find", "findIndex", "includes",
    "indexOf", "slice", "splice", "concat", "reduce",
    "JSON", "Promise", "console", "Object", "Array",
    "parseInt", "parseFloat", "toString", "valueOf",
    "addEventListener", "querySelector", "getElementById",
    "fetch", "then", "catch", "finally",
})

TOUR_MAX_CHUNKS = 8


def get_all_chunks(
    store_path: str | Path,
    path_prefix: str | None = None,
) -> list[dict]:
    """
    Retrieve all chunk metadata from ChromaDB without embeddings.

    Uses collection.get() with include=["metadatas"] — no vectors
    loaded, no API calls. Returns a list of metadata dicts.

    If path_prefix is provided, only return chunks whose
    file_rel_path starts with that prefix (case-insensitive).

    The "calls" field is stored as a comma-joined string in ChromaDB.
    It is split here and returned as list[str], never as a string.
    """
    db = _get_client(store_path)
    col = db.get_or_create_collection(CHUNKS_COLLECTION)
    result = col.get(include=["metadatas"])

    chunks = []
    for meta in result["metadatas"]:
        calls_raw = meta.get("calls", "")
        calls = [c for c in calls_raw.split(",") if c.strip()] if calls_raw else []

        file_rel = meta.get("file_rel_path", "")

        if path_prefix:
            norm_prefix = path_prefix.lower().rstrip("/")
            if not file_rel.lower().startswith(norm_prefix):
                continue

        chunks.append({
            "name": meta.get("name", ""),
            "node_type": meta.get("node_type", ""),
            "file_rel_path": file_rel,
            "file_path": meta.get("file_path", ""),
            "start_line": meta.get("start_line", 0),
            "end_line": meta.get("end_line", 0),
            "calls": calls,
            "docstring": meta.get("docstring", ""),
            "parent_class": meta.get("parent_class", ""),
            "source_text": meta.get("source_text", ""),
            "is_truncated": meta.get("is_truncated", False),
        })

    return chunks


def compute_inbound_counts(chunks: list[dict]) -> dict[str, int]:
    """
    Count how many chunks call each function name.

    For each chunk, iterate its calls list and increment the counter
    for each called name. A function called by 8 chunks has an
    inbound count of 8 — it is architecturally central.

    Returns dict mapping function_name -> inbound_count.
    Only names with count >= 1 are included.
    """
    counts: dict[str, int] = {}
    for chunk in chunks:
        for called_name in chunk["calls"]:
            if called_name and called_name not in BUILTIN_NAMES:
                counts[called_name] = counts.get(called_name, 0) + 1
    return counts


def identify_entry_points(
    chunks: list[dict],
    inbound_counts: dict[str, int],
) -> list[dict]:
    """
    Identify likely entry point chunks using two signals.

    Signal 1 — Heuristic: chunk's file name is in ENTRY_POINT_FILES
    or chunk's name is in ENTRY_POINT_FUNCTIONS.

    Signal 2 — Graph: chunk's name has zero inbound references
    (nothing calls it) AND it has outbound calls (it calls something).
    This identifies source nodes in the call graph.

    Signal 1 matches are ranked higher. Returns deduplicated list,
    capped at 3 entry points.
    """
    signal1 = []
    signal2 = []
    seen_names: set[str] = set()

    for chunk in chunks:
        file_name = Path(chunk["file_rel_path"]).name.lower()
        name = chunk["name"]

        if name in seen_names:
            continue

        is_heuristic = (
            file_name in ENTRY_POINT_FILES or
            name in ENTRY_POINT_FUNCTIONS
        )

        is_graph_source = (
            inbound_counts.get(name, 0) == 0 and
            len(chunk["calls"]) > 0
        )

        if is_heuristic:
            signal1.append(chunk)
            seen_names.add(name)
        elif is_graph_source:
            signal2.append(chunk)
            seen_names.add(name)

    return (signal1 + signal2)[:3]


def select_tour_chunks(
    chunks: list[dict],
    inbound_counts: dict[str, int],
    entry_points: list[dict],
) -> list[dict]:
    """
    Select up to TOUR_MAX_CHUNKS chunks for the LLM context.

    Strategy — maximize structural diversity:
    1. Always include entry point chunks first.
    2. Fill remaining slots with highest-inbound-count chunks,
       one per unique file_rel_path (diversity constraint).
       This prevents 8 chunks from the same file dominating.
    3. If slots remain, fill with next-highest inbound count
       chunks regardless of file diversity.

    Returns list of up to TOUR_MAX_CHUNKS chunks, no duplicates.
    """
    selected = list(entry_points)
    selected_names = {c["name"] for c in selected}
    selected_files = {c["file_rel_path"] for c in selected}

    sorted_chunks = sorted(
        chunks,
        key=lambda c: inbound_counts.get(c["name"], 0),
        reverse=True,
    )

    # Pass 1: one chunk per file, highest inbound count first
    for chunk in sorted_chunks:
        if len(selected) >= TOUR_MAX_CHUNKS:
            break
        if chunk["name"] in selected_names:
            continue
        if chunk["file_rel_path"] in selected_files:
            continue
        selected.append(chunk)
        selected_names.add(chunk["name"])
        selected_files.add(chunk["file_rel_path"])

    # Pass 2: fill remaining slots regardless of file diversity
    for chunk in sorted_chunks:
        if len(selected) >= TOUR_MAX_CHUNKS:
            break
        if chunk["name"] in selected_names:
            continue
        selected.append(chunk)
        selected_names.add(chunk["name"])

    return selected[:TOUR_MAX_CHUNKS]


def build_tour_context(
    chunks: list[dict],
    inbound_counts: dict[str, int],
    repo_path: str | Path,
    path_prefix: str | None = None,
    _all_chunks: list[dict] | None = None,
) -> str:
    """
    Format selected chunks and structural analysis into a context
    string for the LLM tour prompt.

    Includes a structural summary (top functions by inbound count)
    and source previews (first 20 lines) for each selected chunk.

    _all_chunks: Optional full corpus used to resolve file paths for the
    top-inbound summary. When provided, functions that are highly referenced
    but didn't make the selected-8 cut still show their correct file path
    rather than "unknown". Defaults to searching only the selected chunks.
    """
    repo_name = Path(repo_path).name
    scope = path_prefix if path_prefix else "full repo"

    top_by_inbound = sorted(
        inbound_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    # Use the full corpus for lookup so heavily-referenced functions that
    # didn't make the selected-8 cut still resolve to their correct file path.
    lookup_pool = _all_chunks if _all_chunks is not None else chunks

    lines = [
        f"REPO: {repo_name}",
        f"SCOPE: {scope}",
        f"FILES: {len({c['file_rel_path'] for c in chunks})}",
        "",
        "TOP FUNCTIONS BY INBOUND REFERENCES:",
    ]

    for name, count in top_by_inbound:
        chunk_for_name = next(
            (c for c in lookup_pool if c["name"] == name), None
        )
        file_info = chunk_for_name["file_rel_path"] if chunk_for_name else "unknown"
        lines.append(f"  {name} ({file_info}) — called by {count} functions")

    lines.append("")
    lines.append("SELECTED CHUNKS:")

    for i, chunk in enumerate(chunks, 1):
        source_lines = chunk["source_text"].splitlines()
        preview = "\n".join(source_lines[:20])
        if len(source_lines) > 20:
            preview += f"\n... ({len(source_lines) - 20} more lines)"

        lines.append(
            f"\n[{i}] {chunk['name']} ({chunk['node_type']}) "
            f"— {chunk['file_rel_path']}:{chunk['start_line']}-{chunk['end_line']}"
        )
        if chunk.get("docstring"):
            lines.append(f"Docstring: {chunk['docstring'][:200]}")
        lines.append(preview)
        lines.append("---")

    return "\n".join(lines)


def generate_tour(
    store_path: str | Path,
    repo_path: str | Path,
    openai_client: OpenAI,
    path_prefix: str | None = None,
) -> dict:
    """
    Run the full tour pipeline and return a structured briefing.

    Phase 1 is local computation only (no API calls). Phase 2 makes
    a single LLM chat completion call. No embeddings are created.

    Returns dict with keys:
        briefing: str | None — full LLM response text
        briefing_sections: dict | None — parsed sections
        entry_points: list[dict] — identified entry point chunks
        top_functions: list[tuple] — top 5 by inbound count
        chunk_count: int — total chunks in scope
        error: str | None — error message if something failed
    """
    store_path = Path(store_path)

    if not (store_path / "chroma.sqlite3").exists():
        return {
            "briefing": None,
            "briefing_sections": None,
            "entry_points": [],
            "top_functions": [],
            "chunk_count": 0,
            "error": (
                f"No index found at {store_path}. "
                f"Run: repolix index {repo_path}"
            ),
        }

    all_chunks = get_all_chunks(store_path, path_prefix=path_prefix)

    if not all_chunks:
        scope = path_prefix or "this repo"
        return {
            "briefing": None,
            "briefing_sections": None,
            "entry_points": [],
            "top_functions": [],
            "chunk_count": 0,
            "error": f"No indexed chunks found for {scope}.",
        }

    inbound_counts = compute_inbound_counts(all_chunks)
    entry_points = identify_entry_points(all_chunks, inbound_counts)
    selected = select_tour_chunks(all_chunks, inbound_counts, entry_points)
    context = build_tour_context(
        selected, inbound_counts, repo_path, path_prefix,
        _all_chunks=all_chunks,
    )

    from repolix.llm import answer_tour
    result = answer_tour(context, openai_client)

    top_functions = sorted(
        inbound_counts.items(),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    return {
        "briefing": result.get("briefing"),
        "briefing_sections": result.get("briefing_sections"),
        "entry_points": entry_points,
        "top_functions": top_functions,
        "chunk_count": len(all_chunks),
        "error": None,
    }
