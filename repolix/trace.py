"""
trace.py

Call-graph traversal for a named function or class.

Forward trace: BFS outward following calls edges — what does this
function call, what do those call, up to max_depth levels.

Backward trace: reverse lookup — what functions call this one.

Default operation: zero API calls. Pure local computation over
ChromaDB metadata already stored at index time.

Optional --explain flag triggers a single LLM call via answer_trace()
in llm.py to narrate the call chain in plain English.
"""

from collections import deque
from pathlib import Path

from openai import OpenAI

from repolix.store import keyword_search, CHUNKS_COLLECTION, _get_client
from repolix.retriever import display_rel_path_from_meta

TRACE_MAX_DEPTH = 3
TRACE_MAX_NODES = 20


def lookup_chunk_by_name(name: str, store_path) -> dict | None:
    """
    Find a chunk by exact function/class name using keyword_search.

    Uses the same pattern as expand_via_call_graph in retriever.py:
    keyword_search returns candidates, we verify exact name match.

    Returns the first exact match dict or None if not found.
    The returned dict has calls as a list[str] — keyword_search
    already splits the comma-joined string.
    """
    matches = keyword_search(
        query=name,
        store_path=store_path,
        n_results=5,
    )
    for match in matches:
        if match["name"] == name:
            return match
    return None


def forward_trace(
    symbol: str,
    store_path,
    max_depth: int = TRACE_MAX_DEPTH,
    max_nodes: int = TRACE_MAX_NODES,
) -> dict:
    """
    BFS forward traversal from the named symbol following calls edges.

    Algorithm:
      - Queue starts with [(symbol_name, depth=0, parent=None)]
      - visited: set of function names already processed
      - For each node: lookup chunk by name, extract its calls,
        add unvisited calls to queue at depth+1
      - Stop when queue empty, depth >= max_depth, or
        visited count >= max_nodes

    Returns a dict:
        root: str — the starting symbol name
        nodes: dict[str, dict] — name -> {chunk, depth, parent,
               children: list[str], truncated: bool}
        visited_count: int
        truncated: bool — True if max_nodes was hit
        not_found: bool — True if root symbol not in index
    """
    visited: set[str] = set()
    nodes: dict[str, dict] = {}
    truncated = False

    root_chunk = lookup_chunk_by_name(symbol, store_path)
    if root_chunk is None:
        return {
            "root": symbol,
            "nodes": {},
            "visited_count": 0,
            "truncated": False,
            "not_found": True,
        }

    # Queue entries: (name, depth, parent_name)
    queue = deque([(symbol, 0, None)])

    while queue:
        current_name, depth, parent = queue.popleft()

        if current_name in visited:
            # Already visited — add to parent's children as a
            # cycle/already-visited marker but don't re-expand.
            if parent and parent in nodes:
                nodes[parent]["children"].append(current_name)
                nodes[parent]["child_already_visited"] = nodes[parent].get(
                    "child_already_visited", []
                ) + [current_name]
            continue

        if len(visited) >= max_nodes:
            truncated = True
            break

        visited.add(current_name)

        chunk = lookup_chunk_by_name(current_name, store_path)
        nodes[current_name] = {
            "chunk": chunk,
            "depth": depth,
            "parent": parent,
            "children": [],
            "child_already_visited": [],
            "truncated": False,
        }

        if parent and parent in nodes:
            nodes[parent]["children"].append(current_name)

        # Only expand children if we have depth budget remaining.
        if depth < max_depth and chunk is not None:
            calls = chunk.get("calls", [])
            for called_name in calls:
                if not called_name:
                    continue
                if called_name in visited:
                    # Record cycle/already-visited reference for tree rendering
                    # without re-expanding the node.
                    nodes[current_name]["children"].append(called_name)
                    nodes[current_name]["child_already_visited"].append(called_name)
                else:
                    queue.append((called_name, depth + 1, current_name))

        # Mark node as truncated if it has calls we won't expand.
        if depth >= max_depth and chunk and chunk.get("calls"):
            nodes[current_name]["truncated"] = True

    return {
        "root": symbol,
        "nodes": nodes,
        "visited_count": len(visited),
        "truncated": truncated,
        "not_found": False,
    }


def backward_trace(symbol: str, store_path) -> list[dict]:
    """
    Find all chunks that call the named symbol.

    O(n) scan over all stored chunks — fetches all metadata
    without vectors, checks each chunk's calls list for the
    target name. No BFS needed because we're going one level up.

    Returns list of caller dicts, each with name, file_rel_path,
    start_line. Empty list if no callers found.
    """
    from repolix.tour import get_all_chunks

    all_chunks = get_all_chunks(store_path)
    callers = []

    for chunk in all_chunks:
        if symbol in chunk.get("calls", []):
            callers.append({
                "name": chunk["name"],
                "file_rel_path": chunk["file_rel_path"],
                "file_path": chunk["file_path"],
                "start_line": chunk["start_line"],
            })

    return callers


def format_trace_tree(trace_result: dict) -> str:
    """
    Render the forward trace result as a Unicode tree string.

    Uses box-drawing characters: ├── │ └──
    Recursively renders children at each depth level.
    Already-visited nodes shown as [already visited].
    Truncated nodes shown with a hint to increase depth.

    Returns a multiline string ready for Rich console or plain print.
    """
    if trace_result.get("not_found"):
        return f"Symbol '{trace_result['root']}' not found in index."

    nodes = trace_result["nodes"]
    root = trace_result["root"]

    if root not in nodes:
        return f"Symbol '{root}' not found in index."

    lines = []

    def render_node(name: str, prefix: str, is_last: bool):
        node = nodes.get(name)
        connector = "└── " if is_last else "├── "

        if node and node.get("chunk"):
            chunk = node["chunk"]
            file_info = display_rel_path_from_meta(chunk)
            line = chunk.get("start_line", "?")
            lines.append(f"{prefix}{connector}{name}  [{file_info}:{line}]")
        else:
            lines.append(f"{prefix}{connector}{name}  [not indexed]")

        if node is None:
            return

        child_prefix = prefix + ("    " if is_last else "│   ")

        already = node.get("child_already_visited", [])
        real_children = [c for c in node["children"] if c not in already]
        all_display = real_children + [f"{c} [already visited]" for c in already]

        if node.get("truncated"):
            all_display.append("[truncated — use --depth to go deeper]")

        for i, child in enumerate(all_display):
            child_is_last = (i == len(all_display) - 1)
            if isinstance(child, str) and (
                "[already visited]" in child or "[truncated" in child
            ):
                conn = "└── " if child_is_last else "├── "
                lines.append(f"{child_prefix}{conn}{child}")
            else:
                render_node(child, child_prefix, child_is_last)

    # Render root node without a connector prefix.
    root_node = nodes[root]
    if root_node and root_node.get("chunk"):
        chunk = root_node["chunk"]
        file_info = display_rel_path_from_meta(chunk)
        line = chunk.get("start_line", "?")
        lines.append(f"{root}  [{file_info}:{line}]")
    else:
        lines.append(f"{root}  [not indexed]")

    root_children = [
        c for c in root_node["children"]
        if c not in root_node.get("child_already_visited", [])
    ]
    already = root_node.get("child_already_visited", [])
    all_root_display = root_children + [f"{c} [already visited]" for c in already]
    if root_node.get("truncated"):
        all_root_display.append("[truncated — use --depth to go deeper]")

    for i, child in enumerate(all_root_display):
        is_last = (i == len(all_root_display) - 1)
        if isinstance(child, str) and (
            "[already visited]" in child or "[truncated" in child
        ):
            conn = "└── " if is_last else "├── "
            lines.append(f"    {conn}{child}")
        else:
            render_node(child, "", is_last)

    return "\n".join(lines)


def run_trace(
    symbol: str,
    store_path,
    max_depth: int = TRACE_MAX_DEPTH,
    max_nodes: int = TRACE_MAX_NODES,
    include_backward: bool = True,
    openai_client: OpenAI | None = None,
    explain: bool = False,
) -> dict:
    """
    Orchestrate the full trace pipeline.

    Returns dict with keys:
        symbol: str
        forward: dict — result from forward_trace()
        backward: list[dict] — callers from backward_trace()
        tree_str: str — formatted tree from format_trace_tree()
        explanation: str | None — LLM explanation if explain=True
        error: str | None
    """
    forward = forward_trace(symbol, store_path, max_depth, max_nodes)

    if forward.get("not_found"):
        return {
            "symbol": symbol,
            "forward": forward,
            "backward": [],
            "tree_str": format_trace_tree(forward),
            "explanation": None,
            "error": f"'{symbol}' was not found in the index.",
        }

    tree_str = format_trace_tree(forward)
    backward = backward_trace(symbol, store_path) if include_backward else []

    explanation = None
    if explain and openai_client is not None:
        from repolix.llm import answer_trace
        explanation = answer_trace(tree_str, backward, symbol, openai_client)

    return {
        "symbol": symbol,
        "forward": forward,
        "backward": backward,
        "tree_str": tree_str,
        "explanation": explanation,
        "error": None,
    }
