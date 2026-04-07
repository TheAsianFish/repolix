"""
retriever.py

Orchestrates the full retrieval pipeline for a user query.

Milestone 6: hybrid search via Reciprocal Rank Fusion (RRF).

Pipeline:
  1. Receive plain English query string
  2. Run vector similarity search via store.query_chunks (semantic)
  3. Run keyword search via store.keyword_search (exact match)
  4. Merge both ranked lists using RRF into one unified ranking
  5. Re-rank the merged list using metadata signals
  6. Return top 5 chunks with full metadata and scores
"""

from pathlib import Path
from openai import OpenAI
from repolix.store import query_chunks, keyword_search

# How many results to retrieve from each search before merging.
# Over-retrieving gives RRF and re-ranking room to work.
RETRIEVE_N = 10

# Final number of chunks returned to the caller after all ranking.
RETURN_N = 5

# RRF smoothing constant. 60 is the conventional default established
# in the original RRF paper (Cormack et al., 2009). It prevents the
# top-ranked result from dominating too heavily over lower ranks.
RRF_K = 60


def retrieve(
    query: str,
    store_path: str | Path,
    openai_client,
) -> list[dict]:
    """
    Run the full hybrid retrieval pipeline for a plain English query.

    Combines vector similarity search and keyword search via RRF,
    applies metadata re-ranking, then expands via call graph to
    capture implementation functions that vector search misses.

    Args:
        query: Plain English question from the user.
        store_path: Path to the ChromaDB persistence directory.
        openai_client: Initialized OpenAI client.

    Returns:
        List of up to RETURN_N + expansion results sorted by score.
        Each result contains: source, file_path, name, node_type,
        start_line, end_line, calls, docstring, distance,
        rrf_score, rerank_score.
    """
    vector_results = query_chunks(
        query_text=query,
        store_path=store_path,
        openai_client=openai_client,
        n_results=RETRIEVE_N,
    )

    keyword_results = keyword_search(
        query=query,
        store_path=store_path,
        n_results=RETRIEVE_N,
    )

    if not vector_results and not keyword_results:
        return []

    merged = reciprocal_rank_fusion(vector_results, keyword_results)
    ranked = rerank(query, merged)
    top = ranked[:RETURN_N]

    # Expand via call graph to capture called functions that ranked
    # too low during vector search. This pulls _walk_tree into
    # results when chunk_file is retrieved and calls it.
    expanded = expand_via_call_graph(top, store_path, openai_client)

    return expanded


def reciprocal_rank_fusion(
    vector_results: list[dict],
    keyword_results: list[dict],
) -> list[dict]:
    """
    Merge two ranked result lists using Reciprocal Rank Fusion.

    RRF avoids the problem of combining scores from incompatible
    scales (cosine distance vs keyword match count) by ignoring
    raw scores entirely and operating only on rank positions.

    Formula: RRF_score = sum(1 / (k + rank)) across all lists
    where rank is 1-indexed position in each list.

    A result appearing at rank 2 in both lists outscores a result
    appearing at rank 1 in only one list when k=60:
      Both lists:  1/(60+2) + 1/(60+2) = 0.0323
      One list:    1/(60+1) + 0         = 0.0164

    This rewards consistency across search methods over dominance
    in a single method.

    Args:
        vector_results: Ranked list from vector similarity search.
        keyword_results: Ranked list from keyword search.

    Returns:
        Merged list of unique results sorted by rrf_score descending,
        with rrf_score added to each result dict.
    """
    # Use file_path + start_line as the unique key for deduplication.
    # This matches the chunk ID format used in ChromaDB.
    scores: dict[str, float] = {}
    result_map: dict[str, dict] = {}

    for rank, result in enumerate(vector_results, start=1):
        key = f"{result['file_path']}:{result['start_line']}"
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        result_map[key] = result

    for rank, result in enumerate(keyword_results, start=1):
        key = f"{result['file_path']}:{result['start_line']}"
        scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
        result_map[key] = result

    # Attach rrf_score to each result and sort descending.
    merged = []
    for key, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        merged.append({**result_map[key], "rrf_score": score})

    return merged


def rerank(query: str, results: list[dict]) -> list[dict]:
    """
    Re-rank results using metadata signals on top of rrf_score.

    Uses rrf_score as the base instead of raw vector distance.
    Boosts are identical to Milestone 5 — name, file path,
    docstring, and call graph matches.

    Scoring:
      base_score  = rrf_score (already accounts for both search methods)
      +0.3 if any query token appears in chunk name
      +0.2 if any query token appears in file path stem
      +0.15 if any query token appears in docstring
      +0.1 per query token appearing in calls list

    Args:
        query: The original plain English query string.
        results: List of result dicts from reciprocal_rank_fusion.

    Returns:
        Results sorted by rerank_score descending, with rerank_score
        added to each dict.
    """
    query_tokens = [
        t.strip(".,?!:;\"'()[]{}").lower()
        for t in query.split()
        if t.strip(".,?!:;\"'()[]{}").lower()
    ]

    scored = []
    for result in results:
        # Base score is now RRF score, not raw vector distance.
        # RRF score is already a merged signal from both search methods.
        base_score = result.get("rrf_score", 1.0 - result.get("distance", 0.5))

        name = result["name"].lower()
        file_stem = Path(result["file_path"]).stem.lower()
        docstring = (result["docstring"] or "").lower()
        calls = [c.lower() for c in result["calls"]]

        boost = 0.0
        for token in query_tokens:
            if token in name:
                boost += 0.3
            if token in file_stem:
                boost += 0.2
            if token in docstring:
                boost += 0.15
            if any(token in call for call in calls):
                boost += 0.1

        scored.append({**result, "rerank_score": base_score + boost})

    return sorted(scored, key=lambda r: r["rerank_score"], reverse=True)


def expand_via_call_graph(
    results: list[dict],
    store_path: str | Path,
    openai_client,
    max_expansion: int = 3,
) -> list[dict]:
    """
    Expand retrieved results by fetching chunks for functions that
    are called by already-retrieved chunks.

    When chunk_file is retrieved and its calls list contains
    _walk_tree, this function fetches _walk_tree's chunk and adds
    it to the results if not already present. This closes the gap
    where implementation details live in called functions that
    vector search did not rank highly enough.

    Args:
        results: Current list of retrieved and ranked result dicts.
        store_path: ChromaDB store path.
        openai_client: OpenAI client (passed for interface consistency,
            not used directly — keyword_search is call-free).
        max_expansion: Maximum number of additional chunks to add.

    Returns:
        Expanded results list with called chunks appended.
        Original order is preserved — expansions go at the end.
    """
    existing_names = {r["name"] for r in results}
    expansions: list[dict] = []

    for result in results:
        if len(expansions) >= max_expansion:
            break

        for called_name in result.get("calls", []):
            if called_name in existing_names:
                continue
            if len(expansions) >= max_expansion:
                break

            # Search for the called function by exact name.
            # keyword_search does substring matching so we verify
            # the name matches exactly before accepting the result.
            matches = keyword_search(
                query=called_name,
                store_path=store_path,
                n_results=3,
            )

            for match in matches:
                if match["name"] == called_name:
                    # Low score so expansions never displace primary
                    # results — they always appear at the end.
                    expansions.append({
                        **match,
                        "rrf_score": 0.005,
                        "rerank_score": 0.005,
                    })
                    existing_names.add(called_name)
                    break

    return results + expansions


def format_results(results: list[dict]) -> str:
    """
    Format retrieval results as a human-readable string for CLI
    output and LLM context construction.

    Args:
        results: List of result dicts from retrieve().

    Returns:
        Multi-line string ready for printing or passing to LLM.
    """
    if not results:
        return "No results found."

    lines: list[str] = []
    for i, result in enumerate(results, 1):
        file_path = result["file_path"]
        name = result["name"]
        start = result["start_line"]
        end = result["end_line"]
        score = result.get("rerank_score", result.get("rrf_score", 0.0))
        source = result["source"]

        lines.append(f"── Result {i} ──────────────────────────────")
        lines.append(f"File:     {file_path}")
        lines.append(f"Function: {name}  (lines {start}–{end})")
        lines.append(f"Score:    {score:.4f}")
        lines.append("")
        lines.append(source)
        lines.append("")

    return "\n".join(lines)
