"""
llm.py

Constructs prompts from retrieved chunks, calls gpt-5.4-mini,
and parses the response into a structured answer with citations.

The citation format uses numbered labels [1], [2] etc. that the
LLM is instructed to use inline. We map those labels back to
real file paths and line numbers after parsing the response.
"""

import re

from openai import OpenAI

from repolix.retriever import display_rel_path_from_meta

LLM_MODEL = "gpt-5.4-mini"

# Maximum chunks to send to the LLM. Set to 8 to include up to
# 5 primary results + 3 call graph expansions from expand_via_call_graph.
# 8 * 300 tokens = 2400 tokens of context, still well within budget.
MAX_CONTEXT_CHUNKS = 8

SYSTEM_PROMPT = """You are a senior engineer helping a developer
navigate an unfamiliar codebase. You have been given the most
relevant code chunks retrieved from that codebase.

Respond in exactly this structure. Do not add extra sections.
Do not change the header names.

**Answer:** [1-2 sentences maximum. Name the exact file and
function. Be specific. No hedging.]

**How it works:** [2-3 sentences. Explain the mechanism and
the reasoning behind it — not just what the code does, but
why it works that way. This is the insight a senior engineer
would share.]

**Where to look next:** [1-2 sentences. Only include this
section if your answer is incomplete or the developer will
need to look further. If your answer is complete, omit this
section entirely — do not include the header.]

Rules:
- Cite chunks inline using [1], [2] etc. every time your
  answer references that chunk's behavior or code.
- Only cite a chunk if your answer actually references it.
- Do not say "based on the provided chunks" or "from the
  context" or "I couldn't find" or "the chunks don't contain".
- If retrieval is incomplete, tell the developer what to
  search for next and where to start — you are a guide,
  not a search engine returning no results.
- Never fabricate file names, function names, or line numbers
  that are not in the provided chunks.

End your response with a CITATIONS block in exactly this format:
CITATIONS
[1] file_path:start_line-end_line (function_name)
[2] file_path:start_line-end_line (function_name)
Only list labels you actually cited inline above.
"""


def build_prompt(query: str, results: list[dict]) -> tuple[str, list[dict]]:
    """
    Build the user prompt for the LLM from a query and retrieved chunks.

    Labels each chunk [1] through [N] and formats them as readable
    blocks. Returns both the prompt string and the labeled chunks
    list so citations can be resolved after parsing the response.

    Args:
        query: The user's plain English question.
        results: List of result dicts from retriever.retrieve().
                 Only the first MAX_CONTEXT_CHUNKS are used.

    Returns:
        Tuple of (prompt_string, labeled_chunks) where labeled_chunks
        is a list of dicts with label, file_path, start_line, end_line,
        name added for citation resolution.
    """
    chunks = results[:MAX_CONTEXT_CHUNKS]
    labeled: list[dict] = []

    chunk_blocks: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        label = f"[{i}]"
        file_rel = display_rel_path_from_meta(chunk)
        start = chunk["start_line"]
        end = chunk["end_line"]
        name = chunk["name"]
        source = chunk["source"]
        parent = chunk.get("parent_class")

        context = f"{name}"
        if parent:
            context = f"{parent}.{name}"

        block = (
            f"CHUNK {label}\n"
            f"File: {file_rel}  Lines: {start}-{end}\n"
            f"Function: {context}\n"
            f"---\n"
            f"{source}"
        )
        chunk_blocks.append(block)
        labeled.append({
            "label": label,
            "file_path": chunk.get("file_path") or "",
            "file_rel_path": file_rel,
            "start_line": start,
            "end_line": end,
            "name": name,
            "parent_class": parent,
            "is_truncated": bool(chunk.get("is_truncated", False)),
        })

    chunks_text = "\n\n".join(chunk_blocks)
    prompt = (
        f"{chunks_text}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (cite chunks inline using [1], [2] etc.):"
    )

    return prompt, labeled


def parse_citations(response_text: str, labeled_chunks: list[dict]) -> list[dict]:
    """
    Extract citation labels from the LLM response and resolve them
    to real file paths and line numbers.

    Checks exactly the labels we generated (e.g. [1] through [5])
    against the response text. This avoids false positives from
    arbitrary bracketed numbers like [404] or [0] that could appear
    in a technical answer but were never valid citation labels.

    Args:
        response_text: Raw text response from the LLM.
        labeled_chunks: List returned by build_prompt.

    Returns:
        List of citation dicts, each with:
            label, file_rel_path, file_path, start_line, end_line, name,
            parent_class, is_truncated (display_rel_path_from_meta applied).
        Ordered by label number, with no duplicates.
    """
    label_map = {c["label"]: c for c in labeled_chunks}

    # Iterate only over labels we actually generated, in numeric order.
    # A label is "cited" if it appears anywhere in the response text.
    citations = []
    for label in sorted(label_map.keys(), key=lambda x: int(x[1:-1])):
        if label in response_text:
            chunk = label_map[label]
            display_rel = display_rel_path_from_meta(chunk)
            citations.append({
                "label": label,
                "file_rel_path": display_rel,
                "file_path": chunk.get("file_path") or "",
                "start_line": int(chunk.get("start_line", 0)),
                "end_line": int(chunk.get("end_line", 0)),
                "name": chunk.get("name") or "",
                "parent_class": chunk.get("parent_class"),
                "is_truncated": bool(chunk.get("is_truncated", False)),
            })

    return citations


def _strip_citations_block(text: str) -> str:
    """
    Remove the CITATIONS block the LLM appends to its response.

    The system prompt instructs the LLM to end its answer with a
    CITATIONS section. We use that block only to parse citation labels
    via parse_citations. After parsing, we strip it so callers receive
    clean prose without a duplicate citation list.

    Matches any line whose stripped content starts with "CITATIONS"
    (case-insensitive). Everything from that line onward is dropped.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().upper().startswith("CITATIONS"):
            return "\n".join(lines[:i]).rstrip()
    return text


def _parse_sections(answer_text: str) -> dict:
    """
    Parse the structured LLM response into named sections.

    Expects bold markdown headers: **Answer:**, **How it works:**,
    **Where to look next:** (optional). Strips the header from
    each section's content.

    Returns a dict with keys:
        answer: str — the direct answer (always present)
        how_it_works: str | None — the mechanism explanation
        where_to_look: str | None — navigational suggestion if partial

    Falls back gracefully: if structure is not found, puts the
    full text in answer and leaves other keys as None. This handles
    cases where the LLM does not follow the format exactly.
    """
    sections = {
        "answer": answer_text,
        "how_it_works": None,
        "where_to_look": None,
    }

    # Match bold headers case-insensitively, with optional whitespace.
    # We split on any **Header:** pattern to find section boundaries.
    pattern = re.compile(
        r"\*\*(Answer|How it works|Where to look next):\*\*",
        re.IGNORECASE
    )

    matches = list(pattern.finditer(answer_text))
    if not matches:
        # LLM did not follow format — return full text as answer
        return sections

    for i, match in enumerate(matches):
        header = match.group(1).lower().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(answer_text)
        content = answer_text[start:end].strip()

        if "answer" in header and "look" not in header:
            sections["answer"] = content
        elif "how" in header:
            sections["how_it_works"] = content
        elif "look" in header:
            sections["where_to_look"] = content

    return sections


TOUR_SYSTEM_PROMPT = """You are a senior engineer orienting a new
developer to an unfamiliar codebase. You have been given a
structural analysis of the repository including the most
architecturally significant functions and entry points.

Produce a concise orientation briefing in exactly this structure.
Do not add extra sections. Use the exact header names shown.

OVERVIEW
[2-3 sentences. What does this repo do and what problem does it
solve for the developer using it? Be specific about the value:
what can a developer do with this tool that they could not easily
do before? Infer from function names, file names, and docstrings.]

ENTRY POINTS
[Where does execution start? Name the specific file and function.
If multiple entry points exist, list each on its own line.]

MAJOR MODULES
[One sentence per significant file describing its responsibility.
Only include files represented in the provided chunks.]

KEY ABSTRACTIONS
[The 2-3 most important functions or classes. For each, write
2 sentences: first what it does, then why it is architecturally
central — what breaks or becomes impossible if it is removed.
Prefer functions with high inbound call counts.]

START HERE
[Recommended reading order for a developer new to this codebase.
Name 3-5 specific files in order, with one sentence each on why.]

Rules:
- Be specific. Name actual files and functions from the chunks.
- Do not say "based on the provided context" or "from the analysis".
- Write as if you know this codebase well.
- Keep the entire briefing under 400 words.
- Do not include a CITATIONS block.
"""


def answer_tour(context: str, openai_client: OpenAI) -> dict:
    """
    Call the LLM with the tour context and parse the response
    into named sections.

    Args:
        context: Formatted string from build_tour_context().
        openai_client: Initialized OpenAI client.

    Returns:
        Dict with keys:
            briefing: str — full response text
            briefing_sections: dict with keys:
                overview, entry_points, major_modules,
                key_abstractions, start_here
                Each is str | None.
    """
    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": TOUR_SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ],
        temperature=0.2,
        max_completion_tokens=1024,
    )

    briefing = response.choices[0].message.content or ""

    section_keys = [
        "OVERVIEW",
        "ENTRY POINTS",
        "MAJOR MODULES",
        "KEY ABSTRACTIONS",
        "START HERE",
    ]

    pattern = re.compile(
        r"^(" + "|".join(re.escape(k) for k in section_keys) + r")\s*$",
        re.MULTILINE | re.IGNORECASE,
    )

    matches = list(pattern.finditer(briefing))
    sections: dict[str, str | None] = {
        "overview": None,
        "entry_points": None,
        "major_modules": None,
        "key_abstractions": None,
        "start_here": None,
    }

    key_map = {
        "OVERVIEW": "overview",
        "ENTRY POINTS": "entry_points",
        "MAJOR MODULES": "major_modules",
        "KEY ABSTRACTIONS": "key_abstractions",
        "START HERE": "start_here",
    }

    for i, match in enumerate(matches):
        header = match.group(1).upper().strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(briefing)
        content = briefing[start:end].strip()
        key = key_map.get(header)
        if key:
            sections[key] = content

    return {
        "briefing": briefing,
        "briefing_sections": sections,
    }


def answer_query(
    query: str,
    results: list[dict],
    openai_client: OpenAI,
) -> dict:
    """
    Run the full LLM pipeline: build prompt, call gpt-5.4-mini,
    parse citations, return structured response.

    Args:
        query: The user's plain English question.
        results: Retrieved and ranked chunks from retriever.retrieve().
        openai_client: Initialized OpenAI client.

    Returns:
        Dict with keys:
            answer: str — the LLM's response text
            answer_sections: dict | None — parsed sections from response
            citations: list[dict] — resolved citation objects
            chunks_used: int — number of chunks sent to LLM
            confidence: str — "high", "medium", or "low"
            navigation: None — reserved; always None (LLM is called for any non-empty result)
    """
    if not results:
        return {
            "answer": "No relevant code found for this query.",
            "answer_sections": None,
            "citations": [],
            "chunks_used": 0,
            "confidence": "low",
            "navigation": None,
        }

    # Retriever produces rerank_score; "score" is accepted for test compatibility.
    top_score = results[0].get("rerank_score", results[0].get("score", 0.0)) if results else 0.0

    # Medium confidence: proceed but inject caution into prompt.
    system_prompt = SYSTEM_PROMPT
    if top_score < 0.4:
        system_prompt = SYSTEM_PROMPT + (
            "\n\nNote: Retrieval confidence is moderate for this "
            "query. Only make claims you can directly support from "
            "the provided chunks. If uncertain, guide the developer "
            "toward what to look for next rather than speculating."
        )

    prompt, labeled_chunks = build_prompt(query, results)

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_completion_tokens=1024,
    )

    response_text = response.choices[0].message.content or ""

    # Parse citations before stripping — parse_citations finds inline
    # labels [1], [2] etc. throughout the prose, not just in the block.
    citations = parse_citations(response_text, labeled_chunks)

    # Strip the CITATIONS block the LLM appends. The CLI and API both
    # render their own formatted citation sections from the parsed
    # citation objects above. Removing it here fixes both in one place
    # and frees output tokens for actual answer content.
    answer_text = _strip_citations_block(response_text)
    answer_sections = _parse_sections(answer_text)

    return {
        "answer": answer_text,
        "answer_sections": answer_sections,
        "citations": citations,
        "chunks_used": len(labeled_chunks),
        "confidence": (
            "high" if top_score >= 0.4
            else "medium" if top_score >= 0.05
            else "low"
        ),
        "navigation": None,
    }
