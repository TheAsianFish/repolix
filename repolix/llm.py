"""
llm.py

Constructs prompts from retrieved chunks, calls gpt-5.4-mini,
and parses the response into a structured answer with citations.

The citation format uses numbered labels [1], [2] etc. that the
LLM is instructed to use inline. We map those labels back to
real file paths and line numbers after parsing the response.
"""

from openai import OpenAI

LLM_MODEL = "gpt-5.4-mini"

# Maximum chunks to send to the LLM. Set to 8 to include up to
# 5 primary results + 3 call graph expansions from expand_via_call_graph.
# 8 * 300 tokens = 2400 tokens of context, still well within budget.
MAX_CONTEXT_CHUNKS = 8

SYSTEM_PROMPT = """You are a precise code navigation assistant. A developer is asking
about a specific codebase. You have been given the most relevant
code chunks retrieved from that codebase.

Your job:
1. Answer the question directly in 1-3 sentences. Be specific — name
   the file and function where the answer lives.
2. If the answer spans multiple locations, list each one clearly.
3. Do not summarize what the chunks contain. Answer the question.
4. Do not say "based on the provided chunks" or "from the context".
   Just answer as if you know the codebase.
5. If the chunks do not contain enough information to answer
   confidently, say exactly what is missing and what the developer
   should search for next.

Cite chunks inline using their label [1], [2] etc. every time your
answer references that chunk's behavior or code. Only cite a chunk
if your answer actually references its content.

End your answer with a CITATIONS block listing each label you used,
its file path, and its line range. Do not change this format.
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
        file_rel = chunk.get("file_rel_path", chunk["file_path"])
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
            "file_path": chunk["file_path"],
            "file_rel_path": file_rel,
            "start_line": start,
            "end_line": end,
            "name": name,
            "parent_class": parent,
            "is_truncated": chunk.get("is_truncated", False),
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
            label, file_rel_path, start_line, end_line, name.
        Ordered by label number, with no duplicates.
    """
    label_map = {c["label"]: c for c in labeled_chunks}

    # Iterate only over labels we actually generated, in numeric order.
    # A label is "cited" if it appears anywhere in the response text.
    citations = []
    for label in sorted(label_map.keys(), key=lambda x: int(x[1:-1])):
        if label in response_text:
            chunk = label_map[label]
            citations.append({
                "label": label,
                "file_rel_path": chunk["file_rel_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "name": chunk["name"],
                "parent_class": chunk.get("parent_class"),
                "is_truncated": chunk.get("is_truncated", False),
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
            citations: list[dict] — resolved citation objects
            chunks_used: int — number of chunks sent to LLM
    """
    if not results:
        return {
            "answer": "No relevant code found for this query.",
            "citations": [],
            "chunks_used": 0,
        }

    prompt, labeled_chunks = build_prompt(query, results)

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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

    return {
        "answer": answer_text,
        "citations": citations,
        "chunks_used": len(labeled_chunks),
    }
