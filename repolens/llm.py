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

LLM_MODEL = "gpt-5.4-mini"

# Maximum chunks to send to the LLM. Hard cap regardless of
# how many the retriever returns. 5 * 300 tokens = 1500 tokens
# of context, well within gpt-5.4-mini's context window.
MAX_CONTEXT_CHUNKS = 5

SYSTEM_PROMPT = """You are a code assistant that answers questions
about a codebase. You are given a set of relevant code chunks
retrieved from the codebase and a question about the code.

Rules:
- Answer using ONLY the provided code chunks. Do not invent code
  or behavior not shown in the chunks.
- Cite chunks inline using their label [1], [2] etc. wherever
  your answer references that chunk's code or behavior.
- If multiple chunks contribute to the answer, cite all of them.
- If the chunks do not contain enough information to answer
  confidently, say so clearly rather than guessing.
- Be concise and precise. Developers reading your answer are
  technical and do not need basic concepts explained.
- Always end your answer with a CITATIONS section listing each
  label you used with its file path and line range.
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

    Looks for patterns like [1], [2] in the response text. For each
    label found, maps it back to the corresponding chunk metadata.

    Args:
        response_text: Raw text response from the LLM.
        labeled_chunks: List returned by build_prompt.

    Returns:
        List of citation dicts, each with:
            label, file_rel_path, start_line, end_line, name.
        Ordered by label number, deduplicated.
    """
    label_map = {c["label"]: c for c in labeled_chunks}

    found_labels = sorted(
        set(re.findall(r"\[\d+\]", response_text)),
        key=lambda x: int(x[1:-1]),
    )

    citations = []
    for label in found_labels:
        if label in label_map:
            chunk = label_map[label]
            citations.append({
                "label": label,
                "file_rel_path": chunk["file_rel_path"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "name": chunk["name"],
                "parent_class": chunk.get("parent_class"),
            })

    return citations


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
    )

    response_text = response.choices[0].message.content or ""
    citations = parse_citations(response_text, labeled_chunks)

    return {
        "answer": response_text,
        "citations": citations,
        "chunks_used": len(labeled_chunks),
    }
