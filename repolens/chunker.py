"""
chunker.py

Parses Python source files into semantically complete chunks using
Tree-sitter AST parsing. Each chunk represents exactly one function
or class definition — never an arbitrary line slice.
"""

import tiktoken
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

# Build the Python language object once at module load.
# This is the compiled Tree-sitter grammar for Python.
PY_LANGUAGE = Language(tspython.language())

# The AST node types we split on. function_definition is a def block.
# class_definition is a class block. We do not split on nested
# functions inside a class — the whole class is one chunk.
CHUNK_NODE_TYPES = {"function_definition", "class_definition"}

# Hard cap on tokens per chunk. Chunks exceeding this are truncated.
# 300 tokens * 5 chunks = 1500 tokens max context, safe for gpt-4o-mini.
MAX_CHUNK_TOKENS = 300

# cl100k_base is the exact tokenizer gpt-4o-mini uses internally.
# Using it here means our token counts are accurate, not estimated.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


@dataclass
class Chunk:
    """
    A single semantically complete unit of source code.

    Every chunk is exactly one function or class definition.
    Metadata fields are used downstream for re-ranking retrieval results.
    """
    file_path: str       # Absolute path to the source file
    node_type: str       # "function_definition" or "class_definition"
    name: str            # Function or class name
    source: str          # Raw source text of this chunk
    start_line: int      # 1-indexed, inclusive
    end_line: int        # 1-indexed, inclusive
    token_count: int     # Exact token count via tiktoken


def count_tokens(text: str) -> int:
    """Return the exact token count for text using cl100k_base."""
    return len(_TOKENIZER.encode(text))


def extract_name(node, source_bytes: bytes) -> str:
    """
    Extract the name identifier from a function or class AST node.

    Tree-sitter represents names as child nodes with type "identifier".
    We find the first identifier child and decode its bytes to a string.
    """
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return "<unknown>"


def chunk_file(file_path: str | Path) -> list[Chunk]:
    """
    Parse a Python file and return a list of Chunk objects, one per
    top-level or class-level function or class definition.

    Nested functions defined inside another function are not chunked
    separately — they are included in their parent chunk's source text.

    Args:
        file_path: Path to a .py source file.

    Returns:
        List of Chunk objects sorted by start_line.

    Raises:
        ValueError: If the file does not exist or is not a .py file.
        OSError: If the file cannot be read.
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        raise ValueError(f"File does not exist: {file_path}")
    if file_path.suffix != ".py":
        raise ValueError(f"Not a Python file: {file_path}")

    source_bytes = file_path.read_bytes()

    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(source_bytes)

    chunks: list[Chunk] = []
    _walk_tree(tree.root_node, source_bytes, str(file_path), chunks)

    return sorted(chunks, key=lambda c: c.start_line)


def _walk_tree(
    node,
    source_bytes: bytes,
    file_path: str,
    chunks: list[Chunk],
) -> None:
    """
    Recursively walk the AST and extract function and class nodes.

    When a chunk node is found, extract it and stop descending into it.
    This prevents methods inside a class from being double-counted as
    both part of the class chunk and as their own separate chunks.
    """
    for child in node.children:
        if child.type in CHUNK_NODE_TYPES:
            source_text = source_bytes[
                child.start_byte:child.end_byte
            ].decode("utf-8")

            token_count = count_tokens(source_text)

            # Truncate chunks that exceed the hard token cap.
            # This handles pathological cases like 500-line god functions.
            if token_count > MAX_CHUNK_TOKENS:
                encoded = _TOKENIZER.encode(source_text)
                source_text = _TOKENIZER.decode(encoded[:MAX_CHUNK_TOKENS])
                token_count = MAX_CHUNK_TOKENS

            chunks.append(Chunk(
                file_path=file_path,
                node_type=child.type,
                name=extract_name(child, source_bytes),
                source=source_text,
                # Tree-sitter uses 0-indexed lines. Add 1 to convert
                # to the 1-indexed line numbers humans and editors use.
                start_line=child.start_point[0] + 1,
                end_line=child.end_point[0] + 1,
                token_count=token_count,
            ))
            # Stop recursing into this node. Everything inside belongs
            # to this chunk, not to separate chunks of their own.
        else:
            _walk_tree(child, source_bytes, file_path, chunks)
