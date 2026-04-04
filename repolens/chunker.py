"""
chunker.py

Parses Python source files into semantically complete chunks using
Tree-sitter AST parsing. Each chunk represents exactly one function
or class definition — never an arbitrary line slice.

Refactors from Milestone 2:
- Parser is now a module-level singleton instead of per-call instance.
- _walk_tree no longer accepts file_path — resolved once in chunk_file.
"""

import tiktoken
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

PY_LANGUAGE = Language(tspython.language())

# Module-level singletons. Both are stateless between calls so there
# is no reason to instantiate them per file. A 200-file repo was
# creating 200 Parser objects — now it creates one.
_PARSER = Parser(PY_LANGUAGE)
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

CHUNK_NODE_TYPES = {"function_definition", "class_definition"}

MAX_CHUNK_TOKENS = 300


@dataclass
class Chunk:
    """
    A single semantically complete unit of source code.

    Every chunk is exactly one function or class definition.
    Metadata fields support downstream re-ranking and call graph
    expansion.
    """
    file_path: str
    node_type: str          # "function_definition" or "class_definition"
    name: str               # Function or class name
    source: str             # Raw source text of this chunk
    start_line: int         # 1-indexed, inclusive
    end_line: int           # 1-indexed, inclusive
    token_count: int        # Exact token count via tiktoken
    calls: list[str]        # Names of functions called within this chunk
    docstring: str | None   # First string literal if used as docstring
    parent_class: str | None   # Enclosing class name for methods


def count_tokens(text: str) -> int:
    """Return the exact token count for text using cl100k_base."""
    return len(_TOKENIZER.encode(text))


def extract_name(node, source_bytes: bytes) -> str:
    """
    Extract the name identifier from a function or class AST node.

    Tree-sitter represents names as child nodes of type "identifier".
    We find the first such child and decode its bytes to a string.
    """
    for child in node.children:
        if child.type == "identifier":
            return source_bytes[child.start_byte:child.end_byte].decode("utf-8")
    return "<unknown>"


def extract_calls(node, source_bytes: bytes) -> list[str]:
    """
    Walk a function or class node and collect the names of all
    functions called within it.

    Tree-sitter represents a function call as a "call" node whose
    first child is the callable — either an "identifier" (simple call
    like foo()) or an "attribute" node (method call like obj.method()).
    We handle both cases and deduplicate the result.

    Args:
        node: A Tree-sitter node for a function or class definition.
        source_bytes: The full file content as bytes.

    Returns:
        Sorted deduplicated list of callee name strings.
    """
    found: set[str] = set()
    _collect_calls(node, source_bytes, found)
    return sorted(found)


def _collect_calls(node, source_bytes: bytes, found: set[str]) -> None:
    """Recursive helper for extract_calls."""
    for child in node.children:
        if child.type == "call":
            func_node = child.children[0] if child.children else None
            if func_node is not None:
                if func_node.type == "identifier":
                    found.add(
                        source_bytes[
                            func_node.start_byte:func_node.end_byte
                        ].decode("utf-8")
                    )
                elif func_node.type == "attribute":
                    # obj.method() — extract just the method name (last identifier)
                    identifiers = [
                        c for c in func_node.children
                        if c.type == "identifier"
                    ]
                    if identifiers:
                        last = identifiers[-1]
                        found.add(
                            source_bytes[
                                last.start_byte:last.end_byte
                            ].decode("utf-8")
                        )
        _collect_calls(child, source_bytes, found)


def extract_docstring(node, source_bytes: bytes) -> str | None:
    """
    Extract the docstring from a function or class node if one exists.

    A docstring is the first statement in the body if that statement
    is an expression containing a string literal. This matches Python's
    own docstring convention exactly.

    Args:
        node: A Tree-sitter node for a function or class definition.
        source_bytes: The full file content as bytes.

    Returns:
        The docstring text with surrounding quotes stripped,
        or None if no docstring is present.
    """
    body = None
    for child in node.children:
        if child.type == "block":
            body = child
            break

    if body is None or not body.children:
        return None

    # Skip newline/comment/indent nodes to find the first real statement.
    first_stmt = None
    for child in body.children:
        if child.type not in {"newline", "comment", "indent"}:
            first_stmt = child
            break

    if first_stmt is None or first_stmt.type != "expression_statement":
        return None

    for child in first_stmt.children:
        if child.type == "string":
            raw = source_bytes[child.start_byte:child.end_byte].decode("utf-8")
            return raw.strip('"""').strip("'''").strip('"').strip("'").strip()

    return None


def chunk_file(file_path: str | Path) -> list[Chunk]:
    """
    Parse a Python file and return a list of Chunk objects, one per
    top-level or class-level function or class definition.

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
    tree = _PARSER.parse(source_bytes)

    chunks: list[Chunk] = []
    _walk_tree(tree.root_node, source_bytes, str(file_path), chunks)

    return sorted(chunks, key=lambda c: c.start_line)


def _walk_tree(
    node,
    source_bytes: bytes,
    file_path: str,
    chunks: list[Chunk],
    enclosing_class: str | None = None,
) -> None:
    """
    Recursively walk the AST and extract function and class nodes.

    Tracks enclosing class context so methods know which class they
    belong to. This is stored as parent_class metadata and used for
    disambiguation when multiple classes have similarly named methods.

    Stops descending into chunk nodes to prevent double-counting,
    EXCEPT for class_definition nodes — we descend into those to
    capture their methods with parent_class set.

    When a function or class is wrapped in a decorated_definition node
    (i.e. it has one or more decorators), the source text is taken from
    the decorated_definition parent so that decorator lines are included.
    Decorators like @require_auth or @property are often the most
    semantically meaningful part of a definition for retrieval.
    """
    for child in node.children:
        if child.type in CHUNK_NODE_TYPES:
            # If the parent node is a decorated_definition, use it as
            # the source range so decorators are included in the chunk.
            source_node = node if node.type == "decorated_definition" else child

            source_text = source_bytes[
                source_node.start_byte:source_node.end_byte
            ].decode("utf-8")

            token_count = count_tokens(source_text)

            if token_count > MAX_CHUNK_TOKENS:
                encoded = _TOKENIZER.encode(source_text)
                source_text = _TOKENIZER.decode(encoded[:MAX_CHUNK_TOKENS])
                token_count = MAX_CHUNK_TOKENS

            name = extract_name(child, source_bytes)

            chunks.append(Chunk(
                file_path=file_path,
                node_type=child.type,
                name=name,
                source=source_text,
                start_line=source_node.start_point[0] + 1,
                end_line=source_node.end_point[0] + 1,
                token_count=token_count,
                calls=extract_calls(child, source_bytes),
                docstring=extract_docstring(child, source_bytes),
                parent_class=enclosing_class,
            ))

            # For class definitions, descend into the body so methods
            # are chunked separately with parent_class set to this
            # class name. For function definitions, stop — nested
            # functions belong to their parent chunk.
            if child.type == "class_definition":
                _walk_tree(
                    child,
                    source_bytes,
                    file_path,
                    chunks,
                    enclosing_class=name,
                )
        else:
            _walk_tree(
                child,
                source_bytes,
                file_path,
                chunks,
                enclosing_class=enclosing_class,
            )
