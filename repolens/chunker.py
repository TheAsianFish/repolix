"""
chunker.py

Splits source files into semantically meaningful chunks (functions,
classes, top-level statements) using Tree-sitter parse trees so that
each chunk is independently retrievable and fits within an LLM context
window.
"""

# TODO: Implement Tree-sitter-based chunking for Python source files,
# producing a list of Chunk objects (text, file path, start/end lines).
