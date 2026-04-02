"""
Tests for repolens/chunker.py.

Each test writes Python source as a string to a temp file and asserts
on the chunks produced. Tests are fully hermetic — no real repo needed.
"""

import pytest
from pathlib import Path
from repolens.chunker import chunk_file, MAX_CHUNK_TOKENS


def write_py(path: Path, source: str) -> Path:
    """Write Python source text to a .py file, creating dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


class TestChunkFile:

    def test_single_function(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def hello():
    return "hello"
""")
        chunks = chunk_file(f)
        assert len(chunks) == 1
        assert chunks[0].name == "hello"
        assert chunks[0].node_type == "function_definition"
        assert chunks[0].start_line == 1

    def test_multiple_functions(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    pass

def bar():
    pass

def baz():
    pass
""")
        chunks = chunk_file(f)
        assert len(chunks) == 3
        assert [c.name for c in chunks] == ["foo", "bar", "baz"]

    def test_class_is_single_chunk(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
class MyService:
    def __init__(self):
        self.value = 0

    def get(self):
        return self.value
""")
        chunks = chunk_file(f)
        # The whole class is one chunk. Methods are not split separately.
        assert len(chunks) == 1
        assert chunks[0].name == "MyService"
        assert chunks[0].node_type == "class_definition"

    def test_mixed_functions_and_classes(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def standalone():
    pass

class Engine:
    def run(self):
        pass

def another():
    pass
""")
        chunks = chunk_file(f)
        assert len(chunks) == 3
        assert chunks[0].name == "standalone"
        assert chunks[1].name == "Engine"
        assert chunks[2].name == "another"

    def test_start_line_is_one_indexed(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
x = 1

def foo():
    pass
""")
        chunks = chunk_file(f)
        # foo starts on line 3, not line 2 (Tree-sitter is 0-indexed,
        # we convert to 1-indexed)
        assert chunks[0].start_line == 3

    def test_token_count_is_positive(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    return 42
""")
        chunks = chunk_file(f)
        assert chunks[0].token_count > 0

    def test_oversized_chunk_truncated_to_cap(self, tmp_path):
        body = "\n".join(f"    x_{i} = {i}" for i in range(200))
        source = f"def big():\n{body}\n"
        f = write_py(tmp_path / "a.py", source)
        chunks = chunk_file(f)
        assert chunks[0].token_count == MAX_CHUNK_TOKENS

    def test_empty_file_returns_empty_list(self, tmp_path):
        f = write_py(tmp_path / "a.py", "")
        assert chunk_file(f) == []

    def test_module_level_code_not_chunked(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
import os
x = 1
print(x)
""")
        assert chunk_file(f) == []

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            chunk_file(tmp_path / "ghost.py")

    def test_raises_on_non_python_file(self, tmp_path):
        f = tmp_path / "config.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Not a Python file"):
            chunk_file(f)

    def test_chunks_sorted_by_start_line(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def z_func():
    pass

def a_func():
    pass
""")
        chunks = chunk_file(f)
        lines = [c.start_line for c in chunks]
        assert lines == sorted(lines)
