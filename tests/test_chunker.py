"""
Tests for repolens/chunker.py.

Covers original chunking behavior plus new metadata extraction:
call graph collection and docstring extraction.
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


class TestExtractCalls:

    def test_simple_function_call(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    bar()
""")
        chunks = chunk_file(f)
        assert "bar" in chunks[0].calls

    def test_method_call_extracts_method_name(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    self.validate()
""")
        chunks = chunk_file(f)
        assert "validate" in chunks[0].calls

    def test_multiple_calls_deduplicated(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    bar()
    bar()
    baz()
""")
        chunks = chunk_file(f)
        assert chunks[0].calls.count("bar") == 1
        assert "baz" in chunks[0].calls

    def test_no_calls_returns_empty_list(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    x = 1
    return x
""")
        chunks = chunk_file(f)
        assert chunks[0].calls == []

    def test_calls_are_sorted(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    zebra()
    alpha()
    mango()
""")
        chunks = chunk_file(f)
        assert chunks[0].calls == sorted(chunks[0].calls)


class TestExtractDocstring:

    def test_double_quote_docstring(self, tmp_path):
        f = write_py(tmp_path / "a.py", '''\
def foo():
    """This is a docstring."""
    pass
''')
        chunks = chunk_file(f)
        assert chunks[0].docstring == "This is a docstring."

    def test_single_quote_docstring(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    '''Single quote docstring.'''
    pass
""")
        chunks = chunk_file(f)
        assert chunks[0].docstring == "Single quote docstring."

    def test_no_docstring_returns_none(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    x = 1
""")
        chunks = chunk_file(f)
        assert chunks[0].docstring is None

    def test_class_docstring_extracted(self, tmp_path):
        f = write_py(tmp_path / "a.py", '''\
class MyService:
    """Handles core service logic."""
    def run(self):
        pass
''')
        chunks = chunk_file(f)
        assert chunks[0].docstring == "Handles core service logic."
