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
        # Class chunk + one chunk per method, each tagged with parent_class
        assert len(chunks) == 3
        class_chunk = next(c for c in chunks if c.node_type == "class_definition")
        assert class_chunk.name == "MyService"
        method_chunks = [c for c in chunks if c.node_type == "function_definition"]
        assert all(c.parent_class == "MyService" for c in method_chunks)

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
        # standalone, Engine class, Engine.run method (with parent_class), another
        assert len(chunks) == 4
        assert chunks[0].name == "standalone"
        assert chunks[0].parent_class is None
        assert chunks[1].name == "Engine"
        assert chunks[1].parent_class is None
        assert chunks[2].name == "run"
        assert chunks[2].parent_class == "Engine"
        assert chunks[3].name == "another"
        assert chunks[3].parent_class is None

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

    def test_oversized_chunk_is_marked_truncated(self, tmp_path):
        body = "\n".join(f"    x_{i} = {i}" for i in range(200))
        source = f"def big():\n{body}\n"
        f = write_py(tmp_path / "a.py", source)
        chunks = chunk_file(f)
        assert chunks[0].is_truncated is True

    def test_normal_chunk_is_not_truncated(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def foo():
    return 42
""")
        chunks = chunk_file(f)
        assert chunks[0].is_truncated is False

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


class TestChunkCompleteness:
    """
    Regression guard ensuring no function or class definition is silently
    dropped by the chunker. Tests every case that could cause under-chunking:
    decorated functions, decorated classes, decorated methods.

    Because missing a chunk means the LLM never sees that code, we treat
    any under-chunking as a critical bug. Over-chunking is acceptable.
    """

    def test_decorated_function_is_chunked(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def decorator(fn):
    return fn

@decorator
def foo():
    return 1
""")
        chunks = chunk_file(f)
        names = [c.name for c in chunks]
        assert "foo" in names

    def test_decorated_function_source_includes_decorator(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def require_auth(fn):
    return fn

@require_auth
def protected():
    pass
""")
        chunks = chunk_file(f)
        protected = next(c for c in chunks if c.name == "protected")
        assert "@require_auth" in protected.source

    def test_decorated_class_is_chunked(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def singleton(cls):
    return cls

@singleton
class Config:
    pass
""")
        chunks = chunk_file(f)
        names = [c.name for c in chunks]
        assert "Config" in names

    def test_decorated_class_source_includes_decorator(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def singleton(cls):
    return cls

@singleton
class Config:
    pass
""")
        chunks = chunk_file(f)
        config = next(c for c in chunks if c.name == "Config")
        assert "@singleton" in config.source

    def test_decorated_method_inside_class_is_chunked(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
class MyView:
    @staticmethod
    def render():
        pass
""")
        chunks = chunk_file(f)
        names = [c.name for c in chunks]
        assert "render" in names

    def test_decorated_method_source_includes_decorator(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
class MyView:
    @staticmethod
    def render():
        pass
""")
        chunks = chunk_file(f)
        render = next(c for c in chunks if c.name == "render")
        assert "@staticmethod" in render.source

    def test_decorated_method_has_correct_parent_class(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
class MyView:
    @staticmethod
    def render():
        pass
""")
        chunks = chunk_file(f)
        render = next(c for c in chunks if c.name == "render")
        assert render.parent_class == "MyView"

    def test_multiple_decorators_all_included_in_source(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
def dec_a(fn):
    return fn

def dec_b(fn):
    return fn

@dec_a
@dec_b
def multi():
    pass
""")
        chunks = chunk_file(f)
        multi = next(c for c in chunks if c.name == "multi")
        assert "@dec_a" in multi.source
        assert "@dec_b" in multi.source

    def test_start_line_of_decorated_chunk_is_decorator_line(self, tmp_path):
        f = write_py(tmp_path / "a.py", """\
x = 1

@property
def value(self):
    return self._value
""")
        chunks = chunk_file(f)
        value = next(c for c in chunks if c.name == "value")
        # @property is on line 3, not line 4 where `def` starts
        assert value.start_line == 3
