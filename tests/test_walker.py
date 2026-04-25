"""
Tests for repolix/walker.py.

All tests use pytest's tmp_path fixture to create hermetic temporary
directory structures. No test touches any real repository on disk.
"""

import pytest
from pathlib import Path
from repolix.walker import walk_repo, MAX_FILE_SIZE_BYTES, TEST_DIRS


def write_file(path: Path, content: str = "# placeholder") -> None:
    """Create parent directories as needed and write a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestWalkRepo:

    def test_finds_python_files(self, tmp_path):
        write_file(tmp_path / "main.py")
        write_file(tmp_path / "utils.py")
        result = walk_repo(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".py" for p in result)

    def test_ignores_non_python_files(self, tmp_path):
        write_file(tmp_path / "main.py")
        write_file(tmp_path / "README.md")
        write_file(tmp_path / "config.json")
        result = walk_repo(tmp_path)
        assert len(result) == 1
        assert result[0].name == "main.py"

    def test_ignores_pycache_directory(self, tmp_path):
        write_file(tmp_path / "main.py")
        write_file(tmp_path / "__pycache__" / "main.cpython-311.pyc")
        result = walk_repo(tmp_path)
        assert len(result) == 1
        assert result[0].name == "main.py"

    def test_ignores_venv_directory(self, tmp_path):
        write_file(tmp_path / "app.py")
        write_file(tmp_path / "venv" / "lib" / "python3.11" / "site.py")
        result = walk_repo(tmp_path)
        assert len(result) == 1
        assert result[0].name == "app.py"

    def test_ignores_dot_prefixed_directories(self, tmp_path):
        # Any directory starting with "." is treated as hidden/internal
        # and excluded regardless of whether it appears in IGNORED_DIRS.
        write_file(tmp_path / "main.py")
        write_file(tmp_path / ".hidden_tool" / "config.py")
        result = walk_repo(tmp_path)
        assert len(result) == 1
        assert result[0].name == "main.py"

    def test_skips_oversized_files(self, tmp_path):
        write_file(tmp_path / "normal.py", "x = 1")
        large = tmp_path / "generated.py"
        large.parent.mkdir(parents=True, exist_ok=True)
        large.write_bytes(b"x" * (MAX_FILE_SIZE_BYTES + 1))
        result = walk_repo(tmp_path)
        assert len(result) == 1
        assert result[0].name == "normal.py"

    def test_returns_sorted_list(self, tmp_path):
        write_file(tmp_path / "z_last.py")
        write_file(tmp_path / "a_first.py")
        write_file(tmp_path / "m_middle.py")
        result = walk_repo(tmp_path)
        names = [p.name for p in result]
        assert names == sorted(names)

    def test_nested_python_files_all_collected(self, tmp_path):
        write_file(tmp_path / "src" / "core" / "engine.py")
        write_file(tmp_path / "src" / "utils" / "helpers.py")
        write_file(tmp_path / "tests" / "test_engine.py")
        # exclude_tests=False so all three files are included regardless
        result = walk_repo(tmp_path, exclude_tests=False)
        assert len(result) == 3

    def test_raises_on_nonexistent_path(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            walk_repo(tmp_path / "nonexistent_dir")

    def test_raises_when_path_is_file_not_directory(self, tmp_path):
        f = tmp_path / "somefile.py"
        write_file(f)
        with pytest.raises(ValueError, match="not a directory"):
            walk_repo(f)

    def test_empty_repo_returns_empty_list(self, tmp_path):
        result = walk_repo(tmp_path)
        assert result == []

    def test_excludes_test_directory_by_default(self, tmp_path):
        write_file(tmp_path / "src" / "main.py")
        write_file(tmp_path / "tests" / "test_main.py")
        result = walk_repo(tmp_path)
        names = [p.name for p in result]
        assert "main.py" in names
        assert "test_main.py" not in names
        assert len(result) == 1

    def test_include_tests_overrides_exclusion(self, tmp_path):
        write_file(tmp_path / "src" / "main.py")
        write_file(tmp_path / "tests" / "test_main.py")
        result = walk_repo(tmp_path, exclude_tests=False)
        names = [p.name for p in result]
        assert "main.py" in names
        assert "test_main.py" in names
        assert len(result) == 2

    def test_excludes_test_file_by_name(self, tmp_path):
        write_file(tmp_path / "src" / "main.py")
        write_file(tmp_path / "src" / "test_main.py")
        result = walk_repo(tmp_path)
        names = [p.name for p in result]
        assert "main.py" in names
        assert "test_main.py" not in names
        assert len(result) == 1
