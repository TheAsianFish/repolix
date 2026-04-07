"""
walker.py

Traverses a repository directory and returns a sorted list of source
file paths that are worth indexing. Filters out noise directories,
unsupported file types, and files that are too large to be useful.
"""

import os
from pathlib import Path

# Directories that are never worth indexing. These are dependency trees,
# build artifacts, caches, and version control internals. We prune them
# during traversal so os.walk never descends into them at all.
IGNORED_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
    ".next",
    ".nuxt",
    "target",
    "vendor",
}

# File extensions we index in V1. Python only.
# Expanding to TypeScript in V2 means adding ".ts" and ".tsx" here
# and registering the Tree-sitter TypeScript grammar in chunker.py.
ALLOWED_EXTENSIONS = {
    ".py",
}

# Hard cap on individual file size. Files above this threshold are
# almost always generated or minified — not useful for retrieval.
# 500 KB is generous for any hand-written source file.
MAX_FILE_SIZE_BYTES = 500 * 1024


def walk_repo(repo_path: str | Path) -> list[Path]:
    """
    Walk the repository at repo_path and return a sorted list of
    indexable source file paths.

    A file is included if and only if:
      - None of its ancestor directories (relative to repo_path) are
        in IGNORED_DIRS or start with a dot
      - Its file extension is in ALLOWED_EXTENSIONS
      - Its size on disk is <= MAX_FILE_SIZE_BYTES

    Args:
        repo_path: Path to the repository root. Resolved to absolute.

    Returns:
        Sorted list of Path objects. Sorting is deterministic so the
        same repo always produces the same list in the same order.

    Raises:
        ValueError: If repo_path does not exist or is not a directory.
    """
    repo_path = Path(repo_path).resolve()

    if not repo_path.exists():
        raise ValueError(f"Path does not exist: {repo_path}")
    if not repo_path.is_dir():
        raise ValueError(f"Path is not a directory: {repo_path}")

    collected: list[Path] = []

    for root, dirs, files in os.walk(repo_path):
        root_path = Path(root)

        # Mutate dirs in-place. os.walk holds a reference to this list
        # and uses it to decide which subdirectories to visit next.
        # Reassigning dirs = [...] would create a new list object and
        # os.walk would ignore it entirely — the walk would continue
        # into every directory. In-place mutation is required here.
        dirs[:] = sorted([
            d for d in dirs
            if d not in IGNORED_DIRS and not d.startswith(".")
        ])

        for filename in files:
            file_path = root_path / filename

            if file_path.suffix not in ALLOWED_EXTENSIONS:
                continue

            try:
                size = file_path.stat().st_size
            except OSError:
                # File disappeared between directory listing and stat.
                # This is rare but possible on active filesystems.
                continue

            if size > MAX_FILE_SIZE_BYTES:
                continue

            collected.append(file_path)

    return sorted(collected)
