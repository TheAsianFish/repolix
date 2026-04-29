"""
tests/test_trace.py

Tests for repolix/trace.py and the answer_trace() function in llm.py.

All tests use tmp_path and mock ChromaDB/OpenAI.
Never hit real services in tests.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from repolix.trace import (
    lookup_chunk_by_name,
    forward_trace,
    backward_trace,
    format_trace_tree,
    run_trace,
    TRACE_MAX_DEPTH,
    TRACE_MAX_NODES,
)
from repolix.llm import answer_trace


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_chunk(name, calls, file_rel_path="repolix/retriever.py", start_line=10):
    return {
        "name": name,
        "calls": calls,
        "file_path": f"/abs/{file_rel_path}",
        "file_rel_path": file_rel_path,
        "start_line": start_line,
        "end_line": start_line + 5,
        "node_type": "function",
        "source": f"def {name}(): pass",
        "docstring": "",
    }


def _lookup_factory(chunks_by_name: dict):
    """Return a side_effect function for patching lookup_chunk_by_name."""
    def _lookup(name, store_path):
        return chunks_by_name.get(name)
    return _lookup


# ── lookup_chunk_by_name ──────────────────────────────────────────────────────

def test_lookup_chunk_by_name_found(tmp_path):
    chunk = _make_chunk("retrieve", ["query_chunks"])
    with patch("repolix.trace.keyword_search", return_value=[chunk]):
        result = lookup_chunk_by_name("retrieve", tmp_path)
    assert result is not None
    assert result["name"] == "retrieve"


def test_lookup_chunk_by_name_not_found(tmp_path):
    with patch("repolix.trace.keyword_search", return_value=[]):
        result = lookup_chunk_by_name("nonexistent", tmp_path)
    assert result is None


def test_lookup_chunk_by_name_ignores_non_exact_matches(tmp_path):
    """keyword_search may return substring matches; we only want exact name."""
    close_match = _make_chunk("retrieve_all", ["x"])
    with patch("repolix.trace.keyword_search", return_value=[close_match]):
        result = lookup_chunk_by_name("retrieve", tmp_path)
    assert result is None


# ── forward_trace ─────────────────────────────────────────────────────────────

def test_forward_trace_basic(tmp_path):
    chunks = {
        "retrieve": _make_chunk("retrieve", ["query_chunks", "keyword_search"]),
        "query_chunks": _make_chunk("query_chunks", ["_get_client"]),
        "keyword_search": _make_chunk("keyword_search", []),
        "_get_client": _make_chunk("_get_client", []),
    }

    with patch("repolix.trace.lookup_chunk_by_name", side_effect=_lookup_factory(chunks)):
        result = forward_trace("retrieve", tmp_path, max_depth=2)

    assert result["not_found"] is False
    assert "retrieve" in result["nodes"]
    assert "query_chunks" in result["nodes"]
    assert "keyword_search" in result["nodes"]
    assert "_get_client" in result["nodes"]
    children = result["nodes"]["retrieve"]["children"]
    assert "query_chunks" in children
    assert "keyword_search" in children


def test_forward_trace_not_found(tmp_path):
    with patch("repolix.trace.lookup_chunk_by_name", return_value=None):
        result = forward_trace("nonexistent", tmp_path)

    assert result["not_found"] is True
    assert result["nodes"] == {}
    assert result["visited_count"] == 0


def test_forward_trace_cycle_detection(tmp_path):
    chunks = {
        "A": _make_chunk("A", ["B"]),
        "B": _make_chunk("B", ["A"]),
    }

    with patch("repolix.trace.lookup_chunk_by_name", side_effect=_lookup_factory(chunks)):
        result = forward_trace("A", tmp_path, max_depth=5)

    assert result["not_found"] is False
    assert result["visited_count"] == 2
    # A should be in B's child_already_visited (cycle back to A)
    assert "A" in result["nodes"]["B"]["child_already_visited"]


def test_forward_trace_max_nodes_cap(tmp_path):
    # Build a chain A -> B -> C -> ... -> Y (25 nodes)
    names = [chr(ord("A") + i) for i in range(25)]
    chunks = {}
    for i, name in enumerate(names):
        next_calls = [names[i + 1]] if i + 1 < len(names) else []
        chunks[name] = _make_chunk(name, next_calls)

    with patch("repolix.trace.lookup_chunk_by_name", side_effect=_lookup_factory(chunks)):
        result = forward_trace("A", tmp_path, max_nodes=10, max_depth=30)

    assert result["visited_count"] <= 10
    assert result["truncated"] is True


def test_forward_trace_depth_limit(tmp_path):
    # 4 levels deep: root -> L1 -> L2 -> L3
    chunks = {
        "root": _make_chunk("root", ["L1"]),
        "L1": _make_chunk("L1", ["L2"]),
        "L2": _make_chunk("L2", ["L3"]),
        "L3": _make_chunk("L3", ["L4"]),
        "L4": _make_chunk("L4", []),
    }

    with patch("repolix.trace.lookup_chunk_by_name", side_effect=_lookup_factory(chunks)):
        result = forward_trace("root", tmp_path, max_depth=2)

    # Nodes at depth 0, 1, 2 should be present; L3 (depth 3) should not.
    assert "root" in result["nodes"]
    assert "L1" in result["nodes"]
    assert "L2" in result["nodes"]
    assert "L3" not in result["nodes"]

    # All present nodes must have depth <= 2
    for name, node in result["nodes"].items():
        assert node["depth"] <= 2, f"{name} has depth {node['depth']}"


# ── backward_trace ────────────────────────────────────────────────────────────

def test_backward_trace_finds_callers(tmp_path):
    all_chunks = [
        {**_make_chunk("query", ["retrieve", "other"]), "file_rel_path": "repolix/cli.py"},
        {**_make_chunk("query_endpoint", ["retrieve"]), "file_rel_path": "repolix/api.py"},
        {**_make_chunk("unrelated", ["something_else"]), "file_rel_path": "repolix/tour.py"},
    ]

    with patch("repolix.tour.get_all_chunks", return_value=all_chunks):
        result = backward_trace("retrieve", tmp_path)

    assert len(result) == 2
    names = {c["name"] for c in result}
    assert "query" in names
    assert "query_endpoint" in names


def test_backward_trace_no_callers(tmp_path):
    all_chunks = [
        _make_chunk("unrelated", ["something_else"]),
        _make_chunk("another", ["yet_another"]),
    ]

    with patch("repolix.tour.get_all_chunks", return_value=all_chunks):
        result = backward_trace("orphan_func", tmp_path)

    assert result == []


# ── format_trace_tree ─────────────────────────────────────────────────────────

def test_format_trace_tree_basic():
    trace_result = {
        "root": "retrieve",
        "not_found": False,
        "nodes": {
            "retrieve": {
                "chunk": _make_chunk("retrieve", ["query_chunks", "keyword_search"]),
                "depth": 0,
                "parent": None,
                "children": ["query_chunks", "keyword_search"],
                "child_already_visited": [],
                "truncated": False,
            },
            "query_chunks": {
                "chunk": _make_chunk("query_chunks", []),
                "depth": 1,
                "parent": "retrieve",
                "children": [],
                "child_already_visited": [],
                "truncated": False,
            },
            "keyword_search": {
                "chunk": _make_chunk("keyword_search", []),
                "depth": 1,
                "parent": "retrieve",
                "children": [],
                "child_already_visited": [],
                "truncated": False,
            },
        },
        "visited_count": 3,
        "truncated": False,
    }

    output = format_trace_tree(trace_result)

    assert "retrieve" in output
    assert "query_chunks" in output
    assert "keyword_search" in output
    assert "├──" in output or "└──" in output


def test_format_trace_tree_not_found():
    trace_result = {
        "root": "nonexistent",
        "not_found": True,
        "nodes": {},
        "visited_count": 0,
        "truncated": False,
    }

    output = format_trace_tree(trace_result)
    assert "not found in index" in output


def test_format_trace_tree_already_visited():
    trace_result = {
        "root": "A",
        "not_found": False,
        "nodes": {
            "A": {
                "chunk": _make_chunk("A", ["B"]),
                "depth": 0,
                "parent": None,
                "children": ["B"],
                "child_already_visited": [],
                "truncated": False,
            },
            "B": {
                "chunk": _make_chunk("B", ["A"]),
                "depth": 1,
                "parent": "A",
                "children": ["A"],
                "child_already_visited": ["A"],
                "truncated": False,
            },
        },
        "visited_count": 2,
        "truncated": False,
    }

    output = format_trace_tree(trace_result)
    assert "already visited" in output


def test_format_trace_tree_truncated_node():
    trace_result = {
        "root": "root",
        "not_found": False,
        "nodes": {
            "root": {
                "chunk": _make_chunk("root", ["child"]),
                "depth": 0,
                "parent": None,
                "children": ["child"],
                "child_already_visited": [],
                "truncated": False,
            },
            "child": {
                "chunk": _make_chunk("child", ["grandchild"]),
                "depth": 1,
                "parent": "root",
                "children": [],
                "child_already_visited": [],
                "truncated": True,
            },
        },
        "visited_count": 2,
        "truncated": False,
    }

    output = format_trace_tree(trace_result)
    assert "truncated" in output or "--depth" in output


# ── run_trace ─────────────────────────────────────────────────────────────────

def test_run_trace_no_api_calls_by_default(tmp_path):
    chunks = {
        "retrieve": _make_chunk("retrieve", ["query_chunks"]),
        "query_chunks": _make_chunk("query_chunks", []),
    }
    all_chunks_data = [
        {**_make_chunk("caller", ["retrieve"]), "file_rel_path": "repolix/cli.py"},
    ]

    with (
        patch("repolix.trace.lookup_chunk_by_name", side_effect=_lookup_factory(chunks)),
        patch("repolix.tour.get_all_chunks", return_value=all_chunks_data),
    ):
        # Passing None as openai_client — must not raise AttributeError
        result = run_trace("retrieve", tmp_path, explain=False, openai_client=None)

    assert result["error"] is None
    assert result["explanation"] is None
    assert result["symbol"] == "retrieve"
    assert result["tree_str"] != ""


def test_run_trace_not_found_returns_error(tmp_path):
    with patch("repolix.trace.lookup_chunk_by_name", return_value=None):
        result = run_trace("nonexistent", tmp_path)

    assert result["error"] is not None
    assert "not found" in result["error"]
    assert result["forward"]["not_found"] is True


def test_run_trace_backward_included_by_default(tmp_path):
    chunks = {
        "retrieve": _make_chunk("retrieve", []),
    }
    callers = [
        {**_make_chunk("query", ["retrieve"]), "file_rel_path": "repolix/cli.py"},
    ]

    with (
        patch("repolix.trace.lookup_chunk_by_name", side_effect=_lookup_factory(chunks)),
        patch("repolix.tour.get_all_chunks", return_value=callers),
    ):
        result = run_trace("retrieve", tmp_path, include_backward=True)

    assert len(result["backward"]) == 1
    assert result["backward"][0]["name"] == "query"


def test_run_trace_backward_excluded_when_flag_false(tmp_path):
    chunks = {
        "retrieve": _make_chunk("retrieve", []),
    }

    with patch("repolix.trace.lookup_chunk_by_name", side_effect=_lookup_factory(chunks)):
        result = run_trace("retrieve", tmp_path, include_backward=False)

    assert result["backward"] == []


# ── answer_trace ──────────────────────────────────────────────────────────────

def test_answer_trace_calls_openai_once():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "The retrieve function orchestrates..."
    mock_client.chat.completions.create.return_value = mock_response

    result = answer_trace(
        tree_str="retrieve  [repolix/retriever.py:58]\n└── query_chunks  [repolix/store.py:180]",
        backward=[{"name": "query", "file_rel_path": "repolix/cli.py", "start_line": 181}],
        symbol="retrieve",
        openai_client=mock_client,
    )

    mock_client.chat.completions.create.assert_called_once()
    assert isinstance(result, str)
    assert len(result) > 0


def test_answer_trace_empty_backward():
    """answer_trace should handle an empty backward list gracefully."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices[0].message.content = "Some explanation."
    mock_client.chat.completions.create.return_value = mock_response

    result = answer_trace(
        tree_str="root  [some/file.py:1]",
        backward=[],
        symbol="root",
        openai_client=mock_client,
    )

    call_args = mock_client.chat.completions.create.call_args
    user_content = call_args.kwargs["messages"][1]["content"]
    assert "(no callers found in index)" in user_content
    assert isinstance(result, str)
