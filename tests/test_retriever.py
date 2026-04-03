"""
Tests for repolens/retriever.py — Milestone 6: hybrid search.

query_chunks and keyword_search are mocked throughout.
Tests cover RRF merging logic, re-ranking, and format_results.
"""

import pytest
from unittest.mock import MagicMock, patch
from repolens.retriever import (
    retrieve,
    reciprocal_rank_fusion,
    rerank,
    format_results,
    RETURN_N,
    RRF_K,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_result(**kwargs) -> dict:
    defaults = dict(
        source="def authenticate_user(token):\n    return True",
        file_path="/repo/auth.py",
        name="authenticate_user",
        node_type="function_definition",
        start_line=1,
        end_line=2,
        calls=["validate_token"],
        docstring="Validates user credentials.",
        distance=0.2,
    )
    defaults.update(kwargs)
    return defaults


# ── reciprocal_rank_fusion ────────────────────────────────────────────────────

class TestReciprocalRankFusion:

    def test_result_in_both_lists_scores_higher_than_either_alone(self):
        shared = make_result(name="shared_func", start_line=1)
        vector_only = make_result(name="vector_func", start_line=2,
                                  file_path="/repo/b.py")
        keyword_only = make_result(name="keyword_func", start_line=3,
                                   file_path="/repo/c.py")

        vector_results = [shared, vector_only]
        keyword_results = [shared, keyword_only]

        merged = reciprocal_rank_fusion(vector_results, keyword_results)
        names = [r["name"] for r in merged]

        assert names[0] == "shared_func"

    def test_rrf_score_added_to_each_result(self):
        results = [make_result()]
        merged = reciprocal_rank_fusion(results, [])
        assert "rrf_score" in merged[0]

    def test_rrf_score_formula_correct(self):
        result = make_result()
        merged = reciprocal_rank_fusion([result], [])
        expected = 1.0 / (RRF_K + 1)
        assert abs(merged[0]["rrf_score"] - expected) < 1e-9

    def test_deduplicates_results_appearing_in_both_lists(self):
        result = make_result()
        merged = reciprocal_rank_fusion([result], [result])
        assert len(merged) == 1

    def test_empty_vector_results_returns_keyword_results(self):
        result = make_result()
        merged = reciprocal_rank_fusion([], [result])
        assert len(merged) == 1

    def test_empty_keyword_results_returns_vector_results(self):
        result = make_result()
        merged = reciprocal_rank_fusion([result], [])
        assert len(merged) == 1

    def test_both_empty_returns_empty(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_sorted_descending_by_rrf_score(self):
        r1 = make_result(name="func_a", start_line=1)
        r2 = make_result(name="func_b", start_line=2,
                         file_path="/repo/b.py")
        # r1 appears in both lists, r2 only in vector
        merged = reciprocal_rank_fusion([r1, r2], [r1])
        scores = [r["rrf_score"] for r in merged]
        assert scores == sorted(scores, reverse=True)


# ── rerank ────────────────────────────────────────────────────────────────────

class TestRerank:

    def test_uses_rrf_score_as_base_when_present(self):
        result = make_result(rrf_score=0.05)
        ranked = rerank("anything", [result])
        assert ranked[0]["rerank_score"] >= 0.05

    def test_name_match_boosts_score(self):
        no_match = make_result(name="unrelated", rrf_score=0.02,
                               start_line=1)
        match = make_result(name="authenticate_user", rrf_score=0.015,
                            start_line=2, file_path="/repo/b.py")
        ranked = rerank("authenticate", [no_match, match])
        assert ranked[0]["name"] == "authenticate_user"

    def test_rerank_score_added_to_all_results(self):
        results = [make_result(rrf_score=0.02, start_line=i,
                               file_path=f"/repo/{i}.py")
                   for i in range(3)]
        ranked = rerank("query", results)
        assert all("rerank_score" in r for r in ranked)

    def test_sorted_descending_by_rerank_score(self):
        results = [make_result(rrf_score=0.01 * i, start_line=i,
                               file_path=f"/repo/{i}.py")
                   for i in range(1, 5)]
        ranked = rerank("query", results)
        scores = [r["rerank_score"] for r in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_results_returns_empty(self):
        assert rerank("query", []) == []


# ── retrieve integration ──────────────────────────────────────────────────────

class TestRetrieve:

    def test_returns_at_most_return_n(self):
        many = [make_result(name=f"func_{i}", start_line=i,
                            file_path=f"/repo/{i}.py")
                for i in range(10)]
        with patch("repolens.retriever.query_chunks", return_value=many), \
             patch("repolens.retriever.keyword_search", return_value=[]):
            results = retrieve("query", "/fake/db", MagicMock())
        assert len(results) <= RETURN_N

    def test_returns_empty_when_both_searches_empty(self):
        with patch("repolens.retriever.query_chunks", return_value=[]), \
             patch("repolens.retriever.keyword_search", return_value=[]):
            results = retrieve("query", "/fake/db", MagicMock())
        assert results == []

    def test_results_have_rerank_score(self):
        raw = [make_result(name=f"func_{i}", start_line=i,
                           file_path=f"/repo/{i}.py")
               for i in range(3)]
        with patch("repolens.retriever.query_chunks", return_value=raw), \
             patch("repolens.retriever.keyword_search", return_value=[]):
            results = retrieve("query", "/fake/db", MagicMock())
        assert all("rerank_score" in r for r in results)

    def test_keyword_only_results_still_returned(self):
        keyword_result = make_result(name="exact_match_func")
        with patch("repolens.retriever.query_chunks", return_value=[]), \
             patch("repolens.retriever.keyword_search",
                   return_value=[keyword_result]):
            results = retrieve("exact_match_func", "/fake/db", MagicMock())
        assert len(results) == 1
        assert results[0]["name"] == "exact_match_func"


# ── format_results ────────────────────────────────────────────────────────────

class TestFormatResults:

    def test_empty_returns_no_results_message(self):
        assert "No results found" in format_results([])

    def test_contains_file_path(self):
        r = rerank("q", [make_result(rrf_score=0.02)])[0]
        assert "/repo/auth.py" in format_results([r])

    def test_contains_function_name(self):
        r = rerank("q", [make_result(rrf_score=0.02)])[0]
        assert "authenticate_user" in format_results([r])

    def test_contains_line_numbers(self):
        r = rerank("q", [make_result(start_line=10, end_line=20,
                                     rrf_score=0.02)])[0]
        output = format_results([r])
        assert "10" in output
        assert "20" in output

    def test_multiple_results_numbered(self):
        results = [
            rerank("q", [make_result(name=f"func_{i}",
                                     start_line=i,
                                     file_path=f"/repo/{i}.py",
                                     rrf_score=0.02)])[0]
            for i in range(1, 4)
        ]
        output = format_results(results)
        assert "Result 1" in output
        assert "Result 2" in output
        assert "Result 3" in output
