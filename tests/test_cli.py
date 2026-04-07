"""
Tests for repolix/cli.py.

Uses Click's test runner (CliRunner) which invokes CLI commands
in-process without spawning a subprocess. OpenAI and store
functions are mocked throughout.
"""

import pytest
from unittest.mock import patch, MagicMock
from click.testing import CliRunner
from repolix.cli import main, _confidence_label


def mock_index_repo_result(
    total=3, indexed=2, skipped=1, chunks=10, errors=None
):
    return {
        "total_files": total,
        "indexed": indexed,
        "skipped": skipped,
        "total_chunks": chunks,
        "errors": errors or [],
    }


class TestConfidenceLabel:

    def test_high_confidence_above_threshold(self):
        assert _confidence_label(0.5) == "high"
        assert _confidence_label(0.4) == "high"

    def test_medium_confidence_in_range(self):
        assert _confidence_label(0.3) == "medium"
        assert _confidence_label(0.15) == "medium"

    def test_low_confidence_below_threshold(self):
        assert _confidence_label(0.1) == "low"
        assert _confidence_label(0.0) == "low"

    def test_boundary_at_0_4_is_high(self):
        assert _confidence_label(0.4) == "high"

    def test_just_below_0_4_is_medium(self):
        assert _confidence_label(0.399) == "medium"


class TestIndexCommand:

    def test_index_succeeds_with_valid_repo(self, tmp_path):
        with patch("repolix.cli.get_openai_client") as mock_client, \
             patch("repolix.cli.index_repo",
                   return_value=mock_index_repo_result()) as mock_index:
            mock_client.return_value = MagicMock()
            runner = CliRunner()
            result = runner.invoke(main, ["index", str(tmp_path)])
            assert result.exit_code == 0
            assert "Index complete" in result.output

    def test_index_shows_file_counts(self, tmp_path):
        with patch("repolix.cli.get_openai_client"), \
             patch("repolix.cli.index_repo",
                   return_value=mock_index_repo_result(
                       total=5, indexed=4, skipped=1, chunks=20
                   )):
            runner = CliRunner()
            result = runner.invoke(main, ["index", str(tmp_path)])
            assert "5" in result.output
            assert "4" in result.output
            assert "20" in result.output

    def test_index_exits_nonzero_on_errors(self, tmp_path):
        with patch("repolix.cli.get_openai_client"), \
             patch("repolix.cli.index_repo",
                   return_value=mock_index_repo_result(
                       errors=["file.py: parse error"]
                   )):
            runner = CliRunner()
            result = runner.invoke(main, ["index", str(tmp_path)])
            assert result.exit_code == 1

    def test_index_missing_api_key_shows_error(self, tmp_path):
        with patch("repolix.cli.os.getenv", return_value=None):
            runner = CliRunner()
            result = runner.invoke(main, ["index", str(tmp_path)])
            assert "OPENAI_API_KEY" in result.output


class TestQueryCommand:

    def test_query_no_index_shows_helpful_error(self, tmp_path):
        with patch("repolix.cli.get_openai_client"):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["query", "how does auth work", "--repo", str(tmp_path)]
            )
            assert "No index found" in result.output

    def test_query_no_llm_flag_skips_llm(self, tmp_path):
        store = tmp_path / ".repolix"
        store.mkdir()
        (store / "chroma.sqlite3").touch()

        mock_results = [{
            "source": "def foo(): pass",
            "file_path": str(tmp_path / "foo.py"),
            "file_rel_path": "foo.py",
            "name": "foo",
            "node_type": "function_definition",
            "start_line": 1,
            "end_line": 1,
            "calls": [],
            "docstring": None,
            "parent_class": None,
            "distance": 0.1,
            "rrf_score": 0.02,
            "rerank_score": 0.32,
        }]

        with patch("repolix.cli.get_openai_client"), \
             patch("repolix.cli.retrieve", return_value=mock_results), \
             patch("repolix.cli.answer_query") as mock_llm:
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["query", "how does foo work",
                 "--repo", str(tmp_path), "--no-llm"]
            )
            mock_llm.assert_not_called()

    def test_query_shows_answer_and_citations(self, tmp_path):
        store = tmp_path / ".repolix"
        store.mkdir()
        (store / "chroma.sqlite3").touch()

        mock_results = [{
            "source": "def foo(): pass",
            "file_path": str(tmp_path / "foo.py"),
            "file_rel_path": "foo.py",
            "name": "foo",
            "node_type": "function_definition",
            "start_line": 1,
            "end_line": 1,
            "calls": [],
            "docstring": None,
            "parent_class": None,
            "distance": 0.1,
            "rrf_score": 0.02,
            "rerank_score": 0.32,
        }]

        mock_answer = {
            "answer": "foo does something [1].",
            "citations": [{
                "label": "[1]",
                "file_rel_path": "foo.py",
                "start_line": 1,
                "end_line": 1,
                "name": "foo",
                "parent_class": None,
            }],
            "chunks_used": 1,
        }

        with patch("repolix.cli.get_openai_client"), \
             patch("repolix.cli.retrieve", return_value=mock_results), \
             patch("repolix.cli.answer_query", return_value=mock_answer):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["query", "what does foo do", "--repo", str(tmp_path)]
            )
            assert "Answer" in result.output
            assert "Citations" in result.output
            assert "confidence:" in result.output
