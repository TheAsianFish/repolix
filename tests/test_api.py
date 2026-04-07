"""
Tests for repolix/api.py.

Uses FastAPI's TestClient which runs the ASGI app in-process
without a real server. index_repo, retrieve, and answer_query
are mocked throughout.
"""

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient
from repolix.api import app

client = TestClient(app)


def mock_index_stats(**kwargs):
    defaults = dict(
        total_files=3,
        indexed=2,
        skipped=1,
        total_chunks=10,
        errors=[],
    )
    defaults.update(kwargs)
    return defaults


def mock_results():
    return [{
        "source": "def foo(): pass",
        "file_path": "/repo/foo.py",
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


class TestHealthEndpoint:

    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestIndexEndpoint:

    def test_index_valid_repo(self, tmp_path):
        with patch("repolix.api.get_openai_client"), \
             patch("repolix.api.index_repo",
                   return_value=mock_index_stats()):
            response = client.post("/index", json={
                "repo_path": str(tmp_path),
                "force": False,
            })
        assert response.status_code == 200
        data = response.json()
        assert data["total_files"] == 3
        assert data["indexed"] == 2
        assert data["total_chunks"] == 10

    def test_index_invalid_repo_returns_400(self):
        with patch("repolix.api.get_openai_client"):
            response = client.post("/index", json={
                "repo_path": "/nonexistent/path",
                "force": False,
            })
        assert response.status_code == 400

    def test_index_missing_api_key_returns_500(self, tmp_path):
        with patch("repolix.api.os.getenv", return_value=None):
            response = client.post("/index", json={
                "repo_path": str(tmp_path),
            })
        assert response.status_code == 500


class TestQueryEndpoint:

    def test_query_no_index_returns_404(self, tmp_path):
        with patch("repolix.api.get_openai_client"):
            response = client.post("/query", json={
                "question": "how does auth work",
                "repo_path": str(tmp_path),
            })
        assert response.status_code == 404

    def test_query_returns_answer_and_citations(self, tmp_path):
        store = tmp_path / ".repolix"
        store.mkdir()
        (store / "chroma.sqlite3").touch()

        mock_answer = {
            "answer": "foo handles auth [1].",
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

        with patch("repolix.api.get_openai_client"), \
             patch("repolix.api.retrieve", return_value=mock_results()), \
             patch("repolix.api.answer_query", return_value=mock_answer):
            response = client.post("/query", json={
                "question": "how does auth work",
                "repo_path": str(tmp_path),
            })

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "foo handles auth [1]."
        assert len(data["citations"]) == 1
        assert len(data["chunks"]) == 1

    def test_query_no_llm_skips_answer(self, tmp_path):
        store = tmp_path / ".repolix"
        store.mkdir()
        (store / "chroma.sqlite3").touch()

        with patch("repolix.api.get_openai_client"), \
             patch("repolix.api.retrieve", return_value=mock_results()), \
             patch("repolix.api.answer_query") as mock_llm:
            response = client.post("/query", json={
                "question": "query",
                "repo_path": str(tmp_path),
                "no_llm": True,
            })
            mock_llm.assert_not_called()

        assert response.status_code == 200
        assert response.json()["answer"] is None


class TestStatusEndpoint:

    def test_status_not_indexed(self, tmp_path):
        response = client.get(f"/status?repo_path={tmp_path}")
        assert response.status_code == 200
        assert response.json()["indexed"] is False

    def test_status_indexed(self, tmp_path):
        store = tmp_path / ".repolix"
        store.mkdir()
        (store / "chroma.sqlite3").touch()
        response = client.get(f"/status?repo_path={tmp_path}")
        assert response.status_code == 200
        assert response.json()["indexed"] is True
