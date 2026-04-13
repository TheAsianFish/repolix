"""
api.py

FastAPI backend for repolix. Exposes the indexing and query
pipeline over HTTP so the React frontend can consume it.

Endpoints:
  POST /index   — index a repository
  POST /query   — query an indexed repository
  GET  /status  — check if a repo has been indexed
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from openai import OpenAI
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles

from repolix.store import index_repo
from repolix.retriever import retrieve, display_rel_path_from_meta
from repolix.llm import answer_query

load_dotenv()


def resolve_repo_path(repo_path: str) -> Path:
    """
    Resolve repo_path to an absolute directory.

    If the client sends '.' or whitespace only, it means "the API process
    working directory". When REPOLIX_DEFAULT_REPO is set, '.' and empty
    strings use that path instead — useful if uvicorn was started outside
    the repo (see start.sh, which cds to the project root).
    """
    stripped = repo_path.strip()
    default_root = os.environ.get("REPOLIX_DEFAULT_REPO", "").strip()
    if stripped in (".", "") and default_root:
        return Path(default_root).expanduser().resolve()
    return Path(repo_path).expanduser().resolve()


# When installed via pip, the pre-built React bundle is copied into the
# repolix package directory (repolix/dist/) before building the wheel.
# In development, fall back to the Vite output at frontend/dist/.
_PKG_DIST = Path(__file__).parent / "dist"
_DEV_DIST = Path(__file__).parent.parent / "frontend" / "dist"
DIST_DIR = _PKG_DIST if _PKG_DIST.exists() else _DEV_DIST

app = FastAPI(
    title="repolix",
    description="Local-first codebase context engine",
    version="0.1.1",
)

# CORS middleware allows the React frontend at localhost:3000 to make
# requests to the FastAPI backend at localhost:8000. Without this,
# browsers block cross-origin requests by default — this is called
# the Same-Origin Policy. We allow all origins in development since
# both client and server run on the user's local machine.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_openai_client() -> OpenAI:
    """Create an authenticated OpenAI client from the environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY is not set on the server.",
        )
    return OpenAI(api_key=api_key)


def get_store_path(repo_path: str) -> Path:
    """Resolve the ChromaDB store path for a given repo."""
    return Path(repo_path).resolve() / ".repolix"


# ── Request / Response Models ─────────────────────────────────────────────────
# Pydantic models define the shape of request bodies and response
# payloads. FastAPI uses them for automatic validation and serialization.
# If a request body doesn't match the model, FastAPI returns a 422
# error before your handler function is ever called.

class IndexRequest(BaseModel):
    repo_path: str
    force: bool = False


class IndexResponse(BaseModel):
    total_files: int
    indexed: int
    skipped: int
    total_chunks: int
    errors: list[str]


class QueryRequest(BaseModel):
    question: str
    repo_path: str
    no_llm: bool = False


class CitationModel(BaseModel):
    label: str
    file_rel_path: str
    start_line: int
    end_line: int
    name: str
    parent_class: str | None


class ChunkModel(BaseModel):
    source: str
    file_rel_path: str
    name: str
    start_line: int
    end_line: int
    rerank_score: float
    parent_class: str | None


class QueryResponse(BaseModel):
    answer: str | None
    citations: list[CitationModel]
    chunks: list[ChunkModel]
    chunks_used: int


class StatusResponse(BaseModel):
    indexed: bool
    store_path: str
    repo_path: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/index", response_model=IndexResponse)
async def index_endpoint(request: IndexRequest):
    """
    Index a repository.

    Walks the repo, chunks every Python file, embeds the chunks,
    and stores everything in ChromaDB. Skips unchanged files unless
    force=True.
    """
    repo_path = resolve_repo_path(request.repo_path)
    if not repo_path.exists() or not repo_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"repo_path does not exist or is not a directory: {repo_path}",
        )

    store_path = get_store_path(str(repo_path))
    store_path.mkdir(parents=True, exist_ok=True)
    client = get_openai_client()

    stats = index_repo(
        repo_path=repo_path,
        store_path=store_path,
        openai_client=client,
        force=request.force,
    )

    return IndexResponse(**stats)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query an indexed repository with a plain English question.

    Returns a structured response with the LLM answer, citations,
    and the raw retrieved chunks for display in the frontend.
    """
    repo_path = resolve_repo_path(request.repo_path)
    store_path = get_store_path(str(repo_path))

    if not (store_path / "chroma.sqlite3").exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No index at {store_path} (missing chroma.sqlite3). "
                f"Resolved repo: {repo_path}. "
                "Run POST /index or `repolix index` for that path. "
                "Note: '.' is the API server working directory unless "
                "REPOLIX_DEFAULT_REPO is set."
            ),
        )

    client = get_openai_client()

    results = retrieve(
        query=request.question,
        store_path=store_path,
        openai_client=client,
    )

    chunks = [
        ChunkModel(
            source=r["source"],
            file_rel_path=display_rel_path_from_meta(r),
            name=r["name"],
            start_line=r["start_line"],
            end_line=r["end_line"],
            rerank_score=r.get("rerank_score", 0.0),
            parent_class=r.get("parent_class"),
        )
        for r in results
    ]

    if request.no_llm or not results:
        return QueryResponse(
            answer=None,
            citations=[],
            chunks=chunks,
            chunks_used=0,
        )

    output = answer_query(
        query=request.question,
        results=results,
        openai_client=client,
    )

    citations = [CitationModel(**c) for c in output["citations"]]

    return QueryResponse(
        answer=output["answer"],
        citations=citations,
        chunks=chunks,
        chunks_used=output["chunks_used"],
    )


@app.get("/status", response_model=StatusResponse)
async def status_endpoint(repo_path: str):
    """
    Check whether a repository has been indexed.

    Returns indexed=True if a ChromaDB store exists at the expected
    path for the given repo. Used by the frontend to show whether
    indexing is needed before querying.
    """
    resolved = resolve_repo_path(repo_path)
    store_path = get_store_path(str(resolved))
    indexed = (store_path / "chroma.sqlite3").exists()

    return StatusResponse(
        indexed=indexed,
        store_path=str(store_path),
        repo_path=str(resolved),
    )


@app.get("/health")
async def health():
    """Health check endpoint. Returns 200 if the server is running."""
    return {"status": "ok", "version": "0.1.1"}


# ── SPA catch-all ─────────────────────────────────────────────────────────────
# Must come AFTER all API routes so /index, /query, /status, /health are matched
# first. Serves the requested file if it exists in frontend/dist (JS, CSS,
# assets), otherwise returns index.html so React Router handles client-side
# routing for deep-link paths like /dashboard or /profile.

@app.get("/{full_path:path}", include_in_schema=False)
async def serve_spa(full_path: str):
    target = DIST_DIR / full_path
    if target.is_file():
        return FileResponse(str(target))
    index_html = DIST_DIR / "index.html"
    if not index_html.exists():
        raise HTTPException(
            status_code=503,
            detail="Frontend not built. Run: cd frontend && npm run build",
        )
    return HTMLResponse(index_html.read_text())


# Mount static files after all routes. Routes take precedence in FastAPI's
# routing table, so all API paths and the catch-all above are matched first.
# The mount provides explicit static-file serving infrastructure and is used
# when the catch-all delegates to FileResponse for direct asset paths.
if DIST_DIR.exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")
