"""
cli.py

Click-based command line interface for repolens.

Commands:
  repolens index <repo_path>   — index a repository
  repolens query <question>    — query an indexed repository

The store directory defaults to .repolens/ inside the repo being
indexed. This keeps the index co-located with the repo it describes
and makes cleanup obvious — delete .repolens/ to reset the index.
"""

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import OpenAI

from repolens.store import index_repo
from repolens.retriever import retrieve, format_results
from repolens.llm import answer_query

# Load .env file if present. This must happen before any os.getenv
# calls. load_dotenv is a no-op if .env does not exist, so it is
# safe to call unconditionally.
load_dotenv()


def get_openai_client() -> OpenAI:
    """
    Create an authenticated OpenAI client from the environment.

    Raises:
        click.ClickException: If OPENAI_API_KEY is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise click.ClickException(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file or export it in your shell."
        )
    return OpenAI(api_key=api_key)


def _confidence_label(top_score: float) -> str:
    """
    Map the top rerank_score to a human-readable confidence label.

    Thresholds are derived from the scoring weights in retriever.rerank():
      +0.3 name match, +0.2 file path match, +0.15 docstring match.
      RRF base for a top-ranked result in both lists is ~0.033.

      high   >= 0.4  — name or file path matched the query
      medium >= 0.15 — docstring or call graph matched
      low    <  0.15 — vector similarity only, no metadata signal
    """
    if top_score >= 0.4:
        return "high"
    if top_score >= 0.15:
        return "medium"
    return "low"


def resolve_store_path(repo_path: Path, store: str | None) -> Path:
    """
    Resolve the ChromaDB store directory.

    If --store is provided, use that path. Otherwise default to
    .repolens/ inside the repo being indexed. The directory is
    created if it does not exist.
    """
    if store:
        store_path = Path(store).resolve()
    else:
        store_path = repo_path / ".repolens"
    store_path.mkdir(parents=True, exist_ok=True)
    return store_path


@click.group()
@click.version_option(version="0.1.0", prog_name="repolens")
def main():
    """
    repolens — local-first codebase context engine.

    Point it at any Python repo, ask plain English questions,
    get back answers with exact file and line citations.
    """


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--store",
    default=None,
    help="Path to ChromaDB store directory. Defaults to <repo>/.repolens/",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-index all files even if unchanged.",
)
def index(repo_path: str, store: str | None, force: bool):
    """
    Index a repository for querying.

    REPO_PATH is the path to the repository root to index.

    Example:
      repolens index ./myrepo
      repolens index ./myrepo --force
      repolens index ./myrepo --store /tmp/myrepo-index
    """
    repo = Path(repo_path).resolve()
    client = get_openai_client()
    store_path = resolve_store_path(repo, store)

    click.echo(f"Indexing {repo}")
    click.echo(f"Store:   {store_path}")
    click.echo("")

    # We use a mutable container for progress state so the callback
    # closure can update it. Plain int variables in closures are
    # read-only in Python — you cannot rebind them from inside the
    # closure without nonlocal. A list sidesteps this.
    progress_state = [0]

    with click.progressbar(
        length=1,
        label="Indexing",
        show_pos=True,
        show_percent=True,
    ) as bar:

        def progress_callback(current: int, total: int, file_path: str):
            # Update the progress bar length on first call when we
            # know the total. Click allows dynamic length updates.
            if current == 1:
                bar.length = total
                bar.pos = 0
            bar.update(1)
            progress_state[0] = current

        stats = index_repo(
            repo_path=repo,
            store_path=store_path,
            openai_client=client,
            force=force,
            progress_callback=progress_callback,
        )

    click.echo("")
    click.echo("── Index complete ──────────────────────────")
    click.echo(f"Files found:    {stats['total_files']}")
    click.echo(f"Files indexed:  {stats['indexed']}")
    click.echo(f"Files skipped:  {stats['skipped']} (unchanged)")
    click.echo(f"Chunks stored:  {stats['total_chunks']}")

    if stats["errors"]:
        click.echo(f"\nErrors ({len(stats['errors'])}):")
        for err in stats["errors"]:
            click.echo(f"  {err}", err=True)
        sys.exit(1)


@main.command()
@click.argument("question")
@click.option(
    "--repo",
    default=".",
    show_default=True,
    help="Path to the repository that was indexed.",
)
@click.option(
    "--store",
    default=None,
    help="Path to ChromaDB store directory. Defaults to <repo>/.repolens/",
)
@click.option(
    "--no-llm",
    is_flag=True,
    default=False,
    help="Skip LLM call and return raw retrieved chunks only.",
)
@click.option(
    "--n",
    default=5,
    show_default=True,
    help="Number of chunks to retrieve.",
)
def query(question: str, repo: str, store: str | None, no_llm: bool, n: int):
    """
    Query an indexed repository with a plain English question.

    QUESTION is your plain English question about the codebase.

    Examples:
      repolens query "how does authentication work"
      repolens query "where is the database connection set up"
      repolens query "what does UserService do" --no-llm
    """
    repo_path = Path(repo).resolve()
    client = get_openai_client()
    store_path = resolve_store_path(repo_path, store)

    if not (store_path / "chroma.sqlite3").exists():
        raise click.ClickException(
            f"No index found at {store_path}. "
            f"Run: repolens index {repo_path}"
        )

    click.echo(f"Query:  {question}")
    click.echo(f"Store:  {store_path}")
    click.echo("")

    click.echo("Searching...")
    results = retrieve(
        query=question,
        store_path=store_path,
        openai_client=client,
    )

    if no_llm or not results:
        click.echo(format_results(results))
        return

    click.echo("Generating answer...")
    output = answer_query(
        query=question,
        results=results,
        openai_client=client,
    )

    click.echo("── Answer ──────────────────────────────────")
    click.echo(output["answer"])
    click.echo("")
    click.echo("── Citations ───────────────────────────────")

    if output["citations"]:
        for citation in output["citations"]:
            label = citation["label"]
            path = citation["file_rel_path"]
            start = citation["start_line"]
            end = citation["end_line"]
            name = citation["name"]
            parent = citation.get("parent_class")
            context = f"{parent}.{name}" if parent else name
            truncated = " [truncated]" if citation.get("is_truncated") else ""
            click.echo(f"  {label} {path}:{start}-{end}  ({context}){truncated}")
    else:
        click.echo("  No citations extracted.")

    click.echo("")
    top_score = results[0].get("rerank_score", 0.0) if results else 0.0
    confidence = _confidence_label(top_score)
    click.echo(
        f"[confidence: {confidence} · {output['chunks_used']} chunks"
        f" · index: {store_path}]"
    )
