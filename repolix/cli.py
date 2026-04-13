"""
cli.py

Click-based command line interface for repolix.

Commands:
  repolix index <repo_path>   — index a repository
  repolix query <question>    — query an indexed repository

The store directory defaults to .repolix/ inside the repo being
indexed. This keeps the index co-located with the repo it describes
and makes cleanup obvious — delete .repolix/ to reset the index.
"""

import os
import sys
from pathlib import Path

import click
from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn
from rich.rule import Rule

from repolix.store import index_repo
from repolix.retriever import retrieve, format_results, display_rel_path_from_meta
from repolix.llm import answer_query

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
    .repolix/ inside the repo being indexed. The directory is
    created if it does not exist.
    """
    if store:
        store_path = Path(store).resolve()
    else:
        store_path = repo_path / ".repolix"
    store_path.mkdir(parents=True, exist_ok=True)
    return store_path


@click.group()
@click.version_option(version="0.1.1", prog_name="repolix")
def main():
    """
    repolix — local-first codebase context engine.

    Point it at any Python repo, ask plain English questions,
    get back answers with exact file and line citations.
    """


@main.command()
@click.argument("repo_path", type=click.Path(exists=True, file_okay=False))
@click.option(
    "--store",
    default=None,
    help="Path to ChromaDB store directory. Defaults to <repo>/.repolix/",
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
      repolix index ./myrepo
      repolix index ./myrepo --force
      repolix index ./myrepo --store /tmp/myrepo-index
    """
    # Console created here (not at module level) so Click's CliRunner
    # can capture output correctly in tests — CliRunner patches sys.stdout
    # after module load, so a module-level Console would miss that patch.
    console = Console(highlight=False)

    repo = Path(repo_path).resolve()
    client = get_openai_client()
    store_path = resolve_store_path(repo, store)

    console.print(f"[dim]Indexing {repo}[/dim]")

    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    ) as progress:
        # total=None starts the bar in indeterminate (spinner) mode until
        # the first callback fires and we know the real file count.
        task = progress.add_task("Indexing", total=None)

        def progress_callback(current: int, total: int, file_path: str):
            if current == 1:
                progress.update(task, total=total)
            progress.update(task, completed=current)

        stats = index_repo(
            repo_path=repo,
            store_path=store_path,
            openai_client=client,
            force=force,
            progress_callback=progress_callback,
        )

    summary_lines = [
        f"Files found:    {stats['total_files']}",
        f"Files indexed:  {stats['indexed']}",
        f"Files skipped:  {stats['skipped']} (unchanged)",
        f"Chunks stored:  {stats['total_chunks']}",
    ]
    if stats.get("cleaned", 0):
        summary_lines.append(
            f"[dim]Orphans removed: {stats['cleaned']} (deleted/renamed files)[/dim]"
        )
    console.print(Panel("\n".join(summary_lines), title="[bold]Index Complete[/bold]", border_style="dim"))

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
    help="Path to ChromaDB store directory. Defaults to <repo>/.repolix/",
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
      repolix query "how does authentication work"
      repolix query "where is the database connection set up"
      repolix query "what does UserService do" --no-llm
    """
    console = Console(highlight=False)

    repo_path = Path(repo).resolve()
    client = get_openai_client()
    store_path = resolve_store_path(repo_path, store)

    if not (store_path / "chroma.sqlite3").exists():
        raise click.ClickException(
            f"No index found at {store_path}. "
            f"Run: repolix index {repo_path}"
        )

    console.print("[dim]Searching...[/dim]")
    results = retrieve(
        query=question,
        store_path=store_path,
        openai_client=client,
    )

    if no_llm or not results:
        click.echo(format_results(results))
        return

    console.print("[dim]Generating answer...[/dim]")
    output = answer_query(
        query=question,
        results=results,
        openai_client=client,
    )

    console.print(Panel(output["answer"], title="[bold cyan]Answer[/bold cyan]", border_style="cyan"))

    console.print(Rule("[dim]Citations[/dim]", style="dim"))

    if output["citations"]:
        for citation in output["citations"]:
            label = citation["label"]
            path = display_rel_path_from_meta(citation)
            start = citation["start_line"]
            end = citation["end_line"]
            name = citation["name"]
            parent = citation.get("parent_class")
            context = f"{parent}.{name}" if parent else name
            truncated = "  [dim][truncated][/dim]" if citation.get("is_truncated") else ""
            # escape() prevents [1], [2] label brackets from being parsed
            # as Rich markup tags.
            console.print(
                f"  [bold]{escape(label)}[/bold] {path}:{start}-{end}  ({context}){truncated}"
            )
    else:
        console.print("  No citations extracted.")

    top_score = results[0].get("rerank_score", 0.0) if results else 0.0
    confidence = _confidence_label(top_score)
    console.print(f"\n[dim]confidence: {confidence}[/dim]")
