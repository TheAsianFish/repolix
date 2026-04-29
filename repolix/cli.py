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

from repolix import __version__
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
    if top_score >= 0.05:
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
@click.version_option(version=__version__, prog_name="repolix")
def main():
    """
    repolix — local-first codebase context engine.

    Point it at a Python or JavaScript/TypeScript repo, ask plain English questions,
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
@click.option(
    "--include-tests",
    is_flag=True,
    default=False,
    help="Include test files and directories in the index. "
         "Excluded by default to improve retrieval quality.",
)
def index(repo_path: str, store: str | None, force: bool, include_tests: bool):
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
            exclude_tests=not include_tests,
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
    if not include_tests:
        summary_lines.append(
            "[dim]Test dirs excluded  (use --include-tests to override)[/dim]"
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

    sections = output.get("answer_sections")
    navigation = output.get("navigation")
    confidence = output.get("confidence", "low")

    if navigation:
        # Low confidence — render navigational response instead of answer
        nav_lines = [
            f"[yellow]{escape(navigation['message'])}[/yellow]",
            "",
        ]
        for match in navigation.get("closest_matches", []):
            name = escape(match.get("name", ""))
            path = escape(match.get("file_rel_path", ""))
            line = match.get("start_line", 0)
            nav_lines.append(f"  [dim]→[/dim] [bold]{name}[/bold]  [dim]{path}:{line}[/dim]")

        if navigation.get("suggestions"):
            nav_lines.append("")
            nav_lines.append("[dim]Suggestions:[/dim]")
            for s in navigation["suggestions"]:
                nav_lines.append(f"  [dim]·[/dim] {escape(s)}")

        console.print(Panel(
            "\n".join(nav_lines),
            title="[bold yellow]Low Confidence[/bold yellow]",
            border_style="yellow",
        ))

    elif sections:
        # Structured answer — render each section distinctly
        from rich.text import Text

        content = Text()

        # Direct answer — bold, full weight
        direct = sections.get("answer", "").strip()
        if direct:
            content.append(direct + "\n", style="bold white")

        # How it works — normal weight, separated by blank line
        how = sections.get("how_it_works", "")
        if how:
            content.append("\n")
            content.append("How it works\n", style="dim cyan")
            content.append(how.strip() + "\n", style="default")

        # Where to look next — dim, only if present
        where = sections.get("where_to_look", "")
        if where:
            content.append("\n")
            content.append("Where to look next\n", style="dim cyan")
            content.append(where.strip() + "\n", style="dim")

        console.print(Panel(
            content,
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
        ))

    else:
        # Fallback — render raw answer string (handles LLM format failures)
        console.print(Panel(
            escape(output.get("answer") or "No answer returned."),
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
        ))

    if not navigation:
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

    console.print(f"\n[dim]confidence: {escape(output.get('confidence', 'low'))}[/dim]")


@main.command()
@click.argument(
    "repo_path",
    type=click.Path(exists=True, file_okay=False),
    default=".",
)
@click.option(
    "--store",
    default=None,
    help="Path to ChromaDB store directory. Defaults to <repo>/.repolix/",
)
@click.option(
    "--path",
    "scope_path",
    default=None,
    help="Scope the tour to a subdirectory. e.g. --path src/payments",
)
@click.option(
    "--save",
    is_flag=True,
    default=False,
    help="Save the briefing to .repolix/tour.md",
)
def tour(repo_path: str, store: str | None, scope_path: str | None, save: bool):
    """
    Generate a proactive orientation briefing for a repository.

    Analyzes the indexed codebase structure and produces a plain-English
    briefing covering entry points, major modules, and key abstractions.

    The repo must be indexed first: repolix index <repo_path>

    Examples:
      repolix tour .
      repolix tour . --path src/payments
      repolix tour . --save
    """
    from repolix.tour import generate_tour
    from rich.text import Text

    console = Console(highlight=False)
    repo = Path(repo_path).resolve()
    client = get_openai_client()
    store_path = resolve_store_path(repo, store)

    scope_display = f" ({scope_path})" if scope_path else ""
    console.print(f"[dim]Generating tour{scope_display}...[/dim]")

    result = generate_tour(
        store_path=store_path,
        repo_path=repo,
        openai_client=client,
        path_prefix=scope_path,
    )

    if result["error"]:
        raise click.ClickException(result["error"])

    sections = result.get("briefing_sections", {})

    section_display = [
        ("OVERVIEW", sections.get("overview"), "bold white"),
        ("ENTRY POINTS", sections.get("entry_points"), "default"),
        ("MAJOR MODULES", sections.get("major_modules"), "default"),
        ("KEY ABSTRACTIONS", sections.get("key_abstractions"), "default"),
        ("START HERE", sections.get("start_here"), "dim"),
    ]

    content = Text()
    for header, body, style in section_display:
        if body:
            content.append(f"{header}\n", style="dim cyan")
            content.append(f"{body}\n\n", style=style)

    console.print(Panel(
        content,
        title="[bold cyan]Tour[/bold cyan]",
        border_style="cyan",
    ))

    top = result.get("top_functions", [])
    if top:
        console.print(Rule("[dim]Most Referenced[/dim]", style="dim"))
        for name, count in top:
            console.print(
                f"  [bold]{name}[/bold]  "
                f"[dim]called by {count} functions[/dim]"
            )

    console.print(
        f"\n[dim]Analyzed {result['chunk_count']} chunks[/dim]"
    )

    if save:
        save_path = store_path / "tour.md"
        save_path.write_text(result["briefing"] or "")
        console.print(f"[dim]Saved to {save_path}[/dim]")


@main.command()
@click.argument("symbol")
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
    "--depth",
    default=3,
    show_default=True,
    help="Maximum BFS depth for forward traversal.",
)
@click.option(
    "--max-nodes",
    default=20,
    show_default=True,
    help="Maximum number of nodes to visit.",
)
@click.option(
    "--reverse",
    is_flag=True,
    default=False,
    help="Show callers of SYMBOL instead of what SYMBOL calls.",
)
@click.option(
    "--explain",
    is_flag=True,
    default=False,
    help="Add LLM explanation of the call chain (uses 1 API call).",
)
def trace(
    symbol: str,
    repo: str,
    store: str | None,
    depth: int,
    max_nodes: int,
    reverse: bool,
    explain: bool,
):
    """
    Trace the call graph for a named function or class.

    SYMBOL is the exact function or class name to trace.

    Examples:
      repolix trace retrieve
      repolix trace retrieve --depth 5
      repolix trace retrieve --reverse
      repolix trace index_repo --explain
    """
    from repolix.trace import run_trace
    from rich.text import Text

    console = Console(highlight=False)
    repo_path = Path(repo).resolve()
    store_path = resolve_store_path(repo_path, store)

    if not (store_path / "chroma.sqlite3").exists():
        raise click.ClickException(
            f"No index found at {store_path}. "
            f"Run: repolix index {repo_path}"
        )

    client = get_openai_client() if explain else None
    console.print(f"[dim]Tracing {symbol}...[/dim]")

    result = run_trace(
        symbol=symbol,
        store_path=store_path,
        max_depth=depth,
        max_nodes=max_nodes,
        include_backward=not reverse,
        openai_client=client,
        explain=explain,
    )

    if result["error"] and result["forward"].get("not_found"):
        raise click.ClickException(result["error"])

    if reverse:
        console.print(
            Rule(f"[dim]Callers of {symbol}[/dim]", style="dim")
        )
        callers = result["backward"]
        if not callers:
            console.print(f"  [dim]No callers found for '{symbol}'[/dim]")
        else:
            for caller in callers:
                path = caller["file_rel_path"]
                line = caller["start_line"]
                console.print(
                    f"  [bold]{escape(caller['name'])}[/bold]"
                    f"  [dim]{path}:{line}[/dim]"
                )
    else:
        console.print(Panel(
            result["tree_str"],
            title=f"[bold cyan]Trace: {escape(symbol)}[/bold cyan]",
            border_style="cyan",
        ))

        if result["backward"]:
            console.print(
                Rule(f"[dim]Callers of {symbol}[/dim]", style="dim")
            )
            for caller in result["backward"]:
                path = caller["file_rel_path"]
                line = caller["start_line"]
                console.print(
                    f"  [bold]{escape(caller['name'])}[/bold]"
                    f"  [dim]{path}:{line}[/dim]"
                )

    fwd = result["forward"]
    visited = fwd.get("visited_count", 0)
    truncated = fwd.get("truncated", False)
    trunc_note = "  [yellow](truncated)[/yellow]" if truncated else ""
    console.print(
        f"\n[dim]{depth} levels · {visited} nodes{trunc_note}[/dim]"
    )

    if result.get("explanation"):
        console.print(Panel(
            result["explanation"],
            title="[bold cyan]Explanation[/bold cyan]",
            border_style="cyan",
        ))
