# repolix — Project Context

Paste this file at the top of every new Cursor chat and every new
Claude conversation to restore full project context instantly.
Update this file at the end of every milestone before moving on.

---

## What repolix is

A local-first codebase context engine. Point it at any Python or
JavaScript/TypeScript repo, ask plain English questions, get back answers
with exact file and line number citations. Code never leaves the user's machine.
Free and open source. Built for developer tooling.

Published on PyPI as `repolix` (previously developed under the name
`codesight`; renamed before public launch).

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem, tooling, pip distribution |
| AST parsing | Tree-sitter | Fast, accurate, multi-language ready |
| Embeddings | text-embedding-3-small | Cheap, high quality, 1536 dims |
| Vector store | ChromaDB (persistent, in-process) | Local-first, no server needed |
| LLM | gpt-5.4-mini | Fast, cheap, strong instruction following |
| Web server | FastAPI | Async, simple, automatic validation |
| Frontend | React + TypeScript | SPA served by FastAPI from frontend/dist; dev via Vite at localhost:3000 |
| CLI | Click + Rich | Click handles commands/args; Rich handles styled terminal output |
| Install | pip install repolix | One command |

---

## Architecture decisions — locked in

**AST chunking over line splitting.**
Tree-sitter parses each file into a syntax tree. We split only at
function and class boundaries. Every chunk is a semantically complete
unit. This is the most important decision for retrieval quality.

**Class = one chunk. Methods chunked separately with parent_class.**
_walk_tree descends into class bodies so methods are chunked
individually with parent_class set to the enclosing class name.
This enables disambiguation between similarly named methods on
different classes. Example: AuthService.validate and
UserService.validate are distinct chunks with distinct parent_class.

**Module-level singleton for Parser and Tokenizer.**
_PARSER and _TOKENIZER are module-level in chunker.py. Creating
them per file call was wasteful. They are stateless between calls.

**Enriched embedding text.**
We do not embed raw source alone. We prepend node type and name,
then docstring if present, then source. This improves retrieval
because natural language queries map better to natural language
descriptions than to raw syntax.

**Metadata on every chunk.**
Each chunk carries: file_path, file_rel_path, node_type, name,
source, start_line, end_line, token_count, calls, docstring,
parent_class, is_truncated.
calls is stored in ChromaDB as a comma-joined string because
ChromaDB metadata must be primitive types. Split on read, join
on write. parent_class and docstring stored as empty string when
None — ChromaDB does not accept None metadata values.
is_truncated is stored as bool — True when source was cut at the
300-token cap. Surfaced as [truncated] in CLI citation output.

**Incremental indexing via SHA-256 file hashing.**
Every file gets a SHA-256 hash stored in ChromaDB alongside its
chunks. Re-indexing skips files whose hash has not changed.
Only changed files are re-embedded and re-stored.

**Orphan cleanup on every index run.**
After processing all current files, index_repo compares the full
set of stored hash IDs against the walked file set. Any stored path
not present in the current walk (deleted, renamed, moved) is an
orphan — its chunks and its hash entry are deleted. stats["cleaned"]
records the count. The CLI surfaces this only when cleaned > 0 to
avoid noise on normal runs. This prevents stale code from deleted
files appearing in query results.

**Hybrid search: vector similarity + keyword search, merged via RRF.**
Pure vector search misses exact name matches. Pure keyword search
misses semantic similarity. We do both and merge results using
Reciprocal Rank Fusion (k=60). RRF operates on rank positions not
raw scores, sidestepping the normalization problem entirely.

**Keyword search minimum token length is 2 characters.**
Tokens of length 1 are filtered (match too broadly via $contains).
Tokens of length 2+ are included — this covers common Python
identifiers like os, db, id, fn that were previously silently
dropped by the old > 2 guard.

**Call graph expansion after retrieval.**
After retrieving and ranking top N chunks, we inspect each chunk's
calls list and fetch any called functions not already in results
using exact keyword_search by name. Expansions get rerank_score=0.005
so they never displace primary results — they appear at the end.
Max 3 expansions per query to avoid context overflow.

**Metadata re-ranking as a second pass.**
Top 10 chunks retrieved. Re-ranked using metadata signals:
  +0.3 if query token appears in chunk name
  +0.2 if query token appears in file path stem
  +0.15 if query token appears in docstring
  +0.1 per query token appearing in calls list
Base score is rrf_score. Final score is base + boost.
Top 5 sent to LLM after expansion appended.

**Top 5 chunks max sent to LLM.**
Hard cap. 5 chunks * 300 tokens = 1500 tokens max context.
Safe for gpt-5.4-mini. Prevents context overflow.
Call graph expansions are appended after top 5, up to 3 additional.

**Chunk token cap: 300 tokens.**
Hard cap enforced in chunker.py using tiktoken cl100k_base
encoding — the same encoding gpt-5.4-mini uses internally.
Oversized chunks are truncated, not discarded. is_truncated=True
is set on the Chunk so downstream code and the user can detect it.
Metadata (calls, docstring, name) is extracted from the full AST
node before truncation — only source is cut.

**Line numbers are 1-indexed.**
Tree-sitter uses 0-indexed rows (C convention). We add 1 on
every start_point and end_point extraction. All citations the
user sees are 1-indexed.

**Citations use relative paths.**
file_rel_path is computed at index time relative to repo root
and stored as metadata. All citation output uses file_rel_path
not file_path to avoid exposing absolute paths in user-facing output.
If file_rel_path is missing or blank in stored metadata,
display_rel_path_from_meta() in retriever.py derives a short path
from the last two segments of file_path (never the full absolute
path as the display value). build_prompt, parse_citations, the CLI,
and the API chunk payload all use this helper.

**LLM uses max_completion_tokens, not max_tokens.**
gpt-5.4-mini requires max_completion_tokens. Using max_tokens
raises a 400 BadRequest error. All LLM calls use max_completion_tokens.

**LLM CITATIONS block is stripped before returning answer.**
parse_citations extracts inline citation labels [1], [2] etc. from
the full response text. After parsing, _strip_citations_block removes
everything from the first line starting with "CITATIONS" onward.
The CLI and API render their own formatted citation sections.
This prevents citations appearing twice and saves output tokens.

**Confidence label derived from top rerank_score.**
The CLI footer shows confidence: high/medium/low based on the top
retrieved chunk's rerank_score. Thresholds:
  high   >= 0.4   name or file path matched
  medium >= 0.15  docstring or call graph matched
  low    < 0.15   vector similarity only
This replaces the meaningless "N chunks used" counter.

**Query status messages replace progress bar.**
The CLI prints "Searching..." before retrieval and "Generating
answer..." before the LLM call. A 2-step progress bar was not a
real progress indicator and mislabeled the LLM step as "Retrieving".

**Rich library used for CLI output formatting.**
rich>=13.0.0 added as a dependency. Click handles command parsing;
Rich handles all terminal output: Panel for answers and index
summaries, Rule for citation separators, markup for dim/bold/cyan
styling. Console is created inside each command function (not at
module level) so Click's CliRunner can correctly capture output in
tests — CliRunner patches sys.stdout after module load, so a
module-level Console would hold a stale reference to real stdout.

**FastAPI serves the built React SPA as static files.**
api.py mounts frontend/dist at "/" after all API routes. A catch-all
GET /{full_path:path} route returns the requested file if it exists
in dist (JS, CSS, assets), otherwise returns index.html so React
Router handles client-side routing. Routes registered with @app.get()
take precedence over app.mount() in FastAPI's routing table, so all
API endpoints (/index, /query, /status, /health) are always matched
first. The mount is conditional on frontend/dist existing so the
server starts cleanly without a prior npm run build.
Development still uses bash start.sh (Vite dev server + backend).
The static file serving is for distribution: pip-installed users
have no Node.js runtime dependency.

---

## ChromaDB collections

| Collection | Purpose |
|---|---|
| repolix_chunks | Stores chunk source, embeddings, metadata |
| repolix_hashes | Stores one hash per file for incremental indexing |

ChromaDB persists to .repolix/ inside the indexed repo root.
.repolix/ is gitignored.

Chunk IDs: "{absolute_file_path}:{start_line}"
Hash IDs: "{absolute_file_path}"

---

## Module map

| File | Status | Responsibility |
|---|---|---|
| repolix/walker.py | Complete | Filesystem traversal, file filtering |
| repolix/chunker.py | Complete | AST parsing, chunk + metadata extraction, is_truncated flag |
| repolix/store.py | Complete | Embeddings, ChromaDB storage, retrieval, index_repo orchestrator |
| repolix/retriever.py | Complete | Hybrid search, RRF, re-ranking, call graph expansion; display_rel_path_from_meta for safe citation paths |
| repolix/llm.py | Complete | Prompt construction, gpt-5.4-mini call, citation parsing, CITATIONS block stripping; answer_trace for trace explanations |
| repolix/tour.py | Complete | Call-graph analysis, entry point detection, chunk selection, context formatting, generate_tour orchestrator |
| repolix/trace.py | Complete | BFS forward trace, backward trace (reverse lookup), format_trace_tree, run_trace orchestrator |
| repolix/cli.py | Complete | Click CLI — index, query, tour, and trace commands, confidence label |
| repolix/api.py | Complete | FastAPI backend — /index, /query, /tour, /trace, /status, /health; serves built SPA from frontend/dist |
| frontend/src/ | Complete | React + TypeScript SPA; Vite dev server for development; built output served by FastAPI |
| tests/conftest.py | Complete | Creates minimal frontend/dist stub before TestClient initialises |

Note: repolix/embedder.py was deleted. It was an unimplemented stub;
the embedding logic lives in store.py as _embed_texts and build_embed_text.

---

## Test suite status

| File | Tests | Status |
|---|---|---|
| tests/test_walker.py | 11 | Passing |
| tests/test_chunker.py | 23 | Passing |
| tests/test_store.py | 28 | Passing |
| tests/test_retriever.py | 25 | Passing |
| tests/test_llm.py | 36 | Passing |
| tests/test_cli.py | 12 | Passing |
| tests/test_api.py | 9 | Passing |
| tests/test_tour.py | 24 | Passing |
| tests/test_trace.py | 20 | Passing |

Run all tests: pytest tests/ -v
Total: 213 passing

Note: test counts above are approximate. Always trust the actual
pytest output over this table.

---

## Milestone map

| # | Name | Status |
|---|---|---|
| 1 | Project scaffold + walker | Complete |
| 2 | AST chunker | Complete |
| 3 | Metadata extraction | Complete |
| 4 | Embedding pipeline + vector store | Complete |
| 5 | Basic retrieval | Complete |
| 6 | Hybrid search + re-ranking | Complete |
| 7 | LLM integration + citations | Complete |
| 8 | CLI | Complete |
| 9 | FastAPI backend + React frontend | Complete |
| 10 | Polish + ship | Complete |
| 11 | Post-V1 output quality + UX fixes | Complete |
| 12 | Rename codesight → repolix; publish to PyPI as repolix 0.1.0 | Complete |
| 13 | Rich CLI output polish + LLM system prompt update | Complete |
| 14 | repolix 0.1.1 — React UI polish, citation path fixes, loading states | Complete |
| 15 | V2-1: JavaScript and TypeScript indexing support | Complete |
| 16 | repolix 0.2.0 — PyPI minor release shipping JS/TS indexing | Complete |
| 17 | repolix 0.2.1 — Web UI same-origin fetches, CORS localhost/127.0.0.1, VITE_API_URL | Complete |
| 18 | LLM output layer: structured response format, section parsing, confidence gating | Complete |
| 19 | repolix 0.2.2 — tour command: proactive orientation briefing via call-graph analysis | Complete |
| 20 | repolix trace command: BFS call-graph traversal, forward/reverse/explain modes | Complete |

V1 shipped as repolix 0.1.0 on PyPI; **0.1.1** followed (UI polish and fixes).
**0.2.2** shipped `repolix tour`. **0.2.3** (current development line) adds
`repolix trace` — BFS call-graph traversal for any named function, zero API
calls by default, optional `--explain` for a single LLM narration.
Milestone 20 (trace command) complete.

---

## Milestone 13 — Rich CLI output + LLM system prompt

| Change | Files |
|---|---|
| Add rich>=13.0.0 to dependencies | pyproject.toml |
| Rewrite index command output: dim header, Rich Panel summary | repolix/cli.py |
| Rewrite query command output: dim status, cyan Answer Panel, Rule + citation list, dim confidence footer | repolix/cli.py |
| Console created inside each command function (not module-level) for CliRunner test compatibility | repolix/cli.py |
| Replace system prompt: direct navigation assistant tone, no hedging language, explicit next-search guidance | repolix/llm.py |
| Update test assertion "Index complete" → "Index Complete" to match Panel title casing | tests/test_cli.py |

---

## Post-V1 fixes (Milestone 11)

These were identified after V1 ship and resolved before V2 work begins.

| Fix | File(s) | Commit |
|---|---|---|
| max_tokens → max_completion_tokens for gpt-5.4-mini | llm.py, test_llm.py | de01083 |
| Delete unimplemented embedder.py stub | — | 96248de |
| Strip duplicate CITATIONS block from LLM answer | llm.py, test_llm.py | 230a44c |
| Surface is_truncated flag on chunked output | chunker.py, store.py, llm.py, cli.py | 041c695 |
| Allow 2-char tokens in keyword search (was > 2, now >= 2) | store.py, test_store.py | e140622 |
| Fix mock_openai_client to return N embeddings per N inputs | test_store.py | e140622 |
| Replace chunks-used footer with confidence label | cli.py, test_cli.py | cf015c8 |
| Replace fake 2-step progress bar with status messages | cli.py | a154c10 |
|| Disable noUnusedLocals/noUnusedParameters to unblock npm run build | frontend/tsconfig.json | 338c1e4 |
|| Serve built React SPA from FastAPI; catch-all route for client-side routing | repolix/api.py | 12a2ddd |
|| Add conftest.py to create minimal frontend/dist stub before TestClient initialises | tests/conftest.py | c1d1e69 |
|| Complete pyproject.toml for PyPI: authors, classifiers, tiktoken, package-data | pyproject.toml | 5514e3c |
|| Add MANIFEST.in for sdist completeness | MANIFEST.in | 56f50e1 |
|| Fix DIST_DIR to resolve from package dir when installed via pip | repolix/api.py | e90854a |

---

## Milestone 12 — Rename to repolix + PyPI launch

| Change | Files |
|---|---|
| Rename package folder codesight/ → repolix/ | all Python sources |
| Update all imports from codesight.* → repolix.* | repolix/*.py, tests/*.py |
| Rename CLI entrypoint codesight → repolix | pyproject.toml, cli.py |
| Rename ChromaDB collections to repolix_chunks / repolix_hashes | repolix/store.py |
| Rename store dir .codesight/ → .repolix/ | cli.py, api.py, tests/ |
| Update frontend: title, api.ts comment, package.json name | frontend/* |
| Rename pyproject.toml package name codesight → repolix | pyproject.toml |
| Update README, CONTEXT, MANIFEST.in, .gitignore, start.sh | docs/config |
| Published repolix 0.1.0 to PyPI | — |

---

## PyPI release sequence

**You cannot “update” a version already on PyPI.** Each upload must use a
**new version number** in `pyproject.toml` (e.g. patch `0.1.1` for fixes only,
or `0.2.0` for a minor release with new behavior). Old wheels/sdists stay
forever on the index. A CHANGELOG file is optional; PyPI shows **README.md**
on the project page. For this repo, keeping **CONTEXT.md** current is enough unless you want a public `CHANGELOG.md`.

Run these steps in order before every release:

  npm run build --prefix frontend        # rebuild React bundle
  cp -r frontend/dist repolix/dist        # stage bundle inside Python package
  python -m build                        # creates dist/*.whl and dist/*.tar.gz
  twine check dist/*                     # validate metadata before upload
  twine upload dist/*                    # upload to PyPI (prompts for token)

On PyPI, use an API token (not your password). Create one at
https://pypi.org/manage/account/token/ scoped to the repolix project.
Store it in ~/.pypirc or pass as the password when twine prompts
(username = __token__, password = pypi-...).

Test on TestPyPI first: twine upload --repository testpypi dist/*

---

---

## Milestone 15 — V2-1: JavaScript and TypeScript indexing

| Change | Files |
|---|---|
| Add .ts, .tsx, .js, .jsx to ALLOWED_EXTENSIONS | repolix/walker.py |
| Add EXTENSION_TO_LANGUAGE mapping | repolix/chunker.py |
| Replace _PARSER singleton with _PARSER_CACHE dict keyed by language | repolix/chunker.py |
| Add _get_cached_parser(language) helper using tree-sitter-javascript and tree-sitter-typescript | repolix/chunker.py |
| Add _extract_js_calls, _extract_js_name_from_parent, _handle_js_node helpers | repolix/chunker.py |
| Extend _walk_tree to dispatch to Python or JS/TS handlers based on language param | repolix/chunker.py |
| chunk_file returns [] for unknown extensions instead of raising ValueError | repolix/chunker.py |
| Add tree-sitter-javascript and tree-sitter-typescript to dependencies | pyproject.toml |
| Add TestJsChunking test class (12 tests); update test_raises_on_non_python_file | tests/test_chunker.py |
| JS/TS node types chunked: function_declaration, arrow_function, function_expression, class_declaration, method_definition | repolix/chunker.py |
| docstring="" for all JS/TS chunks (JSDoc extraction out of scope for V2-1) | repolix/chunker.py |
| tsx extension maps to separate "tsx" language key to select language_tsx() grammar | repolix/chunker.py |

---

## Milestone 18 — LLM output layer: structured response, section parsing, confidence gating

| Change | Files |
|---|---|
| Replace SYSTEM_PROMPT: senior engineer persona, bold-header structure (Answer/How it works/Where to look next) | repolix/llm.py |
| Add _parse_sections(): splits response on **Header:** boundaries; graceful fallback to full text | repolix/llm.py |
| Add confidence gating to answer_query(): reads results[0]["score"]; skips LLM when score < 0.15 | repolix/llm.py |
| Low-confidence path returns navigation dict with closest_matches and rephrasing suggestions | repolix/llm.py |
| Medium-confidence path (0.15–0.4) appends caution note to system prompt | repolix/llm.py |
| answer_query() now returns answer_sections, confidence, and navigation alongside existing keys | repolix/llm.py |
| Add score=0.5 default to make_result() so existing tests remain high-confidence | tests/test_llm.py |
| Add TestParseSections: full structure, no where_to_look, plain-prose fallback | tests/test_llm.py |
| Add TestAnswerQueryConfidence: low confidence (zero API calls), medium caution injection, sections returned | tests/test_llm.py |

---

## Milestone 19 — tour command: proactive orientation briefing

| Change | Files |
|---|---|
| Add repolix/tour.py: get_all_chunks, compute_inbound_counts, identify_entry_points, select_tour_chunks, build_tour_context, generate_tour | repolix/tour.py |
| Add TOUR_SYSTEM_PROMPT and answer_tour() to llm.py; move import re to module level | repolix/llm.py |
| Add tour CLI command: --path scope, --save flag, Rich panel with 5 sections + Most Referenced footer | repolix/cli.py |
| Add TourRequest, TourResponse Pydantic models and POST /tour endpoint | repolix/api.py |
| Add tests/test_tour.py: 24 tests covering all pipeline functions | tests/test_tour.py |

Key design decisions:
- Phase 1 (local): reads ChromaDB metadata only — no embeddings, no API calls
- Phase 2: single LLM chat completion call with tour-specific prompt
- inbound_counts is a reverse-adjacency count — O(n × avg_calls), no full graph needed
- Two-signal entry point detection: heuristic (file/function name) ranks above graph-source (zero inbound, nonzero outbound)
- Two-pass chunk selection: Pass 1 enforces one-chunk-per-file diversity; Pass 2 fills remaining slots
- build_tour_context accepts _all_chunks for top-function file-path lookup — prevents "unknown" for highly-referenced functions not in the selected 8
- frozenset for ENTRY_POINT_FILES and ENTRY_POINT_FUNCTIONS: O(1) membership test, immutable at module scope
- Lazy import of answer_tour inside generate_tour: defensive against future circular imports between tour.py and llm.py

---

## Milestone 20 — trace command: BFS call-graph traversal

| Change | Files |
|---|---|
| Add repolix/trace.py: lookup_chunk_by_name, forward_trace, backward_trace, format_trace_tree, run_trace | repolix/trace.py |
| Add TRACE_SYSTEM_PROMPT and answer_trace() to llm.py; placed after answer_tour() | repolix/llm.py |
| Add trace CLI command: --depth, --max-nodes, --reverse, --explain; Rich Panel tree + Rule callers section | repolix/cli.py |
| Add TraceRequest, TraceResponse Pydantic models and POST /trace endpoint | repolix/api.py |
| Add tests/test_trace.py: 20 tests covering all pipeline functions | tests/test_trace.py |

Key design decisions:
- forward_trace: BFS over calls edges using lookup_chunk_by_name (same lookup as expand_via_call_graph)
- Cycle detection: already-visited calls recorded inline during expansion (not on dequeue) so child_already_visited is populated correctly before the node is ever re-encountered
- backward_trace: O(n) scan via get_all_chunks from tour.py — no BFS, one level up only
- get_all_chunks import is lazy inside backward_trace (defensive against future circular imports between trace.py and tour.py)
- format_trace_tree: recursive Unicode tree renderer; already-visited shown inline as [already visited]; truncated nodes hint --depth
- run_trace: zero API calls by default; explain=True triggers single answer_trace() LLM call
- Patch target for get_all_chunks in tests is repolix.tour.get_all_chunks (lazy import pattern)
- max_nodes cap test requires explicit high max_depth to prevent depth limit firing before node cap

---

## V2 Roadmap

- TypeScript / JavaScript support (Tree-sitter parser swap) ✓ Done in V2-1
- `repolix tour` — proactive orientation briefing ✓ Done in V2-2
- `repolix trace` — call graph traversal for any named function ✓ Done in V2-3
- VS Code extension wrapper
- Dependency graph visualization
- Secret pattern filter in walker.py (skip files with hardcoded credentials)
- Smart truncation: preserve head + tail of oversized chunks
- Index-time warning for truncated chunks

## V3 Roadmap

- GitHub webhook integration (re-index on push)
- Multi-repo support
- Slack bot

---

## Conventions

- All file paths stored and compared as absolute resolved strings.
- Citations and user-facing output always use file_rel_path.
- Sorted output everywhere for deterministic behavior across runs.
- Tests are hermetic. Every test uses tmp_path. No test touches
  a real repository on disk.
- OpenAI calls are always mocked in tests. Never hit the network
  in a test.
- mock_openai_client uses side_effect to return one embedding per
  input text — not a fixed return_value — so multi-chunk tests work.
- Run pytest after every change before committing.
- Update CONTEXT.md at the end of every milestone or significant
  change. A milestone is not done until CONTEXT.md reflects it.

---

## Git commit conventions

Commit to GitHub after every meaningful unit of work including:
- A complete prompt execution
- A passing test suite for a new feature
- Any fix to a failing test
- Any update to CONTEXT.md
- Any architectural change

Format: conventional commits — ONE LINE ONLY, no body, no bullet points.
  feat: add call graph expansion to retriever
  fix: use file_rel_path in citation output
  refactor: make Parser singleton at module level
  test: add expansion tests to test_retriever
  docs: update CONTEXT.md for Milestone 11
  chore: add httpx to dev dependencies

Semicolons are allowed to join related items on the same line:
  feat: serve React SPA from FastAPI; add catch-all route for client-side routing

Rules:
- One logical change per commit.
- Commit message is exactly one line. No multi-line messages, no -m body flags.
- Never commit with a vague message like "update" or "fix stuff".
- Always run pytest tests/ -v before committing. Green only.
- Always run git status before git add. Verify .env is absent.
- Never use git add . — use git add <specific files>.
- Push to main after every commit unless instructed otherwise.

Sequence:
  pytest tests/ -v
  git status
  git add <specific files>
  git commit -m "type: description"
  git push origin main

---

## What not to do

- Do not split classes into separate method chunks without setting
  parent_class on each method.
- Do not use dirs = [...] in os.walk — use dirs[:] = [...].
- Do not store lists directly in ChromaDB metadata.
- Do not create a Parser or Tokenizer instance per file call.
- Do not embed raw source without enrichment.
- Do not skip the hash check on re-index unless force=True.
- Do not send more than 5 primary chunks to the LLM.
- Do not use 0-indexed line numbers in any user-facing output.
- Do not commit with .env present in git status output.
- Do not bundle unrelated changes into one commit.
- Do not leave CONTEXT.md outdated after a milestone completes.
- Do not use file_path in user-facing citation output — always
  use file_rel_path.
- Do not use max_tokens with gpt-5.4-mini — use max_completion_tokens.
- Do not use a fixed return_value for mock_openai_client — use
  side_effect so the mock returns the right number of embeddings
  for any batch size.
- Do not filter keyword search tokens with len > 2 — the correct
  guard is len >= 2 to preserve 2-character identifiers like os, db.
- Do not print the LLM's raw response as the answer — strip the
  CITATIONS block first via _strip_citations_block.
- Do not write multi-line git commit messages — one line only; use
  semicolons to join related items if needed.
- Do not add a catch-all GET route before the StaticFiles mount without
  making it file-aware — a plain index.html catch-all will serve HTML
  for JS/CSS requests and break the frontend.
- Do not create a module-level Rich Console — create it inside each
  command function so Click's CliRunner patches sys.stdout before the
  Console is constructed, ensuring test output capture works correctly.
- Do not pass citation label strings like [1] directly into Rich markup
  strings — use rich.markup.escape() to prevent them being interpreted
  as markup tags.
- Do not patch "repolix.store.chunk_file" in tests — chunk_file is
  imported locally inside index_repo, so patch "repolix.chunker.chunk_file".
- Do not patch "repolix.trace.get_all_chunks" — get_all_chunks is lazily
  imported inside backward_trace from repolix.tour, so patch
  "repolix.tour.get_all_chunks".
- Do not rely on BFS dequeue to detect already-visited cycle nodes in
  forward_trace — already-visited calls must be recorded in
  child_already_visited during the expansion phase, not on dequeue, because
  already-visited names are never re-enqueued.
- Do not test max_nodes cap with the default max_depth=3 — the depth limit
  will terminate traversal before the node cap fires on short chains.
