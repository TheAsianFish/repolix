# repolix — Project Context

Paste this file at the top of every new Cursor chat and every new
Claude conversation to restore full project context instantly.
Update this file at the end of every milestone before moving on.

---

## What repolix is

A local-first codebase context engine. Point it at any Python repo,
ask plain English questions, get back answers with exact file and
line number citations. Code never leaves the user's machine.
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
| repolix/retriever.py | Complete | Hybrid search, RRF, re-ranking, call graph expansion |
| repolix/llm.py | Complete | Prompt construction, gpt-5.4-mini call, citation parsing, CITATIONS block stripping |
| repolix/cli.py | Complete | Click CLI — index and query commands, confidence label |
| repolix/api.py | Complete | FastAPI backend — /index, /query, /status, /health; serves built SPA from frontend/dist |
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
| tests/test_store.py | 24 | Passing |
| tests/test_retriever.py | 22 | Passing |
| tests/test_llm.py | 21 | Passing |
| tests/test_cli.py | 12 | Passing |
| tests/test_api.py | 9 | Passing |

Run all tests: pytest tests/ -v
Total: 138 passing

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

V1 shipped as repolix 0.1.0 on PyPI. Post-V1 polish complete. Milestone 13 (Rich CLI) complete.

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

## V2 Roadmap

- TypeScript / JavaScript support (Tree-sitter parser swap)
- VS Code extension wrapper
- Dependency graph visualization
- "Start here" guide auto-generated for new engineers
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
