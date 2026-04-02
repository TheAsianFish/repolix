# repolens — Project Context

Paste this file at the top of every new Cursor chat and every new
Claude conversation to restore full project context instantly.
Update this file at the end of every milestone before moving on.

---

## What repolens is

A local-first codebase context engine. Point it at any Python repo,
ask plain English questions, get back answers with exact file and
line number citations. Code never leaves the user's machine.
Free and open source. Built for developer tooling.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | Ecosystem, tooling, pip distribution |
| AST parsing | Tree-sitter | Fast, accurate, multi-language ready |
| Embeddings | text-embedding-3-small | Cheap, high quality, 1536 dims |
| Vector store | ChromaDB (persistent, in-process) | Local-first, no server needed |
| LLM | gpt-5.4-mini | Fast, cheap, strong instruction following |
| Web server | FastAPI | Async, simple, well-documented |
| Frontend | React + TypeScript | Simple local UI at localhost:3000 |
| CLI | Click | Python CLI standard |
| Install | pip install repolens | One command |

---

## Architecture decisions — locked in

**AST chunking over line splitting.**
Tree-sitter parses each file into a syntax tree. We split only at
function and class boundaries. Every chunk is a semantically complete
unit. This is the most important decision for retrieval quality.

**Class = one chunk. Methods inside are not split separately.**
Recursion stops when a chunk node is found. This prevents
double-counting — methods belong to their parent class chunk.

**Module-level singleton for Parser and Tokenizer.**
_PARSER and _TOKENIZER are module-level in chunker.py. Creating
them per file call was wasteful. They are stateless between calls.

**Enriched embedding text.**
We do not embed raw source alone. We prepend node type and name,
then docstring if present, then source. This improves retrieval
because natural language queries map better to natural language
descriptions than to raw syntax.

**Metadata on every chunk.**
Each chunk carries: file_path, node_type, name, source, start_line,
end_line, token_count, calls (functions called), docstring.
calls is stored in ChromaDB as a comma-joined string because
ChromaDB metadata must be primitive types. Split on read, join
on write.

**Incremental indexing via SHA-256 file hashing.**
Every file gets a SHA-256 hash stored in ChromaDB alongside its
chunks. Re-indexing skips files whose hash has not changed.
Only changed files are re-embedded and re-stored.

**Hybrid search: vector similarity + keyword search, merged.**
Pure vector search misses exact name matches. Pure keyword search
misses semantic similarity. We do both and merge results.

**Metadata re-ranking as a second pass.**
Top 10 chunks retrieved. Re-ranked using metadata signals:
file name relevance, function name match, call graph proximity.
Top 5 sent to LLM.

**Top 5 chunks max sent to LLM.**
Hard cap. 5 chunks * 300 tokens = 1500 tokens max context.
Safe for gpt-5.4-mini. Prevents context overflow.

**Chunk token cap: 300 tokens.**
Hard cap enforced in chunker.py using tiktoken cl100k_base
encoding — the same encoding gpt-5.4-mini uses internally.
Oversized chunks are truncated, not discarded.

**Line numbers are 1-indexed.**
Tree-sitter uses 0-indexed rows (C convention). We add 1 on
every start_point and end_point extraction. All citations the
user sees are 1-indexed.

---

## ChromaDB collections

| Collection | Purpose |
|---|---|
| repolens_chunks | Stores chunk source, embeddings, metadata |
| repolens_hashes | Stores one hash per file for incremental indexing |

ChromaDB persists to .repolens/ in the project root.
.repolens/ is gitignored.

Chunk IDs are: "{absolute_file_path}:{start_line}"
This makes IDs deterministic and stable across re-index runs.

Hash IDs are: "{absolute_file_path}"

---

## Module map

| File | Status | Responsibility |
|---|---|---|
| repolens/walker.py | Complete | Filesystem traversal, file filtering |
| repolens/chunker.py | Complete | AST parsing, chunk extraction, metadata |
| repolens/store.py | Complete | Embeddings, ChromaDB storage, retrieval |
| repolens/retriever.py | Complete | Basic retrieval, vector search, metadata re-ranking |
| repolens/cli.py | Pending | Click CLI entry points |
| repolens/llm.py | Pending | LLM call, prompt construction, citations |

Note: llm.py does not exist yet. It will be created at Milestone 7.

---

## Test suite status

| File | Tests | Status |
|---|---|---|
| tests/test_walker.py | 11 | Passing |
| tests/test_chunker.py | 21 | Passing |
| tests/test_store.py | 16 | Passing |
| tests/test_retriever.py | 16 | Passing |

Run all tests: pytest tests/ -v

---

## Milestone map

| # | Name | Status |
|---|---|---|
| 1 | Project scaffold + walker | Complete |
| 2 | AST chunker | Complete |
| 3 | Metadata extraction | Complete |
| 4 | Embedding pipeline + vector store | Complete |
| 5 | Basic retrieval | Complete |
| 6 | Hybrid search + re-ranking | Next |
| 7 | LLM integration + citations | Pending |
| 8 | CLI | Pending |
| 9 | FastAPI backend + React frontend | Pending |
| 10 | Polish + ship | Pending |

---

## Conventions

- All file paths stored and compared as absolute resolved strings.
- Sorted output everywhere — walker, chunker, call lists — for
  deterministic behavior across runs.
- Tests are hermetic. Every test uses tmp_path. No test touches
  a real repository on disk.
- OpenAI calls are always mocked in tests. Never hit the network
  in a test.
- New modules get a stub first, implementation second, tests third.
- Run pytest after every milestone before moving forward.
- Update CONTEXT.md at the end of every milestone. A milestone is
  not done until CONTEXT.md reflects it.

---

## Git commit conventions

Commit to GitHub after every meaningful unit of work. A meaningful
unit is any of the following:
- A complete prompt execution (new module, new test file, refactor)
- A passing test suite for a new feature
- Any fix to a failing test
- Any update to CONTEXT.md itself
- Any architectural change, even if small

Commit message format: use conventional commits.
  feat: add incremental indexing via SHA-256 hashing
  fix: correct 0-indexed line number offset in chunker
  refactor: make Parser and Tokenizer module-level singletons
  test: add metadata extraction tests to test_chunker
  docs: update CONTEXT.md for Milestone 4 completion
  chore: update pyproject.toml with dev dependencies

Rules:
- One logical change per commit. Do not bundle unrelated changes.
- Never commit with a vague message like "update" or "fix stuff".
- Always run pytest tests/ -v before committing. Green tests only.
- Always run git status before git add to verify .env is absent.
- Never use git add . blindly — use git add <specific files>.
- Push to main after every commit. No local-only commits.

Sequence for every commit:
  pytest tests/ -v
  git status
  git add <specific files changed>
  git commit -m "type: description"
  git push origin main

---

## What not to do

- Do not split classes into separate method chunks.
- Do not use dirs = [...] in os.walk — use dirs[:] = [...].
- Do not store lists directly in ChromaDB metadata.
- Do not create a Parser or Tokenizer instance per file call.
- Do not embed raw source without enrichment.
- Do not skip the hash check on re-index unless force=True.
- Do not send more than 5 chunks to the LLM.
- Do not use 0-indexed line numbers in any user-facing output.
- Do not commit with .env present in git status output.
- Do not bundle unrelated changes into one commit.
- Do not leave CONTEXT.md outdated after a milestone completes.
