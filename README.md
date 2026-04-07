# repolix

**Ask plain English questions about any Python codebase. Get answers with exact file and line citations. Runs entirely on your machine.**

```
$ repolix index ./myrepo
$ repolix query "how does authentication work"

Searching...
Generating answer...

── Answer ──────────────────────────────────────────────────────────────
authenticate_user() validates credentials by calling validate_token() [1]
which checks expiry and signature. On success it creates a session via
SessionService.create() [2].

── Citations ───────────────────────────────────────────────────────────
[1] auth/validators.py:14-28    (validate_token)
[2] auth/session.py:45-67       (SessionService.create)

[confidence: high · 5 chunks · index: ./myrepo/.repolix]
```

Your code never leaves your machine. No server. No accounts beyond an OpenAI API key.

---

## Why repolix

Getting dropped into an unfamiliar codebase is painful. Documentation is outdated. Grep finds strings, not meaning. LLM chatbots hallucinate file names and function signatures because they have no access to your actual code.

repolix indexes your code locally using AST-based chunking — every retrieved chunk is a complete function or class, never an arbitrary line slice. It runs entirely on your machine.

---

## How it works

**1. AST chunking**
Tree-sitter parses each file into a syntax tree. repolix splits only at function and class boundaries — every chunk is semantically complete. Methods are tracked with their parent class for disambiguation.

**2. Hybrid search**
Queries run against OpenAI embeddings (vector search) and exact token matching (keyword search) simultaneously. Results are merged using Reciprocal Rank Fusion, a ranking algorithm that rewards consistency across search methods over dominance in just one.

**3. Call graph expansion**
After initial retrieval, repolix inspects each chunk's call graph and fetches called functions that didn't rank highly on their own. This surfaces implementation details that live one function call away from the entry point.

**4. Metadata re-ranking**
Retrieved chunks are re-ranked using function names, file paths, docstrings, and call graph signals before being sent to the LLM.

**5. Cited answers**
The top chunks go to the LLM with instructions to synthesize across all chunks and cite every claim. Citations map back to exact file paths and line numbers.

---

## Quickstart

### Requirements

- Python 3.11+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

> Node.js is **not required** for end users. The web UI is pre-built and bundled inside the package.

### Install

```bash
pip install repolix
```

### Set your API key

```bash
export OPENAI_API_KEY=sk-your-key-here
# or add it to a .env file in your working directory
```

### Index a repo

```bash
repolix index ./path/to/repo
```

### Ask a question

```bash
repolix query "how does authentication work"

# Raw chunks without LLM
repolix query "where is UserService defined" --no-llm

# Force re-index all files
repolix index ./path/to/repo --force
```

### Web UI

```bash
uvicorn repolix.api:app --port 8000
# Open http://localhost:8000
```

**For frontend development** (requires Node.js 18+):

```bash
cd frontend && npm install && cd ..
bash start.sh
# Backend: http://localhost:8000  |  Frontend: http://localhost:3000
```

---

## Install from source

```bash
git clone https://github.com/TheAsianFish/repolix
cd repolix
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Cost

| Action | Approximate cost |
|---|---|
| Index a 30k-line repo | ~$0.02 (one-time) |
| Re-index after a small change | ~$0.001 (changed files only) |
| Each query | ~$0.001 |

Incremental indexing means only changed files are re-embedded on subsequent runs.

---

## Stack

| Layer | Choice |
|---|---|
| AST parsing | Tree-sitter |
| Embeddings | text-embedding-3-small |
| Vector store | ChromaDB (local, no server needed) |
| LLM | gpt-4o-mini |
| Backend | FastAPI |
| Frontend | React + TypeScript |
| CLI | Click |

---

## Output format

Each query produces:

- A prose answer with inline citations `[1]`, `[2]`, etc.
- A citations section with exact file paths and line ranges. Citations marked `[truncated]` mean the function exceeded the 300-token chunk cap.
- A confidence label (`high` / `medium` / `low`) derived from how strongly the retrieved chunks matched the query across function names, file paths, docstrings, and call graph signals.

---

## Limitations

- Python repos only (TypeScript support planned for V2)
- Best on repos up to ~30k lines
- Deeply nested functions are included in their parent chunk
- Large functions (>300 tokens) are truncated at the chunk cap
- Complex cross-file reasoning may require rephrasing the query

---

## Roadmap

**V2** — TypeScript/JavaScript support, VS Code extension, dependency graph visualization

**V3** — GitHub webhook re-indexing, multi-repo support, Slack bot

---

## Contributing

Bug reports and pull requests are welcome. Please open an issue before submitting a large change so we can discuss the approach.

---

## License

MIT © 2026 Patrick Chung
