# repolens

**Ask plain English questions about any Python codebase. Get answers
with exact file and line citations. Runs entirely on your machine.**
```bash
repolens index ./myrepo
repolens query "how does authentication work"
```
── Answer ────────────────────────────────────────────────
authenticate_user() validates credentials by calling validate_token()
[1] which checks expiry and signature. On success it creates a
session via SessionService.create() [2].
── Citations ─────────────────────────────────────────────
[1] auth/validators.py:14-28    (validate_token)
[2] auth/session.py:45-67       (SessionService.create)

Your code never leaves your machine. No server. No accounts beyond
an OpenAI API key.

---

## Why repolens

Getting dropped into an unfamiliar codebase is painful. Documentation
is outdated. Grep finds strings, not meaning. LLM chatbots hallucinate
file names and function signatures because they have no access to your
actual code.

repolens indexes your code locally using AST-based chunking — every
retrieved chunk is a complete function or class, never an arbitrary
line slice. It runs entirely on your machine.

---

## How it works

**1. AST chunking**
Tree-sitter parses each file into a syntax tree. repolens splits only
at function and class boundaries. Every chunk is semantically complete.
Methods are tracked with their parent class for disambiguation.

**2. Hybrid search**
Queries run against OpenAI embeddings (vector search) and exact token
matching (keyword search) simultaneously. Results are merged using
Reciprocal Rank Fusion — a ranking algorithm that rewards consistency
across search methods over dominance in just one.

**3. Call graph expansion**
After initial retrieval, repolens inspects each retrieved chunk's
call graph and fetches called functions that did not rank highly
enough on their own. This surfaces implementation details that live
one function call away from the entry point.

**4. Metadata re-ranking**
Retrieved chunks are re-ranked using function names, file paths,
docstrings, and call graph signals before being sent to the LLM.

**5. Cited answers**
The top chunks go to gpt-5.4-mini with instructions to synthesize
across all chunks and cite every claim. Citations map back to exact
file paths and line numbers.

---

## Quickstart

### Requirements

- Python 3.11+
- Node.js 18+ (web UI only)
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Install
```bash
git clone https://github.com/YOUR_USERNAME/repolens
cd repolens
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### CLI
```bash
# Index a repository (~$0.02 per 30k lines, one-time)
repolens index ./path/to/repo

# Ask a question
repolens query "how does authentication work"

# See raw retrieved chunks without an LLM call
repolens query "where is UserService defined" --no-llm

# Force re-index all files after a major refactor
repolens index ./path/to/repo --force
```

### Web UI
```bash
cd frontend && npm install && cd ..
bash start.sh
# Open http://localhost:3000
```

---

## Cost

| Action | Cost |
|---|---|
| Index 30k line repo | ~$0.02 (one-time) |
| Re-index after small change | ~$0.001 (changed files only) |
| Each query | ~$0.001 |

---

## Stack

| Layer | Choice |
|---|---|
| AST parsing | Tree-sitter |
| Embeddings | text-embedding-3-small |
| Vector store | ChromaDB (local, no server needed) |
| LLM | gpt-5.4-mini |
| Backend | FastAPI |
| Frontend | React + TypeScript |
| CLI | Click |

---

## Limitations

- Python repos only. TypeScript support planned for V2.
- Best on repos up to ~30k lines.
- Deeply nested functions are included in their parent chunk.
- Complex cross-file reasoning may require rephrasing the query.

---

## Roadmap

**V2** — TypeScript support, VS Code extension, dependency graph

**V3** — GitHub webhook re-indexing, multi-repo, Slack bot

---

## License

MIT © 2026 Patrick Chung
