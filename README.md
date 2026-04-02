# repolens

A local-first codebase context engine. Point it at any repo, ask
plain English questions, get back answers with exact file and line
citations. Your code never leaves your machine.

## What it does

- Parses your codebase using AST-based chunking (Tree-sitter) so
  every retrieved chunk is a complete function or class — not an
  arbitrary line slice
- Embeds chunks using OpenAI text-embedding-3-small
- Stores everything locally in ChromaDB — no server, no cloud
- Retrieves using hybrid search: vector similarity plus keyword
  matching, merged and re-ranked
- Answers questions via gpt-4o-mini with citations back to the
  exact file and line number

## Quickstart
```bash
pip install repolens
repolens index ./your-repo
repolens query "where does authentication happen"
```

## Setup
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/repolens
cd repolens
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
# Edit .env and add your OpenAI API key
```

## Stack

Python, Tree-sitter, OpenAI API, ChromaDB, FastAPI, React

## Status

V1 in active development. Python repos supported. 
TypeScript support coming in V2.

## License

MIT
