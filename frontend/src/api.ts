// Type-safe wrappers around the repolens FastAPI backend.
// All functions throw on non-200 responses.

const BASE_URL = 'http://localhost:8000'

export interface Citation {
  label: string
  file_rel_path: string
  start_line: number
  end_line: number
  name: string
  parent_class: string | null
}

export interface Chunk {
  source: string
  file_rel_path: string
  name: string
  start_line: number
  end_line: number
  rerank_score: number
  parent_class: string | null
}

export interface QueryResponse {
  answer: string | null
  citations: Citation[]
  chunks: Chunk[]
  chunks_used: number
}

export interface IndexResponse {
  total_files: number
  indexed: number
  skipped: number
  total_chunks: number
  errors: string[]
}

export interface StatusResponse {
  indexed: boolean
  store_path: string
  repo_path: string
}

export async function checkStatus(repoPath: string): Promise<StatusResponse> {
  const res = await fetch(
    `${BASE_URL}/status?repo_path=${encodeURIComponent(repoPath)}`
  )
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function indexRepo(
  repoPath: string,
  force = false
): Promise<IndexResponse> {
  const res = await fetch(`${BASE_URL}/index`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_path: repoPath, force }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}

export async function queryRepo(
  question: string,
  repoPath: string,
  noLlm = false
): Promise<QueryResponse> {
  const res = await fetch(`${BASE_URL}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      question,
      repo_path: repoPath,
      no_llm: noLlm,
    }),
  })
  if (!res.ok) throw new Error(await res.text())
  return res.json()
}
