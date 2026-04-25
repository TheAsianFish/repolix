// Type-safe wrappers around the repolix FastAPI backend.
// All functions throw on non-200 responses.

// Production: same-origin relative URLs (any uvicorn --port works).
// Dev (Vite): defaults to API on :8000; override with VITE_API_URL in frontend/.env
const BASE_URL = import.meta.env.DEV
  ? (import.meta.env.VITE_API_URL ?? 'http://localhost:8000')
  : ''

/** Prefer FastAPI's `detail` field over raw JSON for UI messages. */
export async function httpErrorMessage(res: Response): Promise<string> {
  const text = await res.text()
  try {
    const parsed = JSON.parse(text) as { detail?: unknown }
    if (typeof parsed.detail === 'string') return parsed.detail
    if (Array.isArray(parsed.detail)) {
      return parsed.detail
        .map((e: { msg?: string }) => e.msg ?? JSON.stringify(e))
        .join('; ')
    }
  } catch {
    /* not JSON */
  }
  return text || `${res.status} ${res.statusText}`
}

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

export interface AnswerSections {
  answer: string
  how_it_works: string | null
  where_to_look: string | null
}

export interface NavigationMatch {
  name: string
  file_rel_path: string
  start_line: number
}

export interface Navigation {
  message: string
  closest_matches: NavigationMatch[]
  suggestions: string[]
}

export interface QueryResponse {
  answer: string | null
  answer_sections: AnswerSections | null
  citations: Citation[]
  chunks: Chunk[]
  chunks_used: number
  confidence: 'high' | 'medium' | 'low'
  navigation: Navigation | null
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
  if (!res.ok) throw new Error(await httpErrorMessage(res))
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
  if (!res.ok) throw new Error(await httpErrorMessage(res))
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
  if (!res.ok) throw new Error(await httpErrorMessage(res))
  return res.json()
}
