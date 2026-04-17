import { useCallback, useEffect, useState } from 'react'
import { checkStatus, indexRepo, queryRepo } from './api'
import type { QueryResponse } from './api'
import { AnswerPanel } from './components/AnswerPanel'
import { ChunkList } from './components/ChunkList'
import { QueryBox } from './components/QueryBox'
import { RepoInput } from './components/RepoInput'
import type { Chunk, Citation } from './types'

function confidenceFromScore(score: number): 'high' | 'medium' | 'low' {
  if (score >= 0.4) return 'high'
  if (score >= 0.15) return 'medium'
  return 'low'
}

function mapCitations(raw: QueryResponse['citations']): Citation[] {
  return raw.map(c => ({
    label: c.label,
    file_rel_path: (c.file_rel_path ?? '').trim(),
    file_path: typeof (c as { file_path?: string }).file_path === 'string'
      ? (c as { file_path?: string }).file_path!
      : '',
    start_line: c.start_line,
    end_line: c.end_line,
    name: c.name,
    is_truncated: Boolean((c as { is_truncated?: boolean }).is_truncated),
  }))
}

function mapChunks(raw: QueryResponse['chunks'], citations: Citation[]): Chunk[] {
  const labelByName = new Map<string, string>()
  for (const c of citations) {
    if (!labelByName.has(c.name)) {
      labelByName.set(c.name, c.label)
    }
  }

  return raw.map((r, idx) => {
    const ext = r as typeof r & { node_type?: string; file_path?: string }
    const citationLabel = labelByName.get(r.name) ?? null
    const is_cited = citationLabel !== null
    return {
      id: `${r.file_rel_path}:${r.start_line}:${idx}`,
      name: r.name,
      node_type: ext.node_type ?? 'code',
      file_rel_path: r.file_rel_path,
      file_path: typeof ext.file_path === 'string' ? ext.file_path : '',
      start_line: r.start_line,
      end_line: r.end_line,
      source: r.source,
      score: r.rerank_score,
      is_cited,
      citation_label: is_cited ? citationLabel : null,
    }
  })
}

export default function App() {
  const [repoPath, setRepoPath] = useState('.')
  const [question, setQuestion] = useState('')
  const [answer, setAnswer] = useState<string | null>(null)
  const [citations, setCitations] = useState<Citation[]>([])
  const [chunks, setChunks] = useState<Chunk[]>([])
  const [confidence, setConfidence] = useState<'high' | 'medium' | 'low'>('low')
  const [isIndexing, setIsIndexing] = useState(false)
  const [isQuerying, setIsQuerying] = useState(false)
  const [indexStatus, setIndexStatus] = useState<string | null>(null)
  const [queryError, setQueryError] = useState<string | null>(null)
  const [hasQueried, setHasQueried] = useState(false)

  const refreshStatusLine = useCallback(async () => {
    try {
      const s = await checkStatus(repoPath)
      if (s.indexed) {
        const base = s.repo_path.split('/').filter(Boolean).pop() ?? s.repo_path
        setIndexStatus(`Index ready — ${base}`)
      } else {
        setIndexStatus(`No index at ${s.store_path}. Click Index to build.`)
      }
    } catch {
      setIndexStatus(
        'Could not reach API (is the backend running? For Vite dev, check VITE_API_URL.)',
      )
    }
  }, [repoPath])

  useEffect(() => {
    void refreshStatusLine()
  }, [refreshStatusLine])

  async function handleIndex() {
    setIsIndexing(true)
    setIndexStatus(null)
    try {
      const stats = await indexRepo(repoPath.trim())
      const rp = repoPath.trim()
      const base = rp.split('/').filter(Boolean).pop() ?? rp
      setIndexStatus(`Index ready — ${base}`)
      if (stats.errors.length > 0) {
        setIndexStatus(prev => `${prev} Errors: ${stats.errors.length}.`)
      }
    } catch (e) {
      setIndexStatus(e instanceof Error ? e.message : 'Index failed')
    } finally {
      setIsIndexing(false)
    }
  }

  async function handleQuery() {
    const q = question.trim()
    if (!q) return
    setIsQuerying(true)
    setQueryError(null)
    setAnswer(null)
    setCitations([])
    setChunks([])
    setConfidence('low')
    setHasQueried(true)

    try {
      const res: QueryResponse = await queryRepo(q, repoPath.trim())
      const cit = mapCitations(res.citations)
      setCitations(cit)
      setChunks(mapChunks(res.chunks, cit))
      setAnswer(res.answer)
      setConfidence(
        res.chunks.length > 0
          ? confidenceFromScore(res.chunks[0].rerank_score)
          : 'low'
      )
    } catch (e) {
      setQueryError(e instanceof Error ? e.message : 'Query failed')
    } finally {
      setIsQuerying(false)
    }
  }

  const showAnswerPanel = isQuerying || (answer !== null && answer !== '')

  return (
    <div className="app-root">
      <header className="app-nav">
        <span className="app-wordmark">repolix</span>
        <a
          className="app-nav-link"
          href="https://github.com/TheAsianFish/repolix"
          target="_blank"
          rel="noreferrer"
        >
          GitHub
        </a>
      </header>
      <div className="app-body">
        <div className="app-col app-col-left">
          <RepoInput
            repoPath={repoPath}
            onRepoPathChange={setRepoPath}
            onIndex={handleIndex}
            isIndexing={isIndexing}
            indexStatus={indexStatus}
          />
          <QueryBox
            question={question}
            onQuestionChange={setQuestion}
            onSubmit={() => void handleQuery()}
            isQuerying={isQuerying}
          />
          {queryError ? <div className="app-query-error">{queryError}</div> : null}
          {showAnswerPanel ? (
            <AnswerPanel
              answer={answer ?? ''}
              citations={citations}
              confidence={confidence}
              isLoading={isQuerying}
            />
          ) : null}
        </div>
        <div className="app-col app-col-right">
          <ChunkList chunks={chunks} hasQueried={hasQueried} isLoading={isQuerying} />
        </div>
      </div>
    </div>
  )
}
