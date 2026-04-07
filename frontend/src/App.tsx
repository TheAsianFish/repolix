import React, { useCallback, useEffect, useState } from 'react'
import { QueryBox } from './components/QueryBox'
import { ResultCard } from './components/ResultCard'
import { CitationList } from './components/CitationList'
import { checkStatus, queryRepo } from './api'
import type { QueryResponse, StatusResponse } from './api'

// Default repo path — user can change this in the UI.
// In production this would come from a config or URL param.
const DEFAULT_REPO_PATH = '.'

export default function App() {
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [repoPath, setRepoPath] = useState(DEFAULT_REPO_PATH)
  const [indexStatus, setIndexStatus] = useState<StatusResponse | null>(null)
  const [statusError, setStatusError] = useState<string | null>(null)
  const [statusLoading, setStatusLoading] = useState(false)

  const refreshStatus = useCallback(async () => {
    setStatusError(null)
    setStatusLoading(true)
    try {
      const s = await checkStatus(repoPath)
      setIndexStatus(s)
    } catch (e) {
      setStatusError(e instanceof Error ? e.message : 'Could not reach API')
    } finally {
      setStatusLoading(false)
    }
  }, [repoPath])

  useEffect(() => {
    let cancelled = false
    setStatusError(null)
    setIndexStatus(null)
    void (async () => {
      setStatusLoading(true)
      try {
        const s = await checkStatus(repoPath)
        if (!cancelled) setIndexStatus(s)
      } catch (e) {
        if (!cancelled) {
          setStatusError(e instanceof Error ? e.message : 'Could not reach API')
        }
      } finally {
        if (!cancelled) setStatusLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [repoPath])

  async function handleQuery(question: string) {
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const result = await queryRepo(question, repoPath)
      setResponse(result)
      if (!indexStatus?.indexed) void refreshStatus()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
      void refreshStatus()
    } finally {
      setLoading(false)
    }
  }

  return (
    <div
      style={{
        maxWidth: '860px',
        margin: '0 auto',
        padding: '2rem 1rem',
        fontFamily:
          '-apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
      }}
    >
      <header style={{ marginBottom: '2rem' }}>
        <h1 style={{ fontSize: '1.6rem', fontWeight: 700, margin: 0 }}>
          🔍 repolix
        </h1>
        <p style={{ color: '#64748b', margin: '0.25rem 0 0' }}>
          Local-first codebase context engine
        </p>
      </header>

      <div style={{ marginBottom: '1rem' }}>
        <label
          style={{ fontSize: '0.85rem', color: '#475569', display: 'block', marginBottom: '0.3rem' }}
        >
          Repository path
        </label>
        <input
          value={repoPath}
          onChange={e => setRepoPath(e.target.value)}
          placeholder="/absolute/path/to/repo"
          style={{
            width: '100%',
            padding: '0.5rem 0.75rem',
            fontSize: '0.9rem',
            border: '1px solid #cbd5e1',
            borderRadius: '6px',
            boxSizing: 'border-box',
            fontFamily: 'monospace',
          }}
        />
        {(statusError || indexStatus) && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '0.5rem', marginTop: '0.35rem' }}>
            {statusError ? (
              <p style={{ fontSize: '0.8rem', color: '#b45309', margin: 0, flex: 1 }}>
                Status: {statusError} (is the backend running on port 8000?)
              </p>
            ) : indexStatus ? (
              <p
                style={{
                  fontSize: '0.8rem',
                  margin: 0,
                  flex: 1,
                  color: indexStatus.indexed ? '#15803d' : '#b45309',
                }}
              >
                {indexStatus.indexed
                  ? `Index ready — ${indexStatus.store_path} (under repo ${indexStatus.repo_path})`
                  : `No index yet — Chroma data should live at ${indexStatus.store_path} (the ".repolix" folder inside ${indexStatus.repo_path}). Run repolix index or POST /index.`}
              </p>
            ) : null}
            <button
              onClick={() => void refreshStatus()}
              disabled={statusLoading}
              title="Refresh index status"
              style={{
                flexShrink: 0,
                padding: '0.15rem 0.5rem',
                fontSize: '0.75rem',
                border: '1px solid #cbd5e1',
                borderRadius: '4px',
                background: '#f8fafc',
                color: '#475569',
                cursor: statusLoading ? 'not-allowed' : 'pointer',
                opacity: statusLoading ? 0.6 : 1,
              }}
            >
              {statusLoading ? '...' : '↻ Refresh'}
            </button>
          </div>
        )}
      </div>

      <QueryBox onSubmit={handleQuery} loading={loading} />

      {error && (
        <div
          style={{
            padding: '0.75rem 1rem',
            background: '#fef2f2',
            border: '1px solid #fecaca',
            borderRadius: '6px',
            color: '#dc2626',
            marginBottom: '1rem',
          }}
        >
          {error}
        </div>
      )}

      {response && (
        <div>
          {response.answer && (
            <div
              style={{
                padding: '1rem',
                background: '#f0f9ff',
                border: '1px solid #bae6fd',
                borderRadius: '8px',
                marginBottom: '1.5rem',
                lineHeight: 1.6,
              }}
            >
              <h2 style={{ fontSize: '1rem', margin: '0 0 0.5rem', color: '#0369a1' }}>
                Answer
              </h2>
              <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                {response.answer}
              </p>
              <CitationList citations={response.citations} />
            </div>
          )}

          {response.chunks.length > 0 && (
            <div>
              <h2 style={{ fontSize: '1rem', color: '#475569', marginBottom: '0.75rem' }}>
                Retrieved chunks ({response.chunks.length})
              </h2>
              {response.chunks.map((chunk, i) => (
                <ResultCard key={`${chunk.file_rel_path}:${chunk.start_line}`} chunk={chunk} rank={i + 1} />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}
