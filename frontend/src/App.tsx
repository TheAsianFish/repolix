import React, { useState } from 'react'
import { QueryBox } from './components/QueryBox'
import { ResultCard } from './components/ResultCard'
import { CitationList } from './components/CitationList'
import { queryRepo } from './api'
import type { QueryResponse } from './api'

// Default repo path — user can change this in the UI.
// In production this would come from a config or URL param.
const DEFAULT_REPO_PATH = '.'

export default function App() {
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [repoPath, setRepoPath] = useState(DEFAULT_REPO_PATH)

  async function handleQuery(question: string) {
    setLoading(true)
    setError(null)
    setResponse(null)

    try {
      const result = await queryRepo(question, repoPath)
      setResponse(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error')
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
