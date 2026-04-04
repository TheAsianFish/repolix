import React from 'react'
import type { Chunk } from '../api'

interface Props {
  chunk: Chunk
  rank: number
}

export function ResultCard({ chunk, rank }: Props) {
  const context = chunk.parent_class
    ? `${chunk.parent_class}.${chunk.name}`
    : chunk.name

  function copyPath() {
    const citation = `${chunk.file_rel_path}:${chunk.start_line}`
    navigator.clipboard.writeText(citation)
  }

  return (
    <div
      style={{
        border: '1px solid #e2e8f0',
        borderRadius: '8px',
        padding: '1rem',
        marginBottom: '1rem',
        background: '#fff',
      }}
    >
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: '0.5rem',
        }}
      >
        <div>
          <span
            style={{
              fontWeight: 600,
              color: '#1e40af',
              marginRight: '0.5rem',
            }}
          >
            [{rank}] {context}
          </span>
          <span style={{ color: '#64748b', fontSize: '0.85rem' }}>
            {chunk.file_rel_path}:{chunk.start_line}–{chunk.end_line}
          </span>
        </div>
        <button
          onClick={copyPath}
          title="Copy file:line to clipboard"
          style={{
            fontSize: '0.75rem',
            padding: '0.2rem 0.5rem',
            border: '1px solid #cbd5e1',
            borderRadius: '4px',
            cursor: 'pointer',
            background: '#f8fafc',
          }}
        >
          Copy path
        </button>
      </div>
      <pre
        style={{
          background: '#f1f5f9',
          borderRadius: '4px',
          padding: '0.75rem',
          overflow: 'auto',
          fontSize: '0.82rem',
          margin: 0,
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
        }}
      >
        {chunk.source}
      </pre>
      <div
        style={{
          marginTop: '0.4rem',
          fontSize: '0.75rem',
          color: '#94a3b8',
          textAlign: 'right',
        }}
      >
        score: {chunk.rerank_score.toFixed(4)}
      </div>
    </div>
  )
}
