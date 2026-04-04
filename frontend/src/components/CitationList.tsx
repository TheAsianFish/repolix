import React from 'react'
import type { Citation } from '../api'

interface Props {
  citations: Citation[]
}

export function CitationList({ citations }: Props) {
  if (citations.length === 0) return null

  return (
    <div style={{ marginTop: '1rem' }}>
      <h3 style={{ fontSize: '0.9rem', color: '#475569', marginBottom: '0.5rem' }}>
        Citations
      </h3>
      <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
        {citations.map(c => {
          const context = c.parent_class
            ? `${c.parent_class}.${c.name}`
            : c.name
          const pathLine = `${c.file_rel_path}:${c.start_line}-${c.end_line}`

          return (
            <li
              key={c.label}
              style={{
                fontSize: '0.85rem',
                padding: '0.3rem 0',
                borderBottom: '1px solid #f1f5f9',
                display: 'flex',
                gap: '0.75rem',
              }}
            >
              <span style={{ fontWeight: 600, color: '#2563eb', minWidth: '2rem' }}>
                {c.label}
              </span>
              <span style={{ color: '#334155' }}>
                {context}
              </span>
              <span
                style={{
                  color: '#94a3b8',
                  fontFamily: 'monospace',
                  fontSize: '0.8rem',
                  cursor: 'pointer',
                }}
                onClick={() => navigator.clipboard.writeText(pathLine)}
                title="Click to copy"
              >
                {pathLine}
              </span>
            </li>
          )
        })}
      </ul>
    </div>
  )
}
