import { useState } from 'react'
import type { CSSProperties } from 'react'
import { ConfidenceTag } from './ConfidenceTag'
import type { Chunk } from '../types'

interface Props {
  chunk: Chunk
  isCited: boolean
  citationLabel: string | null
}

function scoreToConfidence(score: number): 'high' | 'medium' | 'low' {
  if (score >= 0.4) return 'high'
  if (score >= 0.15) return 'medium'
  return 'low'
}

export function ChunkCard({ chunk, isCited, citationLabel }: Props) {
  const [expanded, setExpanded] = useState(isCited)
  const [toggleHover, setToggleHover] = useState(false)
  const showSource = expanded

  const citedStyle: CSSProperties = {
    borderLeft: '3px solid var(--accent)',
    background: 'var(--accent-dim)',
    border: '1px solid rgba(108, 99, 255, 0.25)',
  }

  const uncitedStyle: CSSProperties = {
    borderLeft: '3px solid var(--border)',
    background: 'var(--surface)',
    border: '1px solid var(--border-subtle)',
  }

  const toggleStyle: CSSProperties = {
    border: `1px solid ${toggleHover ? 'var(--accent)' : 'var(--border)'}`,
    borderRadius: 'var(--radius-sm)',
    padding: '2px 10px',
    fontSize: '11px',
    color: toggleHover ? 'var(--text-primary)' : 'var(--text-secondary)',
    background: 'transparent',
    cursor: 'pointer',
    marginTop: '6px',
    display: 'inline-block',
  }

  return (
    <article
      id={citationLabel ? `chunk-${citationLabel}` : undefined}
      style={{
        borderRadius: 'var(--radius)',
        padding: '12px 14px',
        marginBottom: 10,
        ...(isCited ? citedStyle : uncitedStyle),
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: 12,
        }}
      >
        <div style={{ minWidth: 0, flex: 1 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8 }}>
            {citationLabel ? (
              <span
                className="mono"
                style={{ color: 'var(--accent)', fontWeight: 700, fontSize: '13px' }}
              >
                {citationLabel}
              </span>
            ) : null}
            <span
              style={{
                background: 'var(--surface-raised)',
                color: 'var(--text-secondary)',
                fontSize: '10px',
                padding: '1px 6px',
                borderRadius: '999px',
              }}
            >
              {chunk.node_type}
            </span>
            <span style={{ fontWeight: 700, fontSize: '14px' }}>{chunk.name}</span>
          </div>
        </div>
        <div style={{ textAlign: 'right', flexShrink: 0 }}>
          <div
            className="mono"
            style={{
              color: 'var(--text-dim)',
              fontSize: '11px',
            }}
          >
            {chunk.file_rel_path}:{chunk.start_line}
          </div>
          {!isCited ? (
            <div style={{ marginTop: 4, display: 'flex', justifyContent: 'flex-end' }}>
              <ConfidenceTag confidence={scoreToConfidence(chunk.score)} />
            </div>
          ) : null}
        </div>
      </div>
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        onMouseEnter={() => setToggleHover(true)}
        onMouseLeave={() => setToggleHover(false)}
        style={toggleStyle}
      >
        {showSource ? 'hide source' : 'show source'}
      </button>
      {showSource ? (
        <pre
          style={{
            background: 'var(--code-bg)',
            borderRadius: 'var(--radius-sm)',
            padding: '10px 12px',
            overflowX: 'auto',
            fontSize: '12px',
            color: 'var(--text-primary)',
            marginTop: 8,
            whiteSpace: 'pre',
          }}
        >
          <code>{chunk.source}</code>
        </pre>
      ) : null}
    </article>
  )
}
