import { ChunkCard } from './ChunkCard'
import type { Chunk } from '../types'

const skeletonStyle = `
  @keyframes skeleton-pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.7; }
  }
`

interface Props {
  chunks: Chunk[]
  hasQueried: boolean
  isLoading: boolean
}

function labelOrder(label: string | null): number {
  if (!label) return 999
  const m = label.match(/\[(\d+)\]/)
  return m ? parseInt(m[1], 10) : 999
}

export function ChunkList({ chunks, hasQueried, isLoading }: Props) {
  if (isLoading) {
    return (
      <div>
        <style>{skeletonStyle}</style>
        {[92, 75, 60].map((width, i) => (
          <div
            key={i}
            style={{
              background: 'var(--surface)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius)',
              padding: '14px',
              marginBottom: '8px',
            }}
          >
            <div
              style={{
                height: '12px',
                background: 'var(--surface-raised)',
                borderRadius: 'var(--radius-sm)',
                width: `${width}%`,
                animation: `skeleton-pulse 1.4s ease-in-out infinite ${i * 0.15}s`,
              }}
            />
          </div>
        ))}
      </div>
    )
  }

  if (chunks.length === 0 && !hasQueried) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          minHeight: '300px',
          gap: '12px',
          color: 'var(--text-dim)',
          userSelect: 'none',
        }}
      >
        <div style={{ fontSize: '28px', opacity: 0.4 }} aria-hidden>
          {'\u2315'}
        </div>
        <div style={{ fontSize: '13px' }}>Ask a question to see retrieved chunks</div>
        <div style={{ fontSize: '11px', opacity: 0.6 }}>
          Cited chunks will appear expanded · uncited chunks collapsed
        </div>
      </div>
    )
  }

  if (chunks.length === 0 && hasQueried) {
    return (
      <div
        style={{
          color: 'var(--text-dim)',
          textAlign: 'center',
          padding: '48px 16px',
          fontSize: '13px',
        }}
      >
        No chunks returned for this query.
      </div>
    )
  }

  const cited = chunks.filter(c => c.is_cited).sort((a, b) => {
    return labelOrder(a.citation_label) - labelOrder(b.citation_label)
  })
  const uncited = chunks.filter(c => !c.is_cited)

  return (
    <div>
      {cited.map(chunk => (
        <ChunkCard
          key={chunk.id}
          chunk={chunk}
          isCited
          citationLabel={chunk.citation_label}
        />
      ))}
      {cited.length > 0 && uncited.length > 0 ? (
        <div
          style={{
            margin: '20px 0 16px',
            borderTop: '1px solid var(--border-subtle)',
            paddingTop: 12,
            color: 'var(--text-dim)',
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
          }}
        >
          Additional context
        </div>
      ) : null}
      {uncited.map(chunk => (
        <ChunkCard
          key={chunk.id}
          chunk={chunk}
          isCited={false}
          citationLabel={null}
        />
      ))}
    </div>
  )
}
