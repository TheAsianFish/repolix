import type { AnswerSections, Navigation } from '../api'
import type { Citation } from '../types'
import { ConfidenceTag } from './ConfidenceTag'

const skeletonStyle = `
  @keyframes skeleton-pulse {
    0%, 100% { opacity: 0.3; }
    50% { opacity: 0.7; }
  }
`

interface Props {
  answer: string | null
  answer_sections: AnswerSections | null
  citations: Citation[]
  confidence: 'high' | 'medium' | 'low'
  isLoading: boolean
  navigation: Navigation | null
}

function scrollToChunk(label: string) {
  const el = document.getElementById(`chunk-${label}`)
  el?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
}

function renderAnswerWithBadges(text: string) {
  const parts = text.split(/(\[\d+\])/g)
  return parts.map((part, i) => {
    const m = part.match(/^\[(\d+)\]$/)
    if (m) {
      const fullLabel = part
      return (
        <button
          key={`${part}-${i}`}
          type="button"
          onClick={() => scrollToChunk(fullLabel)}
          style={{
            background: 'var(--accent-dim)',
            color: 'var(--accent)',
            borderRadius: 'var(--radius-sm)',
            padding: '0 5px',
            fontSize: '11px',
            cursor: 'pointer',
            fontWeight: 600,
            border: 'none',
            fontFamily: 'inherit',
            verticalAlign: 'baseline',
          }}
          onMouseEnter={e => {
            const t = e.currentTarget
            t.style.background = 'var(--accent)'
            t.style.color = 'white'
          }}
          onMouseLeave={e => {
            const t = e.currentTarget
            t.style.background = 'var(--accent-dim)'
            t.style.color = 'var(--accent)'
          }}
        >
          {part}
        </button>
      )
    }
    return <span key={i}>{part}</span>
  })
}

export function AnswerPanel({
  answer,
  answer_sections,
  citations,
  confidence,
  isLoading,
  navigation,
}: Props) {
  return (
    <div
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius)',
        padding: '16px',
        marginTop: '12px',
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'flex-start',
          justifyContent: 'space-between',
          gap: 12,
          marginBottom: 12,
        }}
      >
        <span
          style={{
            color: 'var(--text-secondary)',
            fontSize: '11px',
            textTransform: 'uppercase',
            letterSpacing: '0.08em',
          }}
        >
          Answer
        </span>
        {isLoading ? (
          <span style={{ width: 56, height: 22 }} aria-hidden />
        ) : (
          <ConfidenceTag confidence={confidence} />
        )}
      </div>

      {isLoading ? (
        <div>
          <style>{skeletonStyle}</style>
          <div
            style={{
              height: '14px',
              background: 'var(--surface-raised)',
              borderRadius: 'var(--radius-sm)',
              marginBottom: '10px',
              width: '92%',
              animation: 'skeleton-pulse 1.4s ease-in-out infinite',
            }}
          />
          <div
            style={{
              height: '14px',
              background: 'var(--surface-raised)',
              borderRadius: 'var(--radius-sm)',
              marginBottom: '10px',
              width: '78%',
              animation: 'skeleton-pulse 1.4s ease-in-out infinite 0.2s',
            }}
          />
          <div
            style={{
              height: '14px',
              background: 'var(--surface-raised)',
              borderRadius: 'var(--radius-sm)',
              width: '55%',
              animation: 'skeleton-pulse 1.4s ease-in-out infinite 0.4s',
            }}
          />
        </div>
      ) : navigation ? (
        /* Low confidence — yellow navigation panel */
        <div
          style={{
            background: 'rgba(245, 158, 11, 0.08)',
            border: '1px solid rgba(245, 158, 11, 0.25)',
            borderRadius: 'var(--radius)',
            padding: '12px 14px',
            marginTop: 8,
          }}
        >
          <div
            style={{
              color: '#f59e0b',
              fontSize: '13px',
              marginBottom: 12,
            }}
          >
            {navigation.message}
          </div>

          {navigation.closest_matches.map((m, i) => (
            <div
              key={i}
              style={{
                fontFamily: 'monospace',
                fontSize: '12px',
                color: 'var(--text-secondary)',
                marginBottom: 4,
              }}
            >
              →{' '}
              <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                {m.name}
              </span>
              {'  '}
              <span style={{ color: 'var(--text-dim)' }}>
                {m.file_rel_path}:{m.start_line}
              </span>
            </div>
          ))}

          {navigation.suggestions.length > 0 && (
            <div style={{ marginTop: 12 }}>
              <div
                style={{
                  color: 'var(--text-secondary)',
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  marginBottom: 6,
                }}
              >
                Suggestions
              </div>
              {navigation.suggestions.map((s, i) => (
                <div
                  key={i}
                  style={{
                    color: 'var(--text-dim)',
                    fontSize: '12px',
                    marginBottom: 4,
                  }}
                >
                  · {s}
                </div>
              ))}
            </div>
          )}
        </div>
      ) : answer_sections ? (
        /* Structured answer — three distinct sections */
        <>
          {answer_sections.answer && (
            <div
              style={{
                color: 'var(--text-primary)',
                fontWeight: 600,
                fontSize: '14px',
                lineHeight: 1.6,
                marginBottom: answer_sections.how_it_works ? 16 : 0,
              }}
            >
              {renderAnswerWithBadges(answer_sections.answer)}
            </div>
          )}

          {answer_sections.how_it_works && (
            <div style={{ marginBottom: answer_sections.where_to_look ? 16 : 0 }}>
              <div
                style={{
                  color: 'var(--text-secondary)',
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  marginBottom: 6,
                }}
              >
                How it works
              </div>
              <div
                style={{
                  color: 'var(--text-primary)',
                  lineHeight: 1.6,
                  fontSize: '13px',
                }}
              >
                {renderAnswerWithBadges(answer_sections.how_it_works)}
              </div>
            </div>
          )}

          {answer_sections.where_to_look && (
            <div>
              <div
                style={{
                  color: 'var(--text-secondary)',
                  fontSize: '11px',
                  textTransform: 'uppercase',
                  letterSpacing: '0.08em',
                  marginBottom: 6,
                }}
              >
                Where to look next
              </div>
              <div
                style={{
                  color: 'var(--text-dim)',
                  lineHeight: 1.6,
                  fontSize: '13px',
                }}
              >
                {renderAnswerWithBadges(answer_sections.where_to_look)}
              </div>
            </div>
          )}
        </>
      ) : (
        /* Fallback — raw answer string */
        <div style={{ color: 'var(--text-primary)', whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
          {renderAnswerWithBadges(answer ?? '')}
        </div>
      )}

      {!navigation && citations.length > 0 && (
        <div style={{ marginTop: 20 }}>
          <div
            style={{
              color: 'var(--text-secondary)',
              fontSize: '11px',
              textTransform: 'uppercase',
              letterSpacing: '0.08em',
              marginBottom: 8,
            }}
          >
            Citations
          </div>
          <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
            {citations.map(c => (
              <li
                key={c.label}
                className="mono"
                style={{
                  fontSize: '12.5px',
                  color: 'var(--text-primary)',
                  marginBottom: 4,
                }}
              >
                {c.label} {c.file_rel_path}:{c.start_line}–{c.end_line} ({c.name})
                {c.is_truncated ? (
                  <span style={{ color: 'var(--text-dim)' }}> [truncated]</span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
