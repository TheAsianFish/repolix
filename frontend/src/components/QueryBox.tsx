import React, { useState } from 'react'

interface Props {
  onSubmit: (question: string) => void
  loading: boolean
}

export function QueryBox({ onSubmit, loading }: Props) {
  const [value, setValue] = useState('')

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === 'Enter' && !e.shiftKey && value.trim()) {
      e.preventDefault()
      onSubmit(value.trim())
    }
  }

  return (
    <div style={{ marginBottom: '1.5rem' }}>
      <textarea
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Ask a question about the codebase... (Enter to submit)"
        disabled={loading}
        rows={3}
        style={{
          width: '100%',
          padding: '0.75rem',
          fontSize: '1rem',
          fontFamily: 'inherit',
          border: '1px solid #ccc',
          borderRadius: '6px',
          resize: 'vertical',
          boxSizing: 'border-box',
        }}
      />
      <button
        onClick={() => value.trim() && onSubmit(value.trim())}
        disabled={loading || !value.trim()}
        style={{
          marginTop: '0.5rem',
          padding: '0.5rem 1.25rem',
          fontSize: '0.95rem',
          cursor: loading ? 'not-allowed' : 'pointer',
          borderRadius: '6px',
          border: 'none',
          background: '#2563eb',
          color: '#fff',
        }}
      >
        {loading ? 'Searching...' : 'Ask'}
      </button>
    </div>
  )
}
