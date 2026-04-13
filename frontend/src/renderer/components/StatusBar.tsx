import React from 'react'
import { usePipelineStore } from '../store/pipelineStore'

export function StatusBar({ backendReady }: { backendReady: boolean }) {
  const { progress, meshInfo, loading } = usePipelineStore()

  return (
    <div style={{
      height: '32px',
      display: 'flex',
      alignItems: 'center',
      padding: '0 16px',
      background: 'var(--bg)',
      borderTop: '1px solid var(--border)',
      fontSize: '11px',
      color: 'var(--text-muted)',
      gap: '16px',
    }}>
      {!backendReady && (
        <span style={{ color: 'var(--accent-orange)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-orange)', animation: 'pulse 1.5s infinite' }} />
          Connecting to backend...
        </span>
      )}
      {backendReady && progress && (
        <>
          <div style={{
            width: '120px',
            height: '3px',
            background: 'var(--border)',
            borderRadius: '2px',
            overflow: 'hidden',
          }}>
            <div style={{
              width: `${progress.pct}%`,
              height: '100%',
              background: 'var(--accent-cyan)',
              borderRadius: '2px',
              transition: 'width 0.3s ease',
            }} />
          </div>
          <span style={{ color: 'var(--text-secondary)' }}>{progress.message || progress.stage}</span>
        </>
      )}
      {backendReady && !progress && meshInfo && (
        <>
          <span>{meshInfo.vertices.toLocaleString()} vertices</span>
          <span style={{ color: 'var(--border-hover)' }}>·</span>
          <span>{meshInfo.faces.toLocaleString()} faces</span>
        </>
      )}
      {backendReady && !progress && !meshInfo && (
        <span>Ready — Drop a mesh file or click Open</span>
      )}
      <div style={{ flex: 1 }} />
      {loading && (
        <span style={{ color: 'var(--accent-cyan)', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-cyan)', animation: 'pulse 1.5s infinite' }} />
          Processing
        </span>
      )}
    </div>
  )
}
