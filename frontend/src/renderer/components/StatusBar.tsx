import React from 'react'
import { usePipelineStore } from '../store/pipelineStore'

export function StatusBar({ backendReady }: { backendReady: boolean }) {
  const { progress, meshInfo, loading } = usePipelineStore()

  return (
    <div
      style={{
        height: '28px',
        display: 'flex',
        alignItems: 'center',
        padding: '0 16px',
        background: '#16213e',
        borderTop: '1px solid #0f3460',
        fontSize: '11px',
        color: '#888',
        gap: '16px',
      }}
    >
      {!backendReady && (
        <span style={{ color: '#f5a623' }}>Waiting for Python backend on port 8321...</span>
      )}
      {backendReady && progress && (
        <>
          <div
            style={{
              width: '120px',
              height: '4px',
              background: '#0f3460',
              borderRadius: '2px',
              overflow: 'hidden',
            }}
          >
            <div
              style={{
                width: `${progress.pct}%`,
                height: '100%',
                background: '#e94560',
                transition: 'width 0.2s',
              }}
            />
          </div>
          <span>{progress.message || progress.stage}</span>
        </>
      )}
      {backendReady && !progress && meshInfo && (
        <>
          <span>Vertices: {meshInfo.vertices.toLocaleString()}</span>
          <span>Faces: {meshInfo.faces.toLocaleString()}</span>
        </>
      )}
      {backendReady && !progress && !meshInfo && (
        <span>Ready — Click "Open Mesh" or "Load Demo" or drag-and-drop an STL file</span>
      )}
      <div style={{ flex: 1 }} />
      {loading && <span style={{ color: '#e94560' }}>Processing...</span>}
    </div>
  )
}
