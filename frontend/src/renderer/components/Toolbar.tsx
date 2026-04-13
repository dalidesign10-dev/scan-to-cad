import React, { useRef } from 'react'
import { usePipelineStore } from '../store/pipelineStore'

export function Toolbar({ backendReady, showInspector, onToggleInspector }: {
  backendReady: boolean
  showInspector: boolean
  onToggleInspector: () => void
}) {
  const { stage, loading, loadMeshFromFile, loadMeshFromPath, exportCAD, error } = usePipelineStore()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) loadMeshFromFile(file)
    e.target.value = ''
  }

  const handleLoadDemo = () => {
    loadMeshFromPath('E:/Raptor/Clio 5/Draft/clio3.stl')
  }

  const canExport = stage !== 'idle'

  const btnStyle: React.CSSProperties = {
    padding: '6px 14px',
    border: '1px solid #2a2a2a',
    borderRadius: '6px',
    background: '#1a1a1a',
    color: '#f0f0f0',
    fontSize: '12px',
    fontWeight: 500,
    cursor: 'pointer',
    transition: 'all 0.15s ease',
    whiteSpace: 'nowrap',
  }

  const btnDisabled: React.CSSProperties = {
    opacity: 0.4,
    cursor: 'not-allowed',
  }

  return (
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      padding: '0 16px',
      background: '#0f0f0f',
      borderBottom: '1px solid #2a2a2a',
      height: '44px',
    }}>
      {/* Logo */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginRight: '16px' }}>
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#06b6d4' }} />
        <span style={{ fontSize: '14px', fontWeight: 600, color: '#f0f0f0', letterSpacing: '-0.3px' }}>
          Scan to CAD
        </span>
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".stl,.ply,.obj"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      <button
        style={{ ...btnStyle, ...((!backendReady || loading) ? btnDisabled : {}) }}
        onClick={() => fileInputRef.current?.click()}
        disabled={!backendReady || loading}
      >
        Open Mesh
      </button>

      <button
        style={{ ...btnStyle, ...((!backendReady || loading) ? btnDisabled : {}) }}
        onClick={handleLoadDemo}
        disabled={!backendReady || loading}
      >
        Load Demo
      </button>

      <div style={{ width: 1, height: 20, background: '#2a2a2a', margin: '0 4px' }} />

      <button
        style={{ ...btnStyle, background: 'transparent', ...((canExport && !loading) ? {} : btnDisabled) }}
        onClick={() => exportCAD('stl')}
        disabled={!canExport || loading}
      >
        Export STL
      </button>

      <button
        style={{ ...btnStyle, background: 'transparent', ...((canExport && !loading) ? {} : btnDisabled) }}
        onClick={() => exportCAD('obj')}
        disabled={!canExport || loading}
      >
        Export OBJ
      </button>

      <div style={{ flex: 1 }} />

      {error && (
        <span style={{
          fontSize: '11px',
          color: '#ef4444',
          background: 'rgba(239,68,68,0.1)',
          padding: '4px 10px',
          borderRadius: '12px',
          maxWidth: '280px',
          overflow: 'hidden',
          textOverflow: 'ellipsis',
          whiteSpace: 'nowrap',
        }}>
          {error}
        </span>
      )}

      {!backendReady && (
        <span style={{ fontSize: '11px', color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#f59e0b' }} />
          Connecting...
        </span>
      )}

      {backendReady && stage !== 'idle' && (
        <span style={{
          fontSize: '10px',
          color: '#999',
          background: '#1a1a1a',
          padding: '3px 8px',
          borderRadius: '4px',
          textTransform: 'uppercase',
          letterSpacing: '0.5px',
        }}>
          {stage}
        </span>
      )}

      {/* Inspector toggle */}
      <button
        onClick={onToggleInspector}
        style={{
          ...btnStyle,
          background: showInspector ? '#242424' : 'transparent',
          padding: '5px 8px',
          fontSize: '14px',
          lineHeight: 1,
        }}
        title={showInspector ? 'Hide inspector' : 'Show inspector'}
      >
        ◧
      </button>
    </div>
  )
}
