import React, { useRef } from 'react'
import { usePipelineStore } from '../store/pipelineStore'

const styles = {
  toolbar: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    padding: '8px 16px',
    background: '#16213e',
    borderBottom: '1px solid #0f3460',
    height: '48px',
  },
  title: {
    fontSize: '14px',
    fontWeight: 700,
    color: '#e94560',
    marginRight: '24px',
  },
  btn: {
    padding: '6px 14px',
    border: 'none',
    borderRadius: '4px',
    background: '#0f3460',
    color: '#e0e0e0',
    fontSize: '12px',
    cursor: 'pointer',
    transition: 'background 0.15s',
  } as React.CSSProperties,
  btnPrimary: {
    background: '#e94560',
    color: '#fff',
  },
  btnDisabled: {
    opacity: 0.5,
    cursor: 'not-allowed',
  },
}

export function Toolbar({ backendReady }: { backendReady: boolean }) {
  const { stage, loading, loadMeshFromFile, loadMeshFromPath, exportCAD, runAllPipeline, error } = usePipelineStore()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) loadMeshFromFile(file)
    e.target.value = ''
  }

  const handleLoadDemo = () => {
    loadMeshFromPath('E:/Raptor/Clio 5/Draft/clio3.stl')
  }

  const canExport = stage === 'fitted' || stage === 'features_detected'

  return (
    <div style={styles.toolbar}>
      <span style={styles.title}>GEOMAGIC CLAUDE</span>

      <input
        ref={fileInputRef}
        type="file"
        accept=".stl,.ply,.obj"
        onChange={handleFileSelect}
        style={{ display: 'none' }}
      />

      <button
        style={{ ...styles.btn, opacity: backendReady && !loading ? 1 : 0.5 }}
        onClick={() => fileInputRef.current?.click()}
        disabled={!backendReady || loading}
      >
        Open Mesh
      </button>

      <button
        style={{ ...styles.btn, background: '#4ecca3', color: '#111', opacity: backendReady && !loading ? 1 : 0.5 }}
        onClick={handleLoadDemo}
        disabled={!backendReady || loading}
      >
        Load Demo (clio3.stl)
      </button>

      <button
        style={{ ...styles.btn, background: '#f5a623', color: '#111', fontWeight: 'bold', opacity: backendReady && !loading ? 1 : 0.5 }}
        onClick={runAllPipeline}
        disabled={!backendReady || loading}
        title="Load demo + run all pipeline steps + build polyhedral CAD"
      >
        ▶ Run All
      </button>

      <button
        style={{
          ...styles.btn,
          ...styles.btnPrimary,
          ...(canExport && !loading ? {} : styles.btnDisabled),
        }}
        onClick={() => exportCAD('stl')}
        disabled={!canExport || loading}
        title="Export tessellated CAD preview as STL"
      >
        Export STL
      </button>

      <button
        style={{
          ...styles.btn,
          ...styles.btnPrimary,
          ...(canExport && !loading ? {} : styles.btnDisabled),
        }}
        onClick={() => exportCAD('obj')}
        disabled={!canExport || loading}
        title="Export tessellated CAD preview as OBJ"
      >
        Export OBJ
      </button>

      <button
        style={{
          ...styles.btn,
          ...styles.btnPrimary,
          ...(canExport && !loading ? {} : styles.btnDisabled),
        }}
        onClick={() => exportCAD('step')}
        disabled={!canExport || loading}
        title="Export true B-Rep STEP (requires pythonocc)"
      >
        Export STEP
      </button>

      <div style={{ flex: 1 }} />

      {error && (
        <span style={{ fontSize: '11px', color: '#ff6b6b', maxWidth: '300px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          Error: {error}
        </span>
      )}

      {!backendReady && (
        <span style={{ fontSize: '11px', color: '#f5a623' }}>
          Connecting to backend...
        </span>
      )}

      {backendReady && stage !== 'idle' && (
        <span style={{ fontSize: '11px', color: '#888' }}>
          Stage: {stage}
        </span>
      )}
    </div>
  )
}
