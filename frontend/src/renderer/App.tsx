import React, { useEffect, useState } from 'react'
import { Toolbar } from './components/Toolbar'
import { PipelinePanel } from './components/PipelinePanel'
import { Viewport3D } from './components/Viewport3D'
import { FeatureInspector } from './components/FeatureInspector'
import { StatusBar } from './components/StatusBar'

declare global {
  interface Window {
    api?: {
      openFileDialog: () => Promise<string | null>
      saveFileDialog: (name: string) => Promise<string | null>
      onProgress: (cb: (data: any) => void) => () => void
      readBinaryFile: (path: string) => ArrayBuffer
    }
  }
}

export default function App() {
  const [backendReady, setBackendReady] = useState(false)
  const [showInspector, setShowInspector] = useState(true)
  const [panelCollapsed, setPanelCollapsed] = useState(false)

  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch('http://localhost:8321/api/ping')
        if (res.ok) setBackendReady(true)
      } catch {
        setTimeout(check, 2000)
      }
    }
    check()
  }, [])

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg)' }}>
      <Toolbar backendReady={backendReady} showInspector={showInspector} onToggleInspector={() => setShowInspector(v => !v)} />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <PipelinePanel collapsed={panelCollapsed} onToggleCollapse={() => setPanelCollapsed(v => !v)} />
        <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
          <Viewport3D />
          {showInspector && <FeatureInspector onClose={() => setShowInspector(false)} />}
        </div>
      </div>
      <StatusBar backendReady={backendReady} />
    </div>
  )
}
