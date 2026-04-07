import React, { useEffect, useState } from 'react'
import { Toolbar } from './components/Toolbar'
import { PipelinePanel } from './components/PipelinePanel'
import { Viewport3D } from './components/Viewport3D'
import { FeatureInspector } from './components/FeatureInspector'
import { StatusBar } from './components/StatusBar'
import { usePipelineStore } from './store/pipelineStore'

declare global {
  interface Window {
    api?: {
      openFileDialog: () => Promise<string | null>
      saveFileDialog: (name: string) => Promise<string | null>
      pythonCall: (method: string, params?: any) => Promise<any>
      pythonPing: () => Promise<any>
      onProgress: (cb: (data: any) => void) => () => void
      readBinaryFile: (path: string) => ArrayBuffer
    }
  }
}

export default function App() {
  const setProgress = usePipelineStore((s) => s.setProgress)
  const [backendReady, setBackendReady] = useState(false)

  useEffect(() => {
    // Listen for Electron progress events if available
    if (window.api?.onProgress) {
      const cleanup = window.api.onProgress((data) => setProgress(data))
      return cleanup
    }
  }, [setProgress])

  // Global Ctrl+Z / Ctrl+Y (and Ctrl+Shift+Z) for undo/redo
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const mod = e.ctrlKey || e.metaKey
      if (!mod) return
      const target = e.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
      const key = e.key.toLowerCase()
      if (key === 'z' && !e.shiftKey) {
        e.preventDefault()
        usePipelineStore.getState().undo()
      } else if (key === 'y' || (key === 'z' && e.shiftKey)) {
        e.preventDefault()
        usePipelineStore.getState().redo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Check backend connectivity
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
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
      <Toolbar backendReady={backendReady} />
      <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
        <PipelinePanel />
        <Viewport3D />
        <FeatureInspector />
      </div>
      <StatusBar backendReady={backendReady} />
    </div>
  )
}
