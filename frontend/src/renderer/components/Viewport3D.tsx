import React, { useRef, useEffect } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { SceneManager } from '../three/SceneManager'

export function Viewport3D() {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<SceneManager | null>(null)
  const meshTransferUrl = usePipelineStore((s) => s.meshTransferUrl)
  const intentOverlay = usePipelineStore((s) => s.intentOverlay)
  const showIntentRegionColors = usePipelineStore((s) => s.showIntentRegionColors)
  const showIntentGizmos = usePipelineStore((s) => s.showIntentGizmos)
  const intentColorMode = usePipelineStore((s) => s.intentColorMode)
  const deviationResult = usePipelineStore((s) => s.deviationResult)
  const showDeviationHeatmap = usePipelineStore((s) => s.showDeviationHeatmap)
  const reconstructionEvents = usePipelineStore((s) => s.reconstructionEvents)
  const lastEventCount = useRef(0)

  useEffect(() => {
    if (!containerRef.current) return
    const scene = new SceneManager(containerRef.current)
    sceneRef.current = scene
    return () => scene.dispose()
  }, [])

  // Load mesh when transfer URL changes
  useEffect(() => {
    if (!meshTransferUrl || !sceneRef.current) return
    sceneRef.current.loadMeshFromUrl(meshTransferUrl)
  }, [meshTransferUrl])

  // Phase E0 — push overlay payload + visibility into the scene.
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setIntentOverlay(intentOverlay)
  }, [intentOverlay])

  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setIntentRegionColors(showIntentRegionColors)
  }, [showIntentRegionColors, intentOverlay])

  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setIntentGizmosVisible(showIntentGizmos)
  }, [showIntentGizmos, intentOverlay])

  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setIntentColorMode(intentColorMode)
  }, [intentColorMode, intentOverlay])

  // Deviation heatmap overlay
  useEffect(() => {
    if (!sceneRef.current) return
    if (showDeviationHeatmap && deviationResult?.face_colors_b64) {
      sceneRef.current.applyDeviationColors(deviationResult.face_colors_b64, deviationResult.n_faces)
    } else {
      sceneRef.current.clearDeviationColors()
    }
  }, [showDeviationHeatmap, deviationResult])

  // Live reconstruction — render new surfaces as they arrive
  useEffect(() => {
    if (!sceneRef.current) return
    const newEvents = reconstructionEvents.slice(lastEventCount.current)
    for (const ev of newEvents) {
      if (ev.status === 'built' && ev.geometry) {
        sceneRef.current.addReconstructedSurface(
          ev.geometry.vertices, ev.geometry.faces, ev.classification
        )
      }
    }
    lastEventCount.current = reconstructionEvents.length
    // Clear when events reset to empty
    if (reconstructionEvents.length === 0 && lastEventCount.current > 0) {
      sceneRef.current.clearReconstructedSurfaces()
      lastEventCount.current = 0
    }
  }, [reconstructionEvents])

  // Drag and drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  // Click handler (patch picking removed — can repurpose for E0 region picking later)
  const downPos = useRef<{ x: number; y: number } | null>(null)
  const handlePointerDown = (e: React.PointerEvent) => {
    downPos.current = { x: e.clientX, y: e.clientY }
  }
  const handlePointerUp = (e: React.PointerEvent) => {
    const start = downPos.current
    downPos.current = null
    if (!start || !sceneRef.current) return
    // Ignore if user dragged (orbiting), only treat as click on small movement
    const dx = e.clientX - start.x
    const dy = e.clientY - start.y
    if (dx * dx + dy * dy > 9) return
    // TODO: E0 region picking can be wired here
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    const file = e.dataTransfer.files[0]
    if (file) {
      usePipelineStore.getState().loadMeshFromFile(file)
    }
  }

  return (
    <div
      ref={containerRef}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      onPointerDown={handlePointerDown}
      onPointerUp={handlePointerUp}
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: '#111111',
        cursor: 'grab',
      }}
    />
  )
}
