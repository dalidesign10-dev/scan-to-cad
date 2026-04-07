import React, { useRef, useEffect } from 'react'
import { usePipelineStore } from '../store/pipelineStore'
import { SceneManager } from '../three/SceneManager'

export function Viewport3D() {
  const containerRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<SceneManager | null>(null)
  const meshTransferUrl = usePipelineStore((s) => s.meshTransferUrl)
  const labelsUrl = usePipelineStore((s) => s.labelsUrl)
  const patches = usePipelineStore((s) => s.patches)
  const primitives = usePipelineStore((s) => s.primitives)
  const infiniteSurfaces = usePipelineStore((s) => s.infiniteSurfaces)
  const edges = usePipelineStore((s) => s.edges)
  const showInfiniteSurfaces = usePipelineStore((s) => s.showInfiniteSurfaces)
  const cadPreview = usePipelineStore((s) => s.cadPreview)
  const showCadPreview = usePipelineStore((s) => s.showCadPreview)
  const polyhedralCad = usePipelineStore((s) => s.polyhedralCad)
  const showPolyhedralCad = usePipelineStore((s) => s.showPolyhedralCad)
  const selectedPatchId = usePipelineStore((s) => s.selectedPatchId)
  const selectedPatchIds = usePipelineStore((s) => s.selectedPatchIds)
  const mergeMap = usePipelineStore((s) => s.mergeMap)
  const point2cylResult = usePipelineStore((s) => s.point2cylResult)
  const showPoint2Cyl = usePipelineStore((s) => s.showPoint2Cyl)

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

  // Apply segmentation overlay (with fillet highlighting)
  useEffect(() => {
    if (!labelsUrl || !sceneRef.current) return
    sceneRef.current.applySegmentationColorsFromUrl(labelsUrl, patches)
  }, [labelsUrl, patches])

  // Cheap recolor when selection or merge map changes
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.updateOverlayOptions({ mergeMap, selectedPatchIds })
  }, [mergeMap, selectedPatchIds])

  // Point2Cyl result + visibility
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.showPoint2Cyl(point2cylResult)
  }, [point2cylResult])

  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setPoint2CylVisible(showPoint2Cyl)
  }, [showPoint2Cyl])

// Show primitive gizmo only for selected patch (or none)
  useEffect(() => {
    if (!sceneRef.current) return
    if (selectedPatchId == null) {
      sceneRef.current.showPrimitiveGizmos([])
      return
    }
    const selected = primitives.filter((p) => p.patch_id === selectedPatchId)
    sceneRef.current.showPrimitiveGizmos(selected)
  }, [primitives, selectedPatchId])

  // Show infinite surfaces from feature detection
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.showInfiniteSurfaces(infiniteSurfaces || [])
  }, [infiniteSurfaces])

  // Toggle infinite surface visibility
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setInfiniteSurfacesVisible(showInfiniteSurfaces)
  }, [showInfiniteSurfaces])

  // Show edge curves from intersection
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.showEdgeCurves(edges || [])
  }, [edges])

  // CAD preview surfaces
  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.showCadPreview(cadPreview || [])
  }, [cadPreview])

  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setCadPreviewVisible(showCadPreview)
  }, [showCadPreview])

  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.showPolyhedralCad(polyhedralCad)
  }, [polyhedralCad])

  useEffect(() => {
    if (!sceneRef.current) return
    sceneRef.current.setPolyhedralCadVisible(showPolyhedralCad)
  }, [showPolyhedralCad])

  // Drag and drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  // Click-to-select patch (only when segmentation is active)
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
    const rect = (e.currentTarget as HTMLDivElement).getBoundingClientRect()
    const ndcX = ((e.clientX - rect.left) / rect.width) * 2 - 1
    const ndcY = -((e.clientY - rect.top) / rect.height) * 2 + 1

    if (!labelsUrl) return
    const pid = sceneRef.current.pickPatchAt(ndcX, ndcY)
    if (pid == null) {
      usePipelineStore.getState().clearPatchSelection()
      return
    }
    const additive = e.shiftKey || e.ctrlKey || e.metaKey
    usePipelineStore.getState().togglePatchSelection(pid, additive)
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
        flex: 1,
        background: '#0a0a1a',
        position: 'relative',
        cursor: 'grab',
      }}
    />
  )
}
