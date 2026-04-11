import { create } from 'zustand'

const API = 'http://localhost:8321/api'

export interface PrimitiveResult {
  patch_id: number
  type: string
  normal?: number[]
  d?: number
  axis?: number[]
  center?: number[]
  radius?: number
  inlier_ratio: number
  face_count?: number
  centroid?: number[]
  bbox_min?: number[]
  bbox_max?: number[]
  patch_size?: number
}

export interface PatchInfo {
  id: number
  face_count: number
  classification: string
  mean_k1?: number
  mean_k2?: number
  is_fillet?: boolean
  avg_normal?: number[]
}

export interface FeatureInfo {
  type: string
  patch_id: number
  adjacent_patches: number[]
  estimated_radius?: number
  angle_degrees?: number
  face_count: number
}

interface PipelineState {
  stage: 'idle' | 'loaded' | 'preprocessed' | 'segmented' | 'fitted' | 'features_detected' | 'exported'
  loading: boolean
  progress: { stage: string; pct: number; message: string } | null
  error: string | null

  // Transfer data URLs (for binary mesh/labels)
  meshTransferUrl: string | null
  meshInfo: { vertices: number; faces: number; is_watertight: boolean } | null

  labelsUrl: string | null
  patches: PatchInfo[]
  primitives: PrimitiveResult[]
  features: FeatureInfo[]
  infiniteSurfaces: any[]
  edges: any[]
  showInfiniteSurfaces: boolean
  cadPreview: any[]
  showCadPreview: boolean
  polyhedralCad: { vertices: number[][]; faces: number[][] } | null
  showPolyhedralCad: boolean
  selectedPatchId: number | null

  // Multi-selection for merging patches into logical faces
  selectedPatchIds: number[]
  mergeMap: Record<number, number>  // patch_id -> group_id

  // User-created planes (mirror of backend SESSION["user_planes"]) and selection
  userPlanes: Array<{ id: number; normal: number[]; d: number; centroid: number[]; source_patch_ids: number[] }>
  selectedPlaneIds: number[]

  // Undo/redo
  history: any[]
  future: any[]

  // Point2Cyl AI inference
  point2cylResult: any | null
  showPoint2Cyl: boolean

  // Phase E0 — mechanical intent reconstruction
  intentSummary: any | null
  intentRegions: any[] | null
  intentBoundaries: any[] | null
  intentOverlay: any | null
  showIntentRegionColors: boolean
  showIntentGizmos: boolean

  setProgress: (p: { stage: string; pct: number; message: string } | null) => void
  setSelectedPatch: (id: number | null) => void
  setError: (e: string | null) => void

  togglePatchSelection: (id: number, additive: boolean) => void
  clearPatchSelection: () => void
  mergeSelectedPatches: () => Promise<void>
  clearMergeGroups: () => Promise<void>
  createInfinitePlaneFromSelection: () => Promise<void>
  intersectUserPlanes: () => Promise<void>
  clearUserPlanes: () => Promise<void>
  togglePlaneSelection: (id: number, additive: boolean) => void
  clearPlaneSelection: () => void
  undo: () => Promise<void>
  redo: () => Promise<void>
  runPoint2Cyl: () => Promise<void>
  togglePoint2Cyl: () => void
  buildPolyhedralBrep: (params?: any) => Promise<void>
  filletSharpEdges: (params?: any) => Promise<void>
  chamferSharpEdges: (params?: any) => Promise<void>
  exportBrep: (format: 'stl' | 'step' | 'brep') => Promise<void>

  loadMeshFromFile: (file: File) => Promise<void>
  loadMeshFromPath: (path: string) => Promise<void>
  runCleanup: (params?: any) => Promise<void>
  runPreprocess: (params?: any) => Promise<void>
  runSegmentation: (params?: any) => Promise<void>
  runFitting: () => Promise<void>
  runExtractCage: (params?: any) => Promise<void>
  runFeatureDetection: () => Promise<void>
  runIntersection: () => Promise<void>
  runCadPreview: () => Promise<void>
  toggleInfiniteSurfaces: () => void
  toggleCadPreview: () => void
  runPolyhedralCad: () => Promise<void>
  togglePolyhedralCad: () => void
  runAllPipeline: () => Promise<void>
  exportSTEP: () => Promise<void>
  exportCAD: (format: 'stl' | 'obj' | 'ply' | 'step') => Promise<void>

  // Phase E0 actions
  runIntentSegmentation: (params?: any) => Promise<void>
  toggleIntentRegionColors: () => void
  toggleIntentGizmos: () => void
}

function extractFilename(transferPath: string): string {
  // Extract filename from full path like "C:\\Users\\...\\abc123.bin"
  return transferPath.replace(/\\/g, '/').split('/').pop() || ''
}

// Slice of state we treat as undoable
function snapshotState(s: any) {
  return {
    infiniteSurfaces: s.infiniteSurfaces,
    edges: s.edges,
    mergeMap: { ...s.mergeMap },
    userPlanes: s.userPlanes,
    selectedPlaneIds: [...s.selectedPlaneIds],
    selectedPatchIds: [...s.selectedPatchIds],
  }
}

const HISTORY_LIMIT = 50

async function syncUserPlanesToBackend(planes: any[]) {
  try {
    await fetch(`${API}/set_user_planes`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ planes }),
    })
  } catch {}
}

export const usePipelineStore = create<PipelineState>((set) => ({
  stage: 'idle',
  loading: false,
  progress: null,
  error: null,
  meshTransferUrl: null,
  meshInfo: null,
  labelsUrl: null,
  patches: [],
  primitives: [],
  features: [],
  infiniteSurfaces: [],
  edges: [],
  showInfiniteSurfaces: false,
  cadPreview: [],
  showCadPreview: false,
  polyhedralCad: null,
  showPolyhedralCad: false,
  selectedPatchId: null,
  selectedPatchIds: [],
  mergeMap: {},
  userPlanes: [],
  selectedPlaneIds: [],
  history: [],
  future: [],
  point2cylResult: null,
  showPoint2Cyl: false,

  intentSummary: null,
  intentRegions: null,
  intentBoundaries: null,
  intentOverlay: null,
  showIntentRegionColors: false,
  showIntentGizmos: false,

  setProgress: (p) => set({ progress: p }),
  setSelectedPatch: (id) => set({ selectedPatchId: id }),
  setError: (e) => set({ error: e }),

  togglePatchSelection: (id, additive) => set((s) => {
    if (!additive) {
      // Single-select: replace, or clear if clicking the same one
      const same = s.selectedPatchIds.length === 1 && s.selectedPatchIds[0] === id
      return { selectedPatchIds: same ? [] : [id], selectedPatchId: same ? null : id }
    }
    const exists = s.selectedPatchIds.includes(id)
    const next = exists ? s.selectedPatchIds.filter((x) => x !== id) : [...s.selectedPatchIds, id]
    return { selectedPatchIds: next, selectedPatchId: next[next.length - 1] ?? null }
  }),

  clearPatchSelection: () => set({ selectedPatchIds: [], selectedPatchId: null }),

  mergeSelectedPatches: async () => {
    const cur = usePipelineStore.getState()
    const ids = cur.selectedPatchIds
    if (ids.length < 2) return
    try {
      const res = await fetch(`${API}/merge_patches`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patch_ids: ids }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const mm: Record<number, number> = {}
      for (const k of Object.keys(result.merge_map || {})) mm[Number(k)] = result.merge_map[k]
      set((s) => ({
        history: [...s.history, snapshotState(cur)].slice(-HISTORY_LIMIT),
        future: [],
        mergeMap: mm,
        selectedPatchIds: [],
        selectedPatchId: null,
      }))
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  createInfinitePlaneFromSelection: async () => {
    const cur = usePipelineStore.getState()
    const ids = cur.selectedPatchIds
    if (ids.length === 0) return
    try {
      const res = await fetch(`${API}/infinite_plane_from_patches`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patch_ids: ids }),
      })
      if (!res.ok) throw new Error(await res.text())
      const surf = await res.json()
      const newPlane = {
        id: surf.plane_id,
        normal: surf.normal,
        d: surf.d,
        centroid: surf.centroid || [0, 0, 0],
        source_patch_ids: surf.source_patch_ids || ids,
      }
      set((s) => ({
        history: [...s.history, snapshotState(cur)].slice(-HISTORY_LIMIT),
        future: [],
        infiniteSurfaces: [...s.infiniteSurfaces, surf],
        userPlanes: [...s.userPlanes, newPlane],
        showInfiniteSurfaces: true,
      }))
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  togglePlaneSelection: (id, additive) => set((s) => {
    if (!additive) {
      const same = s.selectedPlaneIds.length === 1 && s.selectedPlaneIds[0] === id
      return { selectedPlaneIds: same ? [] : [id] }
    }
    const exists = s.selectedPlaneIds.includes(id)
    return {
      selectedPlaneIds: exists
        ? s.selectedPlaneIds.filter((x) => x !== id)
        : [...s.selectedPlaneIds, id],
    }
  }),

  clearPlaneSelection: () => set({ selectedPlaneIds: [] }),

  undo: async () => {
    const s = usePipelineStore.getState()
    if (s.history.length === 0) return
    const prev = s.history[s.history.length - 1]
    const cur = snapshotState(s)
    set({
      history: s.history.slice(0, -1),
      future: [...s.future, cur].slice(-HISTORY_LIMIT),
      ...prev,
    })
    await syncUserPlanesToBackend(prev.userPlanes)
  },

  redo: async () => {
    const s = usePipelineStore.getState()
    if (s.future.length === 0) return
    const next = s.future[s.future.length - 1]
    const cur = snapshotState(s)
    set({
      future: s.future.slice(0, -1),
      history: [...s.history, cur].slice(-HISTORY_LIMIT),
      ...next,
    })
    await syncUserPlanesToBackend(next.userPlanes)
  },

  buildPolyhedralBrep: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'brep', pct: 5, message: 'Building polyhedral B-Rep…' } })
    try {
      const res = await fetch(`${API}/build_polyhedral_brep`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_faces: 400, ...params }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        polyhedralCad: { vertices: result.vertices, faces: result.faces },
        showPolyhedralCad: true,
        progress: { stage: 'brep', pct: 100, message: `${result.n_faces} OCC faces (input ${result.decimated_faces})` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  filletSharpEdges: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'fillet', pct: 10, message: 'Filleting sharp edges…' } })
    try {
      const res = await fetch(`${API}/fillet_sharp_edges`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ radius: 0.3, min_dihedral_deg: 100, ...params }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        polyhedralCad: { vertices: result.vertices, faces: result.faces },
        showPolyhedralCad: true,
        progress: { stage: 'fillet', pct: 100, message: `${result.n_edges_modified} edges filleted` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  chamferSharpEdges: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'chamfer', pct: 10, message: 'Chamfering sharp edges…' } })
    try {
      const res = await fetch(`${API}/chamfer_sharp_edges`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ distance: 0.3, min_dihedral_deg: 100, ...params }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        polyhedralCad: { vertices: result.vertices, faces: result.faces },
        showPolyhedralCad: true,
        progress: { stage: 'chamfer', pct: 100, message: `${result.n_edges_modified} edges chamfered` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  exportBrep: async (format) => {
    set({ loading: true, error: null, progress: { stage: 'export', pct: 30, message: `Exporting B-Rep as ${format.toUpperCase()}…` } })
    try {
      const res = await fetch(`${API}/export_brep`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ format }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      window.open(`${API}/download_cad/${result.filename}`, '_blank')
      set({ progress: { stage: 'export', pct: 100, message: `${format.toUpperCase()} downloaded` } })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  runPoint2Cyl: async () => {
    set({ loading: true, error: null, progress: { stage: 'point2cyl', pct: 20, message: 'Running Point2Cyl AI…' } })
    try {
      const res = await fetch(`${API}/run_point2cyl`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        point2cylResult: result,
        showPoint2Cyl: true,
        progress: { stage: 'point2cyl', pct: 100, message: `${result.n_segments} extrusions detected` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  togglePoint2Cyl: () => set((s) => ({ showPoint2Cyl: !s.showPoint2Cyl })),

  intersectUserPlanes: async () => {
    const cur = usePipelineStore.getState()
    const ids = cur.selectedPlaneIds
    if (ids.length < 2) {
      set({ error: 'Select at least 2 planes to intersect' })
      return
    }
    try {
      const res = await fetch(`${API}/intersect_user_planes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ plane_ids: ids }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set((s) => ({
        history: [...s.history, snapshotState(cur)].slice(-HISTORY_LIMIT),
        future: [],
        edges: result.edges || [],
      }))
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  clearUserPlanes: async () => {
    const cur = usePipelineStore.getState()
    try {
      await fetch(`${API}/clear_user_planes`, { method: 'POST' })
      set((s) => ({
        history: [...s.history, snapshotState(cur)].slice(-HISTORY_LIMIT),
        future: [],
        infiniteSurfaces: [],
        edges: [],
        userPlanes: [],
        selectedPlaneIds: [],
      }))
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  clearMergeGroups: async () => {
    try {
      await fetch(`${API}/clear_merges`, { method: 'POST' })
      set({ mergeMap: {}, selectedPatchIds: [], selectedPatchId: null })
    } catch (e: any) {
      set({ error: e.message })
    }
  },

  loadMeshFromFile: async (file: File) => {
    set({ loading: true, error: null, progress: { stage: 'load', pct: 10, message: 'Uploading...' } })
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API}/load_mesh`, { method: 'POST', body: form })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const filename = extractFilename(result.transfer_path)
      set({
        stage: 'loaded',
        meshTransferUrl: `${API}/transfer/${filename}`,
        meshInfo: { vertices: result.vertices, faces: result.faces, is_watertight: result.is_watertight },
        labelsUrl: null, patches: [], primitives: [], features: [], infiniteSurfaces: [],
        progress: { stage: 'load', pct: 100, message: 'Mesh loaded' },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 1500)
    }
  },

  loadMeshFromPath: async (path: string) => {
    set({ loading: true, error: null, progress: { stage: 'load', pct: 10, message: 'Loading...' } })
    try {
      const res = await fetch(`${API}/load_mesh?path=${encodeURIComponent(path)}`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const filename = extractFilename(result.transfer_path)
      set({
        stage: 'loaded',
        meshTransferUrl: `${API}/transfer/${filename}`,
        meshInfo: { vertices: result.vertices, faces: result.faces, is_watertight: result.is_watertight },
        labelsUrl: null, patches: [], primitives: [], features: [], infiniteSurfaces: [],
        progress: { stage: 'load', pct: 100, message: 'Mesh loaded' },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 1500)
    }
  },

  runCleanup: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'cleanup', pct: 5, message: 'Cleaning up scan…' } })
    try {
      const res = await fetch(`${API}/cleanup_mesh`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          poisson_depth: 10,
          sample_count: 400000,
          density_cutoff: 0.0,
          taubin_iters: 5,
          keep_largest_only: true,
          ...params,
        }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const filename = extractFilename(result.transfer_path)
      set({
        stage: 'preprocessed',
        meshTransferUrl: `${API}/transfer/${filename}`,
        meshInfo: { vertices: result.vertices, faces: result.faces, is_watertight: result.is_watertight },
        // Reset derived state — user must re-run downstream stages
        labelsUrl: null,
        patches: [],
        primitives: [],
        features: [],
        infiniteSurfaces: [],
        edges: [],
        userPlanes: [],
        selectedPlaneIds: [],
        mergeMap: {},
        selectedPatchIds: [],
        selectedPatchId: null,
        progress: { stage: 'cleanup', pct: 100, message: `${result.vertices.toLocaleString()} verts, ${result.is_watertight ? 'watertight' : 'NOT watertight'}` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  runPreprocess: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'preprocess', pct: 10, message: 'Preprocessing...' } })
    try {
      const res = await fetch(`${API}/preprocess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ denoise: 'taubin', iterations: 10, fill_holes: true, ...params }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const filename = extractFilename(result.transfer_path)
      set({
        stage: 'preprocessed',
        meshTransferUrl: `${API}/transfer/${filename}`,
        meshInfo: { vertices: result.vertices, faces: result.faces, is_watertight: result.is_watertight },
        progress: { stage: 'preprocess', pct: 100, message: 'Done' },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 1500)
    }
  },

  runSegmentation: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'segment', pct: 10, message: 'Segmenting...' } })
    try {
      const res = await fetch(`${API}/segment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ angle_threshold: 60, grow_angle: 8, min_patch_faces: 200, ...params }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const filename = extractFilename(result.labels_path)
      set({
        stage: 'segmented',
        labelsUrl: `${API}/transfer/${filename}`,
        patches: result.patches,
        mergeMap: {},
        selectedPatchIds: [],
        selectedPatchId: null,
        userPlanes: [],
        selectedPlaneIds: [],
        infiniteSurfaces: [],
        edges: [],
        history: [],
        future: [],
        progress: { stage: 'segment', pct: 100, message: `${result.n_patches} patches found` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 1500)
    }
  },

  runFitting: async () => {
    set({ loading: true, error: null, progress: { stage: 'fit', pct: 10, message: 'Fitting primitives...' } })
    try {
      const res = await fetch(`${API}/fit_primitives`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        stage: 'fitted',
        primitives: result.primitives,
        progress: { stage: 'fit', pct: 100, message: 'Fitting complete' },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 1500)
    }
  },

  runExtractCage: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'cage', pct: 5, message: 'Extracting cage…' } })
    try {
      const res = await fetch(`${API}/extract_cage`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(params),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const filename = extractFilename(result.labels_path)
      set({
        // Cage labels overwrite the segmentation labels — re-render the overlay
        labelsUrl: `${API}/transfer/${filename}`,
        patches: result.patches,
        // Reset derived selection state — primitive fits are gone now
        primitives: [],
        mergeMap: {},
        selectedPatchIds: [],
        selectedPatchId: null,
        userPlanes: [],
        selectedPlaneIds: [],
        infiniteSurfaces: [],
        edges: [],
        progress: {
          stage: 'cage',
          pct: 100,
          message: `${result.n_cage_faces} cage faces (${result.coverage_pct.toFixed(0)}% coverage)`,
        },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  runFeatureDetection: async () => {
    set({ loading: true, error: null, progress: { stage: 'features', pct: 10, message: 'Detecting features...' } })
    try {
      const res = await fetch(`${API}/detect_features`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        stage: 'features_detected',
        features: result.features,
        infiniteSurfaces: result.infinite_surfaces || [],
        progress: { stage: 'features', pct: 100, message: 'Detection complete' },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 1500)
    }
  },

  runIntersection: async () => {
    set({ loading: true, error: null, progress: { stage: 'intersect', pct: 10, message: 'Intersecting surfaces...' } })
    try {
      const res = await fetch(`${API}/intersect_surfaces`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        edges: result.edges || [],
        progress: { stage: 'intersect', pct: 100, message: `${result.n_edges} edge curves` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  toggleInfiniteSurfaces: () => set((s) => ({ showInfiniteSurfaces: !s.showInfiniteSurfaces })),

  toggleCadPreview: () => set((s) => ({ showCadPreview: !s.showCadPreview })),

  togglePolyhedralCad: () => set((s) => ({ showPolyhedralCad: !s.showPolyhedralCad })),

  runAllPipeline: async () => {
    const store = usePipelineStore.getState()
    set({ loading: true, error: null })
    try {
      // 1. Load demo if no mesh
      if (!store.meshTransferUrl) {
        await store.loadMeshFromPath('E:/Raptor/Clio 5/Draft/clio3.stl')
      }
      // 2. Preprocess
      await usePipelineStore.getState().runPreprocess()
      // 3. Segment
      await usePipelineStore.getState().runSegmentation()
      // 4. Fit
      await usePipelineStore.getState().runFitting()
      // 5. Polyhedral CAD
      await usePipelineStore.getState().runPolyhedralCad()
      set({ progress: { stage: 'all', pct: 100, message: 'All steps complete' } })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  runPolyhedralCad: async () => {
    set({ loading: true, error: null, progress: { stage: 'cad', pct: 30, message: 'Building polyhedral CAD...' } })
    try {
      const res = await fetch(`${API}/build_polyhedral_cad`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        polyhedralCad: { vertices: result.vertices, faces: result.faces },
        showPolyhedralCad: true,
        progress: { stage: 'cad', pct: 100, message: `${result.n_faces} CAD faces` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  runCadPreview: async () => {
    set({ loading: true, error: null, progress: { stage: 'cad', pct: 10, message: 'Building CAD preview...' } })
    try {
      const res = await fetch(`${API}/build_cad_preview`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        cadPreview: result.surfaces || [],
        showCadPreview: true,
        progress: { stage: 'cad', pct: 100, message: `${result.n_surfaces} CAD surfaces` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  exportCAD: async (format) => {
    set({ loading: true, error: null, progress: { stage: 'export', pct: 30, message: `Exporting ${format.toUpperCase()}...` } })
    try {
      if (format === 'step') {
        // Try true B-Rep STEP export (needs pythonocc); fall back to tessellated
        let res = await fetch(`${API}/reconstruct_brep`, { method: 'POST' })
        if (!res.ok) throw new Error(await res.text())
        res = await fetch(`${API}/export_step`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({}),
        })
        if (!res.ok) throw new Error(await res.text())
        window.open(`${API}/download_step`, '_blank')
      } else {
        const res = await fetch(`${API}/export_cad`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ format }),
        })
        if (!res.ok) throw new Error(await res.text())
        const result = await res.json()
        window.open(`${API}/download_cad/${result.filename}`, '_blank')
      }
      set({ progress: { stage: 'export', pct: 100, message: `${format.toUpperCase()} exported!` } })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  exportSTEP: async () => {
    set({ loading: true, error: null, progress: { stage: 'export', pct: 10, message: 'Reconstructing...' } })
    try {
      // Reconstruct B-Rep first
      let res = await fetch(`${API}/reconstruct_brep`, { method: 'POST' })
      if (!res.ok) throw new Error(await res.text())

      set({ progress: { stage: 'export', pct: 50, message: 'Exporting STEP...' } })

      // Export
      res = await fetch(`${API}/export_step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) throw new Error(await res.text())

      // Download the file
      window.open(`${API}/download_step`, '_blank')
      set({ stage: 'exported', progress: { stage: 'export', pct: 100, message: 'STEP exported!' } })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  // ───────────────────────── Phase E0 ─────────────────────────
  runIntentSegmentation: async (params = {}) => {
    set({ loading: true, error: null, progress: { stage: 'intent', pct: 5, message: 'Building proxy mesh…' } })
    try {
      const res = await fetch(`${API}/intent/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ target_proxy_faces: 30000, min_region_faces: 12, ...params }),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      // Pull the overlay payload separately so the tinted-mesh / gizmos
      // path doesn't bloat the /intent/run response.
      const ovlRes = await fetch(`${API}/intent/overlays`)
      const overlay = ovlRes.ok ? await ovlRes.json() : null
      set({
        intentSummary: result.summary || null,
        intentRegions: result.regions || [],
        intentBoundaries: result.boundaries || [],
        intentOverlay: overlay,
        showIntentRegionColors: true,
        showIntentGizmos: true,
        progress: {
          stage: 'intent',
          pct: 100,
          message: `${result.summary?.n_regions ?? 0} regions, ${result.summary?.n_high_plane_fits ?? 0}+${result.summary?.n_high_cylinder_fits ?? 0} high fits`,
        },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  toggleIntentRegionColors: () => set((s) => ({ showIntentRegionColors: !s.showIntentRegionColors })),
  toggleIntentGizmos: () => set((s) => ({ showIntentGizmos: !s.showIntentGizmos })),
}))
