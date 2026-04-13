import { create } from 'zustand'

const API = 'http://localhost:8321/api'

interface PipelineState {
  stage: 'idle' | 'loaded' | 'preprocessed' | 'segmented' | 'fitted' | 'features_detected' | 'exported'
  loading: boolean
  progress: { stage: string; pct: number; message: string } | null
  error: string | null

  // Transfer data URLs (for binary mesh)
  meshTransferUrl: string | null
  meshInfo: { vertices: number; faces: number; is_watertight: boolean } | null

  // Phase E0 — mechanical intent reconstruction
  intentSummary: any | null
  intentRegions: any[] | null
  intentBoundaries: any[] | null
  intentOverlay: any | null
  showIntentRegionColors: boolean
  showIntentGizmos: boolean
  intentColorMode: 'region' | 'family'

  // Deviation analysis
  deviationResult: {
    mean_deviation: number
    max_deviation: number
    p95_deviation: number
    pct_snapped: number
    face_colors_b64: string
    n_faces: number
    color_scale_max: number
    tolerance_bands: Record<string, number>
  } | null
  showDeviationHeatmap: boolean

  // Phase E1 — snap
  snapResult: {
    n_snapped: number
    n_edge: number
    n_corner: number
    pct_snapped: number
    mean_displacement: number
  } | null
  snappedMeshTransferUrl: string | null
  showSnappedMesh: boolean

  setProgress: (p: { stage: string; pct: number; message: string } | null) => void
  setError: (e: string | null) => void

  loadMeshFromFile: (file: File) => Promise<void>
  loadMeshFromPath: (path: string) => Promise<void>
  runCleanup: (params?: any) => Promise<void>
  exportCAD: (format: 'stl' | 'obj' | 'ply' | 'step') => Promise<void>

  // Phase E0 actions
  runIntentSegmentation: (params?: any) => Promise<void>
  toggleIntentRegionColors: () => void
  toggleIntentGizmos: () => void
  setIntentColorMode: (mode: 'region' | 'family') => void

  // Deviation analysis actions
  runDeviationAnalysis: () => Promise<void>
  toggleDeviationHeatmap: () => void

  // Live reconstruction
  reconstructionEvents: Array<{
    region_id: number
    step: number
    total: number
    classification: string
    confidence: number
    reasoning: string
    status: string
    deviation?: { mean: number; max: number; pct_within_0_1mm: number }
    geometry?: { vertices: number[][]; faces: number[][] }
  }>
  reconstructionRunning: boolean
  reconstructionProgress: { step: number; total: number } | null
  runLiveReconstruction: (apiKey?: string) => void
  stopReconstruction: () => void

  // AI classification
  classifyResult: {
    n_classified: number
    n_changed: number
    type_counts: Record<string, number>
    classifications: Array<{ region_id: number; classification: string; confidence: number; reasoning: string }>
  } | null
  runAIClassify: (apiKey?: string) => Promise<void>

  // Phase E1 actions
  runSnapToSurfaces: () => Promise<void>
  toggleSnappedMesh: () => void

  // Phase E3 — STEP export
  stepResult: {
    n_faces_built: number
    n_faces_failed: number
    step_size: number
  } | null
  runExportStep: () => Promise<void>
  downloadStep: () => void

  // Phase E2 — trim
  trimResult: {
    n_trimmed_faces: number
    n_with_holes: number
    total_boundary_vertices: number
    n_plane_faces: number
    n_cylinder_faces: number
    n_cone_faces: number
  } | null
  trimBoundaryData: any[] | null
  runConstructFaces: () => Promise<void>
}

function extractFilename(transferPath: string): string {
  // Extract filename from full path like "C:\\Users\\...\\abc123.bin"
  return transferPath.replace(/\\/g, '/').split('/').pop() || ''
}

export const usePipelineStore = create<PipelineState>((set) => ({
  stage: 'idle',
  loading: false,
  progress: null,
  error: null,
  meshTransferUrl: null,
  meshInfo: null,

  intentSummary: null,
  intentRegions: null,
  intentBoundaries: null,
  intentOverlay: null,
  showIntentRegionColors: false,
  showIntentGizmos: false,
  intentColorMode: 'region',

  deviationResult: null,
  showDeviationHeatmap: false,

  reconstructionEvents: [],
  reconstructionRunning: false,
  reconstructionProgress: null,

  classifyResult: null,

  snapResult: null,
  snappedMeshTransferUrl: null,
  showSnappedMesh: false,

  stepResult: null,

  trimResult: null,
  trimBoundaryData: null,

  setProgress: (p) => set({ progress: p }),
  setError: (e) => set({ error: e }),

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
        progress: { stage: 'cleanup', pct: 100, message: `${result.vertices.toLocaleString()} verts, ${result.is_watertight ? 'watertight' : 'NOT watertight'}` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
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
  setIntentColorMode: (mode) => set({ intentColorMode: mode }),

  // ───────────────────────── Deviation Analysis ─────────────────────────
  runDeviationAnalysis: async () => {
    set({ loading: true, error: null, progress: { stage: 'deviation', pct: 30, message: 'Computing deviation...' } })
    try {
        const res = await fetch(`${API}/intent/deviation`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' })
        if (!res.ok) throw new Error(await res.text())
        const result = await res.json()
        set({
            deviationResult: {
                mean_deviation: result.mean_deviation ?? 0,
                max_deviation: result.max_deviation ?? 0,
                p95_deviation: result.p95_deviation ?? 0,
                pct_snapped: result.pct_snapped ?? 0,
                face_colors_b64: result.face_colors_b64 ?? '',
                n_faces: result.n_faces ?? 0,
                color_scale_max: result.color_scale_max ?? 1,
                tolerance_bands: result.tolerance_bands ?? {},
            },
            showDeviationHeatmap: true,
            progress: { stage: 'deviation', pct: 100, message: `Mean: ${(result.mean_deviation ?? 0).toFixed(3)}` },
        })
    } catch (e: any) { set({ error: e.message }) }
    finally { set({ loading: false }); setTimeout(() => set({ progress: null }), 2500) }
  },

  toggleDeviationHeatmap: () => set((s) => ({ showDeviationHeatmap: !s.showDeviationHeatmap })),

  // ───────────────────────── Live Reconstruction ─────────────────────────
  runLiveReconstruction: (apiKey?: string) => {
    const url = new URL(`${API}/intent/reconstruct_live`)
    if (apiKey) url.searchParams.set('api_key', apiKey)
    url.searchParams.set('use_ai', 'true')

    set({ reconstructionRunning: true, reconstructionEvents: [], error: null,
          progress: { stage: 'reconstruct', pct: 0, message: 'Starting live reconstruction...' } })

    const source = new EventSource(url.toString())
    // Store ref for stopping
    ;(window as any).__reconstructionSource = source

    source.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        if (data.status === 'complete') {
          set((s) => ({
            reconstructionRunning: false,
            reconstructionProgress: null,
            progress: { stage: 'reconstruct', pct: 100, message: `Done! ${data.total_built} surfaces built` },
          }))
          source.close()
          setTimeout(() => set({ progress: null }), 3000)
          return
        }
        set((s) => ({
          reconstructionEvents: [...s.reconstructionEvents, data],
          reconstructionProgress: { step: data.step, total: data.total },
          progress: {
            stage: 'reconstruct',
            pct: Math.round((data.step / data.total) * 100),
            message: `R${data.region_id}: ${data.classification} (${data.status})`,
          },
        }))
      } catch (e) { /* ignore parse errors */ }
    }

    source.onerror = () => {
      set({ reconstructionRunning: false, error: 'Reconstruction stream ended' })
      source.close()
      setTimeout(() => set({ progress: null }), 2000)
    }
  },

  stopReconstruction: () => {
    const source = (window as any).__reconstructionSource
    if (source) { source.close(); delete (window as any).__reconstructionSource }
    set({ reconstructionRunning: false, progress: null })
  },

  // ───────────────────────── AI Classification ─────────────────────────
  runAIClassify: async (apiKey?: string) => {
    set({ loading: true, error: null, progress: { stage: 'classify', pct: 20, message: 'Classifying regions with AI…' } })
    try {
      const body: any = {}
      if (apiKey) body.api_key = apiKey
      const res = await fetch(`${API}/intent/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        classifyResult: {
          n_classified: result.n_classified ?? 0,
          n_changed: result.n_changed ?? 0,
          type_counts: result.type_counts ?? {},
          classifications: result.classifications ?? [],
        },
        progress: { stage: 'classify', pct: 100, message: `${result.n_classified} regions classified (${result.n_changed} changed)` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  // ───────────────────────── Phase E1 — Snap ─────────────────────────
  runSnapToSurfaces: async () => {
    set({ loading: true, error: null, progress: { stage: 'snap', pct: 10, message: 'Snapping to surfaces…' } })
    try {
      const res = await fetch(`${API}/intent/snap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const filename = extractFilename(result.transfer_path)
      set({
        snapResult: {
          n_snapped: result.n_snapped ?? 0,
          n_edge: result.n_edge ?? 0,
          n_corner: result.n_corner ?? 0,
          pct_snapped: result.pct_snapped ?? 0,
          mean_displacement: result.mean_displacement ?? 0,
        },
        snappedMeshTransferUrl: `${API}/transfer/${filename}`,
        showSnappedMesh: true,
        meshTransferUrl: `${API}/transfer/${filename}`,
        progress: { stage: 'snap', pct: 100, message: `${(result.pct_snapped ?? 0).toFixed(1)}% vertices snapped` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  toggleSnappedMesh: () => set((s) => {
    if (!s.snappedMeshTransferUrl) return {}
    const showSnapped = !s.showSnappedMesh
    return {
      showSnappedMesh: showSnapped,
      meshTransferUrl: showSnapped ? s.snappedMeshTransferUrl : s.meshTransferUrl,
    }
  }),

  // ───────────────────────── Phase E3 — STEP Export ─────────────────────────
  runExportStep: async () => {
    set({ loading: true, error: null, progress: { stage: 'step', pct: 10, message: 'Building B-Rep + exporting STEP…' } })
    try {
      const res = await fetch(`${API}/intent/export_step`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      set({
        stepResult: {
          n_faces_built: result.n_faces_built ?? 0,
          n_faces_failed: result.n_faces_failed ?? 0,
          step_size: result.step_size ?? 0,
        },
        progress: { stage: 'step', pct: 100, message: `STEP exported (${result.n_faces_built} faces)` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },

  downloadStep: () => {
    window.open(`${API}/intent/download_step`, '_blank')
  },

  // ───────────────────────── Phase E2 — Trim ─────────────────────────
  runConstructFaces: async () => {
    set({ loading: true, error: null, progress: { stage: 'trim', pct: 10, message: 'Constructing trimmed faces…' } })
    try {
      const res = await fetch(`${API}/intent/trim`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) throw new Error(await res.text())
      const result = await res.json()
      const byType = result.by_surface_type ?? {}
      set({
        trimResult: {
          n_trimmed_faces: result.n_trimmed_faces ?? 0,
          n_with_holes: result.n_with_holes ?? 0,
          total_boundary_vertices: result.total_boundary_vertices ?? 0,
          n_plane_faces: byType.plane ?? 0,
          n_cylinder_faces: byType.cylinder ?? 0,
          n_cone_faces: byType.cone ?? 0,
        },
        trimBoundaryData: result.faces ?? null,
        progress: { stage: 'trim', pct: 100, message: `${result.n_trimmed_faces ?? 0} trimmed faces` },
      })
    } catch (e: any) {
      set({ error: e.message })
    } finally {
      set({ loading: false })
      setTimeout(() => set({ progress: null }), 2500)
    }
  },
}))
