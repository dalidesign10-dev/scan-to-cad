import * as THREE from 'three'

/**
 * Phase E0 — debug overlays for the intent reconstruction layer.
 *
 * Two visual layers:
 *   1. Per-face region tint applied to the loaded full-resolution mesh.
 *      Color is driven by primitive type + confidence class so the user can
 *      see at a glance which regions the system is confident about.
 *   2. Per-region gizmos (plane normals, cylinder axes) and sharp-edge
 *      segments rendered in a dedicated group.
 *
 * Important: this never replaces the segmentation overlay machinery. It
 * lives in its own group and on its own toggle so the existing patch
 * workflow keeps working.
 */

export interface IntentRegionInfo {
  id: number
  type: 'plane' | 'cylinder' | 'unknown'
  confidence_class: 'high' | 'medium' | 'low' | 'rejected'
  score: number
  rmse: number
  n_full_faces: number
  area_fraction: number
  gizmo: null | {
    kind: 'plane_normal' | 'cylinder_axis'
    origin: number[]
    direction: number[]
    radius?: number
    height?: number
  }
}

export interface IntentSharpEdges {
  starts: number[][]
  ends: number[][]
  confidence: number[]
  n: number
}

export interface IntentOverlayPayload {
  available: boolean
  n_full_faces: number
  full_face_region_b64: string | null
  regions: IntentRegionInfo[]
  sharp_edges: IntentSharpEdges | null
  summary: any
}

const COLOR_PLANE_HIGH = new THREE.Color(0x4ecca3)
const COLOR_PLANE_MED = new THREE.Color(0x2a8a6e)
const COLOR_PLANE_LOW = new THREE.Color(0x1a5544)

const COLOR_CYL_HIGH = new THREE.Color(0xe94560)
const COLOR_CYL_MED = new THREE.Color(0xa53344)
const COLOR_CYL_LOW = new THREE.Color(0x66222a)

const COLOR_UNKNOWN = new THREE.Color(0x666677)
const COLOR_REJECTED = new THREE.Color(0x333344)

function colorForRegion(r: IntentRegionInfo): THREE.Color {
  if (r.type === 'plane') {
    if (r.confidence_class === 'high') return COLOR_PLANE_HIGH
    if (r.confidence_class === 'medium') return COLOR_PLANE_MED
    return COLOR_PLANE_LOW
  }
  if (r.type === 'cylinder') {
    if (r.confidence_class === 'high') return COLOR_CYL_HIGH
    if (r.confidence_class === 'medium') return COLOR_CYL_MED
    return COLOR_CYL_LOW
  }
  if (r.confidence_class === 'rejected') return COLOR_REJECTED
  return COLOR_UNKNOWN
}

function decodeInt32B64(b64: string): Int32Array {
  const bin = atob(b64)
  const len = bin.length
  const buf = new Uint8Array(len)
  for (let i = 0; i < len; i++) buf[i] = bin.charCodeAt(i)
  return new Int32Array(buf.buffer)
}

/**
 * Recolor the segmentation overlay mesh in-place using intent regions.
 *
 * Reuses the same non-indexed BufferGeometry the SegmentationOverlay set
 * up. Required: the overlay has already been built (mesh.geometry has
 * userData.isSegmentationOverlay).
 */
export function applyIntentRegionColors(
  mesh: THREE.Mesh,
  payload: IntentOverlayPayload,
) {
  const geometry = mesh.geometry as THREE.BufferGeometry
  if (!geometry.userData?.isSegmentationOverlay) {
    console.warn('IntentOverlay: mesh has no segmentation overlay yet')
    return
  }
  const colorAttr = geometry.attributes.color as THREE.BufferAttribute
  if (!colorAttr) return
  const colors = colorAttr.array as Float32Array
  const nFaces = colors.length / 9

  if (!payload.full_face_region_b64) return
  const faceRegion = decodeInt32B64(payload.full_face_region_b64)
  if (faceRegion.length !== nFaces) {
    console.warn('IntentOverlay: face region length mismatch', faceRegion.length, '!=', nFaces)
    return
  }
  const regionById = new Map<number, IntentRegionInfo>()
  for (const r of payload.regions) regionById.set(r.id, r)

  for (let fi = 0; fi < nFaces; fi++) {
    const rid = faceRegion[fi]
    const r = regionById.get(rid)
    const c = r ? colorForRegion(r) : COLOR_UNKNOWN
    for (let vi = 0; vi < 3; vi++) {
      const dst = (fi * 3 + vi) * 3
      colors[dst + 0] = c.r
      colors[dst + 1] = c.g
      colors[dst + 2] = c.b
    }
  }
  colorAttr.needsUpdate = true
}

export function clearIntentGizmos(group: THREE.Group) {
  while (group.children.length > 0) {
    const child = group.children[0]
    if ((child as any).geometry) (child as any).geometry.dispose?.()
    if ((child as any).material) {
      const m = (child as any).material
      if (Array.isArray(m)) m.forEach((x) => x.dispose?.())
      else m.dispose?.()
    }
    group.remove(child)
  }
}

/**
 * Render gizmos: plane normals as short lines + dots, cylinder axes as
 * long lines passing through the origin. Confidence class controls color
 * saturation; only high/medium are drawn (low fits would just be noise).
 */
export function renderIntentGizmos(
  group: THREE.Group,
  payload: IntentOverlayPayload,
  sceneScale: number,
) {
  clearIntentGizmos(group)
  const planeLen = sceneScale * 0.06
  const cylLen = sceneScale * 0.45

  for (const r of payload.regions) {
    if (!r.gizmo) continue
    if (r.confidence_class !== 'high' && r.confidence_class !== 'medium') continue
    const o = new THREE.Vector3(r.gizmo.origin[0], r.gizmo.origin[1], r.gizmo.origin[2])
    const d = new THREE.Vector3(r.gizmo.direction[0], r.gizmo.direction[1], r.gizmo.direction[2]).normalize()

    if (r.gizmo.kind === 'plane_normal') {
      const end = o.clone().addScaledVector(d, planeLen)
      const geom = new THREE.BufferGeometry().setFromPoints([o, end])
      const mat = new THREE.LineBasicMaterial({
        color: colorForRegion(r),
        depthTest: true,
      })
      group.add(new THREE.Line(geom, mat))
    } else if (r.gizmo.kind === 'cylinder_axis') {
      const a = o.clone().addScaledVector(d, -cylLen * 0.5)
      const b = o.clone().addScaledVector(d, cylLen * 0.5)
      const geom = new THREE.BufferGeometry().setFromPoints([a, b])
      const mat = new THREE.LineBasicMaterial({
        color: colorForRegion(r),
        depthTest: true,
      })
      group.add(new THREE.Line(geom, mat))
    }
  }
}

/**
 * Render the sharp boundary edges as short LineSegments. Confidence is
 * encoded in alpha so the eye can distinguish marginal from decisive ones.
 */
export function renderIntentSharpEdges(
  group: THREE.Group,
  payload: IntentOverlayPayload,
) {
  // We append the sharp-edge LineSegments under the same group as the
  // gizmos so a single visibility toggle covers both.
  if (!payload.sharp_edges || payload.sharp_edges.n === 0) return
  const { starts, ends, confidence } = payload.sharp_edges
  const n = starts.length
  const positions = new Float32Array(n * 6)
  const colors = new Float32Array(n * 6)
  for (let i = 0; i < n; i++) {
    positions[i * 6 + 0] = starts[i][0]
    positions[i * 6 + 1] = starts[i][1]
    positions[i * 6 + 2] = starts[i][2]
    positions[i * 6 + 3] = ends[i][0]
    positions[i * 6 + 4] = ends[i][1]
    positions[i * 6 + 5] = ends[i][2]
    const c = Math.min(1.0, Math.max(0.0, confidence[i]))
    // monochrome amber, brightness = confidence
    const r = 1.0
    const g = 0.65 * c + 0.35
    const b = 0.1 * c
    colors[i * 6 + 0] = r * c + (1 - c) * 0.4
    colors[i * 6 + 1] = g * c + (1 - c) * 0.2
    colors[i * 6 + 2] = b
    colors[i * 6 + 3] = colors[i * 6 + 0]
    colors[i * 6 + 4] = colors[i * 6 + 1]
    colors[i * 6 + 5] = colors[i * 6 + 2]
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  const mat = new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false })
  const lines = new THREE.LineSegments(geom, mat)
  lines.renderOrder = 998
  group.add(lines)
}
