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

export interface IntentGizmo {
  kind: 'plane_normal' | 'cylinder_axis' | 'cone_axis'
  origin: number[]
  direction: number[]
  radius?: number
  height?: number
  half_angle_deg?: number
}

export interface IntentRegionInfo {
  id: number
  type: 'plane' | 'cylinder' | 'cone' | 'unknown'
  confidence_class: 'high' | 'medium' | 'low' | 'rejected'
  score: number
  rmse: number
  n_full_faces: number
  area_fraction: number
  // -1 when the pipeline didn't assign a family (non-HIGH or rejected).
  surface_family_id: number
  gizmo: null | IntentGizmo
}

export interface IntentSurfaceFamilyInfo {
  id: number
  type: 'plane' | 'cylinder' | 'cone' | 'unknown'
  region_ids: number[]
  representative_region_id: number
  total_area_fraction: number
  n_members: number
  gizmo: null | IntentGizmo
}

export interface IntentSharpEdges {
  starts: number[][]
  ends: number[][]
  confidence: number[]
  n: number
}

/**
 * Analytic intersection between two SurfaceFamilies — currently only
 * plane/plane (a line segment clipped to the mesh AABB), but the
 * shape is forward-compatible with sampled curves (polyline) so
 * plane/cylinder etc. can land without schema churn.
 */
export interface IntentFamilyEdge {
  family_a: number
  family_b: number
  type_a: 'plane' | 'cylinder' | 'cone' | 'unknown'
  type_b: 'plane' | 'cylinder' | 'cone' | 'unknown'
  kind: string  // e.g. "plane_plane"
  points: number[][]  // polyline, length ≥ 2
  n_points: number
  n_supporting_boundaries: number
}

export interface IntentOverlayPayload {
  available: boolean
  n_full_faces: number
  full_face_region_b64: string | null
  regions: IntentRegionInfo[]
  surface_families?: IntentSurfaceFamilyInfo[]
  family_edges?: IntentFamilyEdge[]
  sharp_edges: IntentSharpEdges | null
  summary: any
}

export type IntentColorMode = 'region' | 'family'

const COLOR_PLANE_HIGH = new THREE.Color(0x4ecca3)
const COLOR_PLANE_MED = new THREE.Color(0x2a8a6e)
const COLOR_PLANE_LOW = new THREE.Color(0x1a5544)

const COLOR_CYL_HIGH = new THREE.Color(0xe94560)
const COLOR_CYL_MED = new THREE.Color(0xa53344)
const COLOR_CYL_LOW = new THREE.Color(0x66222a)

const COLOR_CONE_HIGH = new THREE.Color(0xf39c12)
const COLOR_CONE_MED = new THREE.Color(0xa8690a)
const COLOR_CONE_LOW = new THREE.Color(0x553306)

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
  if (r.type === 'cone') {
    if (r.confidence_class === 'high') return COLOR_CONE_HIGH
    if (r.confidence_class === 'medium') return COLOR_CONE_MED
    return COLOR_CONE_LOW
  }
  if (r.confidence_class === 'rejected') return COLOR_REJECTED
  return COLOR_UNKNOWN
}

/**
 * Deterministic colour for a surface family id. The golden-ratio hue
 * stride gives well-separated colours for adjacent ids, and the family
 * TYPE controls saturation/lightness so plane families stay greenish,
 * cylinder families reddish, and cone families amber — the eye still
 * reads type first, family id second.
 */
const _familyColorCache = new Map<string, THREE.Color>()
function colorForFamily(
  familyId: number,
  type: 'plane' | 'cylinder' | 'cone' | 'unknown',
): THREE.Color {
  const key = `${type}:${familyId}`
  const hit = _familyColorCache.get(key)
  if (hit) return hit
  const GOLDEN = 0.61803398875
  // Hue anchor per type so each family type lives in its own arc of
  // colour wheel. Plane ≈ green, cylinder ≈ red, cone ≈ amber.
  let baseHue = 0.0
  let sat = 0.55
  let light = 0.55
  if (type === 'plane') { baseHue = 0.42; sat = 0.55; light = 0.55 }
  else if (type === 'cylinder') { baseHue = 0.97; sat = 0.60; light = 0.55 }
  else if (type === 'cone') { baseHue = 0.09; sat = 0.65; light = 0.55 }
  else { baseHue = 0.7; sat = 0.20; light = 0.40 }
  // Spread ids across a narrow arc centred on baseHue so the type is
  // still recognisable at a glance.
  const arc = 0.08
  const offset = ((familyId * GOLDEN) % 1) * 2 - 1  // [-1, 1]
  const hue = (baseHue + offset * arc + 1) % 1
  const col = new THREE.Color().setHSL(hue, sat, light)
  _familyColorCache.set(key, col)
  return col
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
 *
 * In 'region' mode (default) each face is tinted by its region's type +
 * confidence. In 'family' mode faces are tinted by the surface family
 * they belong to — two physically separate pads on the same plane get
 * the same colour, so coplanarity is visible at a glance. Regions that
 * have no family (family_id < 0, i.e. non-HIGH) fall back to the
 * region-colour path so the user still sees MED/LOW fits.
 */
export function applyIntentRegionColors(
  mesh: THREE.Mesh,
  payload: IntentOverlayPayload,
  mode: IntentColorMode = 'region',
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
    let c: THREE.Color
    if (!r) {
      c = COLOR_UNKNOWN
    } else if (mode === 'family' && r.surface_family_id >= 0) {
      c = colorForFamily(r.surface_family_id, r.type)
    } else {
      c = colorForRegion(r)
    }
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
 * Internal: draw one gizmo line. Shared by the region and family paths
 * so the geometry choice (plane short line, cylinder axis, cone apex
 * cone) stays consistent between modes.
 */
function _drawGizmoLine(
  group: THREE.Group,
  gz: IntentGizmo,
  color: THREE.Color,
  sceneScale: number,
) {
  const planeLen = sceneScale * 0.06
  const cylLen = sceneScale * 0.45
  const o = new THREE.Vector3(gz.origin[0], gz.origin[1], gz.origin[2])
  const d = new THREE.Vector3(gz.direction[0], gz.direction[1], gz.direction[2]).normalize()
  const mat = new THREE.LineBasicMaterial({ color, depthTest: true })

  if (gz.kind === 'plane_normal') {
    const end = o.clone().addScaledVector(d, planeLen)
    const geom = new THREE.BufferGeometry().setFromPoints([o, end])
    group.add(new THREE.Line(geom, mat))
  } else if (gz.kind === 'cylinder_axis') {
    const a = o.clone().addScaledVector(d, -cylLen * 0.5)
    const b = o.clone().addScaledVector(d, cylLen * 0.5)
    const geom = new THREE.BufferGeometry().setFromPoints([a, b])
    group.add(new THREE.Line(geom, mat))
  } else if (gz.kind === 'cone_axis') {
    // Cone axis: origin is the apex (not a centerpoint). Draw from
    // the apex outward along +direction so the line lives inside
    // the cone rather than poking through the apex into empty space.
    // Length is scaled by half-angle so wide cones get longer lines
    // (they span more volume) and narrow cones stay short.
    const halfDeg = gz.half_angle_deg ?? 30
    const lenScale = 0.6 + 0.4 * Math.min(1, halfDeg / 45)
    const tip = o.clone()
    const base = o.clone().addScaledVector(d, cylLen * lenScale)
    const geom = new THREE.BufferGeometry().setFromPoints([tip, base])
    group.add(new THREE.Line(geom, mat))
  }
}

/**
 * Render gizmos: plane normals as short lines + dots, cylinder axes as
 * long lines passing through the origin. Confidence class controls color
 * saturation; only high/medium are drawn (low fits would just be noise).
 *
 * In 'family' mode one gizmo is drawn per SurfaceFamily (using the
 * family's canonical params) instead of one per region, so 20 parallel
 * planes collapse to a single arrow instead of stacking on top of each
 * other. Regions whose fit didn't land in a family (non-HIGH) still get
 * their own gizmo so MED fits remain visible.
 */
export function renderIntentGizmos(
  group: THREE.Group,
  payload: IntentOverlayPayload,
  sceneScale: number,
  mode: IntentColorMode = 'region',
) {
  clearIntentGizmos(group)

  if (mode === 'family' && payload.surface_families && payload.surface_families.length > 0) {
    // One gizmo per family, using canonical params.
    for (const fam of payload.surface_families) {
      if (!fam.gizmo) continue
      const color = colorForFamily(fam.id, fam.type)
      _drawGizmoLine(group, fam.gizmo, color, sceneScale)
    }
    // Also show MED region gizmos that have no family — otherwise the
    // user would lose sight of the "second-tier" fits entirely.
    for (const r of payload.regions) {
      if (!r.gizmo) continue
      if (r.surface_family_id >= 0) continue
      if (r.confidence_class !== 'medium') continue
      _drawGizmoLine(group, r.gizmo, colorForRegion(r), sceneScale)
    }
    return
  }

  for (const r of payload.regions) {
    if (!r.gizmo) continue
    if (r.confidence_class !== 'high' && r.confidence_class !== 'medium') continue
    _drawGizmoLine(group, r.gizmo, colorForRegion(r), sceneScale)
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

/**
 * Render family-level analytic intersection edges (plane/plane lines
 * clipped to the mesh AABB). These are the "ideal" B-Rep edges implied
 * by the current family set — they do not depend on per-region sharp
 * detection, so they stay clean even on noisy scans.
 *
 * Cyan so they're distinguishable from the amber sharp-edge overlay
 * and the region-level gizmo lines.
 */
const COLOR_FAMILY_EDGE = new THREE.Color(0x48dbfb)
export function renderIntentFamilyEdges(
  group: THREE.Group,
  payload: IntentOverlayPayload,
) {
  const fe = payload.family_edges
  if (!fe || fe.length === 0) return
  // All family edges are drawn in one batched LineSegments. Each
  // polyline of N points contributes (N-1) segments, so we first
  // count segments to size the buffer.
  let nSeg = 0
  for (const e of fe) if (e.n_points >= 2) nSeg += (e.n_points - 1)
  if (nSeg === 0) return
  const positions = new Float32Array(nSeg * 6)
  let write = 0
  for (const e of fe) {
    if (e.n_points < 2) continue
    for (let i = 0; i < e.n_points - 1; i++) {
      const a = e.points[i]
      const b = e.points[i + 1]
      positions[write++] = a[0]
      positions[write++] = a[1]
      positions[write++] = a[2]
      positions[write++] = b[0]
      positions[write++] = b[1]
      positions[write++] = b[2]
    }
  }
  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  const mat = new THREE.LineBasicMaterial({
    color: COLOR_FAMILY_EDGE,
    transparent: true,
    opacity: 0.9,
    depthTest: false,
  })
  const lines = new THREE.LineSegments(geom, mat)
  // Slightly above the sharp-edge overlay so the clean analytic
  // lines aren't hidden by noisy sharp-edge clusters.
  lines.renderOrder = 999
  group.add(lines)
}
