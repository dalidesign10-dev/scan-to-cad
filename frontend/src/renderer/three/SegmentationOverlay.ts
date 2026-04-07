import * as THREE from 'three'

// Categorical color palette (20 distinct colors)
const PALETTE = [
  0x4ecca3, 0xe94560, 0xf5a623, 0x7b68ee, 0x00bcd4,
  0xff6b6b, 0x48dbfb, 0xfeca57, 0xff9ff3, 0x54a0ff,
  0x5f27cd, 0x01a3a4, 0xf368e0, 0xff6348, 0x2ed573,
  0x1e90ff, 0xffa502, 0x7bed9f, 0x70a1ff, 0xeccc68,
]

// Special semantic color for fillets
const FILLET_COLOR = 0xf368e0

interface PatchClassification {
  id: number
  classification: string
  is_fillet?: boolean
}

export interface OverlayOptions {
  patches?: PatchClassification[]
  mergeMap?: Record<number, number>  // patch_id -> group_id
  selectedPatchIds?: number[]
}

/**
 * Build a non-indexed geometry from `mesh` (once) and store the labels +
 * source positions/normals on userData for cheap recoloring later.
 */
export function applySegmentationOverlay(
  mesh: THREE.Mesh,
  labelsBuffer: ArrayBuffer | Int32Array,
  options: OverlayOptions = {},
) {
  const labels = labelsBuffer instanceof Int32Array
    ? labelsBuffer
    : new Int32Array(labelsBuffer)

  const geometry = mesh.geometry as THREE.BufferGeometry
  // If we've already built the non-indexed overlay geometry once, reuse it.
  const alreadyOverlay = geometry.userData?.isSegmentationOverlay === true

  let nFaces: number
  let newPositions: Float32Array
  let newNormals: Float32Array
  let target: THREE.BufferGeometry

  if (alreadyOverlay) {
    target = geometry
    newPositions = target.attributes.position.array as Float32Array
    newNormals = target.attributes.normal.array as Float32Array
    nFaces = newPositions.length / 9
  } else {
    if (!geometry.index) {
      console.error('Geometry must be indexed for first segmentation overlay')
      return
    }
    const indices = geometry.index.array
    nFaces = indices.length / 3
    const positions = geometry.attributes.position.array as Float32Array
    const normals = geometry.attributes.normal?.array as Float32Array | undefined

    newPositions = new Float32Array(nFaces * 9)
    newNormals = new Float32Array(nFaces * 9)

    for (let fi = 0; fi < nFaces; fi++) {
      for (let vi = 0; vi < 3; vi++) {
        const srcIdx = indices[fi * 3 + vi]
        const dstIdx = fi * 3 + vi
        newPositions[dstIdx * 3 + 0] = positions[srcIdx * 3 + 0]
        newPositions[dstIdx * 3 + 1] = positions[srcIdx * 3 + 1]
        newPositions[dstIdx * 3 + 2] = positions[srcIdx * 3 + 2]
        if (normals) {
          newNormals[dstIdx * 3 + 0] = normals[srcIdx * 3 + 0]
          newNormals[dstIdx * 3 + 1] = normals[srcIdx * 3 + 1]
          newNormals[dstIdx * 3 + 2] = normals[srcIdx * 3 + 2]
        }
      }
    }

    target = new THREE.BufferGeometry()
    target.setAttribute('position', new THREE.BufferAttribute(newPositions, 3))
    target.setAttribute('normal', new THREE.BufferAttribute(newNormals, 3))
    target.setAttribute('color', new THREE.BufferAttribute(new Float32Array(nFaces * 9), 3))
    target.userData.isSegmentationOverlay = true

    mesh.geometry.dispose()
    mesh.geometry = target

    const material = mesh.material as THREE.MeshPhongMaterial
    material.vertexColors = true
    material.color.set(0xffffff)
    material.needsUpdate = true
  }

  // Cache labels for picking + recoloring
  target.userData.labels = labels

  recolorOverlay(mesh, options)
}

/**
 * Recolor an existing overlay mesh based on merge map + selection.
 * Cheap: only writes the color attribute.
 */
export function recolorOverlay(mesh: THREE.Mesh, options: OverlayOptions = {}) {
  const geometry = mesh.geometry as THREE.BufferGeometry
  if (!geometry.userData?.isSegmentationOverlay) return
  const labels: Int32Array | undefined = geometry.userData.labels
  if (!labels) return

  const colorAttr = geometry.attributes.color as THREE.BufferAttribute
  const colors = colorAttr.array as Float32Array
  const nFaces = colors.length / 9

  const filletIds = new Set<number>()
  if (options.patches) {
    for (const p of options.patches) {
      if (p.is_fillet || p.classification === 'fillet') filletIds.add(p.id)
    }
  }

  const mergeMap = options.mergeMap || {}
  const selected = new Set<number>(options.selectedPatchIds || [])
  const tmp = new THREE.Color()

  for (let fi = 0; fi < nFaces; fi++) {
    const label = fi < labels.length ? labels[fi] : 0
    // Resolve to merge group rep so merged patches share a color
    const groupId = mergeMap[label] ?? label
    const colorHex = filletIds.has(label)
      ? FILLET_COLOR
      : PALETTE[groupId % PALETTE.length]
    tmp.set(colorHex)

    // Highlight selected patches by lerping toward white
    if (selected.has(label)) {
      tmp.lerp(new THREE.Color(0xffffff), 0.55)
    }

    for (let vi = 0; vi < 3; vi++) {
      const dst = (fi * 3 + vi) * 3
      colors[dst + 0] = tmp.r
      colors[dst + 1] = tmp.g
      colors[dst + 2] = tmp.b
    }
  }
  colorAttr.needsUpdate = true
}

/**
 * Look up the patch id for a given face index (raycast hit).
 */
export function patchIdForFace(mesh: THREE.Mesh, faceIndex: number): number | null {
  const geometry = mesh.geometry as THREE.BufferGeometry
  const labels: Int32Array | undefined = geometry.userData?.labels
  if (!labels) return null
  if (faceIndex < 0 || faceIndex >= labels.length) return null
  return labels[faceIndex]
}
