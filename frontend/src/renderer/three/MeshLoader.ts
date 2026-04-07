import * as THREE from 'three'

/**
 * Load mesh from our custom binary format:
 * Header: 3 x uint32 (nVerts, nFaces, hasNormals)
 * Vertices: float32 * nVerts * 3
 * Normals: float32 * nVerts * 3 (if hasNormals)
 * Faces: uint32 * nFaces * 3
 */
export function loadMeshFromBinaryFile(buffer: ArrayBuffer): THREE.BufferGeometry {
  const view = new DataView(buffer)
  let offset = 0

  const nVerts = view.getUint32(offset, true); offset += 4
  const nFaces = view.getUint32(offset, true); offset += 4
  const hasNormals = view.getUint32(offset, true); offset += 4

  // Read vertices
  const vertices = new Float32Array(buffer, offset, nVerts * 3)
  offset += nVerts * 3 * 4

  // Read normals
  let normals: Float32Array | null = null
  if (hasNormals) {
    normals = new Float32Array(buffer, offset, nVerts * 3)
    offset += nVerts * 3 * 4
  }

  // Read faces
  const indices = new Uint32Array(buffer, offset, nFaces * 3)

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(vertices), 3))
  if (normals) {
    geometry.setAttribute('normal', new THREE.BufferAttribute(new Float32Array(normals), 3))
  } else {
    geometry.computeVertexNormals()
  }
  geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(indices), 1))

  return geometry
}
