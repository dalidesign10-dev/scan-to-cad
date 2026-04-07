import * as THREE from 'three'

export function clearPolyhedralCad(group: THREE.Group) {
  while (group.children.length > 0) {
    const child = group.children[0]
    group.remove(child)
    if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose()
    const mat = (child as THREE.Mesh).material as THREE.Material | THREE.Material[]
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
    else if (mat) mat.dispose()
  }
}

export function renderPolyhedralCad(
  group: THREE.Group,
  data: { vertices: number[][]; faces: number[][] }
) {
  clearPolyhedralCad(group)
  if (!data || !data.vertices || !data.faces) return
  if (data.vertices.length < 3 || data.faces.length < 1) return

  const positions = new Float32Array(data.vertices.length * 3)
  for (let i = 0; i < data.vertices.length; i++) {
    positions[i * 3] = data.vertices[i][0]
    positions[i * 3 + 1] = data.vertices[i][1]
    positions[i * 3 + 2] = data.vertices[i][2]
  }
  const indices = new Uint32Array(data.faces.length * 3)
  for (let i = 0; i < data.faces.length; i++) {
    indices[i * 3] = data.faces[i][0]
    indices[i * 3 + 1] = data.faces[i][1]
    indices[i * 3 + 2] = data.faces[i][2]
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setIndex(new THREE.BufferAttribute(indices, 1))
  geometry.computeVertexNormals()

  const material = new THREE.MeshStandardMaterial({
    color: 0xb8c5d6,
    metalness: 0.2,
    roughness: 0.4,
    flatShading: true,
    side: THREE.DoubleSide,
  })
  const mesh = new THREE.Mesh(geometry, material)
  group.add(mesh)

  // Sharp edge overlay
  const edgeGeo = new THREE.EdgesGeometry(geometry, 15)
  const edgeMat = new THREE.LineBasicMaterial({ color: 0x222831, linewidth: 1 })
  const edges = new THREE.LineSegments(edgeGeo, edgeMat)
  group.add(edges)
}
