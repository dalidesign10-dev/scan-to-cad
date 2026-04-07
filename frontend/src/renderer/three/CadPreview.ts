import * as THREE from 'three'

export function clearCadPreview(group: THREE.Group) {
  while (group.children.length > 0) {
    const child = group.children[0]
    group.remove(child)
    if ((child as THREE.Mesh).geometry) (child as THREE.Mesh).geometry.dispose()
    const mat = (child as THREE.Mesh).material as THREE.Material | THREE.Material[]
    if (Array.isArray(mat)) mat.forEach((m) => m.dispose())
    else if (mat) mat.dispose()
  }
}

export function renderCadPreview(group: THREE.Group, surfaces: any[]) {
  clearCadPreview(group)
  for (const surf of surfaces) {
    const verts = surf.vertices as number[][]
    const faces = surf.faces as number[][]
    if (!verts || !faces || verts.length < 3 || faces.length < 1) continue

    const positions = new Float32Array(verts.length * 3)
    for (let i = 0; i < verts.length; i++) {
      positions[i * 3] = verts[i][0]
      positions[i * 3 + 1] = verts[i][1]
      positions[i * 3 + 2] = verts[i][2]
    }
    const indices = new Uint32Array(faces.length * 3)
    for (let i = 0; i < faces.length; i++) {
      indices[i * 3] = faces[i][0]
      indices[i * 3 + 1] = faces[i][1]
      indices[i * 3 + 2] = faces[i][2]
    }

    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geometry.setIndex(new THREE.BufferAttribute(indices, 1))
    geometry.computeVertexNormals()

    const c = surf.color as number[]
    const color = new THREE.Color(c[0], c[1], c[2])
    const material = new THREE.MeshStandardMaterial({
      color,
      metalness: 0.1,
      roughness: 0.55,
      flatShading: false,
      side: THREE.DoubleSide,
    })
    const mesh = new THREE.Mesh(geometry, material)
    mesh.userData.patchId = surf.patch_id
    group.add(mesh)
  }
}
