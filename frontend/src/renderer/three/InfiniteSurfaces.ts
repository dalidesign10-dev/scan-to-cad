import * as THREE from 'three'

export interface InfiniteSurface {
  type: 'infinite_plane' | 'infinite_cylinder' | 'infinite_sphere' | 'trimmed_plane'
  patch_id?: number
  patch_ids?: number[]
  face_count?: number
  // plane
  normal?: number[]
  d?: number
  point?: number[]
  u_axis?: number[]
  v_axis?: number[]
  corners?: number[][]
  extent?: number
  // cylinder
  axis?: number[]
  center?: number[]
  radius?: number
  length?: number
  // trimmed_plane
  vertices?: number[][]
  triangles?: number[][]
  boundary_indices?: number[]
}

const TYPE_COLORS: Record<string, number> = {
  infinite_plane: 0x4ecca3,
  infinite_cylinder: 0xe94560,
  infinite_sphere: 0xf5a623,
  trimmed_plane: 0xfeca57,
}

export function clearInfiniteSurfaces(group: THREE.Group) {
  while (group.children.length > 0) {
    const child = group.children[0]
    if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
      child.geometry.dispose()
      if (child.material instanceof THREE.Material) child.material.dispose()
    }
    group.remove(child)
  }
}

export function renderInfiniteSurfaces(group: THREE.Group, surfaces: InfiniteSurface[]) {
  for (const surf of surfaces) {
    const color = TYPE_COLORS[surf.type] || 0x888888
    if (surf.type === 'infinite_plane') {
      renderInfinitePlane(group, surf, color)
    } else if (surf.type === 'infinite_cylinder') {
      renderInfiniteCylinder(group, surf, color)
    } else if (surf.type === 'infinite_sphere') {
      renderInfiniteSphere(group, surf, color)
    } else if (surf.type === 'trimmed_plane') {
      renderTrimmedPlane(group, surf, color)
    }
  }
}

function renderTrimmedPlane(group: THREE.Group, surf: InfiniteSurface, color: number) {
  if (!surf.vertices || !surf.triangles) return
  const verts = surf.vertices
  const tris = surf.triangles

  const positions = new Float32Array(tris.length * 9)
  for (let i = 0; i < tris.length; i++) {
    const [a, b, c] = tris[i]
    positions[i * 9 + 0] = verts[a][0]
    positions[i * 9 + 1] = verts[a][1]
    positions[i * 9 + 2] = verts[a][2]
    positions[i * 9 + 3] = verts[b][0]
    positions[i * 9 + 4] = verts[b][1]
    positions[i * 9 + 5] = verts[b][2]
    positions[i * 9 + 6] = verts[c][0]
    positions[i * 9 + 7] = verts[c][1]
    positions[i * 9 + 8] = verts[c][2]
  }

  const geom = new THREE.BufferGeometry()
  geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geom.computeVertexNormals()

  const mat = new THREE.MeshPhongMaterial({
    color,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
    depthWrite: false,
    shininess: 60,
  })
  const mesh = new THREE.Mesh(geom, mat)
  mesh.userData.patchIds = surf.patch_ids
  group.add(mesh)

  // Boundary outline (uses boundary_indices to draw the loop)
  if (surf.boundary_indices && surf.boundary_indices.length >= 2) {
    const loop = surf.boundary_indices
    const lineVerts: number[] = []
    for (let i = 0; i < loop.length; i++) {
      const a = verts[loop[i]]
      const b = verts[loop[(i + 1) % loop.length]]
      lineVerts.push(a[0], a[1], a[2], b[0], b[1], b[2])
    }
    const lineGeom = new THREE.BufferGeometry()
    lineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(lineVerts), 3))
    const lineMat = new THREE.LineBasicMaterial({ color, linewidth: 2 })
    const line = new THREE.LineSegments(lineGeom, lineMat)
    group.add(line)
  }
}

function renderInfinitePlane(group: THREE.Group, surf: InfiniteSurface, color: number) {
  if (!surf.corners || surf.corners.length !== 4) return

  const geom = new THREE.BufferGeometry()
  const verts = new Float32Array([
    ...surf.corners[0],
    ...surf.corners[1],
    ...surf.corners[2],
    ...surf.corners[0],
    ...surf.corners[2],
    ...surf.corners[3],
  ])
  geom.setAttribute('position', new THREE.BufferAttribute(verts, 3))
  geom.computeVertexNormals()

  // Translucent fill
  const fillMat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.12,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
  const mesh = new THREE.Mesh(geom, fillMat)
  mesh.userData.patchId = surf.patch_id
  group.add(mesh)

  // Edge outline
  const edges = new THREE.EdgesGeometry(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(...surf.corners[0]),
      new THREE.Vector3(...surf.corners[1]),
      new THREE.Vector3(...surf.corners[2]),
      new THREE.Vector3(...surf.corners[3]),
      new THREE.Vector3(...surf.corners[0]),
    ]),
  )
  // Use line loop instead
  const lineGeom = new THREE.BufferGeometry()
  const loop = [
    ...surf.corners[0], ...surf.corners[1],
    ...surf.corners[1], ...surf.corners[2],
    ...surf.corners[2], ...surf.corners[3],
    ...surf.corners[3], ...surf.corners[0],
  ]
  lineGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(loop), 3))
  const lineMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 })
  const line = new THREE.LineSegments(lineGeom, lineMat)
  line.userData.patchId = surf.patch_id
  group.add(line)
}

function renderInfiniteCylinder(group: THREE.Group, surf: InfiniteSurface, color: number) {
  if (!surf.axis || !surf.center || !surf.radius || !surf.length) return

  const geom = new THREE.CylinderGeometry(surf.radius, surf.radius, surf.length, 32, 1, true)
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.15,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
  const mesh = new THREE.Mesh(geom, mat)
  mesh.position.set(surf.center[0], surf.center[1], surf.center[2])

  const axisVec = new THREE.Vector3(surf.axis[0], surf.axis[1], surf.axis[2]).normalize()
  const up = new THREE.Vector3(0, 1, 0)
  const quat = new THREE.Quaternion().setFromUnitVectors(up, axisVec)
  mesh.quaternion.copy(quat)
  mesh.userData.patchId = surf.patch_id
  group.add(mesh)

  // Wire outline
  const wireGeo = new THREE.CylinderGeometry(surf.radius, surf.radius, surf.length, 16, 1, true)
  const edges = new THREE.EdgesGeometry(wireGeo)
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 }),
  )
  line.position.copy(mesh.position)
  line.quaternion.copy(mesh.quaternion)
  line.userData.patchId = surf.patch_id
  group.add(line)
}

function renderInfiniteSphere(group: THREE.Group, surf: InfiniteSurface, color: number) {
  if (!surf.center || !surf.radius) return
  const geom = new THREE.SphereGeometry(surf.radius, 24, 16)
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.15,
    side: THREE.DoubleSide,
    depthWrite: false,
  })
  const mesh = new THREE.Mesh(geom, mat)
  mesh.position.set(surf.center[0], surf.center[1], surf.center[2])
  group.add(mesh)

  const edges = new THREE.EdgesGeometry(new THREE.SphereGeometry(surf.radius, 12, 8))
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }),
  )
  line.position.copy(mesh.position)
  group.add(line)
}
