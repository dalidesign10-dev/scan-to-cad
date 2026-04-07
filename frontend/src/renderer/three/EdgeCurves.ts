import * as THREE from 'three'

export interface EdgeCurve {
  patch_a: number
  patch_b: number
  type_a: string
  type_b: string
  points: number[][]
  n_points: number
}

export function clearEdgeCurves(group: THREE.Group) {
  while (group.children.length > 0) {
    const child = group.children[0]
    if (child instanceof THREE.Line || child instanceof THREE.LineSegments) {
      child.geometry.dispose()
      if (child.material instanceof THREE.Material) child.material.dispose()
    }
    group.remove(child)
  }
}

export function renderEdgeCurves(group: THREE.Group, edges: EdgeCurve[]) {
  const mat = new THREE.LineBasicMaterial({
    color: 0xffff00, // bright yellow for edges
    linewidth: 3,
    transparent: true,
    opacity: 0.95,
    depthTest: false,
  })

  for (const edge of edges) {
    if (!edge.points || edge.points.length < 2) continue

    const flat: number[] = []
    for (const p of edge.points) {
      flat.push(p[0], p[1], p[2])
    }
    const geom = new THREE.BufferGeometry()
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(flat), 3))

    const line = new THREE.Line(geom, mat.clone())
    line.renderOrder = 999 // draw on top
    group.add(line)
  }
}
