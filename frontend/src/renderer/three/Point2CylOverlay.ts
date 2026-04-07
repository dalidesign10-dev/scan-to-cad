import * as THREE from 'three'

export interface P2CSegment {
  id: number
  n_points: number
  axis: number[]
  center: number[]
  extent: number
  radius: number
  points: number[][]
}

export interface P2CResult {
  n_segments: number
  segments: P2CSegment[]
}

const PALETTE = [
  0x4ecca3, 0xe94560, 0xf5a623, 0x7b68ee, 0x00bcd4,
  0xff6b6b, 0x48dbfb, 0xfeca57, 0xff9ff3, 0x54a0ff,
]

export function clearPoint2Cyl(group: THREE.Group) {
  while (group.children.length > 0) {
    const c = group.children[0]
    if ((c as any).geometry) (c as any).geometry.dispose?.()
    if ((c as any).material) {
      const m = (c as any).material
      if (Array.isArray(m)) m.forEach((mm) => mm.dispose?.())
      else m.dispose?.()
    }
    group.remove(c)
  }
}

export function renderPoint2Cyl(group: THREE.Group, result: P2CResult) {
  if (!result?.segments) return

  for (let i = 0; i < result.segments.length; i++) {
    const seg = result.segments[i]
    const color = PALETTE[i % PALETTE.length]

    // 1) Colored point cloud for this segment
    if (seg.points && seg.points.length > 0) {
      const flat = new Float32Array(seg.points.length * 3)
      for (let j = 0; j < seg.points.length; j++) {
        flat[j * 3 + 0] = seg.points[j][0]
        flat[j * 3 + 1] = seg.points[j][1]
        flat[j * 3 + 2] = seg.points[j][2]
      }
      const geom = new THREE.BufferGeometry()
      geom.setAttribute('position', new THREE.BufferAttribute(flat, 3))
      const mat = new THREE.PointsMaterial({
        color,
        size: 1.2,
        sizeAttenuation: true,
      })
      const pts = new THREE.Points(geom, mat)
      pts.userData.segId = seg.id
      group.add(pts)
    }

    // 2) Extrusion cylinder gizmo
    if (seg.radius > 0 && seg.extent > 0) {
      const cyl = new THREE.CylinderGeometry(seg.radius, seg.radius, seg.extent, 32, 1, true)
      const mat = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.18,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      const mesh = new THREE.Mesh(cyl, mat)
      mesh.position.set(seg.center[0], seg.center[1], seg.center[2])
      const axis = new THREE.Vector3(seg.axis[0], seg.axis[1], seg.axis[2]).normalize()
      const up = new THREE.Vector3(0, 1, 0)
      mesh.quaternion.copy(new THREE.Quaternion().setFromUnitVectors(up, axis))
      mesh.userData.segId = seg.id
      group.add(mesh)

      // Wire outline
      const edges = new THREE.EdgesGeometry(
        new THREE.CylinderGeometry(seg.radius, seg.radius, seg.extent, 24, 1, true),
      )
      const line = new THREE.LineSegments(
        edges,
        new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }),
      )
      line.position.copy(mesh.position)
      line.quaternion.copy(mesh.quaternion)
      group.add(line)

      // Axis arrow
      const arrowLen = seg.extent * 0.55
      const arrow = new THREE.ArrowHelper(axis, mesh.position, arrowLen, color, arrowLen * 0.15, arrowLen * 0.08)
      group.add(arrow)
    }
  }
}
