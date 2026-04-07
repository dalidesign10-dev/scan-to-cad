import * as THREE from 'three'
import type { PrimitiveResult } from '../store/pipelineStore'

const TYPE_COLORS: Record<string, number> = {
  plane: 0x4ecca3,
  cylinder: 0xe94560,
  sphere: 0xf5a623,
  bspline: 0x54a0ff,
  freeform: 0x888888,
}

export function clearGizmos(group: THREE.Group) {
  while (group.children.length > 0) {
    const child = group.children[0]
    if (child instanceof THREE.Mesh || child instanceof THREE.LineSegments) {
      child.geometry.dispose()
      if (child.material instanceof THREE.Material) {
        child.material.dispose()
      }
    }
    group.remove(child)
  }
}

export function createPrimitiveGizmos(group: THREE.Group, primitives: PrimitiveResult[]) {
  for (const prim of primitives) {
    if (!prim.centroid || !prim.patch_size || prim.patch_size < 0.01) continue
    // Show gizmos for plane, cylinder, sphere (since we only render selected one now)
    if (prim.type !== 'plane' && prim.type !== 'cylinder' && prim.type !== 'sphere') continue

    const color = TYPE_COLORS[prim.type] || 0x888888

    if (prim.type === 'plane' && prim.normal) {
      createPlaneGizmo(group, prim, color)
    } else if (prim.type === 'cylinder' && prim.axis && prim.center && prim.radius) {
      createCylinderGizmo(group, prim, color)
    } else if (prim.type === 'sphere' && prim.center && prim.radius) {
      createSphereGizmo(group, prim, color)
    }
  }
}

function createPlaneGizmo(group: THREE.Group, prim: PrimitiveResult, color: number) {
  const size = Math.min((prim.patch_size || 10) * 0.4, 15)
  const geometry = new THREE.PlaneGeometry(size, size)
  const edges = new THREE.EdgesGeometry(geometry)
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })
  )

  const c = prim.centroid!
  line.position.set(c[0], c[1], c[2])

  const n = new THREE.Vector3(prim.normal![0], prim.normal![1], prim.normal![2])
  const target = new THREE.Vector3(c[0] + n.x, c[1] + n.y, c[2] + n.z)
  line.lookAt(target)
  group.add(line)
}

function createCylinderGizmo(group: THREE.Group, prim: PrimitiveResult, color: number) {
  const rawRadius = prim.radius!
  const patchSize = prim.patch_size || 10

  // Sanity check: if radius is larger than the patch, this is a bad fit
  if (rawRadius > patchSize * 1.2 || rawRadius < 0.01) return

  const radius = rawRadius
  // Cap height to patch size
  const height = Math.min((prim as any).height || patchSize * 0.5, patchSize)

  const geometry = new THREE.CylinderGeometry(radius, radius, height, 16, 1, true)
  const edges = new THREE.EdgesGeometry(geometry)
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })
  )

  // Position at the patch centroid (not the cylinder axis center which may be far away)
  const c = prim.centroid!
  line.position.set(c[0], c[1], c[2])

  const axisVec = new THREE.Vector3(prim.axis![0], prim.axis![1], prim.axis![2]).normalize()
  const up = new THREE.Vector3(0, 1, 0)
  const quat = new THREE.Quaternion().setFromUnitVectors(up, axisVec)
  line.quaternion.copy(quat)
  group.add(line)
}

function createSphereGizmo(group: THREE.Group, prim: PrimitiveResult, color: number) {
  const radius = prim.radius!
  const geometry = new THREE.SphereGeometry(radius, 12, 8)
  const edges = new THREE.EdgesGeometry(geometry)
  const line = new THREE.LineSegments(
    edges,
    new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 })
  )

  const c = prim.center!
  line.position.set(c[0], c[1], c[2])
  group.add(line)
}
