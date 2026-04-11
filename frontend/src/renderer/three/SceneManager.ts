import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { loadMeshFromBinaryFile } from './MeshLoader'
import { applySegmentationOverlay, recolorOverlay, patchIdForFace, OverlayOptions } from './SegmentationOverlay'
import { createPrimitiveGizmos, clearGizmos } from './PrimitiveGizmos'
import { renderInfiniteSurfaces, clearInfiniteSurfaces, InfiniteSurface } from './InfiniteSurfaces'
import { renderEdgeCurves, clearEdgeCurves, EdgeCurve } from './EdgeCurves'
import { renderCadPreview, clearCadPreview } from './CadPreview'
import { renderPolyhedralCad, clearPolyhedralCad } from './PolyhedralCad'
import { renderPoint2Cyl, clearPoint2Cyl, P2CResult } from './Point2CylOverlay'
import {
  IntentOverlayPayload,
  IntentColorMode,
  applyIntentRegionColors,
  renderIntentGizmos,
  renderIntentSharpEdges,
  clearIntentGizmos,
} from './IntentOverlay'
import type { PrimitiveResult } from '../store/pipelineStore'

export class SceneManager {
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer
  private controls: OrbitControls
  private container: HTMLElement
  private animationId: number = 0
  private meshGroup: THREE.Group
  private gizmoGroup: THREE.Group
  private infiniteSurfaceGroup: THREE.Group
  private edgeGroup: THREE.Group
  private cadPreviewGroup: THREE.Group
  private polyhedralCadGroup: THREE.Group
  private pickedPointsGroup: THREE.Group
  private point2cylGroup: THREE.Group
  private intentGizmoGroup: THREE.Group
  private currentMesh: THREE.Mesh | null = null
  private currentIntentPayload: IntentOverlayPayload | null = null
  private intentRegionColorsActive: boolean = false
  private intentColorMode: IntentColorMode = 'region'
  private pendingLabelsUrl: string | null = null
  private currentPatches: any[] | null = null
  private currentLabels: Int32Array | null = null
  private currentOverlayOptions: OverlayOptions = {}
  private raycaster = new THREE.Raycaster()

  constructor(container: HTMLElement) {
    this.container = container

    this.scene = new THREE.Scene()
    this.scene.background = new THREE.Color(0x0a0a1a)

    const aspect = container.clientWidth / container.clientHeight
    this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 10000)
    this.camera.position.set(100, 100, 100)

    this.renderer = new THREE.WebGLRenderer({ antialias: true })
    this.renderer.setSize(container.clientWidth, container.clientHeight)
    this.renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(this.renderer.domElement)

    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.1

    const ambient = new THREE.AmbientLight(0x404040, 2)
    this.scene.add(ambient)

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 1.5)
    dirLight1.position.set(100, 200, 150)
    this.scene.add(dirLight1)

    const dirLight2 = new THREE.DirectionalLight(0x8888ff, 0.8)
    dirLight2.position.set(-100, -50, -100)
    this.scene.add(dirLight2)

    const grid = new THREE.GridHelper(500, 50, 0x222244, 0x111133)
    this.scene.add(grid)

    this.meshGroup = new THREE.Group()
    this.scene.add(this.meshGroup)
    this.gizmoGroup = new THREE.Group()
    this.scene.add(this.gizmoGroup)
    this.infiniteSurfaceGroup = new THREE.Group()
    this.scene.add(this.infiniteSurfaceGroup)
    this.edgeGroup = new THREE.Group()
    this.scene.add(this.edgeGroup)
    this.cadPreviewGroup = new THREE.Group()
    this.cadPreviewGroup.visible = false
    this.scene.add(this.cadPreviewGroup)
    this.polyhedralCadGroup = new THREE.Group()
    this.polyhedralCadGroup.visible = false
    this.scene.add(this.polyhedralCadGroup)
    this.pickedPointsGroup = new THREE.Group()
    this.scene.add(this.pickedPointsGroup)
    this.point2cylGroup = new THREE.Group()
    this.scene.add(this.point2cylGroup)
    this.intentGizmoGroup = new THREE.Group()
    this.intentGizmoGroup.visible = false
    this.scene.add(this.intentGizmoGroup)

    const resizeObserver = new ResizeObserver(() => this.handleResize())
    resizeObserver.observe(container)

    this.animate()
  }

  private handleResize() {
    const w = this.container.clientWidth
    const h = this.container.clientHeight
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    this.renderer.setSize(w, h)
  }

  private animate() {
    this.animationId = requestAnimationFrame(() => this.animate())
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  async loadMeshFromUrl(url: string) {
    this.meshGroup.clear()
    this.gizmoGroup.clear()
    this.currentMesh = null
    // Don't clear pendingLabelsUrl — we want it applied after mesh loads

    try {
      const response = await fetch(url)
      const buffer = await response.arrayBuffer()
      const geometry = loadMeshFromBinaryFile(buffer)

      const material = new THREE.MeshPhongMaterial({
        color: 0x6688cc,
        specular: 0x222222,
        shininess: 40,
        side: THREE.DoubleSide,
        vertexColors: false,
      })

      const mesh = new THREE.Mesh(geometry, material)
      this.meshGroup.add(mesh)
      this.currentMesh = mesh

      // Auto-fit camera
      const box = new THREE.Box3().setFromObject(mesh)
      const center = box.getCenter(new THREE.Vector3())
      const size = box.getSize(new THREE.Vector3())
      const maxDim = Math.max(size.x, size.y, size.z)

      this.controls.target.copy(center)
      this.camera.position.set(
        center.x + maxDim * 1.2,
        center.y + maxDim * 0.8,
        center.z + maxDim * 1.2,
      )
      this.camera.lookAt(center)
      this.controls.update()

      const grid = this.scene.children.find((c) => c instanceof THREE.GridHelper)
      if (grid) {
        grid.position.y = box.min.y
      }

      // If a labels URL was set before mesh was ready, apply now
      if (this.pendingLabelsUrl) {
        const url = this.pendingLabelsUrl
        this.pendingLabelsUrl = null
        await this.applySegmentationColorsFromUrl(url, this.currentPatches || undefined)
      }
    } catch (err) {
      console.error('Failed to load mesh:', err)
    }
  }

  async applySegmentationColorsFromUrl(url: string, patches?: any[]) {
    if (patches) this.currentPatches = patches
    if (!this.currentMesh) {
      // Defer until mesh is loaded
      this.pendingLabelsUrl = url
      return
    }
    try {
      const response = await fetch(url)
      const buffer = await response.arrayBuffer()
      this.currentLabels = new Int32Array(buffer)
      this.currentOverlayOptions = {
        ...this.currentOverlayOptions,
        patches: this.currentPatches || undefined,
      }
      applySegmentationOverlay(this.currentMesh, this.currentLabels, this.currentOverlayOptions)
    } catch (err) {
      console.error('Failed to apply segmentation:', err)
    }
  }

  /** Update merge map / selection without re-fetching labels. */
  updateOverlayOptions(opts: Partial<OverlayOptions>) {
    this.currentOverlayOptions = { ...this.currentOverlayOptions, ...opts }
    if (this.currentMesh && this.currentLabels) {
      recolorOverlay(this.currentMesh, this.currentOverlayOptions)
    }
  }

  /** Raycast and return the 3D hit point on the mesh, or null. */
  pickPointAt(ndcX: number, ndcY: number): [number, number, number] | null {
    if (!this.currentMesh) return null
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera)
    const hits = this.raycaster.intersectObject(this.currentMesh, false)
    if (!hits.length) return null
    const p = hits[0].point
    return [p.x, p.y, p.z]
  }

  /** Update the picked-points marker overlay. */
  showPickedPoints(points: number[][]) {
    // Clear without disposing — markers reuse a fresh geometry each call below
    while (this.pickedPointsGroup.children.length > 0) {
      this.pickedPointsGroup.remove(this.pickedPointsGroup.children[0])
    }
    if (!points || points.length === 0) return
    // Size relative to scene
    const box = new THREE.Box3()
    if (this.currentMesh) box.setFromObject(this.currentMesh)
    const size = box.getSize(new THREE.Vector3())
    const r = Math.max(size.x, size.y, size.z) * 0.012 || 1
    const mat = new THREE.MeshBasicMaterial({ color: 0xff3355, depthTest: false })
    const geom = new THREE.SphereGeometry(r, 12, 8)
    for (const p of points) {
      const m = new THREE.Mesh(geom, mat)
      m.position.set(p[0], p[1], p[2])
      m.renderOrder = 999
      this.pickedPointsGroup.add(m)
    }
  }

  /** Raycast a normalized device coordinate against the segmented mesh. */
  pickPatchAt(ndcX: number, ndcY: number): number | null {
    if (!this.currentMesh) return null
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera)
    const hits = this.raycaster.intersectObject(this.currentMesh, false)
    if (!hits.length) return null
    const hit = hits[0]
    if (hit.faceIndex == null) return null
    return patchIdForFace(this.currentMesh, hit.faceIndex)
  }

  showPrimitiveGizmos(primitives: PrimitiveResult[]) {
    clearGizmos(this.gizmoGroup)
    if (!primitives || primitives.length === 0) return
    createPrimitiveGizmos(this.gizmoGroup, primitives)
  }

  showInfiniteSurfaces(surfaces: InfiniteSurface[]) {
    clearInfiniteSurfaces(this.infiniteSurfaceGroup)
    if (!surfaces || surfaces.length === 0) return
    renderInfiniteSurfaces(this.infiniteSurfaceGroup, surfaces)
  }

  setInfiniteSurfacesVisible(visible: boolean) {
    this.infiniteSurfaceGroup.visible = visible
  }

  showPolyhedralCad(data: { vertices: number[][]; faces: number[][] } | null) {
    clearPolyhedralCad(this.polyhedralCadGroup)
    if (!data) return
    renderPolyhedralCad(this.polyhedralCadGroup, data)
  }

  setPolyhedralCadVisible(visible: boolean) {
    this.polyhedralCadGroup.visible = visible
    // When polyhedral CAD is on, hide everything else
    this.meshGroup.visible = !visible
    this.cadPreviewGroup.visible = !visible && this.cadPreviewGroup.visible
  }

  showCadPreview(surfaces: any[]) {
    clearCadPreview(this.cadPreviewGroup)
    if (!surfaces || surfaces.length === 0) return
    renderCadPreview(this.cadPreviewGroup, surfaces)
  }

  setCadPreviewVisible(visible: boolean) {
    this.cadPreviewGroup.visible = visible
    // Hide the original noisy mesh when CAD preview is on
    this.meshGroup.visible = !visible
  }

  showPoint2Cyl(result: P2CResult | null) {
    clearPoint2Cyl(this.point2cylGroup)
    if (!result) return
    renderPoint2Cyl(this.point2cylGroup, result)
  }

  setPoint2CylVisible(v: boolean) {
    this.point2cylGroup.visible = v
  }

  /** Set the current intent overlay payload from the backend.
   *  Stores it so the user can toggle region-colour and gizmo visibility
   *  independently. */
  setIntentOverlay(payload: IntentOverlayPayload | null) {
    this.currentIntentPayload = payload
    clearIntentGizmos(this.intentGizmoGroup)
    if (!payload) return
    // Compute scene scale from the current mesh bbox so gizmo lines don't
    // get lost on small parts.
    let scale = 100
    if (this.currentMesh) {
      const box = new THREE.Box3().setFromObject(this.currentMesh)
      const size = box.getSize(new THREE.Vector3())
      scale = Math.max(size.x, size.y, size.z) || 100
    }
    renderIntentGizmos(this.intentGizmoGroup, payload, scale, this.intentColorMode)
    renderIntentSharpEdges(this.intentGizmoGroup, payload)
    if (this.intentRegionColorsActive && this.currentMesh) {
      applyIntentRegionColors(this.currentMesh, payload, this.intentColorMode)
    }
  }

  setIntentRegionColors(active: boolean) {
    this.intentRegionColorsActive = active
    if (active && this.currentMesh && this.currentIntentPayload) {
      applyIntentRegionColors(this.currentMesh, this.currentIntentPayload, this.intentColorMode)
    } else if (!active && this.currentMesh && this.currentLabels) {
      // Restore the regular segmentation overlay coloring.
      recolorOverlay(this.currentMesh, this.currentOverlayOptions)
    }
  }

  setIntentColorMode(mode: IntentColorMode) {
    if (mode === this.intentColorMode) return
    this.intentColorMode = mode
    // Re-render whatever is currently shown using the new mode.
    if (!this.currentIntentPayload) return
    let scale = 100
    if (this.currentMesh) {
      const box = new THREE.Box3().setFromObject(this.currentMesh)
      const size = box.getSize(new THREE.Vector3())
      scale = Math.max(size.x, size.y, size.z) || 100
    }
    renderIntentGizmos(this.intentGizmoGroup, this.currentIntentPayload, scale, mode)
    renderIntentSharpEdges(this.intentGizmoGroup, this.currentIntentPayload)
    if (this.intentRegionColorsActive && this.currentMesh) {
      applyIntentRegionColors(this.currentMesh, this.currentIntentPayload, mode)
    }
  }

  setIntentGizmosVisible(v: boolean) {
    this.intentGizmoGroup.visible = v
  }

  showEdgeCurves(edges: EdgeCurve[]) {
    clearEdgeCurves(this.edgeGroup)
    if (!edges || edges.length === 0) return
    renderEdgeCurves(this.edgeGroup, edges)
  }

  dispose() {
    cancelAnimationFrame(this.animationId)
    this.renderer.dispose()
    this.controls.dispose()
    this.container.removeChild(this.renderer.domElement)
  }
}
