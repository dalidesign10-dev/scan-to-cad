import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { loadMeshFromBinaryFile } from './MeshLoader'
import { applySegmentationOverlay, recolorOverlay, OverlayOptions } from './SegmentationOverlay'
import {
  IntentOverlayPayload,
  IntentColorMode,
  applyIntentRegionColors,
  renderIntentGizmos,
  renderIntentSharpEdges,
  renderIntentFamilyEdges,
  clearIntentGizmos,
} from './IntentOverlay'

export class SceneManager {
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private renderer: THREE.WebGLRenderer
  private controls: OrbitControls
  private container: HTMLElement
  private animationId: number = 0
  private meshGroup: THREE.Group
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
    this.scene.background = new THREE.Color(0x111111)

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

    const grid = new THREE.GridHelper(500, 50, 0x2a2a2a, 0x1a1a1a)
    this.scene.add(grid)

    this.meshGroup = new THREE.Group()
    this.scene.add(this.meshGroup)
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
    this.currentMesh = null

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
    renderIntentFamilyEdges(this.intentGizmoGroup, payload)
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
    renderIntentFamilyEdges(this.intentGizmoGroup, this.currentIntentPayload)
    if (this.intentRegionColorsActive && this.currentMesh) {
      applyIntentRegionColors(this.currentMesh, this.currentIntentPayload, mode)
    }
  }

  setIntentGizmosVisible(v: boolean) {
    this.intentGizmoGroup.visible = v
  }

  applyDeviationColors(faceColorsB64: string, nFaces: number) {
    if (!this.currentMesh) return
    const binary = atob(faceColorsB64)
    const bytes = new Uint8Array(binary.length)
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)

    const geometry = this.currentMesh.geometry
    const faceCount = geometry.index ? geometry.index.count / 3 : geometry.attributes.position.count / 3

    // Create per-vertex color attribute
    const positions = geometry.attributes.position
    const colors = new Float32Array(positions.count * 3)

    if (geometry.index) {
        const index = geometry.index.array
        for (let f = 0; f < Math.min(nFaces, faceCount); f++) {
            const r = bytes[f * 3] / 255
            const g = bytes[f * 3 + 1] / 255
            const b = bytes[f * 3 + 2] / 255
            for (let v = 0; v < 3; v++) {
                const vi = index[f * 3 + v]
                colors[vi * 3] = r
                colors[vi * 3 + 1] = g
                colors[vi * 3 + 2] = b
            }
        }
    }

    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    const mat = this.currentMesh.material as THREE.MeshPhongMaterial
    mat.vertexColors = true
    mat.color.setHex(0xffffff)
    mat.needsUpdate = true
    geometry.attributes.color.needsUpdate = true
  }

  clearDeviationColors() {
    if (!this.currentMesh) return
    const mat = this.currentMesh.material as THREE.MeshPhongMaterial
    mat.vertexColors = false
    mat.color.setHex(0x6688cc)
    mat.needsUpdate = true
  }

  private reconstructGroup: THREE.Group = new THREE.Group()
  private reconstructGroupAdded = false

  /** Add a reconstructed surface to the viewport (called per-region during live reconstruction) */
  addReconstructedSurface(vertices: number[][], faces: number[][], classification: string) {
    if (!this.reconstructGroupAdded) {
      this.scene.add(this.reconstructGroup)
      this.reconstructGroupAdded = true
    }

    const geometry = new THREE.BufferGeometry()
    const positions = new Float32Array(vertices.length * 3)
    for (let i = 0; i < vertices.length; i++) {
      positions[i * 3] = vertices[i][0]
      positions[i * 3 + 1] = vertices[i][1]
      positions[i * 3 + 2] = vertices[i][2]
    }
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))

    const indices: number[] = []
    for (const f of faces) {
      indices.push(f[0], f[1], f[2])
    }
    geometry.setIndex(indices)
    geometry.computeVertexNormals()

    // Color by classification type
    const colorMap: Record<string, number> = {
      PLANE: 0x4488cc,
      CYLINDER: 0x44cc88,
      CONE: 0xcc8844,
      FILLET: 0x8844cc,
      CHAMFER: 0xcccc44,
    }
    const color = colorMap[classification] ?? 0x888888

    const material = new THREE.MeshPhongMaterial({
      color,
      specular: 0x333333,
      shininess: 60,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.85,
    })

    const mesh = new THREE.Mesh(geometry, material)
    this.reconstructGroup.add(mesh)
  }

  /** Clear all reconstructed surfaces */
  clearReconstructedSurfaces() {
    while (this.reconstructGroup.children.length > 0) {
      const child = this.reconstructGroup.children[0]
      this.reconstructGroup.remove(child)
      if (child instanceof THREE.Mesh) {
        child.geometry.dispose()
        ;(child.material as THREE.Material).dispose()
      }
    }
  }

  dispose() {
    cancelAnimationFrame(this.animationId)
    this.renderer.dispose()
    this.controls.dispose()
    this.container.removeChild(this.renderer.domElement)
  }
}
