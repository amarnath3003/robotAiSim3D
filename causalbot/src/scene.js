import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { state } from './state.js'

const loader = new GLTFLoader()

export async function initScene() {
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.shadowMap.enabled = true
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.1
  document.body.appendChild(renderer.domElement)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x111111)
  scene.fog = new THREE.FogExp2(0x111111, 0.04)

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100)
  camera.position.set(0, 5, 7)
  camera.lookAt(0, 0, 0)

  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.set(0, 0.5, 0)
  controls.enableDamping = true
  controls.dampingFactor = 0.07
  controls.maxPolarAngle = Math.PI / 2.05
  controls.minDistance = 2
  controls.maxDistance = 15
  controls.update()

  // Lighting
  scene.add(new THREE.AmbientLight(0xffffff, 0.5))

  const sun = new THREE.DirectionalLight(0xfff5e0, 1.4)
  sun.position.set(4, 10, 4)
  sun.castShadow = true
  sun.shadow.mapSize.setScalar(2048)
  sun.shadow.camera.near = 0.5
  sun.shadow.camera.far = 30
  sun.shadow.camera.left = sun.shadow.camera.bottom = -8
  sun.shadow.camera.right = sun.shadow.camera.top = 8
  scene.add(sun)

  const fillLight = new THREE.PointLight(0x4466ff, 0.5, 15)
  fillLight.position.set(-3, 4, -3)
  scene.add(fillLight)

  // Load environment
  const env = await loader.loadAsync('/environment.glb')
  env.scene.traverse(c => {
    if (c.isMesh) {
      c.castShadow = true
      c.receiveShadow = true
    }
  })
  scene.add(env.scene)

  state.scene.three = scene
  state.scene.camera = camera
  state.scene.renderer = renderer
  state.scene.controls = controls

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  })

  console.log('Scene ready')
}

export function renderScene() {
  state.scene.controls?.update()
  state.scene.renderer?.render(state.scene.three, state.scene.camera)
}

// Get Three.js mesh by object id
export function getMesh(id) {
  return state.scene.three?.getObjectByName(id) || null
}

// Sync a world object's state position TO its Three.js mesh
export function syncMeshToState(id) {
  const obj = state.world.objects[id]
  const mesh = getMesh(id)
  if (!obj || !mesh) return
  mesh.position.set(...obj.position)
}