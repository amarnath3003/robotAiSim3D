import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { state } from './state.js'

const loader = new GLTFLoader()
const rgbeLoader = new RGBELoader()

export async function initScene() {
  const renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(window.innerWidth, window.innerHeight)
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.shadowMap.enabled = true
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 0.9
  document.body.appendChild(renderer.domElement)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x111111)
  scene.fog = new THREE.FogExp2(0x111111, 0.04)

  const hdri = await rgbeLoader.loadAsync('/sky3.hdr')
  hdri.mapping = THREE.EquirectangularReflectionMapping
  scene.environment = hdri
  scene.background = hdri

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

  scene.add(new THREE.AmbientLight(0xffffff, 0.5))

  const fillLight = new THREE.PointLight(0x4466ff, 0.5, 15)
  fillLight.position.set(-3, 4, -3)
  scene.add(fillLight)

  // Load environment
  const env = await loader.loadAsync('/environment2.glb')
  env.scene.traverse(c => {
    if (c.isMesh) {
      c.castShadow = true
      c.receiveShadow = true
      
      // Hide table for now
      if (c.name.toLowerCase().includes('table')) {
        c.visible = false
      }
    }
  })
  scene.add(env.scene)

  const composer = new EffectComposer(renderer)
  composer.addPass(new RenderPass(scene, camera))
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.9,
    0.25,
    0.8
  )
  composer.addPass(bloomPass)

  state.scene.three = scene
  state.scene.camera = camera
  state.scene.renderer = renderer
  state.scene.controls = controls
  state.scene.composer = composer

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
    composer.setSize(window.innerWidth, window.innerHeight)
  })

  console.log('Scene ready')
}

let lastMode = 'ai'

export function renderScene() {
  state.scene.controls.enabled = true
  
  if (state.controlMode !== lastMode) {
    if (state.controlMode === 'ai') {
      const p = state.robot.position
      state.scene.controls.target.set(p[0], p[1] + 0.4, p[2])
    }
    lastMode = state.controlMode
  }

  if (state.controlMode === 'debug') {
    const target = state.debugRobot.position
    state.scene.controls.target.lerp(new THREE.Vector3(target[0], target[1] + 0.4, target[2]), 0.1)
  }

  state.scene.controls?.update()

  if (state.scene.composer) {
    state.scene.composer.render()
  } else {
    state.scene.renderer?.render(state.scene.three, state.scene.camera)
  }
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