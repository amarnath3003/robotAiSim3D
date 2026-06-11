/**
 * render/scene.js — Three.js Scene Setup & Rendering
 * 
 * Handles:
 * - Scene creation with HDRI environment
 * - Camera setup (perspective + orbit controls)
 * - Post-processing (bloom, tone mapping)
 * - Lighting
 * - Render loop (called by engine)
 * 
 * Robot-agnostic: renders whatever is in the scene.
 */

import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { RGBELoader } from 'three/examples/jsm/loaders/RGBELoader.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'

// ─── State ─────────────────────────────────────────────────────────────────────

let _scene = null
let _camera = null
let _renderer = null
let _controls = null
let _composer = null
let _initialized = false

// Camera-follow state
let _followTarget   = null                  // RobotInstance to track
let _lastFollowPos  = new THREE.Vector3()   // robot position at last frame

// 3J-6: track active focusCamera RAF handle so stacked calls can be cancelled
let _focusCameraHandle = null

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the Three.js scene with all rendering infrastructure.
 * @param {Object} options - {hdriPath, enableBloom, cameraPosition}
 * @returns {Promise<{scene: THREE.Scene, camera: THREE.Camera, renderer: THREE.WebGLRenderer}>}
 */
export async function initScene(options = {}) {
  const {
    hdriPath = '/sky3.hdr',
    enableBloom = true,
    cameraPosition = [4, 3, 4],
    cameraTarget = [0, 0.5, 0],
  } = options
  
  // Scene
  _scene = new THREE.Scene()
  _scene.background = new THREE.Color(0x111118)
  
  // Camera
  _camera = new THREE.PerspectiveCamera(
    50,
    window.innerWidth / window.innerHeight,
    0.1,
    200
  )
  _camera.position.set(...cameraPosition)
  _camera.lookAt(...cameraTarget)
  
  // Renderer
  _renderer = new THREE.WebGLRenderer({
    antialias: true,
    powerPreference: 'high-performance',
  })
  _renderer.setSize(window.innerWidth, window.innerHeight)
  _renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  _renderer.shadowMap.enabled = true
  _renderer.shadowMap.type = THREE.PCFSoftShadowMap
  _renderer.toneMapping = THREE.ACESFilmicToneMapping
  _renderer.toneMappingExposure = 1.2
  _renderer.outputColorSpace = THREE.SRGBColorSpace
  
  document.body.appendChild(_renderer.domElement)
  
  // Controls
  _controls = new OrbitControls(_camera, _renderer.domElement)
  _controls.enableDamping = true
  _controls.dampingFactor = 0.05
  _controls.target.set(...cameraTarget)
  _controls.minDistance = 2
  _controls.maxDistance = 20
  _controls.maxPolarAngle = Math.PI * 0.45
  
  // Lighting
  setupLighting(_scene)
  
  // HDRI Environment
  await loadHDRI(hdriPath)
  
  // Post-processing
  if (enableBloom) {
    setupPostProcessing()
  }
  
  // Floor (visual only — physics floor is separate)
  createVisualFloor(_scene)
  
  // Resize handling
  // 3J-1: store reference so destroyScene() can remove it later
  window.addEventListener('resize', handleResize)
  
  _initialized = true
  console.log('[Render] Scene initialized')
  
  return { scene: _scene, camera: _camera, renderer: _renderer }
}

/**
 * Set the robot whose XZ position the camera should track.
 * The camera preserves its current orbit angle and distance — only the
 * target (and camera offset) shift by the robot's movement delta each frame.
 * Pass null to disable following.
 * @param {import('../core/adapter.js').RobotInstance|null} robot
 */
export function setFollowTarget(robot) {
  _followTarget = robot
  if (robot) _lastFollowPos.copy(robot.position)
}

/**
 * Render one frame. Called by the engine each tick.
 * @param {number} dt - Delta time
 */
export function renderFrame(dt) {
  if (!_initialized) return

  // Shift camera + orbit target by robot's XZ movement delta
  _updateCameraFollow()

  _controls.update()
  
  if (_composer) {
    _composer.render()
  } else {
    _renderer.render(_scene, _camera)
  }
}

/**
 * Get the Three.js scene.
 */
export function getScene() {
  return _scene
}

/**
 * Get the camera.
 */
export function getCamera() {
  return _camera
}

/**
 * Get the renderer.
 */
export function getRenderer() {
  return _renderer
}

/**
 * Focus camera on a specific position with a smooth ease-out lerp.
 * @param {THREE.Vector3} target   - World position to look at
 * @param {number}        distance - How far the camera should be from the target
 * @param {number}        frames   - Duration in animation frames (default 20 ≈ 330 ms)
 */
export function focusCamera(target, distance = 5, frames = 20) {
  if (!_controls || !_camera) return

  // 3J-6: cancel any in-progress focus animation before starting a new one
  if (_focusCameraHandle != null) {
    cancelAnimationFrame(_focusCameraHandle)
    _focusCameraHandle = null
  }

  const startTarget = _controls.target.clone()
  const startPos    = _camera.position.clone()

  // Preserve the current camera-to-target offset direction, scale to `distance`
  const offset = startPos.clone().sub(startTarget)
  const offsetDir = offset.length() > 0.001 ? offset.normalize() : new THREE.Vector3(0, 0.6, 1).normalize()
  const endPos    = target.clone().add(offsetDir.multiplyScalar(distance))

  let frame = 0

  function animate() {
    frame++
    // Ease-out cubic: fast start, smooth finish
    const t    = frame / frames
    const ease = 1 - Math.pow(1 - t, 3)

    _controls.target.lerpVectors(startTarget, target, ease)
    _camera.position.lerpVectors(startPos, endPos, ease)
    _controls.update()

    if (frame < frames) {
      _focusCameraHandle = requestAnimationFrame(animate)
    } else {
      _focusCameraHandle = null
    }
  }

  _focusCameraHandle = requestAnimationFrame(animate)
}

// ─── Internal: Camera Follow ───────────────────────────────────────────────────

/**
 * Shift the orbit controls target (and camera position) by the robot's XZ delta.
 * Runs every render frame to keep the robot centred in the view.
 * Only XZ is tracked — Y (height) is left to OrbitControls so the user's
 * elevation angle is preserved.
 */
function _updateCameraFollow() {
  if (!_followTarget || !_controls || !_camera) return

  const rp = _followTarget.position
  const dx = rp.x - _lastFollowPos.x
  const dz = rp.z - _lastFollowPos.z

  if (Math.abs(dx) > 1e-5 || Math.abs(dz) > 1e-5) {
    _controls.target.x += dx
    _controls.target.z += dz
    _camera.position.x += dx
    _camera.position.z += dz
    _lastFollowPos.set(rp.x, rp.y, rp.z)
  }
}

// ─── Internal Setup ────────────────────────────────────────────────────────────

function setupLighting(scene) {
  // Ambient light
  const ambient = new THREE.AmbientLight(0x404060, 0.4)
  scene.add(ambient)
  
  // Main directional light (sun)
  const sun = new THREE.DirectionalLight(0xffffff, 1.5)
  sun.position.set(5, 10, 5)
  sun.castShadow = true
  sun.shadow.mapSize.width = 2048
  sun.shadow.mapSize.height = 2048
  sun.shadow.camera.near = 0.5
  sun.shadow.camera.far = 50
  sun.shadow.camera.left = -10
  sun.shadow.camera.right = 10
  sun.shadow.camera.top = 10
  sun.shadow.camera.bottom = -10
  sun.shadow.bias = -0.0001
  scene.add(sun)
  
  // Fill light (softer, from opposite side)
  const fill = new THREE.DirectionalLight(0x8888ff, 0.3)
  fill.position.set(-3, 5, -3)
  scene.add(fill)
  
  // Point light for robot area (warm accent)
  const accent = new THREE.PointLight(0xff8844, 0.5, 8)
  accent.position.set(0, 3, 0)
  scene.add(accent)
}

async function loadHDRI(path) {
  try {
    const rgbeLoader = new RGBELoader()
    const hdrTexture = await new Promise((resolve, reject) => {
      rgbeLoader.load(path, resolve, undefined, reject)
    })
    
    hdrTexture.mapping = THREE.EquirectangularReflectionMapping

    // 3J-5: dispose old HDRI texture before replacing it to avoid GPU leak
    if (_scene.environment && _scene.environment !== _scene.background) {
      _scene.environment.dispose()
    }
    if (_scene.background?.isTexture) {
      _scene.background.dispose()
    }

    _scene.environment = hdrTexture
    _scene.background = hdrTexture
    
    console.log(`[Render] HDRI loaded: ${path}`)
  } catch (e) {
    console.warn(`[Render] HDRI load failed (${path}), using solid background`)
    _scene.background = new THREE.Color(0x111118)
  }
}

function setupPostProcessing() {
  _composer = new EffectComposer(_renderer)
  
  const renderPass = new RenderPass(_scene, _camera)
  _composer.addPass(renderPass)
  
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.3,    // strength
    0.4,    // radius
    0.85    // threshold
  )
  _composer.addPass(bloomPass)
}

function createVisualFloor(scene) {
  // 48×48 covers the full ±16 arena + visible margin beyond the boundary walls
  const SIZE = 48

  const geometry = new THREE.PlaneGeometry(SIZE, SIZE, 1, 1)
  const material = new THREE.MeshStandardMaterial({
    color:     0x1a1a2e,
    roughness: 0.9,
    metalness: 0.05,
  })
  const floor = new THREE.Mesh(geometry, material)
  floor.rotation.x = -Math.PI / 2
  floor.receiveShadow = true
  floor.name = 'floor'
  scene.add(floor)

  // Grid helper — major lines every 2 m, 24 divisions = 48 m total
  const grid = new THREE.GridHelper(SIZE, 24, 0x2a2a4a, 0x222238)
  grid.position.y = 0.003   // just above floor surface to avoid z-fighting
  grid.name = 'floor_grid'
  scene.add(grid)
}

function handleResize() {
  if (!_camera || !_renderer) return
  
  _camera.aspect = window.innerWidth / window.innerHeight
  _camera.updateProjectionMatrix()
  _renderer.setSize(window.innerWidth, window.innerHeight)
  
  if (_composer) {
    _composer.setSize(window.innerWidth, window.innerHeight)
  }
}
