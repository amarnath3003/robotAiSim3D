/**
 * perception/cv_camera.js — Local Computer Vision via Transformers.js
 *
 * Captures the Three.js WebGL canvas from a dedicated robot-eye camera
 * and processes it LOCALLY using an object detection neural network (YOLOS-tiny).
 * 
 * Key design principles:
 * - 100% Local Inference — No API keys, no external calls.
 * - Pixel Color Sampling — Reads exact RGB values from the image to determine color.
 * - Positions are APPROXIMATE (±0.3 m noise) — just like real robot CV
 * - Runs async (non-blocking) — never stalls the render loop
 */

import * as THREE from 'three'
import { updatePerceptionMemory } from '../core/state.js'
import { getManifest } from '../core/manifest.js'

// ─── Configuration ─────────────────────────────────────────────────────────────

const CV_INTERVAL_MS  = 150    // Run mock CV every 150ms to save CPU
const CV_NOISE_M      = 0.30   // Gaussian position noise radius (m) — visual uncertainty
const CAPTURE_SIZE    = 256    // Off-screen render target size (pixels, square)

// ─── Module State ──────────────────────────────────────────────────────────────

let _renderer     = null    // THREE.WebGLRenderer (shared — we hijack render target briefly)
let _getRobot     = null    // () => RobotInstance
let _getScene     = null    // () => THREE.Scene
let _eyeCamera    = null    // THREE.PerspectiveCamera — robot's eye view
let _renderTarget = null    // THREE.WebGLRenderTarget — off-screen capture
let _pixelCanvas  = null    // HTMLCanvasElement — internal clean buffer (no boxes)
let _pixelCtx     = null    // CanvasRenderingContext2D
let _pipCanvas    = null    // HTMLCanvasElement — visible overlay in DOM
let _pipCtx       = null    // CanvasRenderingContext2D

let _lastCVTime   = 0
let _active       = false   // true when model is loaded in worker
let _cvBusy       = false   // prevent concurrent calls
let _eyeFovRad    = 100 * (Math.PI / 180)  // eye camera FOV (radians) — for size estimation
let _camRange     = 8.0                    // detection range gate (metres)

let _worker       = null    // Web Worker for background inference
let _pendingTasks = new Map() // Maps message ID to Promise resolve/reject
let _taskId       = 0
let _lastBoxes    = []      // Cached bounding boxes for 60fps rendering

// ─── Public API ────────────────────────────────────────────────────────────────

export function initCVCamera(renderer, getRobotFn, getSceneFn) {
  _renderer = renderer
  _getRobot = getRobotFn
  _getScene = getSceneFn

  const manifest   = getManifest()
  const camSensor  = manifest?.sensors?.find(s => s.type === 'camera')
  // NOTE: config.fov is the raycast-fan FOV (can be ~179°). Using that as a
  // perspective-projection FOV made everything project to a few pixels and
  // broke colour sampling. The eye camera uses cvFov (default 100°) instead.
  const eyeFOV     = Math.min(camSensor?.config?.cvFov || 100, 130)
  _eyeFovRad       = eyeFOV * (Math.PI / 180)
  _camRange        = camSensor?.config?.range || 8.0

  _eyeCamera = new THREE.PerspectiveCamera(eyeFOV, 1.0, 0.05, 30)

  _renderTarget = new THREE.WebGLRenderTarget(CAPTURE_SIZE, CAPTURE_SIZE, {
    minFilter: THREE.LinearFilter,
    magFilter: THREE.LinearFilter,
    format: THREE.RGBAFormat,
    colorSpace: THREE.SRGBColorSpace
  })

  // Clean hidden canvas for inference extraction
  _pixelCanvas        = document.createElement('canvas')
  _pixelCanvas.width  = CAPTURE_SIZE
  _pixelCanvas.height = CAPTURE_SIZE
  _pixelCtx           = _pixelCanvas.getContext('2d', { willReadFrequently: true })

  // Visible PiP Canvas for the DOM
  _pipCanvas          = document.createElement('canvas')
  _pipCanvas.id       = 'cv-pip'
  _pipCanvas.width    = CAPTURE_SIZE
  _pipCanvas.height   = CAPTURE_SIZE
  _pipCtx             = _pipCanvas.getContext('2d')
  
  _pipCanvas.style.position = 'absolute'
  _pipCanvas.style.bottom = '20px'
  _pipCanvas.style.right = '20px'
  _pipCanvas.style.width = '256px'
  _pipCanvas.style.height = '256px'
  _pipCanvas.style.border = '2px solid rgba(0, 255, 204, 0.5)'
  _pipCanvas.style.borderRadius = '8px'
  _pipCanvas.style.zIndex = '9999'
  _pipCanvas.style.pointerEvents = 'none'
  _pipCanvas.style.boxShadow = '0 4px 12px rgba(0,0,0,0.5)'
  
  const existingPip = document.getElementById('cv-pip')
  if (existingPip) {
    existingPip.replaceWith(_pipCanvas)
  } else {
    document.body.appendChild(_pipCanvas)
  }

  /* --- REAL CV (Transformers.js) Commented out ---
  console.log('[CVCamera] ⏳ Spawning Web Worker for local vision model...')
  _worker = new Worker(new URL('./cv_worker.js', import.meta.url), { type: 'module' })

  _worker.onmessage = (event) => {
    const data = event.data
    if (data.type === 'ready') {
      _active = true
      console.log('[CVCamera] ✓ Background vision worker active!')
    } else if (data.id !== undefined) {
      const task = _pendingTasks.get(data.id)
      if (task) {
        if (data.error) task.reject(new Error(data.error))
        else task.resolve(data.output)
        _pendingTasks.delete(data.id)
      }
    }
  }

  _worker.onerror = (error) => {
    console.error('[CVCamera] ❌ Web Worker Error:', error)
  }
  */

  // --- MOCK CV (Raycasting) ---
  console.log('[CVCamera] ⏳ Initializing Mock Raycast Vision...')
  _active = true
}

export function cvTick() {
  // if (!_worker) return
  
  // Always update the 60fps live feed for the user
  _updatePiPFeed()

  if (!_active || _cvBusy) return
  const now = Date.now()
  if (now - _lastCVTime < CV_INTERVAL_MS) return
  _lastCVTime = now
  _cvBusy = true

  // Run the background inference
  _runCV()
    .finally(() => { _cvBusy = false })
    .catch(() => {})
}

export async function captureAndAnalyze() {
  if (!_active) return []
  return _runCV()
}

export function isCVActive() { return _active }

// ─── Core CV Pipeline ──────────────────────────────────────────────────────────

function _updatePiPFeed() {
  const robot = _getRobot?.()
  const scene = _getScene?.()
  if (!robot || !scene || !_eyeCamera || !_renderTarget || !_renderer) return

  // 1. Position eye camera
  const rp  = robot.position
  const ori = robot.orientation

  const capturePos = new THREE.Vector3(rp.x, rp.y + 0.55, rp.z)
  const forwardOffset = new THREE.Vector3(0, 0, 1).applyQuaternion(ori)
  capturePos.add(forwardOffset.multiplyScalar(0.45))

  const captureOri = new THREE.Quaternion().copy(ori)
  captureOri.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), Math.PI))

  _eyeCamera.position.copy(capturePos)
  _eyeCamera.quaternion.copy(captureOri)
  _eyeCamera.updateMatrixWorld()

  // 2. Render robot's eye view
  const prevTarget = _renderer.getRenderTarget()
  const prevClear  = _renderer.autoClear
  
  const robotMesh = robot.mesh
  const wasVisible = robotMesh ? robotMesh.visible : true
  if (robotMesh) robotMesh.visible = false

  _renderer.autoClear = true
  _renderer.setRenderTarget(_renderTarget)
  try {
    _renderer.render(scene, _eyeCamera)
  } finally {
    if (robotMesh) robotMesh.visible = wasVisible
    _renderer.setRenderTarget(prevTarget)
    _renderer.autoClear = prevClear
  }

  // 3. Read pixels
  const pixelBuf = new Uint8Array(CAPTURE_SIZE * CAPTURE_SIZE * 4)
  _renderer.readRenderTargetPixels(_renderTarget, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE, pixelBuf)

  const imgData = _pixelCtx.createImageData(CAPTURE_SIZE, CAPTURE_SIZE)
  for (let row = 0; row < CAPTURE_SIZE; row++) {
    const srcRow = CAPTURE_SIZE - 1 - row
    imgData.data.set(
      pixelBuf.subarray(srcRow * CAPTURE_SIZE * 4, (srcRow + 1) * CAPTURE_SIZE * 4),
      row * CAPTURE_SIZE * 4
    )
  }
  
  // 4. Update the hidden clean canvas
  _pixelCtx.putImageData(imgData, 0, 0)
  
  // 5. Update the visible PiP canvas
  _pipCtx.drawImage(_pixelCanvas, 0, 0)
  
  // 6. Draw bounding boxes on the PiP canvas
  for (const det of _lastBoxes) {
    const { label, score, box, color } = det
    _pipCtx.strokeStyle = 'lime'
    _pipCtx.lineWidth = 2
    _pipCtx.strokeRect(box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin)

    const text = `${color} ${label} (${(score*100).toFixed(0)}%)`
    _pipCtx.fillStyle = 'rgba(0, 0, 0, 0.7)'
    const textWidth = _pipCtx.measureText(text).width
    _pipCtx.fillRect(box.xmin, box.ymin > 14 ? box.ymin - 14 : 0, textWidth + 4, 14)
    _pipCtx.fillStyle = 'lime'
    _pipCtx.font = '10px Arial'
    _pipCtx.fillText(text, box.xmin + 2, box.ymin > 14 ? box.ymin - 4 : 10)
  }
}

async function _runCV() {
  const robot = _getRobot?.()
  const scene = _getScene?.()
  if (!robot || !scene || !_eyeCamera) return []

  // Snapshot orientation for raycasting (camera is already positioned by _updatePiPFeed)
  const capturePos = _eyeCamera.position.clone()
  const captureOri = _eyeCamera.quaternion.clone()

  /* --- REAL CV START ---
  // 4. Encode to JPEG base64 and Dispatch to Worker ────────────────────────
  // The _pixelCanvas holds a CLEAN image (no bounding boxes) updated by _updatePiPFeed
  const dataUrl = _pixelCanvas.toDataURL('image/jpeg', 0.8)
  const currentId = ++_taskId

  const output = await new Promise((resolve, reject) => {
    _pendingTasks.set(currentId, { resolve, reject })
    _worker.postMessage({ id: currentId, dataUrl })
  })

  if (!output || !output.length) {
    _lastBoxes = []
    return []
  }

  // 5. Process Detections (Color Sampling & 3D Mapping) ────────────────────
  _eyeCamera.position.copy(capturePos)
  _eyeCamera.quaternion.copy(captureOri)
  _eyeCamera.updateMatrixWorld()

  const raycaster = new THREE.Raycaster()
  const results = []
  const newBoxes = []

  for (const det of output) {
    const { label, score, box } = det

    const cx = Math.floor((box.xmin + box.xmax) / 2)
    const cy = Math.floor((box.ymin + box.ymax) / 2)
    
    // Sample exact RGB pixel color from the clean 2D context
    const pixel = _pixelCtx.getImageData(cx, cy, 1, 1).data
    const color = _rgbToColorName(pixel[0], pixel[1], pixel[2])
    
    // Save to draw array
    newBoxes.push({ label, score, box, color })

    const ndcX = (cx / CAPTURE_SIZE) * 2 - 1
    const ndcY = -(cy / CAPTURE_SIZE) * 2 + 1

    // Raycast to find depth in the 3D scene (acting as an RGB-D depth sensor)
    raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), _eyeCamera)
    const intersects = raycaster.intersectObjects(scene.children, true)
    
    // Find first valid hit that isn't the robot itself, the floor, or the walls
    let hit = null
    for (const intersect of intersects) {
      const objName = intersect.object.name || ''
      if (intersect.object.userData?.isRobot || objName === 'Floor' || objName.includes('Wall') || objName.includes('Boundary')) continue
      hit = intersect
      break
    }

    if (!hit) continue

    const distM = hit.distance

    // Approximate world position with noise — visual position is never exact
    const noise   = () => (Math.random() - 0.5) * 2 * CV_NOISE_M
    const approxX = hit.point.x + noise()
    const approxZ = hit.point.z + noise()
    const approxY = 0.3  // floor-level objects

    const objectId = _labelToId(label, color)
    if (!objectId) continue

    // Feed into perception memory
    updatePerceptionMemory(objectId, { x: approxX, y: approxY, z: approxZ }, score)

    console.log(`[CVCamera] 👁 [Local] "${color} ${label}" → ${objectId} ~${distM.toFixed(1)}m (conf ${score.toFixed(2)})`)

    results.push({ objectId, label, color, distanceCategory: 'medium', approxX, approxZ, confidence: score })
  }

  _lastBoxes = newBoxes

  return results
  --- REAL CV END --- */

  // --- MOCK CV (Frustum Projection) ---
  _eyeCamera.position.copy(capturePos)
  _eyeCamera.quaternion.copy(captureOri)
  _eyeCamera.updateMatrixWorld()

  const frustum = new THREE.Frustum()
  const projScreenMatrix = new THREE.Matrix4()
  projScreenMatrix.multiplyMatrices(_eyeCamera.projectionMatrix, _eyeCamera.matrixWorldInverse)
  frustum.setFromProjectionMatrix(projScreenMatrix)

  const detected = new Map() // objId -> { name, minX, maxX, minY, maxY, distM, center }
  const raycaster = new THREE.Raycaster()

  const IGNORE_PREFIXES = ['robot', 'debug', 'sky', 'ground', 'floor', 'env', 'wall', 'boundary', 'room', 'plane']

  scene.traverse((child) => {
    if (!child.isMesh || !child.visible) return
    
    const lowerName = (child.name || '').toLowerCase()
    if (child.userData?.isRobot || IGNORE_PREFIXES.some(p => lowerName.includes(p))) return

    if (!frustum.intersectsObject(child)) return

    // Ignore excessively large meshes (like the background environment or floor)
    // Interactable objects are small; if it's > 5 meters, it's likely part of the scene geometry.
    child.geometry.computeBoundingBox()
    const box3 = new THREE.Box3().setFromObject(child)
    const size = new THREE.Vector3()
    box3.getSize(size)
    if (size.x > 8 || size.y > 8 || size.z > 8) return

    // Find the meaningful root object name
    let rootObj = child
    while(rootObj.parent && rootObj.parent.type !== 'Scene' && !rootObj.name.startsWith('object_')) {
        if (rootObj.name.includes('box') || rootObj.name.includes('ball')) break
        rootObj = rootObj.parent
    }
    
    const rootName = rootObj.name || child.name

    const center = new THREE.Vector3()
    box3.getCenter(center)

    // Occlusion check: raycast from camera to center
    const dir = new THREE.Vector3().subVectors(center, _eyeCamera.position)
    const distToCenter = dir.length()
    dir.normalize()
    raycaster.set(_eyeCamera.position, dir)
    raycaster.far = distToCenter + 0.1 // Only cast as far as the object
    const intersects = raycaster.intersectObjects(scene.children, true)
    
    let occluded = true
    for (const hit of intersects) {
       const hName = (hit.object.name || '').toLowerCase()
       if (hit.object.userData?.isRobot || IGNORE_PREFIXES.some(p => hName.includes(p))) continue
       
       if (hit.object === child || hit.object.parent === rootObj) {
           occluded = false
       }
       break
    }

    if (occluded) return

    // Calculate 2D bounds by projecting the 8 corners of the 3D bounding box
    const corners = [
        new THREE.Vector3(box3.min.x, box3.min.y, box3.min.z),
        new THREE.Vector3(box3.min.x, box3.min.y, box3.max.z),
        new THREE.Vector3(box3.min.x, box3.max.y, box3.min.z),
        new THREE.Vector3(box3.min.x, box3.max.y, box3.max.z),
        new THREE.Vector3(box3.max.x, box3.min.y, box3.min.z),
        new THREE.Vector3(box3.max.x, box3.min.y, box3.max.z),
        new THREE.Vector3(box3.max.x, box3.max.y, box3.min.z),
        new THREE.Vector3(box3.max.x, box3.max.y, box3.max.z),
    ]

    let minX = Infinity, minY = Infinity
    let maxX = -Infinity, maxY = -Infinity

    corners.forEach(corner => {
        corner.project(_eyeCamera)
        // Convert NDC back to pixel space
        const px = (corner.x * 0.5 + 0.5) * CAPTURE_SIZE
        const py = -(corner.y * 0.5 - 0.5) * CAPTURE_SIZE
        minX = Math.min(minX, px)
        maxX = Math.max(maxX, px)
        minY = Math.min(minY, py)
        maxY = Math.max(maxY, py)
    })

    // Range gate — the camera cannot recognise objects beyond its rated range
    if (distToCenter > _camRange * 1.25) return

    if (!detected.has(rootName)) {
        detected.set(rootName, { name: rootName, minX, maxX, minY, maxY, distM: distToCenter, center: center.clone() })
    } else {
        const entry = detected.get(rootName)
        entry.minX = Math.min(entry.minX, minX)
        entry.maxX = Math.max(entry.maxX, maxX)
        entry.minY = Math.min(entry.minY, minY)
        entry.maxY = Math.max(entry.maxY, maxY)
        // Keep the centre of the closest sub-mesh (repeated averaging drifted)
        if (distToCenter < entry.distM) {
          entry.distM = distToCenter
          entry.center.copy(center)
        }
    }
  })

  const results = []
  const newBoxes = []

  for (const [name, entry] of detected) {
    // If the object's 2D bounds are completely outside the canvas, skip
    if (entry.maxX < 0 || entry.minX > CAPTURE_SIZE || entry.maxY < 0 || entry.minY > CAPTURE_SIZE) continue

    // Clamp the bounding box to the screen canvas limits
    const padding = 6
    const box = {
      xmin: Math.max(0, entry.minX - padding),
      xmax: Math.min(CAPTURE_SIZE, entry.maxX + padding),
      ymin: Math.max(0, entry.minY - padding),
      ymax: Math.min(CAPTURE_SIZE, entry.maxY + padding)
    }

    // Skip if box is way too small (e.g. noise or mostly off-screen)
    if (box.xmax - box.xmin < 4 || box.ymax - box.ymin < 4) continue

    // Determine basic label
    let label = 'object'
    const n = name.toLowerCase()
    if (n.includes('box') || n.includes('crate')) label = 'box'
    else if (n.includes('ball')) label = 'sports ball'
    else if (n.includes('goal')) label = 'stop sign'
    else if (n.includes('pillar')) label = 'pillar'

    // Multi-point colour sampling with voting — robust against specular
    // highlights, shadow edges and background bleed (centre pixel alone
    // frequently misread shaded objects).
    const color = _sampleBoxColor(box)

    // Simulated confidence score based on distance (closer = more confident)
    const score = Math.min(0.99, Math.max(0.55, 1.0 - (entry.distM / (_camRange * 1.5))))

    newBoxes.push({ label, score, box, color })

    const noise   = () => (Math.random() - 0.5) * 2 * CV_NOISE_M
    const approxX = entry.center.x + noise()
    const approxZ = entry.center.z + noise()
    const approxY = entry.center.y

    // Physical size estimate from angular width (like a real RGB-D pipeline):
    // a sphere of radius r at distance d subtends 2·asin(r/d).
    const angW = ((entry.maxX - entry.minX) / CAPTURE_SIZE) * _eyeFovRad
    const estRadius = Math.min(0.9, Math.max(0.1, entry.distM * Math.sin(Math.min(angW, Math.PI * 0.5) / 2)))

    // ID: trust the detector's tracked instance name when it follows scene
    // naming (ball_*/box_*/...) — prevents two brown boxes collapsing into one
    // id. Colour+label mapping is the fallback for unnamed meshes.
    const objectId = /^(ball|box|crate|pillar|goal|object)_/i.test(name)
      ? name
      : (_labelToId(label, color) || name)

    updatePerceptionMemory(
      objectId,
      { x: approxX, y: approxY, z: approxZ },
      score,
      { colorName: color, label, radius: estRadius }
    )

    console.log(`[CVCamera] 👁 [Mock] "${color} ${label}" → ${objectId} ~${entry.distM.toFixed(1)}m r~${estRadius.toFixed(2)} (conf ${score.toFixed(2)})`)

    results.push({ objectId, label, color, distanceCategory: 'medium', approxX, approxZ, confidence: score })
  }

  _lastBoxes = newBoxes
  return results
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Maps COCO labels and sampled color to canonical scene object IDs.
 */
function _labelToId(label, color) {
  if (!label) return null
  const lower = label.toLowerCase()
  const col   = color.toLowerCase()

  // Balls / spheres (COCO labels: sports ball, apple, orange, etc.)
  if (/ball|apple|orange|fruit|clock|vase/.test(lower)) {
    if (!col || col === 'unknown') return `ball_unknown_${Date.now()}`
    return `ball_${col}`
  }

  // Goal marker
  if (/stop sign|kite|fire hydrant/.test(lower) && col === 'green') return 'goal_marker'

  // Boxes / crates (COCO labels: suitcase, tv, microwave, oven, refrigerator, box)
  if (/suitcase|tv|microwave|oven|refrigerator|box|book|laptop/.test(lower)) {
    if (col === 'orange' || col === 'brown' || col === 'yellow') return 'box_A'
    if (col === 'black' || col === 'dark' || col === 'gray')  return 'box_B'
    return `box_${col || 'unknown'}`
  }

  // Pillars / columns (COCO: bottle, cup, vase)
  if (/bottle|cup|vase/.test(lower)) return 'pillar_A'

  // Default fallback if we detect *something* and know its color
  if (col) return `object_${col}_${Date.now()}`

  return null
}

/**
 * Sample several points inside a bounding box and vote on the colour name.
 * The centre pixel gets extra weight; 'unknown' votes are discarded.
 * @param {{xmin:number,xmax:number,ymin:number,ymax:number}} box
 * @returns {string}
 */
function _sampleBoxColor(box) {
  const OFFSETS = [
    [0.50, 0.50, 1.5],   // centre (weighted)
    [0.34, 0.50, 1.0],
    [0.66, 0.50, 1.0],
    [0.50, 0.34, 1.0],
    [0.50, 0.66, 1.0],
  ]
  const votes = new Map()

  for (const [fx, fy, w] of OFFSETS) {
    const sx = Math.round(Math.min(CAPTURE_SIZE - 1, Math.max(0, box.xmin + (box.xmax - box.xmin) * fx)))
    const sy = Math.round(Math.min(CAPTURE_SIZE - 1, Math.max(0, box.ymin + (box.ymax - box.ymin) * fy)))
    try {
      const p = _pixelCtx.getImageData(sx, sy, 1, 1).data
      const c = _rgbToColorName(p[0], p[1], p[2])
      if (c && c !== 'unknown') votes.set(c, (votes.get(c) || 0) + w)
    } catch (e) { /* off-canvas sample — skip */ }
  }

  let best = 'unknown'
  let bestVotes = 0
  for (const [c, v] of votes) {
    if (v > bestVotes) { bestVotes = v; best = c }
  }
  return best
}

/**
 * Converts sampled RGB values to a semantic colour name.
 * Classifies in HSV space: hue decides the family, saturation/value separate
 * white/gray/black/brown. Far more robust under scene lighting and shading
 * than nearest-RGB matching (which called shaded green "gray").
 */
function _rgbToColorName(r, g, b) {
  const rn = r / 255, gn = g / 255, bn = b / 255
  const max = Math.max(rn, gn, bn)
  const min = Math.min(rn, gn, bn)
  const d = max - min
  const v = max
  const s = max === 0 ? 0 : d / max

  let h = 0
  if (d > 0) {
    if (max === rn)      h = 60 * (((gn - bn) / d) % 6)
    else if (max === gn) h = 60 * ((bn - rn) / d + 2)
    else                 h = 60 * ((rn - gn) / d + 4)
    if (h < 0) h += 360
  }

  if (v < 0.12) return 'black'
  if (s < 0.16) return v > 0.82 ? 'white' : (v > 0.28 ? 'gray' : 'black')

  if (h < 14 || h >= 345) return (v > 0.78 && s < 0.55) ? 'pink' : 'red'
  if (h < 42)  return v < 0.62 ? 'brown' : 'orange'
  if (h < 70)  return 'yellow'
  if (h < 170) return 'green'
  if (h < 255) return 'blue'
  if (h < 292) return 'purple'
  return 'pink'
}
