import * as THREE from 'three'
import { state, getRobotPos, getObject } from './state.js'
import { setRobotPos } from './state.js'

// ─── Constants ────────────────────────────────────────────────────────────────
const LIDAR_RAYS   = 11
const LIDAR_FOV    = 165 * (Math.PI / 180) // radians
const LIDAR_RANGE  = 5.0
const LIDAR_HEIGHT = 0.45  // eye height above base
// Bounds are big enough for either room or 17×17 maze (~8m radius)
const ROOM_BOUNDS  = window.MAZE_MODE
  ? { minX: -9, maxX: 9, minZ: -9, maxZ: 9 }
  : { minX: -2.8, maxX: 2.8, minZ: -2.8, maxZ: 2.8 }
const MAX_LINEAR   = 2.5   // m/s forward
const MAX_ANGULAR  = 2.0   // rad/s turning

// ─── State ────────────────────────────────────────────────────────────────────
let ws = null
let _lidarMeshCache = null // cached list of obstacle meshes for raycasting
let _pendingAction  = null // action waiting to be applied in the next physics tick
let _stepReady      = false // flag: physics has advanced, ready to send state
let _raycaster      = new THREE.Raycaster()

// ─── WebSocket Client ─────────────────────────────────────────────────────────
export function initRL() {
  ws = new WebSocket('ws://localhost:8765')
  ws.binaryType = 'arraybuffer'

  ws.onopen = () => {
    console.log('[RL] Connected to Gymnasium Server')
    _lidarMeshCache = null // invalidate on reconnect

    // In maze mode, tell Python where the goal is
    if (window.MAZE_MODE && window.getMazeGoal) {
      const goal = window.getMazeGoal()
      if (goal) {
        setTimeout(() => { // small delay so Python handler is ready
          ws.send(JSON.stringify({ type: 'maze_goal', x: goal.x, z: goal.z }))
          console.log(`[RL] Sent maze_goal to Python: (${goal.x.toFixed(2)}, ${goal.z.toFixed(2)})`)
        }, 500)
      }
    }
  }

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data)

      if (msg.type === 'reset') {
        _handleReset()
      } else if (msg.type === 'action') {
        if (state.controlMode === 'rl') {
          const [linear, angular] = msg.action
          _pendingAction = {
            linear: Math.max(-MAX_LINEAR, Math.min(MAX_LINEAR, linear)),
            angular: Math.max(-MAX_ANGULAR, Math.min(MAX_ANGULAR, angular)),
          }
          // State will be sent back after the physics step (via updateRL)
          _stepReady = false
        }
      }
    } catch (e) {
      console.error('[RL] Parse error:', e)
    }
  }

  ws.onclose = () => {
    console.log('[RL] Disconnected — reconnecting in 3s')
    _pendingAction = null
    setTimeout(initRL, 3000)
  }

  ws.onerror = (e) => console.error('[RL] WebSocket error:', e)
}

// ─── Prompt dispatch ─────────────────────────────────────────────────────────
export function sendPromptRL(text) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'prompt', text }))
  }
}

// ─── Reset handler ────────────────────────────────────────────────────────────
function _handleReset() {
  // Teleport the AI robot back to start
  const startX = 0, startY = 0.35, startZ = 1.8
  setRobotPos(startX, startY, startZ)
  state.robot.rotation = 0

  // Sync the Rapier kinematic body immediately
  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({ x: startX, y: startY, z: startZ })
  }

  // Also reset debug robot position away from RL robot so they don't interfere
  state.debugRobot.position[0] = 1.5
  state.debugRobot.position[1] = 0.35
  state.debugRobot.position[2] = 1.5

  _pendingAction = null

  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset_done', observation: getObservation() }))
  }
}

function getObservation() {
  const rx = state.robot.position[0]
  const rz = state.robot.position[2]
  const heading = state.robot.rotation || 0
  return [rx, rz, heading, ...castLidar()]
}

// ─── Physics update (called every substep in main.js) ─────────────────────────
export function updateRL(delta) {
  if (state.controlMode !== 'rl') return

  if (_pendingAction) {
    const { linear, angular } = _pendingAction

    // Update robot heading
    state.robot.rotation = (state.robot.rotation || 0) + angular * delta

    // Compute velocity in world space from heading + linear speed
    const heading = state.robot.rotation
    const dx = Math.sin(heading) * linear * delta
    const dz = Math.cos(heading) * linear * delta

    // Apply movement, clamped to room bounds
    const newX = Math.max(ROOM_BOUNDS.minX, Math.min(ROOM_BOUNDS.maxX, state.robot.position[0] + dx))
    const newZ = Math.max(ROOM_BOUNDS.minZ, Math.min(ROOM_BOUNDS.maxZ, state.robot.position[2] + dz))
    setRobotPos(newX, state.robot.position[1], newZ)

    // Sync kinematic Rapier body so physics collisions still register
    if (state.robot._body) {
      state.robot._body.setNextKinematicTranslation({ x: newX, y: state.robot.position[1], z: newZ })
    }

    // Flag that we should send state after this batch of substeps
    _stepReady = true
  }
}

// ─── Called once per rendered frame (after all substeps) ─────────────────────
export function flushRLState() {
  if (state.controlMode !== 'rl') return
  if (!_stepReady || !_pendingAction) return
  if (ws?.readyState !== WebSocket.OPEN) return

  _pendingAction = null
  _stepReady = false
  ws.send(JSON.stringify({ type: 'state', observation: getObservation() }))
}

// ─── Lidar sensor ─────────────────────────────────────────────────────────────
/**
 * Cast LIDAR_RAYS rays in a LIDAR_FOV arc in front of the robot.
 * Returns an array of LIDAR_RAYS distances (meters), clamped to LIDAR_RANGE.
 */
export function castLidar() {
  const scene = state.scene.three
  const maxDist = Array(LIDAR_RAYS).fill(LIDAR_RANGE)
  if (!scene) return maxDist

  // Build or reuse mesh cache (obstacles + walls — never robots)
  if (!_lidarMeshCache) {
    _lidarMeshCache = []
    scene.traverse(child => {
      if (!child.isMesh) return
      const name = (child.name || '').toLowerCase()
      // Exclude RL robot itself and debug robot
      if (name.includes('robot') || name.includes('airobot') || name.includes('debugrobot')) return
      _lidarMeshCache.push(child)
    })
  }

  const pos = getRobotPos()
  const eyePos = new THREE.Vector3(pos.x, pos.y + LIDAR_HEIGHT, pos.z)
  const heading = state.robot.rotation || 0
  const halfFov = LIDAR_FOV / 2
  const step = LIDAR_FOV / (LIDAR_RAYS - 1)

  _raycaster.near = 0.05
  _raycaster.far  = LIDAR_RANGE

  const distances = []
  for (let i = 0; i < LIDAR_RAYS; i++) {
    const rayAngle = heading - halfFov + step * i
    const dir = new THREE.Vector3(Math.sin(rayAngle), 0, Math.cos(rayAngle))
    _raycaster.set(eyePos, dir)

    const hits = _raycaster.intersectObjects(_lidarMeshCache, false)
    distances.push(hits.length > 0 ? parseFloat(hits[0].distance.toFixed(3)) : LIDAR_RANGE)
  }

  return distances
}

/** Call after scene changes (new objects added etc.) */
export function invalidateRLCache() {
  _lidarMeshCache = null
}
