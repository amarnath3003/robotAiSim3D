/**
 * rl.js — WebSocket bridge between the Three.js physics sim and Python/Gymnasium.
 *
 * Fixes over previous version:
 *  1. ROOM_BOUNDS read lazily (after main.js sets window.MAZE_MODE) — not at import time
 *  2. Lidar cache invalidated after maze walls are added
 *  3. controlMode defaults to 'ai'; RL mode only activates when Python connects
 *  4. _pendingAction cleared correctly after flushRLState so stale actions don't repeat
 *  5. Reset handler respects maze start position when available
 *  6. sendPromptRL is a no-op in non-RL mode (safe to call regardless)
 */

import * as THREE from 'three'
import { state, getRobotPos, setRobotPos } from './state.js'

// ─── Constants ────────────────────────────────────────────────────────────────
const LIDAR_RAYS   = 11
const LIDAR_FOV    = 165 * (Math.PI / 180)
const LIDAR_RANGE  = 5.0
const LIDAR_HEIGHT = 0.45
const MAX_LINEAR   = 2.5
const MAX_ANGULAR  = 2.0

// Room bounds read lazily so MAZE_MODE is already set by main.js
function getRoomBounds() {
  return window.MAZE_MODE
    ? { minX: -25, maxX: 25, minZ: -25, maxZ: 25 }
    : { minX: -2.8, maxX: 2.8, minZ: -2.8, maxZ: 2.8 }
}

// ─── Module state ─────────────────────────────────────────────────────────────
let ws              = null
let _lidarCache     = null   // cached meshes for raycasting
let _pendingAction  = null   // set by Python 'action' message
let _stepReady      = false  // set after physics step, cleared after flush
let _raycaster      = new THREE.Raycaster()
let _connected      = false  // true while Python client is connected

// ─── Init ─────────────────────────────────────────────────────────────────────
export function initRL() {
  _tryConnect()
}

function _tryConnect() {
  ws = new WebSocket('ws://localhost:8765')
  ws.binaryType = 'arraybuffer'

  ws.onopen = () => {
    _connected = true
    _lidarCache = null  // rebuild after reconnect; scene may have changed
    console.log('[RL] Python client connected — switching to RL mode')

    // Only switch to RL mode when Python is actually connected
    state.controlMode = 'rl'

    if (window.MAZE_MODE && window.getMazeGoal) {
      const goal = window.getMazeGoal()
      if (goal) {
        // Small delay so Python ws handler is ready
        setTimeout(() => {
          _send({ type: 'maze_goal', x: goal.x, z: goal.z })
          console.log(`[RL] Sent maze_goal: (${goal.x.toFixed(2)}, ${goal.z.toFixed(2)})`)
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
            linear:  Math.max(-MAX_LINEAR,  Math.min(MAX_LINEAR,  linear)),
            angular: Math.max(-MAX_ANGULAR, Math.min(MAX_ANGULAR, angular)),
          }
          _stepReady = false
        }
      }
    } catch (e) {
      console.error('[RL] Parse error:', e)
    }
  }

  ws.onclose = () => {
    if (_connected) {
      console.log('[RL] Python disconnected — reverting to AI mode')
    }
    _connected      = false
    _pendingAction  = null
    _stepReady      = false

    // Revert to AI mode so the LLM input still works
    if (state.controlMode === 'rl') {
      state.controlMode = 'ai'
    }

    // Reconnect after 3 s — Python may restart
    setTimeout(_tryConnect, 3000)
  }

  ws.onerror = () => {
    // Suppress error spam when Python isn't running; onclose fires next
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function _send(obj) {
  if (ws?.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(obj))
  }
}

// ─── Reset ────────────────────────────────────────────────────────────────────
function _handleReset() {
  // In maze mode use the maze start; otherwise room centre-front
  let sx = 0, sy = 0.35, sz = 1.8
  if (window.MAZE_MODE && window.getMazeStart) {
    const s = window.getMazeStart()
    sx = s.x; sz = s.z
  }

  setRobotPos(sx, sy, sz)
  state.robot.rotation  = 0
  state.robot.heldObject = null
  state.robot.armAngle  = 0

  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({ x: sx, y: sy, z: sz })
  }

  // Move debug robot out of the way
  state.debugRobot.position[0] = 1.5
  state.debugRobot.position[1] = 0.35
  state.debugRobot.position[2] = 1.5

  _pendingAction = null
  _lidarCache    = null   // scene may have changed between episodes

  _send({ type: 'reset_done', observation: _getObservation() })
}

// ─── Observation ──────────────────────────────────────────────────────────────
function _getObservation() {
  const p = getRobotPos()
  return [p.x, p.z, state.robot.rotation || 0, ...castLidar()]
}

// ─── Physics update (called every substep) ────────────────────────────────────
export function updateRL(delta) {
  if (state.controlMode !== 'rl' || !_pendingAction) return

  const { linear, angular } = _pendingAction

  // Integrate heading
  state.robot.rotation = (state.robot.rotation || 0) + angular * delta

  // World-space displacement
  const heading = state.robot.rotation
  const bounds  = getRoomBounds()
  const newX = Math.max(bounds.minX, Math.min(bounds.maxX,
    state.robot.position[0] + Math.sin(heading) * linear * delta))
  const newZ = Math.max(bounds.minZ, Math.min(bounds.maxZ,
    state.robot.position[2] + Math.cos(heading) * linear * delta))

  setRobotPos(newX, state.robot.position[1], newZ)

  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({
      x: newX, y: state.robot.position[1], z: newZ,
    })
  }

  _stepReady = true
}

// ─── Flush — called once per rendered frame (after all substeps) ──────────────
export function flushRLState() {
  if (state.controlMode !== 'rl') return
  if (!_stepReady || !_pendingAction) return
  if (ws?.readyState !== WebSocket.OPEN) return

  // Clear BEFORE sending so a fast Python reply doesn't double-clear
  _pendingAction = null
  _stepReady     = false

  _send({ type: 'state', observation: _getObservation() })
}

// ─── Prompt forwarding (used by ui.js in RL mode) ─────────────────────────────
export function sendPromptRL(text) {
  if (state.controlMode !== 'rl') return
  _send({ type: 'prompt', text })
}

// ─── Lidar sensor ─────────────────────────────────────────────────────────────
/**
 * Cast LIDAR_RAYS rays across LIDAR_FOV in front of the robot.
 * Returns array of distances (m), capped at LIDAR_RANGE.
 */
export function castLidar() {
  const scene   = state.scene.three
  const maxDist = Array(LIDAR_RAYS).fill(LIDAR_RANGE)
  if (!scene) return maxDist

  // Build or reuse mesh cache — excludes both robot meshes
  if (!_lidarCache) {
    _lidarCache = []
    scene.traverse(child => {
      if (!child.isMesh) return
      const n = (child.name || '').toLowerCase()
      if (n.includes('robot')) return
      _lidarCache.push(child)
    })
  }

  const pos    = getRobotPos()
  const eyePos = new THREE.Vector3(pos.x, pos.y + LIDAR_HEIGHT, pos.z)
  const heading = state.robot.rotation || 0
  const halfFov = LIDAR_FOV / 2
  const step    = LIDAR_FOV / (LIDAR_RAYS - 1)

  _raycaster.near = 0.05
  _raycaster.far  = LIDAR_RANGE

  return Array.from({ length: LIDAR_RAYS }, (_, i) => {
    const angle = heading - halfFov + step * i
    const dir   = new THREE.Vector3(Math.sin(angle), 0, Math.cos(angle))
    _raycaster.set(eyePos, dir)
    const hits = _raycaster.intersectObjects(_lidarCache, false)
    return hits.length > 0 ? parseFloat(hits[0].distance.toFixed(3)) : LIDAR_RANGE
  })
}

/** Call after adding objects or maze walls to the scene */
export function invalidateRLCache() {
  _lidarCache = null
}

/** True while a Python Gymnasium client is connected */
export function isRLConnected() {
  return _connected
}