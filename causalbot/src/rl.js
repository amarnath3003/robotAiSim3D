/**
 * rl.js — WebSocket bridge to Python/Gymnasium.
 *
 * Key design decisions:
 *  - controlMode switches to 'rl' ONLY when Python connects (ws.onopen)
 *  - controlMode reverts to 'ai' when Python disconnects
 *  - sendPromptRL works even before mode has switched (queues if not open)
 *  - ROOM_BOUNDS read lazily so window.MAZE_MODE is set by the time we need it
 *  - Lidar cache rebuilt after every reconnect and on demand
 *  - _pendingAction cleared BEFORE sending state to avoid stale re-use
 */

import * as THREE from 'three'
import { state, getRobotPos, setRobotPos } from './state.js'

// ─── Constants ────────────────────────────────────────────────────────────────
const LIDAR_RAYS    = 11
const LIDAR_FOV_RAD = 165 * (Math.PI / 180)
const LIDAR_RANGE   = 5.0
const LIDAR_HEIGHT  = 0.45
const MAX_LINEAR    = 2.5
const MAX_ANGULAR   = 2.0
const WS_URL        = 'ws://localhost:8765'
const RECONNECT_MS  = 3000

function getRoomBounds() {
  return window.MAZE_MODE
    ? { minX: -25, maxX: 25, minZ: -25, maxZ: 25 }
    : { minX: -2.8, maxX: 2.8, minZ: -2.8, maxZ: 2.8 }
}

// ─── Module state ─────────────────────────────────────────────────────────────
let _ws             = null
let _lidarCache     = null
let _pendingAction  = null
let _stepReady      = false
let _connected      = false
const _raycaster    = new THREE.Raycaster()

// ─── Telemetry (read by RL dashboard) ────────────────────────────────────────
const _telemetry = {
  connected:    false,
  mode:         'IDLE',        // 'IDLE' | 'EXECUTING'
  goal:         { x: null, z: null },
  lastAction:   { linear: 0, angular: 0 },
  lastLidar:    Array(11).fill(5.0),
  stepCount:    0,
  totalSteps:   0,
  lastReward:   0,
  robotPos:     { x: 0, z: 0 },
  distToGoal:   null,
  maxSteps:     1000,
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function initRL() {
  _connect()
}

export function updateRL(delta) {
  if (state.controlMode !== 'rl' || !_pendingAction) return

  const { linear, angular } = _pendingAction
  const bounds  = getRoomBounds()
  const heading = (state.robot.rotation || 0) + angular * delta
  state.robot.rotation = heading

  const nx = Math.max(bounds.minX, Math.min(bounds.maxX,
    state.robot.position[0] + Math.sin(heading) * linear * delta))
  const nz = Math.max(bounds.minZ, Math.min(bounds.maxZ,
    state.robot.position[2] + Math.cos(heading) * linear * delta))

  setRobotPos(nx, state.robot.position[1], nz)

  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({
      x: nx, y: state.robot.position[1], z: nz,
    })
  }
  _stepReady = true
}

export function flushRLState() {
  if (state.controlMode !== 'rl') return
  if (!_stepReady || !_pendingAction) return
  if (_ws?.readyState !== WebSocket.OPEN) return

  // Clear BEFORE send — fast Python reply must not see stale action
  _pendingAction = null
  _stepReady     = false

  const obs = _buildObservation()
  _telemetry.stepCount++
  _telemetry.totalSteps++
  _telemetry.robotPos  = { x: obs[0], z: obs[1] }
  _telemetry.lastLidar = obs.slice(3)

  _send({ type: 'state', observation: obs })
}

export function sendPromptRL(text) {
  // Works regardless of controlMode — user may type before Python connects
  if (_ws?.readyState === WebSocket.OPEN) {
    _send({ type: 'prompt', text })
    console.log('[RL] Prompt sent to Python:', text)
  } else {
    console.warn('[RL] sendPromptRL: Python not connected. Start run_agent.py.')
    // Surface the error in the UI
    const el = document.getElementById('status-bar')
    if (el) el.textContent = '⚠ Python agent not running — start run_agent.py first'
  }
}

export function invalidateRLCache() {
  _lidarCache = null
}

export function isRLConnected() {
  return _connected
}

export function getRLTelemetry() {
  return _telemetry
}

/** Send a goal directly from the dashboard (bypasses LLM). */
export function setRLGoalOverride(x, z) {
  if (_ws?.readyState === WebSocket.OPEN) {
    _send({ type: 'goal_override', x: parseFloat(x), z: parseFloat(z) })
    _telemetry.goal = { x: parseFloat(x), z: parseFloat(z) }
    _telemetry.mode = 'EXECUTING'
  }
}

/** Push updated env params to Python. */
export function setRLParams(params) {
  if (_ws?.readyState === WebSocket.OPEN) {
    _send({ type: 'params', ...params })
  }
  if (params.max_steps !== undefined) _telemetry.maxSteps = params.max_steps
}

// ─── Internal ─────────────────────────────────────────────────────────────────

function _connect() {
  _ws = new WebSocket(WS_URL)

  _ws.onopen = () => {
    _connected   = true
    _lidarCache  = null
    _telemetry.connected = true
    _telemetry.stepCount = 0
    console.log('[RL] Python connected → switching to RL mode')
    state.controlMode = 'rl'

    // Update status bar
    const el = document.getElementById('status-bar')
    if (el) el.textContent = '🤖 RL mode — type a goal in the input below'

    // Maze mode: tell Python where the goal is
    if (window.MAZE_MODE && window.getMazeGoal) {
      const g = window.getMazeGoal()
      if (g) {
        setTimeout(() => {
          _send({ type: 'maze_goal', x: g.x, z: g.z })
          console.log(`[RL] Sent maze_goal (${g.x.toFixed(2)}, ${g.z.toFixed(2)})`)
        }, 500)
      }
    }
  }

  _ws.onmessage = (event) => {
    let msg
    try { msg = JSON.parse(event.data) } catch { return }

    if (msg.type === 'reset') {
      _handleReset()
      _telemetry.mode = 'IDLE'
      _telemetry.stepCount = 0
    } else if (msg.type === 'action' && state.controlMode === 'rl') {
      const [lin, ang] = msg.action
      const clampedLin = Math.max(-MAX_LINEAR,  Math.min(MAX_LINEAR,  lin))
      const clampedAng = Math.max(-MAX_ANGULAR, Math.min(MAX_ANGULAR, ang))
      _pendingAction = { linear: clampedLin, angular: clampedAng }
      _telemetry.lastAction = { linear: clampedLin, angular: clampedAng }
      _telemetry.mode = 'EXECUTING'
      _stepReady = false
    } else if (msg.type === 'telemetry') {
      // Python pushes back reward + goal info
      if (msg.goal)   _telemetry.goal       = msg.goal
      if (msg.reward !== undefined) _telemetry.lastReward = msg.reward
      if (msg.dist   !== undefined) _telemetry.distToGoal = msg.dist
      if (msg.mode)  _telemetry.mode = msg.mode
    }
  }

  _ws.onclose = () => {
    const wasConnected = _connected
    _connected     = false
    _pendingAction = null
    _stepReady     = false
    _telemetry.connected = false
    _telemetry.mode = 'IDLE'

    if (wasConnected) {
      console.log('[RL] Python disconnected → reverting to AI mode')
      if (state.controlMode === 'rl') {
        state.controlMode = 'ai'
      }
      const el = document.getElementById('status-bar')
      if (el) el.textContent = 'Ready (Python disconnected)'
    }

    setTimeout(_connect, RECONNECT_MS)
  }

  _ws.onerror = () => {
    // onclose fires after onerror — suppress noise
  }
}

function _send(obj) {
  if (_ws?.readyState === WebSocket.OPEN) {
    _ws.send(JSON.stringify(obj))
  }
}

function _handleReset() {
  let sx = 0, sy = 0.35, sz = 1.8

  if (window.MAZE_MODE && window.getMazeStart) {
    const s = window.getMazeStart()
    if (s) { sx = s.x; sz = s.z }
  }

  setRobotPos(sx, sy, sz)
  state.robot.rotation   = 0
  state.robot.heldObject = null
  state.robot.armAngle   = 0
  state.robot.eyeColor   = 0x4488ff

  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({ x: sx, y: sy, z: sz })
  }

  _pendingAction = null
  _lidarCache    = null

  _send({ type: 'reset_done', observation: _buildObservation() })
}

function _buildObservation() {
  const p     = getRobotPos()
  const lidar = castLidar()
  _telemetry.robotPos  = { x: p.x, z: p.z }
  _telemetry.lastLidar = lidar
  return [p.x, p.z, state.robot.rotation || 0, ...lidar]
}

// ─── Lidar ────────────────────────────────────────────────────────────────────

export function castLidar() {
  const scene   = state.scene.three
  const maxDist = Array(LIDAR_RAYS).fill(LIDAR_RANGE)
  if (!scene) return maxDist

  if (!_lidarCache) {
    _lidarCache = []
    scene.traverse(child => {
      if (!child.isMesh) return
      const n = (child.name || '').toLowerCase()
      if (n.includes('robot')) return
      _lidarCache.push(child)
    })
  }

  const pos     = getRobotPos()
  const eyePos  = new THREE.Vector3(pos.x, pos.y + LIDAR_HEIGHT, pos.z)
  const heading = state.robot.rotation || 0
  const halfFov = LIDAR_FOV_RAD / 2
  const step    = LIDAR_FOV_RAD / (LIDAR_RAYS - 1)

  _raycaster.near = 0.05
  _raycaster.far  = LIDAR_RANGE

  return Array.from({ length: LIDAR_RAYS }, (_, i) => {
    const angle = heading - halfFov + step * i
    const dir   = new THREE.Vector3(Math.sin(angle), 0, Math.cos(angle))
    _raycaster.set(eyePos, dir)
    const hits  = _raycaster.intersectObjects(_lidarCache, false)
    return hits.length ? parseFloat(hits[0].distance.toFixed(3)) : LIDAR_RANGE
  })
}