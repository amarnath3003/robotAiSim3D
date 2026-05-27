/**
 * rl.js — WebSocket bridge to Python/Gymnasium.
 *
 * Key design decisions:
 *  - controlMode switches to 'rl' ONLY when Python connects (ws.onopen)
 *  - controlMode reverts to 'ai' when Python disconnects
 *  - Each episode reset: old dynamic walls removed, new walls + goal ball rendered
 *  - Action space: [linear, angular, arm_rotation, jump] — 4-dim
 *  - DEATH (collision): red screen flash, death counter incremented
 *  - SUCCESS (goal): green flash, success counter incremented
 *  - ROOM_BOUNDS read lazily so window.MAZE_MODE is already set
 *  - _pendingAction cleared BEFORE sending state to avoid stale re-use
 */

import * as THREE from 'three'
import { state, getRobotPos, setRobotPos } from './state.js'
import { addRLWalls, removeRLWalls } from './physics.js'

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

// Dynamic episode assets (cleared every reset)
let _wallMeshes     = []     // Three.js meshes
let _wallBodies     = []     // Rapier body handles
let _goalMarker     = null   // Three.js goal sphere
let _goalBeam       = null   // vertical glow beam

// ─── Telemetry (read by RL dashboard) ────────────────────────────────────────
const _telemetry = {
  connected:    false,
  mode:         'IDLE',
  goal:         { x: null, z: null },
  lastAction:   { linear: 0, angular: 0, armRot: 0, jump: 0 },
  lastLidar:    Array(11).fill(5.0),
  stepCount:    0,
  totalSteps:   0,
  lastReward:   0,
  robotPos:     { x: 0, z: 0 },
  distToGoal:   null,
  maxSteps:     500,
  deaths:       0,
  successes:    0,
  episode:      0,
  epReward:     0,
}

// ─── Public API ───────────────────────────────────────────────────────────────

export function initRL() {
  _createGoalMarker()
  _connect()
}

export function updateRL(delta) {
  if (state.controlMode !== 'rl' || !_pendingAction) return

  const { linear, angular, armRot, jump } = _pendingAction
  const bounds  = getRoomBounds()
  const heading = (state.robot.rotation || 0) + angular * delta
  state.robot.rotation = heading

  const nx = Math.max(bounds.minX, Math.min(bounds.maxX,
    state.robot.position[0] + Math.sin(heading) * linear * delta))
  const nz = Math.max(bounds.minZ, Math.min(bounds.maxZ,
    state.robot.position[2] + Math.cos(heading) * linear * delta))

  setRobotPos(nx, state.robot.position[1], nz)

  // Apply arm rotation
  if (armRot !== undefined) {
    state.robot.armAngle = armRot
  }

  // Apply jump (small upward impulse via kinematic body)
  if (jump > 0.5 && state.robot._body) {
    const p = state.robot.position
    // Nudge upward — gravity will bring it back
    setRobotPos(p[0], Math.min(p[1] + 0.12, 1.2), p[2])
  }

  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({
      x: state.robot.position[0],
      y: state.robot.position[1],
      z: state.robot.position[2],
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
  if (_ws?.readyState === WebSocket.OPEN) {
    _send({ type: 'prompt', text })
    console.log('[RL] Prompt sent to Python:', text)
  } else {
    console.warn('[RL] sendPromptRL: Python not connected. Start run_agent.py.')
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
    _moveGoalMarker(parseFloat(x), parseFloat(z))
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

    const el = document.getElementById('status-bar')
    if (el) el.textContent = '🤖 RL mode — type any goal command in the input below'

    if (window.MAZE_MODE && window.getMazeGoal) {
      const g = window.getMazeGoal()
      if (g) {
        setTimeout(() => {
          _send({ type: 'maze_goal', x: g.x, z: g.z })
        }, 500)
      }
    }
  }

  _ws.onmessage = (event) => {
    let msg
    try { msg = JSON.parse(event.data) } catch { return }

    if (msg.type === 'reset') {
      _handleReset(msg)
      _telemetry.mode = 'IDLE'
      _telemetry.stepCount = 0

    } else if (msg.type === 'action' && state.controlMode === 'rl') {
      const [lin, ang, arm, jmp] = msg.action
      const clampedLin = Math.max(0,          Math.min(MAX_LINEAR,  lin ?? 0))
      const clampedAng = Math.max(-MAX_ANGULAR, Math.min(MAX_ANGULAR, ang ?? 0))
      const armRot     = Math.max(-Math.PI,   Math.min(Math.PI,   arm ?? 0))
      const jump       = Math.max(0,           Math.min(1,         jmp ?? 0))
      _pendingAction = { linear: clampedLin, angular: clampedAng, armRot, jump }
      _telemetry.lastAction = { linear: clampedLin, angular: clampedAng, armRot, jump }
      _telemetry.mode = 'EXECUTING'
      _stepReady = false

    } else if (msg.type === 'telemetry') {
      if (msg.goal)                _telemetry.goal       = msg.goal
      if (msg.reward !== undefined) _telemetry.lastReward = msg.reward
      if (msg.dist   !== undefined) _telemetry.distToGoal = msg.dist
      if (msg.mode)                _telemetry.mode       = msg.mode
      if (msg.deaths !== undefined) _telemetry.deaths     = msg.deaths
      if (msg.successes !== undefined) _telemetry.successes = msg.successes
      if (msg.episode !== undefined)   _telemetry.episode  = msg.episode
      if (msg.ep_reward !== undefined) _telemetry.epReward = msg.ep_reward

      // Trigger visual effects based on outcome
      if (msg.outcome === 'death')   _triggerDeathFlash()
      if (msg.outcome === 'success') _triggerSuccessFlash()
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
      if (state.controlMode === 'rl') state.controlMode = 'ai'
      const el = document.getElementById('status-bar')
      if (el) el.textContent = 'Ready (Python disconnected)'
    }

    setTimeout(_connect, RECONNECT_MS)
  }

  _ws.onerror = () => { /* onclose fires after onerror */ }
}

function _send(obj) {
  if (_ws?.readyState === WebSocket.OPEN) {
    _ws.send(JSON.stringify(obj))
  }
}

// ─── Episode Reset — build walls + place goal ─────────────────────────────────

function _handleReset(msg) {
  const scene = state.scene.three

  // ── 1. Remove old dynamic walls ──────────────────────────────────────────
  for (const m of _wallMeshes) {
    if (m.parent) scene?.remove(m)
    m.geometry?.dispose()
    m.material?.dispose()
  }
  _wallMeshes = []

  removeRLWalls(_wallBodies)
  _wallBodies = []

  // ── 2. Reset robot position ───────────────────────────────────────────────
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

  // ── 3. Place goal ball at new position ───────────────────────────────────
  const goal = msg.goal
  if (goal) {
    _telemetry.goal = { x: goal.x, z: goal.z }
    _moveGoalMarker(goal.x, goal.z)

    // Also move the world ball mesh if it exists
    const ballMesh = scene?.getObjectByName('object_ball')
    if (ballMesh) {
      ballMesh.position.set(goal.x, 0.2, goal.z)
    }
    // Move its Rapier body too
    const ballObj = state.world.objects['object_ball']
    if (ballObj?._body) {
      ballObj._body.setTranslation({ x: goal.x, y: 0.2, z: goal.z }, true)
      ballObj._body.setLinvel({ x: 0, y: 0, z: 0 }, true)
      ballObj._body.setAngvel({ x: 0, y: 0, z: 0 }, true)
    }
    ballObj && (ballObj.position[0] = goal.x, ballObj.position[1] = 0.2, ballObj.position[2] = goal.z)
  }

  // ── 4. Build new random walls ─────────────────────────────────────────────
  const walls = msg.walls || []
  if (scene) {
    const wallMat = new THREE.MeshStandardMaterial({
      color: 0x334466,
      roughness: 0.85,
      metalness: 0.1,
      emissive: 0x0a1122,
      emissiveIntensity: 0.3,
    })

    for (const w of walls) {
      const geo  = new THREE.BoxGeometry(w.w, w.h, w.d)
      const mesh = new THREE.Mesh(geo, wallMat.clone())
      mesh.position.set(w.x, w.h / 2, w.z)
      mesh.castShadow    = true
      mesh.receiveShadow = true
      mesh.name = `rl_wall_${_wallMeshes.length}`
      scene.add(mesh)
      _wallMeshes.push(mesh)
    }
  }

  // Add Rapier colliders for new walls
  _wallBodies = addRLWalls(walls)

  // Invalidate lidar cache — scene changed
  _lidarCache = null
  invalidateRLCache()

  // ── 5. Confirm reset done ─────────────────────────────────────────────────
  _send({ type: 'reset_done', observation: _buildObservation() })

  console.log(`[RL] Episode reset | Goal (${goal?.x?.toFixed(2)}, ${goal?.z?.toFixed(2)}) | ${walls.length} walls`)
}

// ─── Goal marker (glowing sphere) ────────────────────────────────────────────

function _createGoalMarker() {
  const scene = state.scene.three
  if (!scene) return

  // Glowing sphere
  const geo = new THREE.SphereGeometry(0.18, 16, 16)
  const mat = new THREE.MeshStandardMaterial({
    color:             0xff6600,
    emissive:          0xff3300,
    emissiveIntensity: 2.0,
    roughness:         0.2,
    metalness:         0.0,
  })
  _goalMarker = new THREE.Mesh(geo, mat)
  _goalMarker.name    = 'rl_goal_marker'
  _goalMarker.visible = false
  scene.add(_goalMarker)

  // Vertical beam
  const beamGeo = new THREE.CylinderGeometry(0.02, 0.06, 3.0, 8)
  const beamMat = new THREE.MeshBasicMaterial({
    color:       0xff6600,
    transparent: true,
    opacity:     0.25,
  })
  _goalBeam         = new THREE.Mesh(beamGeo, beamMat)
  _goalBeam.name    = 'rl_goal_beam'
  _goalBeam.visible = false
  scene.add(_goalBeam)
}

function _moveGoalMarker(x, z) {
  if (_goalMarker) {
    _goalMarker.position.set(x, 0.18, z)
    _goalMarker.visible = true
  }
  if (_goalBeam) {
    _goalBeam.position.set(x, 1.5, z)
    _goalBeam.visible = true
  }
}

// ─── Flash effects ────────────────────────────────────────────────────────────

function _triggerDeathFlash() {
  let el = document.getElementById('rl-flash-overlay')
  if (!el) {
    el = document.createElement('div')
    el.id = 'rl-flash-overlay'
    el.style.cssText = `
      position:fixed;inset:0;pointer-events:none;z-index:9999;
      background:rgba(220,30,30,0.0);transition:none;
    `
    document.body.appendChild(el)
  }
  // Flash sequence: pop to full → fade
  el.style.transition = 'none'
  el.style.background = 'rgba(220,30,30,0.65)'
  requestAnimationFrame(() => {
    el.style.transition = 'background 0.5s ease-out'
    el.style.background = 'rgba(220,30,30,0.0)'
  })

  // Show ☠ death counter in status bar
  const status = document.getElementById('status-bar')
  if (status) {
    status.textContent = `☠ Death #${_telemetry.deaths} — resetting...`
    status.style.color = '#ff4444'
    setTimeout(() => { status.style.color = '' }, 1500)
  }

  // Shake the canvas
  const canvas = document.querySelector('canvas')
  if (canvas) {
    canvas.style.transition = 'none'
    canvas.style.transform  = 'translate(-4px, 2px)'
    setTimeout(() => {
      canvas.style.transition = 'transform 0.3s ease-out'
      canvas.style.transform  = ''
    }, 60)
  }
}

function _triggerSuccessFlash() {
  let el = document.getElementById('rl-flash-overlay')
  if (!el) {
    el = document.createElement('div')
    el.id = 'rl-flash-overlay'
    el.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:9999;'
    document.body.appendChild(el)
  }
  el.style.transition = 'none'
  el.style.background = 'rgba(50,210,120,0.55)'
  requestAnimationFrame(() => {
    el.style.transition = 'background 0.7s ease-out'
    el.style.background = 'rgba(50,210,120,0.0)'
  })

  const status = document.getElementById('status-bar')
  if (status) {
    status.textContent = `✓ Goal reached! Success #${_telemetry.successes}`
    status.style.color = '#44ff88'
    setTimeout(() => { status.style.color = '' }, 2000)
  }
}

// ─── Observation builder ──────────────────────────────────────────────────────

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
      if (n === 'rl_goal_marker' || n === 'rl_goal_beam') return
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