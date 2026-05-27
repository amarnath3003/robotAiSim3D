/**
 * rl.js — WebSocket bridge to Python/Gymnasium.
 *
 * Bugs fixed:
 *  B1: lidar cache now invalidated AFTER walls are added to the scene
 *  B5: ball position update uses proper if-block (not comma operator)
 *  B7: gravity recovery for jump — Y lerps back to floor level each frame
 *  B10: death flash uses double-RAF to guarantee browser sees initial state
 *
 * Key design decisions:
 *  - controlMode → 'rl' ONLY on ws.onopen; reverts to 'ai' on ws.onclose
 *  - Episode reset: old walls destroyed, new walls + goal ball rendered
 *  - Action space: [linear, angular, arm_rotation, jump] — 4-dim
 *  - ROOM_BOUNDS read lazily (window.MAZE_MODE already set by main.js)
 *  - _pendingAction cleared BEFORE flush send to prevent stale re-use
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
const FLOOR_Y       = 0.35   // nominal robot floor height
const WS_URL        = 'ws://localhost:8765'
const RECONNECT_MS  = 3000

function getRoomBounds() {
  return window.MAZE_MODE
    ? { minX: -25, maxX: 25, minZ: -25, maxZ: 25 }
    : { minX: -2.8, maxX: 2.8, minZ: -2.8, maxZ: 2.8 }
}

// ─── Module state ─────────────────────────────────────────────────────────────
let _ws            = null
let _lidarCache    = null
let _pendingAction = null
let _stepReady     = false
let _connected     = false
const _raycaster   = new THREE.Raycaster()

// Dynamic episode assets (rebuilt every reset)
let _wallMeshes    = []    // Three.js wall meshes
let _wallBodies    = []    // Rapier rigid body handles
let _goalMarker    = null  // glowing sphere at goal
let _goalBeam      = null  // vertical light beam
let _goalPulse     = 0     // pulsing animation phase

// Episode history (last 10 outcomes) for client-side display
const _history = []
const HISTORY_LEN = 10

// ─── Telemetry (read by RL dashboard) ────────────────────────────────────────
const _telemetry = {
  connected:   false,
  mode:        'IDLE',
  goal:        { x: null, z: null },
  lastAction:  { linear: 0, angular: 0, armRot: 0, jump: 0 },
  lastLidar:   Array(11).fill(5.0),
  stepCount:   0,
  totalSteps:  0,
  lastReward:  0,
  robotPos:    { x: 0, z: 0 },
  distToGoal:  null,
  maxSteps:    500,
  deaths:      0,
  successes:   0,
  episode:     0,
  epReward:    0,
  winRate:     0,
  history:     [],  // client-side copy
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

  // B7 fix: gravity recovery — lerp Y back toward floor
  const curY  = state.robot.position[1]
  const newY  = jump > 0.5
    ? Math.min(curY + 0.18, FLOOR_Y + 0.5)   // jump nudge
    : curY + (FLOOR_Y - curY) * Math.min(1, delta * 8)  // gravity

  setRobotPos(nx, newY, nz)

  // Arm rotation
  state.robot.armAngle = armRot

  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({
      x: state.robot.position[0],
      y: state.robot.position[1],
      z: state.robot.position[2],
    })
  }

  // Animate goal marker pulse
  _goalPulse += delta * 3.0
  if (_goalMarker) {
    const s = 1.0 + 0.12 * Math.sin(_goalPulse)
    _goalMarker.scale.setScalar(s)
    _goalMarker.material.emissiveIntensity = 1.5 + 0.8 * Math.sin(_goalPulse)
  }
  if (_goalBeam) {
    _goalBeam.material.opacity = 0.15 + 0.12 * Math.sin(_goalPulse * 0.7)
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
    console.log('[RL] Prompt →', text)
  } else {
    console.warn('[RL] Python not connected — start run_agent.py')
    const el = document.getElementById('status-bar')
    if (el) el.textContent = '⚠ Python agent not running — start run_agent.py first'
  }
}

export function invalidateRLCache() {
  _lidarCache = null
}

export function isRLConnected() { return _connected }
export function getRLTelemetry() { return _telemetry }

export function setRLGoalOverride(x, z) {
  if (_ws?.readyState === WebSocket.OPEN) {
    _send({ type: 'goal_override', x: +x, z: +z })
    _telemetry.goal = { x: +x, z: +z }
    _moveGoalMarker(+x, +z)
  }
}

export function setRLParams(params) {
  if (_ws?.readyState === WebSocket.OPEN) {
    _send({ type: 'params', ...params })
  }
  if (params.max_steps !== undefined) _telemetry.maxSteps = params.max_steps
}

// ─── WebSocket connection ─────────────────────────────────────────────────────

function _connect() {
  _ws = new WebSocket(WS_URL)

  _ws.onopen = () => {
    _connected             = true
    _lidarCache            = null
    _telemetry.connected   = true
    _telemetry.stepCount   = 0
    state.controlMode      = 'rl'

    console.log('[RL] Python connected → RL mode')
    const el = document.getElementById('status-bar')
    if (el) el.textContent = '🤖 RL mode active — type any goal command below'
  }

  _ws.onmessage = ({ data }) => {
    let msg
    try { msg = JSON.parse(data) } catch { return }

    if (msg.type === 'reset') {
      _handleReset(msg)
      _telemetry.mode      = 'IDLE'
      _telemetry.stepCount = 0

    } else if (msg.type === 'action' && state.controlMode === 'rl') {
      const [lin = 0, ang = 0, arm = 0, jmp = 0] = msg.action
      const clampedLin = Math.max(0,           Math.min(MAX_LINEAR,  lin))
      const clampedAng = Math.max(-MAX_ANGULAR, Math.min(MAX_ANGULAR, ang))
      const armRot     = Math.max(-Math.PI,    Math.min(Math.PI,    arm))
      const jump       = Math.max(0,           Math.min(1,          jmp))
      _pendingAction = { linear: clampedLin, angular: clampedAng, armRot, jump }
      _telemetry.lastAction = { linear: clampedLin, angular: clampedAng, armRot, jump }
      _telemetry.mode = 'EXECUTING'
      _stepReady      = false

    } else if (msg.type === 'telemetry') {
      if (msg.goal      !== undefined) _telemetry.goal       = msg.goal
      if (msg.reward    !== undefined) _telemetry.lastReward = msg.reward
      if (msg.dist      !== undefined) _telemetry.distToGoal = msg.dist
      if (msg.mode      !== undefined) _telemetry.mode       = msg.mode
      if (msg.deaths    !== undefined) _telemetry.deaths     = msg.deaths
      if (msg.successes !== undefined) _telemetry.successes  = msg.successes
      if (msg.episode   !== undefined) _telemetry.episode    = msg.episode
      if (msg.ep_reward !== undefined) _telemetry.epReward   = msg.ep_reward
      if (msg.win_rate  !== undefined) _telemetry.winRate    = msg.win_rate

      // Client-side history tracking
      if (msg.outcome) {
        _history.push(msg.outcome)
        if (_history.length > HISTORY_LEN) _history.shift()
        _telemetry.history = [..._history]
      }

      if (msg.outcome === 'death')   _triggerDeathFlash()
      if (msg.outcome === 'success') _triggerSuccessFlash()
    }
  }

  _ws.onclose = () => {
    const was = _connected
    _connected             = false
    _pendingAction         = null
    _stepReady             = false
    _telemetry.connected   = false
    _telemetry.mode        = 'IDLE'

    if (was) {
      console.log('[RL] Python disconnected → AI mode')
      if (state.controlMode === 'rl') state.controlMode = 'ai'
      const el = document.getElementById('status-bar')
      if (el) el.textContent = 'Ready (Python disconnected)'
    }
    setTimeout(_connect, RECONNECT_MS)
  }

  _ws.onerror = () => {}  // onclose fires next
}

function _send(obj) {
  if (_ws?.readyState === WebSocket.OPEN) {
    _ws.send(JSON.stringify(obj))
  }
}

// ─── Episode reset — walls + goal ball ───────────────────────────────────────

function _handleReset(msg) {
  const scene = state.scene.three

  // 1. Remove old wall meshes
  for (const m of _wallMeshes) {
    if (m.parent) scene?.remove(m)
    m.geometry?.dispose()
    if (Array.isArray(m.material)) m.material.forEach(mat => mat.dispose())
    else m.material?.dispose()
  }
  _wallMeshes = []

  // Remove old Rapier bodies
  removeRLWalls(_wallBodies)
  _wallBodies = []

  // 2. Reset robot to start
  const sx = 0, sy = FLOOR_Y, sz = 1.8
  setRobotPos(sx, sy, sz)
  state.robot.rotation   = 0
  state.robot.heldObject = null
  state.robot.armAngle   = 0
  state.robot.eyeColor   = 0x4488ff

  if (state.robot._body) {
    state.robot._body.setNextKinematicTranslation({ x: sx, y: sy, z: sz })
  }

  // Also reset the Three.js robot mesh rotation
  const robotMesh = scene?.getObjectByName('aiRobot')
  if (robotMesh) {
    robotMesh.rotation.y = 0
  }

  _pendingAction = null

  // 3. Move goal ball + goal marker to new position
  const goal = msg.goal
  if (goal) {
    _telemetry.goal = { x: goal.x, z: goal.z }
    _moveGoalMarker(goal.x, goal.z)

    // Move the physical ball mesh
    const ballMesh = scene?.getObjectByName('object_ball')
    if (ballMesh) {
      ballMesh.position.set(goal.x, 0.18, goal.z)
    }

    // B5 fix: proper if-block (not comma operator)
    const ballObj = state.world.objects['object_ball']
    if (ballObj) {
      if (ballObj._body) {
        ballObj._body.setTranslation({ x: goal.x, y: 0.18, z: goal.z }, true)
        ballObj._body.setLinvel(     { x: 0, y: 0, z: 0 }, true)
        ballObj._body.setAngvel(     { x: 0, y: 0, z: 0 }, true)
      }
      ballObj.position[0] = goal.x
      ballObj.position[1] = 0.18
      ballObj.position[2] = goal.z
    }
  }

  // 4. Create new wall meshes
  const walls = msg.walls || []
  if (scene && walls.length > 0) {
    for (let i = 0; i < walls.length; i++) {
      const w   = walls[i]
      const geo = new THREE.BoxGeometry(w.w, w.h, w.d)
      // Alternate two slightly different materials for visual variety
      const mat = new THREE.MeshStandardMaterial({
        color:             i % 2 === 0 ? 0x2a3a55 : 0x1e3040,
        roughness:         0.9,
        metalness:         0.05,
        emissive:          0x0a1830,
        emissiveIntensity: 0.4,
      })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(w.x, w.h / 2, w.z)
      mesh.castShadow    = true
      mesh.receiveShadow = true
      mesh.name          = `rl_wall_${i}`
      scene.add(mesh)
      _wallMeshes.push(mesh)
    }
  }

  // 5. Add Rapier colliders for new walls
  _wallBodies = addRLWalls(walls)

  // B1 fix: invalidate lidar cache AFTER walls are in the scene
  _lidarCache = null

  // 6. Confirm reset to Python
  const obs = _buildObservation()
  _send({ type: 'reset_done', observation: obs })

  console.log(`[RL] Reset | goal (${goal?.x?.toFixed(2)}, ${goal?.z?.toFixed(2)}) | ${walls.length} walls`)
}

// ─── Goal marker ──────────────────────────────────────────────────────────────

function _createGoalMarker() {
  const scene = state.scene.three
  if (!scene) {
    // Retry once scene is ready
    requestAnimationFrame(_createGoalMarker)
    return
  }

  const geo = new THREE.SphereGeometry(0.18, 20, 20)
  const mat = new THREE.MeshStandardMaterial({
    color:             0xff7700,
    emissive:          0xff4400,
    emissiveIntensity: 2.0,
    roughness:         0.15,
    metalness:         0.0,
  })
  _goalMarker         = new THREE.Mesh(geo, mat)
  _goalMarker.name    = 'rl_goal_marker'
  _goalMarker.visible = false
  scene.add(_goalMarker)

  // Vertical beam
  const beamGeo = new THREE.CylinderGeometry(0.015, 0.05, 4.0, 8)
  const beamMat = new THREE.MeshBasicMaterial({
    color:       0xff7700,
    transparent: true,
    opacity:     0.22,
    depthWrite:  false,
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
    _goalBeam.position.set(x, 2.0, z)
    _goalBeam.visible = true
  }
}

// ─── Flash effects ────────────────────────────────────────────────────────────

function _getFlashEl() {
  let el = document.getElementById('rl-flash-overlay')
  if (!el) {
    el = document.createElement('div')
    el.id = 'rl-flash-overlay'
    el.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:9999;background:transparent;'
    document.body.appendChild(el)
  }
  return el
}

function _triggerDeathFlash() {
  const el = _getFlashEl()

  // B10 fix: double-RAF guarantees browser paints the initial red state before transition
  el.style.transition = 'none'
  el.style.background = 'rgba(210, 20, 20, 0.7)'
  requestAnimationFrame(() => requestAnimationFrame(() => {
    el.style.transition = 'background 0.55s ease-out'
    el.style.background = 'transparent'
  }))

  // Status bar
  const status = document.getElementById('status-bar')
  if (status) {
    status.textContent = `☠ Death #${_telemetry.deaths} — restarting...`
    status.style.color = '#ff4444'
    setTimeout(() => { status.style.color = '' }, 1500)
  }

  // Canvas shake (CSS transform)
  const canvas = document.querySelector('canvas')
  if (canvas) {
    canvas.style.transition = 'none'
    canvas.style.transform  = 'translate(-5px, 3px) rotate(-0.3deg)'
    setTimeout(() => {
      canvas.style.transition = 'transform 0.35s ease-out'
      canvas.style.transform  = ''
    }, 50)
  }

  // Brief eye colour change on robot
  state.robot.eyeColor = 0xff0000
  setTimeout(() => { state.robot.eyeColor = 0x4488ff }, 600)
}

function _triggerSuccessFlash() {
  const el = _getFlashEl()
  el.style.transition = 'none'
  el.style.background = 'rgba(30, 220, 120, 0.55)'
  requestAnimationFrame(() => requestAnimationFrame(() => {
    el.style.transition = 'background 0.8s ease-out'
    el.style.background = 'transparent'
  }))

  const status = document.getElementById('status-bar')
  if (status) {
    status.textContent = `✓ Goal reached! #${_telemetry.successes}`
    status.style.color = '#44ff88'
    setTimeout(() => { status.style.color = '' }, 2000)
  }

  // Happy eye colour
  state.robot.eyeColor = 0x00ff88
  setTimeout(() => { state.robot.eyeColor = 0x4488ff }, 1200)
}

// ─── Observation builder ──────────────────────────────────────────────────────

function _buildObservation() {
  const p     = getRobotPos()
  const lidar = castLidar()
  _telemetry.robotPos  = { x: p.x, z: p.z }
  _telemetry.lastLidar = lidar
  // Format: [rx, rz, heading, lidar×11]  — Python _ingest_obs expects this
  return [p.x, p.z, state.robot.rotation || 0, ...lidar]
}

// ─── Lidar ────────────────────────────────────────────────────────────────────

export function castLidar() {
  const scene    = state.scene.three
  const maxDist  = Array(LIDAR_RAYS).fill(LIDAR_RANGE)
  if (!scene) return maxDist

  // Build mesh cache (lazily, invalidated on scene changes)
  if (!_lidarCache) {
    _lidarCache = []
    scene.traverse(child => {
      if (!child.isMesh) return
      const n = (child.name || '').toLowerCase()
      // Exclude robot itself and goal visual markers
      if (n.includes('robot'))      return
      if (n.includes('goal_marker')) return
      if (n.includes('goal_beam'))   return
      if (n.includes('debug'))       return
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
    const dir   = new THREE.Vector3(Math.sin(angle), 0, Math.cos(angle)).normalize()
    _raycaster.set(eyePos, dir)
    const hits = _raycaster.intersectObjects(_lidarCache, false)
    return hits.length > 0 ? parseFloat(hits[0].distance.toFixed(3)) : LIDAR_RANGE
  })
}