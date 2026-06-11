/**
 * rl/bridge.js — Manifest-Driven WebSocket Bridge to Python RL Agent
 * 
 * Connects the browser physics simulation to a Python training script
 * via WebSocket. Action and observation spaces are defined by the manifest,
 * not hardcoded.
 * 
 * Key improvements over old rl.js:
 * - Action/observation dimensions from manifest
 * - Modular reward computation (imported from reward.js)
 * - Clean separation: bridge handles communication, observer handles observations
 * - Supports multiple training modes (navigation, manipulation, etc.)
 */

// XM-2: removed dead getActionSpace + getObservationSpace imports (getObservationSpace
//        never used; getActionSpace was only used to seed _actionChannels which is now
//        handled via the manifest object directly)
import { getManifest } from '../core/manifest.js'
import { setState, getState } from '../core/state.js'
import { buildRLObservationForBridge } from '../perception/observer.js'
import { castLidar } from '../perception/vision.js'
import { computeReward, resetRewardState } from './reward.js'

// ─── Configuration ─────────────────────────────────────────────────────────────

// WS-1: prefer VITE_RL_WS_URL env var so the URL can be changed without
//        touching source code (e.g., remote training server, docker container).
const WS_URL = import.meta.env.VITE_RL_WS_URL || 'ws://localhost:8765'
const RECONNECT_DELAY = 3000

// ─── State ─────────────────────────────────────────────────────────────────────

let _ws = null
let _connected = false
let _reconnectTimer = null
let _scene = null
let _robot = null

// RL episode state
let _goalPosition = { x: 0, z: -1.5 }
let _episode = 0
let _stepCount = 0
let _episodeReward = 0
let _training = false

// Action clamping (from manifest)
let _actionChannels = []

// Callbacks
let _onAction = null          // Called when Python sends an action
let _onReset = null           // Called when episode resets
let _onConnect = null
let _onDisconnect = null

// ─── Jump Arc State ────────────────────────────────────────────────────────────

// LB-7: _jumpOriginY default matches _FLOOR_Y=0.60 so a jump arc started
//        before any reset still lands the robot on the floor, not underground
let _jumpActive  = false
let _jumpT       = 0          // current frame within arc (0 … JUMP_FRAMES)
let _jumpOriginY = 0.60       // Y at jump start — matches FLOOR_Y constant
const JUMP_FRAMES  = 30       // ~0.5 s at 60 Hz
const JUMP_HEIGHT  = 0.4      // metres

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the RL bridge.
 * @param {THREE.Scene} scene
 * @param {import('../core/adapter.js').RobotInstance} robot
 * @param {Object} options
 */
export function initRLBridge(scene, robot, options = {}) {
  _scene = scene
  _robot = robot
  _onAction = options.onAction || null
  _onReset = options.onReset || null
  _onConnect = options.onConnect || null
  _onDisconnect = options.onDisconnect || null
  
  // Load action space from manifest
  const manifest = getManifest()
  if (manifest?.actionSpace?.channels) {
    _actionChannels = manifest.actionSpace.channels
  }
  
  console.log(`[RL Bridge] Initialized. Action space: ${_actionChannels.length}D`)
  console.log(`[RL Bridge] Connecting to ${WS_URL}...`)
  
  connect()
}

/**
 * Send the current observation to Python (call after physics step).
 * Also advances any active jump arc so it runs at physics frame rate.
 */
export function sendObservation() {
  if (!_connected || !_ws || !_robot) return
  // AP-1: only send when socket is genuinely open
  if (_ws.readyState !== WebSocket.OPEN) return
  // NG-2: _scene might be null if called before scene is ready
  if (!_scene) return

  // Tick jump arc (must run every frame, not just when an action arrives)
  _tickJumpArc()

  // Build observation from manifest-defined space
  const lidar = castLidar(_robot.position, getYawFromRobot(_robot), _scene)
  const obs = buildRLObservationForBridge(_robot, lidar, _goalPosition)
  
  // Send state message
  const msg = {
    type: 'state',
    observation: Array.from(obs),
    step: _stepCount,
  }
  
  _ws.send(JSON.stringify(msg))
}

/**
 * Check terminal conditions and send reward.
 * @param {number} dt
 * @returns {{done: boolean, reason: string|null}}
 */
export function checkTermination(dt) {
  if (!_training || !_robot) return { done: false, reason: null }
  
  const lidar = castLidar(_robot.position, getYawFromRobot(_robot), _scene)
  
  const { reward, terminated, truncated, outcome } = computeReward(
    _robot,
    _goalPosition,
    lidar,
    _stepCount,
    dt
  )
  
  _episodeReward += reward
  _stepCount++
  
  if (terminated || truncated) {
    // Update stats
    setState('rl.episode', _episode)
    setState('rl.episodeReward', _episodeReward)
    
    return { done: true, reason: outcome }
  }
  
  return { done: false, reason: null }
}

/**
 * Is the RL bridge connected to Python?
 */
export function isRLConnected() {
  return _connected
}

/**
 * Is RL training active?
 */
export function isTraining() {
  return _training
}

/**
 * Disconnect the RL bridge.
 */
export function disconnectRL() {
  if (_ws) {
    _ws.close()
    _ws = null
  }
  if (_reconnectTimer) {
    clearTimeout(_reconnectTimer)
    _reconnectTimer = null
  }
  _connected = false
}

// ─── WebSocket Communication ───────────────────────────────────────────────────

function connect() {
  try {
    _ws = new WebSocket(WS_URL)
    
    _ws.onopen = () => {
      _connected = true
      _training = true
      setState('rl.connected', true)
      setState('rl.training', true)
      console.log('[RL Bridge] Connected to Python agent')
      _onConnect?.()
    }
    
    _ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data)
        handleMessage(msg)
      } catch (e) {
        console.error('[RL Bridge] Failed to parse message:', e)
      }
    }
    
    _ws.onclose = () => {
      _connected = false
      _training = false
      setState('rl.connected', false)
      setState('rl.training', false)
      console.log('[RL Bridge] Disconnected')
      _onDisconnect?.()
      
      // Auto-reconnect
      _reconnectTimer = setTimeout(connect, RECONNECT_DELAY)
    }
    
    _ws.onerror = (err) => {
      console.warn('[RL Bridge] WebSocket error — will retry')
    }
    
  } catch (e) {
    console.warn('[RL Bridge] Connection failed — retrying in', RECONNECT_DELAY, 'ms')
    _reconnectTimer = setTimeout(connect, RECONNECT_DELAY)
  }
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'action':
      handleAction(msg.action || [])
      break
    
    case 'reset':
      handleReset(msg)
      break
    
    case 'telemetry':
      // Python sending stats back (win rate, etc.)
      if (msg.episode) setState('rl.episode', msg.episode)
      if (msg.win_rate) setState('rl.stats.winRate', msg.win_rate)
      break
    
    case 'params':
      // Training parameter updates
      if (msg.train_mode !== undefined) {
        _training = msg.train_mode
        setState('rl.training', _training)
      }
      break
      
    default:
      console.warn(`[RL Bridge] Unknown message type: "${msg.type}"`)
  }
}

function handleAction(actionArray) {
  if (!_robot) return
  
  // Clamp actions to manifest-defined ranges
  const clampedAction = new Float32Array(_actionChannels.length)
  for (let i = 0; i < _actionChannels.length; i++) {
    const channel = _actionChannels[i]
    const raw = actionArray[i] ?? 0
    clampedAction[i] = Math.max(channel.low, Math.min(channel.high, raw))
  }
  
  // Apply action to robot through the manifest-defined channels
  applyAction(clampedAction)
  
  // Notify callback
  _onAction?.(clampedAction)
}

function handleReset(msg) {
  _episode++
  _stepCount = 0
  _episodeReward = 0
  
  // Update goal
  if (msg.goal) {
    _goalPosition = { x: msg.goal.x, z: msg.goal.z }
  }
  
  // Reset robot position — LB-6: use FLOOR_Y=0.60, not 0.35
  if (_robot) {
    _robot.position.set(0, 0.60, 1.8)
    _robot.velocity.set(0, 0, 0)
    _robot.angularVelocity = 0
    
    if (_robot.physicsBody) {
      _robot.physicsBody.setNextKinematicTranslation({ x: 0, y: 0.60, z: 1.8 })
    }
  }

  // SM-2: reset reward tracking state so _prevDist is correct for the new goal
  if (_robot) {
    const dx = _goalPosition.x - _robot.position.x
    const dz = _goalPosition.z - _robot.position.z
    resetRewardState(Math.sqrt(dx * dx + dz * dz))
  }
  
  // Notify callback (for wall rebuilding, etc.)
  _onReset?.(msg)
  
  // NG-2: guard both _scene and readyState before sending reset_done
  if (_ws && _ws.readyState === WebSocket.OPEN && _robot && _scene) {
    const lidar = castLidar(_robot.position, getYawFromRobot(_robot), _scene)
    const obs = buildRLObservationForBridge(_robot, lidar, _goalPosition)
    
    _ws.send(JSON.stringify({
      type: 'reset_done',
      observation: Array.from(obs),
    }))
  }
  
  setState('rl.episode', _episode)
}

// ─── Action Application ────────────────────────────────────────────────────────

function applyAction(action) {
  if (!_robot) return
  
  // Map action channels to robot commands based on manifest actionSpace
  for (let i = 0; i < _actionChannels.length; i++) {
    const channel = _actionChannels[i]
    const value = action[i]
    
    switch (channel.name) {
      case 'linear_speed':
        _robot.moveForward(value)
        break
      case 'angular_speed':
        _robot.rotate(value)
        break
      case 'arm_rotation':
        // Apply to arm joints
        _robot.setGroupTarget('left_arm', value * (180 / Math.PI))
        _robot.setGroupTarget('right_arm', value * (180 / Math.PI))
        break
      case 'jump':
        // Request a jump arc — the arc itself is applied in _tickJumpArc()
        if (value > 0.5 && !_jumpActive) {
          _jumpActive  = true
          _jumpT       = 0
          _jumpOriginY = _robot.position.y
        }
        break
      default:
        // Generic: try to find a matching joint or group
        _robot.setJointTarget(channel.name, value)
    }
  }
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Advance the jump arc by one frame.
 * Applies a sine-curve Y offset via setNextKinematicTranslation so the
 * Rapier body is correctly updated (not just the Three.js position).
 */
function _tickJumpArc() {
  if (!_jumpActive || !_robot?.physicsBody) return

  _jumpT++
  const t       = _jumpT / JUMP_FRAMES
  const yOffset = Math.sin(t * Math.PI) * JUMP_HEIGHT
  const cur     = _robot.physicsBody.translation()

  _robot.physicsBody.setNextKinematicTranslation({
    x: cur.x,
    y: _jumpOriginY + yOffset,
    z: cur.z,
  })

  if (_jumpT >= JUMP_FRAMES) {
    _jumpActive = false
    _jumpT      = 0
    // Snap back to origin Y cleanly
    _robot.physicsBody.setNextKinematicTranslation({
      x: cur.x,
      y: _jumpOriginY,
      z: cur.z,
    })
  }
}

function getYawFromRobot(robot) {
  const q = robot.orientation
  const siny = 2.0 * (q.w * q.y + q.x * q.z)
  const cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
  return Math.atan2(siny, cosy)
}
