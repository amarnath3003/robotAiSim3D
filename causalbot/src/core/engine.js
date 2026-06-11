/**
 * core/engine.js — Main Simulation Engine
 * 
 * Orchestrates the simulation loop in the correct order:
 * 1. Apply motor commands to physics
 * 2. Step physics simulation
 * 3. Sync robot state from physics
 * 4. Update perception (sensors fire)
 * 5. Update brain/skills (if executing)
 * 6. Render
 * 
 * Provides a clean lifecycle that all systems hook into,
 * regardless of which robot is loaded.
 */

import * as THREE from 'three'
import { setState } from './state.js'

// ─── Engine State ──────────────────────────────────────────────────────────────

const _systems = {
  motor: null,      // Motor controller update function
  physics: null,    // Physics step function
  objects: null,    // Dynamic interactable sync (mesh ← physics) + robot push
  perception: null, // Perception update function
  brain: null,      // Brain/skill execution update
  rl: null,         // RL bridge update
  render: null,     // Render function
  ui: null,         // UI update function
}

let _robot = null         // Active RobotInstance
let _clock = null
let _running = false
let _paused = false

// Fixed timestep for physics (60 Hz)
const PHYSICS_HZ = 60
const PHYSICS_DT = 1 / PHYSICS_HZ
let _accumulator = 0

// Performance tracking
let _frameCount = 0
let _lastFpsTime = 0
let _fps = 0

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the engine with a loaded robot.
 * @param {import('./adapter.js').RobotInstance} robot
 */
export function initEngine(robot) {
  _robot = robot
  _clock = new THREE.Clock()
  _frameCount = 0
  _lastFpsTime = performance.now()
  
  console.log(`[Engine] Initialized with robot: "${robot.manifest.name}"`)
}

/**
 * Register a system update function.
 * @param {'motor'|'physics'|'perception'|'brain'|'rl'|'render'|'ui'} name
 * @param {function(number, import('./adapter.js').RobotInstance): void} updateFn
 */
export function registerSystem(name, updateFn) {
  if (!(name in _systems)) {
    console.warn(`[Engine] Unknown system: "${name}"`)
    return
  }
  _systems[name] = updateFn
  console.log(`[Engine] Registered system: ${name}`)
}

/**
 * Start the main simulation loop.
 */
export function startEngine() {
  if (_running) return
  // RC-1/NG-1: guard against null clock (initEngine not yet called)
  if (!_clock) {
    console.error('[Engine] startEngine() called before initEngine() — aborting')
    return
  }
  _running = true
  _paused = false
  console.log('[Engine] Starting simulation loop')
  _clock.start()
  requestAnimationFrame(tick)
}

/**
 * Stop the simulation loop.
 */
export function stopEngine() {
  _running = false
  console.log('[Engine] Stopped')
}

/**
 * Pause/unpause the simulation (rendering continues, physics/brain stop).
 */
export function togglePause() {
  _paused = !_paused
  console.log(`[Engine] ${_paused ? 'Paused' : 'Resumed'}`)
}

/**
 * Get the active robot instance.
 * @returns {import('./adapter.js').RobotInstance|null}
 */
export function getActiveRobot() {
  return _robot
}

/**
 * Get current FPS.
 * @returns {number}
 */
export function getFPS() {
  return _fps
}

/**
 * Is the engine currently running?
 */
export function isRunning() {
  return _running
}

// ─── Main Loop ─────────────────────────────────────────────────────────────────

function tick() {
  if (!_running) return
  requestAnimationFrame(tick)
  
  const rawDelta = _clock.getDelta()
  // Clamp delta to prevent spiral of death (e.g., tab was in background)
  const delta = Math.min(rawDelta, 0.1)
  
  // FPS tracking
  _frameCount++
  const now = performance.now()
  if (now - _lastFpsTime >= 1000) {
    _fps = _frameCount
    _frameCount = 0
    _lastFpsTime = now
  }
  
  if (!_paused && _robot) {
    // ─── Fixed timestep physics ──────────────────────────────────────────
    _accumulator += delta
    
    while (_accumulator >= PHYSICS_DT) {
      // 1. Apply motor commands
      if (_systems.motor) _systems.motor(PHYSICS_DT, _robot)
      
      // 2. Apply robot movement to physics body
      _robot.applyToPhysics(PHYSICS_DT)
      
      // 3. Step physics world
      if (_systems.physics) _systems.physics(PHYSICS_DT, _robot)
      
      // 4. Sync robot state from physics
      _robot.syncFromPhysics()
      
      // 5. Update joints (interpolate toward targets)
      _robot.updateJoints(PHYSICS_DT)
      
      _accumulator -= PHYSICS_DT
    }
    
    // Mirror live robot pose into global state (once per frame, not every physics sub-step)
    // SM-1: pass plain {x,y,z} snapshot — never the live Vector3 reference
    const p = _robot.position
    setState('robot.position', { x: p.x, y: p.y, z: p.z })
    setState('robot.heading', _yawFromQuat(_robot.orientation))
    
    // ─── Variable timestep systems (once per frame) ──────────────────────
    
    // 6. Sync interactable mesh positions from physics + apply robot push
    if (_systems.objects) _systems.objects(delta, _robot)
    
    // 7. Perception (sensor update)
    if (_systems.perception) _systems.perception(delta, _robot)
    
    // 8. Brain / skill execution
    if (_systems.brain) _systems.brain(delta, _robot)
    
    // 9. RL bridge
    if (_systems.rl) _systems.rl(delta, _robot)
  }
  
  // 9. Render (always, even when paused)
  if (_systems.render) _systems.render(delta, _robot)
  
  // 10. UI update
  if (_systems.ui) _systems.ui(delta, _robot)
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

/**
 * Extract yaw (Y-axis rotation) from a quaternion.
 * @param {THREE.Quaternion} q
 * @returns {number} Yaw in radians
 */
function _yawFromQuat(q) {
  const siny = 2.0 * (q.w * q.y + q.x * q.z)
  const cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
  return Math.atan2(siny, cosy)
}
