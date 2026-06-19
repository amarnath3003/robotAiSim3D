/**
 * perception/observer.js — Unified Observation Builder
 * 
 * Combines all sensor data into structured observations:
 * - For the RL agent: a normalized float vector (defined by manifest observationSpace)
 * - For the LLM brain: a structured JSON object with semantic meaning
 * 
 * This is the single source of truth for "what the robot currently perceives."
 * Both RL and LLM read from this — never from raw scene data.
 */

import { getManifest, getObservationSpace } from '../core/manifest.js'
import { getKnownObjects, decayPerceptionMemory } from '../core/state.js'
import { castVision, castLidar } from './vision.js'
import { cvTick, initCVCamera } from './cv_camera.js'
import { updateDynamicObstacles } from '../nav/pathfinder.js'
import { updateLidarRays, updateFovCone } from '../debug/visualizer.js'

// ─── State ─────────────────────────────────────────────────────────────────────

let _lastObservation = null        // Latest observation vector (for RL)
let _lastPerception = null         // Latest structured perception (for LLM)
let _updateCounter = 0
let _scene = null

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the observer with the scene reference.
 * @param {THREE.Scene} scene
 * @param {THREE.WebGLRenderer} [renderer] - Optional renderer for real CV camera
 * @param {function} [getRobotFn] - Optional function returning current robot instance
 */
export function initObserver(scene, renderer, getRobotFn) {
  _scene = scene

  // Initialise real CV camera if renderer is provided
  if (renderer && getRobotFn) {
    initCVCamera(renderer, getRobotFn, () => scene)
  }

  console.log('[Observer] Initialized')
}

/**
 * Update all perception (call once per frame or at sensor update rate).
 * This is the main perception tick.
 * 
 * @param {number} dt - Delta time in seconds
 * @param {import('../core/adapter.js').RobotInstance} robot - Active robot
 */
export function updatePerception(dt, robot) {
  if (!robot || !_scene) return
  
  _updateCounter++
  
  // Decay memory over time (objects not re-observed fade)
  // Reduced rate: 0.015/s keeps objects reliable across multi-step tasks (was 0.03)
  decayPerceptionMemory(0.015, dt)
  
  // ── Real CV camera tick (async, non-blocking, throttled to 600ms) ───────────────
  // This sends a WebGL canvas screenshot to Gemini Vision API for real object detection.
  // Results are merged into perception memory alongside raycaster hits.
  cvTick()

  // ── Raycaster vision (fast, every N frames) ────────────────────────────
  // Raycaster provides high-frequency positional updates (faster than Vision API).
  const visionConfig = getManifest()?.sensors?.find(s => s.type === 'camera')
  const visionRate = visionConfig?.config?.updateRate || 30
  const visionInterval = Math.round(60 / visionRate)  // frames between updates
  
  if (_updateCounter % visionInterval === 0) {
    const visibleObjects = castVision(
      robot.position,
      getYawFromQuaternion(robot.orientation),
      _scene
    )
    _lastPerception = {
      visibleObjects,
      knownObjects: getKnownObjects(0.1),
      robotPosition: { x: robot.position.x, y: robot.position.y, z: robot.position.z },
      timestamp: Date.now(),
    }
  }
  
  // Cast LiDAR (always, for RL) — typically 60Hz
  const lidarDistances = castLidar(
    robot.position,
    getYawFromQuaternion(robot.orientation),
    _scene
  )

  // Feed LiDAR hits into the dynamic obstacle layer of the pathfinder.
  // This keeps A* aware of moving objects without rebuilding the static grid.
  const lidarCfg = getManifest()?.sensors?.find(s => s.type === 'lidar')
  if (lidarCfg) {
    const { fov = 360, range = 5, rays = 11 } = lidarCfg.config || {}
    // Manifest stores FOV in degrees; pathfinder.updateDynamicObstacles expects radians
    updateDynamicObstacles(
      robot.position.x,
      robot.position.z,
      getYawFromQuaternion(robot.orientation),
      lidarDistances,
      fov * (Math.PI / 180), range, rays
    )
  }

  // Update debug sensor visualizers in Three.js scene
  const yaw = getYawFromQuaternion(robot.orientation)
  updateLidarRays(robot.position, yaw, lidarDistances)
  updateFovCone(robot.position, yaw)

  // Build RL observation vector
  _lastObservation = buildRLObservation(robot, lidarDistances)
}

/**
 * Get the latest RL observation vector.
 * @returns {Float32Array|null}
 */
export function getRLObservation() {
  return _lastObservation
}

/**
 * Get the latest structured perception (for LLM/brain).
 * @returns {Object|null}
 */
export function getPerceptionState() {
  return _lastPerception
}

/**
 * Build a complete observation for sending to the RL Python agent.
 * Format matches manifest observationSpace definition.
 * 
 * @param {import('../core/adapter.js').RobotInstance} robot
 * @param {Float32Array} lidarDistances
 * @param {{x: number, z: number}} goalPosition - Current goal (from RL bridge)
 * @returns {Float32Array}
 */
export function buildRLObservationForBridge(robot, lidarDistances, goalPosition) {
  const obsSpace = getObservationSpace()
  if (!obsSpace) return new Float32Array(13)  // Fallback
  
  const dim = obsSpace.dimensions || 13
  const obs = new Float32Array(dim)
  
  // Channel 0: target distance
  const dx = (goalPosition?.x || 0) - robot.position.x
  const dz = (goalPosition?.z || 0) - robot.position.z
  const dist = Math.sqrt(dx * dx + dz * dz)
  obs[0] = dist
  
  // Channel 1: target relative angle
  const yaw = getYawFromQuaternion(robot.orientation)
  const targetAngle = Math.atan2(dx, dz) - yaw
  obs[1] = normalizeAngle(targetAngle)
  
  // Channels 2+: lidar distances
  for (let i = 0; i < lidarDistances.length && (i + 2) < dim; i++) {
    obs[i + 2] = lidarDistances[i]
  }
  
  return obs
}

// ─── Internal ──────────────────────────────────────────────────────────────────

function buildRLObservation(robot, lidarDistances) {
  // Generic observation without specific goal (used for perception update)
  const obsSpace = getObservationSpace()
  const dim = obsSpace?.dimensions || 13
  const obs = new Float32Array(dim)
  
  // For general perception, use nearest known object as implicit "target"
  const known = getKnownObjects(0.3)
  if (known.length > 0) {
    const nearest = known.reduce((a, b) => {
      const distA = Math.sqrt((a.position.x - robot.position.x) ** 2 + (a.position.z - robot.position.z) ** 2)
      const distB = Math.sqrt((b.position.x - robot.position.x) ** 2 + (b.position.z - robot.position.z) ** 2)
      return distA < distB ? a : b
    })
    
    const dx = nearest.position.x - robot.position.x
    const dz = nearest.position.z - robot.position.z
    obs[0] = Math.sqrt(dx * dx + dz * dz)
    const yaw = getYawFromQuaternion(robot.orientation)
    obs[1] = normalizeAngle(Math.atan2(dx, dz) - yaw)
  }
  
  // LiDAR data
  for (let i = 0; i < lidarDistances.length && (i + 2) < dim; i++) {
    obs[i + 2] = lidarDistances[i]
  }
  
  return obs
}

function getYawFromQuaternion(quat) {
  // Extract Y-axis rotation from quaternion
  const siny = 2.0 * (quat.w * quat.y + quat.x * quat.z)
  const cosy = 1.0 - 2.0 * (quat.y * quat.y + quat.z * quat.z)
  return Math.atan2(siny, cosy)
}

function normalizeAngle(angle) {
  // LB-4: JS % can return negative values for negative inputs, breaking
  //        the fold-to-[−π, π] calculation. Use the while-loop form instead.
  while (angle >  Math.PI) angle -= 2 * Math.PI
  while (angle < -Math.PI) angle += 2 * Math.PI
  return angle
}
