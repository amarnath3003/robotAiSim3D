/**
 * perception/vision.js — Manifest-Configured Vision Sensor
 * 
 * Raycasting-based vision system configured from the robot manifest.
 * Replaces the old hardcoded visionSensor.js.
 * 
 * Features:
 * - Configurable FOV, range, ray count from manifest sensors
 * - Object identification via mesh hierarchy traversal
 * - Distance estimation and position inference
 * - LiDAR mode for RL (raw distances) vs. vision mode (object detection)
 * - Automatic cache invalidation when scene changes
 */

import * as THREE from 'three'
import { getManifest, getSensor } from '../core/manifest.js'
import { updatePerceptionMemory } from '../core/state.js'
import { getWorld, getRapier } from '../physics/world.js'

// ─── Internal State ────────────────────────────────────────────────────────────

const _raycaster = new THREE.Raycaster()
let _sceneMeshes = []           // Cached flat list of intersectable meshes
let _cacheValid = false
let _lastConfig = null

// Mesh names to ignore (robot's own body, environment shell)
const IGNORE_PREFIXES = ['robot', 'debug', 'sky', 'ground', 'floor', 'env_shell', 'wall_rl']

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize or update the vision system from the current manifest.
 * Call after scene is loaded and robot manifest is available.
 * @param {THREE.Scene} scene
 */
export function initVision(scene) {
  _cacheValid = false
  rebuildMeshCache(scene)
}

/**
 * Invalidate the mesh cache (call when objects are added/removed from scene).
 */
export function invalidateVisionCache() {
  _cacheValid = false
}

/**
 * Cast vision rays and return perceived objects.
 * Uses the front_vision sensor config from manifest.
 * 
 * @param {THREE.Vector3} robotPosition - Robot's world position
 * @param {number} facingAngle - Robot's Y-axis rotation (radians)
 * @param {THREE.Scene} scene
 * @returns {Array<{id: string, name: string, distance: number, angle: number, position: THREE.Vector3, confidence: number}>}
 */
export function castVision(robotPosition, facingAngle, scene) {
  if (!_cacheValid) rebuildMeshCache(scene)
  
  const config = getVisionConfig()
  const { fov, range, rays, position: sensorOffset } = config
  
  // Sensor world position
  const sensorPos = new THREE.Vector3(
    robotPosition.x + (sensorOffset?.x || 0),
    robotPosition.y + (sensorOffset?.y || 0.5),
    robotPosition.z + (sensorOffset?.z || 0)
  )
  
  const halfFov = (fov / 2) * (Math.PI / 180)
  const angleStep = (fov * (Math.PI / 180)) / Math.max(rays - 1, 1)
  // EC-4: for rays=1 the single ray should point straight ahead (centre of FOV)
  const startAngle = rays > 1 ? facingAngle - halfFov : facingAngle
  
  const detectedObjects = new Map()  // objectId → {hits, totalDist, positions}
  
  for (let i = 0; i < rays; i++) {
    const angle = startAngle + angleStep * i
    // Fix: Robot is oriented towards +Z. The rays should point forward.
    const direction = new THREE.Vector3(
      Math.sin(angle),
      0,
      Math.cos(angle)
    ).normalize()
    
    _raycaster.set(sensorPos, direction)
    _raycaster.far = range
    
    const intersects = _raycaster.intersectObjects(_sceneMeshes, false)
    
    if (intersects.length > 0) {
      const hit = intersects[0]
      const rootName = getRootObjectName(hit.object)
      
      if (rootName && !shouldIgnore(rootName)) {
        if (!detectedObjects.has(rootName)) {
          detectedObjects.set(rootName, {
            hits: 0,
            totalDist: 0,
            positions: [],
            minDist: Infinity,
            colorName: hit.object?.userData?.colorName || null,
          })
        }
        
        const entry = detectedObjects.get(rootName)
        entry.hits++
        entry.totalDist += hit.distance
        entry.positions.push(hit.point.clone())
        entry.minDist = Math.min(entry.minDist, hit.distance)
        // Capture colorName from mesh userData on first hit
        if (!entry.colorName && hit.object?.userData?.colorName) {
          entry.colorName = hit.object.userData.colorName
        }
      }
    }
  }
  
  // Convert to output format
  const results = []
  for (const [name, data] of detectedObjects) {
    const avgDist  = data.totalDist / data.hits
    const confidence = Math.min(1.0, data.hits / (rays * 0.3))

    // Estimate object center position from hit points
    const center = new THREE.Vector3()
    for (const p of data.positions) center.add(p)
    center.divideScalar(data.positions.length)

    // Add small noise to simulate real sensor uncertainty (±0.15 m)
    // Real robot sensors are never perfectly accurate
    const NOISE = 0.15
    center.x += (Math.random() - 0.5) * 2 * NOISE
    center.z += (Math.random() - 0.5) * 2 * NOISE

    // Angle relative to robot facing
    const dx = center.x - robotPosition.x
    const dz = center.z - robotPosition.z
    const objectAngle = Math.atan2(dx, dz) - facingAngle

    // Try to get colorName from mesh userData for richer LLM context
    const colorName = data.colorName || null
    const displayName = colorName
      ? `${colorName} ${name.replace(/^object_|^ball_[a-z]+$|^box_[a-z]+$/i, m => m.includes('ball') ? 'ball' : 'box')}`
      : name.replace('object_', '')

    results.push({
      id: name,
      name: displayName,
      colorName,
      distance: avgDist,
      angle: normalizeAngle(objectAngle),
      position: center,
      confidence,
    })

    // Update perception memory (carry colour metadata when the mesh has it)
    updatePerceptionMemory(name, {
      x: center.x,
      y: center.y,
      z: center.z,
    }, confidence, colorName ? { colorName } : null)
  }

  return results
}

/**
 * Cast LiDAR rays and return raw distance array.
 * Uses the lidar sensor config from manifest.
 *
 * Primary implementation raycasts against the PHYSICS world (Rapier):
 * - fast (broadphase-accelerated), scales to dense ray fans at 60 Hz
 * - measures the exact collision geometry the robot can actually hit
 * - immune to stale mesh caches when the scene is rebuilt (RL episodes)
 * - the robot's own capsule and any carried object are excluded (both are
 *   kinematic bodies; everything else in the world is fixed or dynamic)
 * Falls back to Three.js raycasting if the physics world isn't ready.
 *
 * @param {THREE.Vector3} robotPosition
 * @param {number} facingAngle
 * @param {THREE.Scene} scene
 * @returns {Float32Array} Array of distances (length = sensor ray count)
 */
export function castLidar(robotPosition, facingAngle, scene) {
  const config = getLidarConfig()
  const { fov, range, rays, position: sensorOffset } = config

  const sx = robotPosition.x + (sensorOffset?.x || 0)
  const sy = robotPosition.y + (sensorOffset?.y ?? 0.5)
  const sz = robotPosition.z + (sensorOffset?.z || 0)

  const distances = new Float32Array(rays)
  const halfFov = (fov / 2) * (Math.PI / 180)
  const angleStep = (fov * (Math.PI / 180)) / Math.max(rays - 1, 1)
  // EC-4: single-ray LiDAR should point straight ahead
  const startAngle = rays > 1 ? facingAngle - halfFov : facingAngle

  const world = getWorld()
  const RAPIER = getRapier()

  if (world && RAPIER) {
    // ── Physics-based LiDAR ────────────────────────────────────────────────
    if (!_lidarRay) _lidarRay = new RAPIER.Ray({ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 1 })
    _lidarRay.origin.x = sx
    _lidarRay.origin.y = sy
    _lidarRay.origin.z = sz

    for (let i = 0; i < rays; i++) {
      const angle = startAngle + angleStep * i
      // LB-3: forward is +sin/+cos — must match castVision AND
      // pathfinder.updateDynamicObstacles hit reconstruction.
      _lidarRay.dir.x = Math.sin(angle)
      _lidarRay.dir.y = 0
      _lidarRay.dir.z = Math.cos(angle)

      const hit = world.castRay(
        _lidarRay, range, true,
        undefined, undefined, undefined, undefined,
        _lidarFilterPredicate
      )
      const toi = hit ? (hit.timeOfImpact ?? hit.toi ?? range) : range
      distances[i] = Math.min(toi, range)
    }
    return distances
  }

  // ── Fallback: Three.js raycaster (pre-physics boot only) ─────────────────
  if (!_cacheValid) rebuildMeshCache(scene)
  const sensorPos = new THREE.Vector3(sx, sy, sz)

  for (let i = 0; i < rays; i++) {
    const angle = startAngle + angleStep * i
    const direction = new THREE.Vector3(Math.sin(angle), 0, Math.cos(angle)).normalize()

    _raycaster.set(sensorPos, direction)
    _raycaster.far = range

    const intersects = _raycaster.intersectObjects(_sceneMeshes, false)
    let d = range
    for (const hit of intersects) {
      const rootName = getRootObjectName(hit.object)
      if (rootName && shouldIgnore(rootName)) continue
      d = hit.distance
      break
    }
    distances[i] = d
  }

  return distances
}

// Reusable Rapier ray (allocated once)
let _lidarRay = null

// Exclude kinematic bodies from LiDAR: the robot itself is kinematic, and any
// object the robot carries is switched to kinematic while held — both would
// otherwise read as phantom point-blank obstacles.
function _lidarFilterPredicate(collider) {
  const body = collider.parent && collider.parent()
  if (body && typeof body.isKinematic === 'function' && body.isKinematic()) return false
  return true
}

/**
 * Downsample a dense LiDAR scan to N sector readings (minimum per sector —
 * conservative: a sector reports the closest obstacle inside it).
 * Used to keep the RL observation vector at its manifest-defined size while
 * the actual sensor runs at much higher resolution.
 *
 * @param {Float32Array} distances  Full-resolution scan
 * @param {number} outCount         Target number of readings
 * @param {number} [clampMax]       Clamp output values (RL obs space "high")
 * @returns {Float32Array}
 */
export function downsampleLidar(distances, outCount, clampMax = Infinity) {
  const out = new Float32Array(outCount)
  if (!distances || distances.length === 0) return out

  const n = distances.length
  if (n <= outCount) {
    for (let i = 0; i < outCount; i++) {
      out[i] = Math.min(distances[Math.min(i, n - 1)], clampMax)
    }
    return out
  }

  for (let j = 0; j < outCount; j++) {
    const a = Math.floor((j * n) / outCount)
    const b = Math.max(a + 1, Math.floor(((j + 1) * n) / outCount))
    let m = Infinity
    for (let i = a; i < b && i < n; i++) {
      if (distances[i] < m) m = distances[i]
    }
    out[j] = Math.min(m, clampMax)
  }
  return out
}

// ─── Configuration Helpers ─────────────────────────────────────────────────────

function getVisionConfig() {
  const manifest = getManifest()
  const sensor = manifest?.sensors?.find(s => s.type === 'camera') || {}
  
  return {
    fov: sensor.config?.fov || 150,
    range: sensor.config?.range || 8.0,
    rays: sensor.config?.rays || 36,
    position: sensor.position || { x: 0, y: 0.55, z: 0.15 },
    updateRate: sensor.config?.updateRate || 30,
  }
}

function getLidarConfig() {
  const manifest = getManifest()
  const sensor = manifest?.sensors?.find(s => s.type === 'lidar') || {}
  
  return {
    fov: sensor.config?.fov || 360,
    range: sensor.config?.range || 5.0,
    rays: sensor.config?.rays || 11,
    position: sensor.position || { x: 0, y: 0.5, z: 0 },
    updateRate: sensor.config?.updateRate || 60,
  }
}

// ─── Internal Helpers ──────────────────────────────────────────────────────────

function rebuildMeshCache(scene) {
  if (!scene) return
  
  _sceneMeshes = []
  scene.traverse((child) => {
    if (child.isMesh && child.visible) {
      _sceneMeshes.push(child)
    }
  })
  
  _cacheValid = true
}

function getRootObjectName(mesh) {
  let current = mesh
  while (current) {
    if (current.name && current.name.startsWith('object_')) {
      return current.name
    }
    // Check for meaningful names at each level
    if (current.name && !current.name.startsWith('Scene') && current.name.length > 2) {
      // Check if parent is the scene root
      if (!current.parent || current.parent.type === 'Scene' || current.parent.type === 'Group') {
        return current.name
      }
    }
    current = current.parent
  }
  return mesh.name || null
}

function shouldIgnore(name) {
  const lower = name.toLowerCase()
  return IGNORE_PREFIXES.some(prefix => lower.startsWith(prefix))
}

function normalizeAngle(angle) {
  while (angle > Math.PI) angle -= 2 * Math.PI
  while (angle < -Math.PI) angle += 2 * Math.PI
  return angle
}
