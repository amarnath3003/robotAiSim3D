import * as THREE from 'three'
import { state, getRobotPos } from '../state.js'

export const VISION_CONFIG = {
  fovDegrees:   150,
  maxRange:     4.5,
  rayCount:     36,
  heightOffset: 0.5,
}

// Prefixes that mark environment geometry (not interactive objects)
const ENV_PREFIXES = ['floor', 'wall', 'environment', 'room', 'ground', 'ceiling', 'env_', 'sky', 'light']

// Cached flat list of candidate meshes — rebuilt when scene changes
let _meshCache   = null
let _lastScene   = null

/**
 * Build (or return cached) list of candidate meshes to raycast against.
 * Excludes robot parts and environment geometry.
 */
function getCandidateMeshes(scene) {
  if (scene === _lastScene && _meshCache) return _meshCache
  _lastScene  = scene
  _meshCache  = []
  scene.traverse(child => {
    if (!child.isMesh) return
    // Climb up to find this mesh's "object root" — first named ancestor
    const rootName = getRootName(child)
    if (!rootName) return
    _meshCache.push(child)
  })
  return _meshCache
}

/** Invalidate mesh cache (call after adding/removing objects from scene) */
export function invalidateVisionCache() {
  _meshCache = null
}

/**
 * Walk up the Three.js hierarchy from a mesh node to find the
 * meaningful "object root name" — the first named ancestor that
 * is NOT the Scene itself and NOT an environment/robot name.
 * Returns null if the mesh belongs to the environment or robot.
 */
function getRootName(mesh) {
  let node = mesh
  let bestName = null
  while (node && node.type !== 'Scene') {
    const name = node.name || ''
    if (name) {
      const lower = name.toLowerCase()
      // If any level is robot → reject entire chain
      if (lower.includes('robot')) return null
      
      // If we find an explicit interactive object, accept it immediately.
      // This prevents higher-level GLTF root nodes (like "environment_root") from rejecting it.
      if (lower.startsWith('object_')) return name

      // If any level is environment → reject entire chain
      if (ENV_PREFIXES.some(p => lower.startsWith(p) || lower.includes(p))) return null
      // Otherwise record highest valid named ancestor
      bestName = name
    }
    node = node.parent
  }
  return bestName
}

/**
 * Cast vision rays from the robot's eye position + facing angle.
 * Returns perceived objects with estimated positions and confidence.
 *
 * @param {number} facingAngle  - Robot Y rotation (radians)
 * @returns {Array<{meshName, estimatedPos, distance, confidence, angle}>}
 */
export function castVision(facingAngle) {
  const scene = state.scene.three
  if (!scene) return []

  const rp     = getRobotPos()
  const eyePos = new THREE.Vector3(rp.x, rp.y + VISION_CONFIG.heightOffset, rp.z)

  const raycaster   = new THREE.Raycaster()
  raycaster.far     = VISION_CONFIG.maxRange
  raycaster.near    = 0.05

  const fovRad  = (VISION_CONFIG.fovDegrees * Math.PI) / 180
  const halfFov = fovRad / 2
  const step    = fovRad / (VISION_CONFIG.rayCount - 1)

  // Build candidate mesh list once per call (cached per scene reference)
  const candidates = getCandidateMeshes(scene)
  if (!candidates.length) return []

  // rootName → { distance, worldPos, rayAngle, hitCount }
  const accumulator = new Map()

  for (let i = 0; i < VISION_CONFIG.rayCount; i++) {
    const rayAngle = facingAngle - halfFov + step * i
    const dir = new THREE.Vector3(Math.sin(rayAngle), 0, Math.cos(rayAngle))
    raycaster.set(eyePos, dir)

    const intersects = raycaster.intersectObjects(candidates, false)
    if (!intersects.length) continue

    // Find the first valid hit that is NOT the currently held object
    let hitMesh = null, point = null, distance = null, rootName = null
    for (const hit of intersects) {
      const rn = getRootName(hit.object)
      if (!rn) continue
      if (rn === state.robot.heldObject) continue // Ignore what we are holding!
      
      hitMesh = hit.object
      point = hit.point
      distance = hit.distance
      rootName = rn
      break
    }

    if (!rootName) continue

    const acc = accumulator.get(rootName)
    if (acc) {
      acc.hitCount++
      if (distance < acc.distance) {
        acc.distance = distance
        acc.worldPos = point.clone()
        acc.rayAngle = rayAngle
      }
    } else {
      accumulator.set(rootName, {
        distance,
        worldPos: point.clone(),
        rayAngle,
        hitCount: 1,
      })
    }
  }

  // Convert accumulated hits → perceived object list
  const perceived = []
  accumulator.forEach((acc, rootName) => {
    const confidence = Math.min(1.0, acc.hitCount / 4)   // 4+ rays = fully confident
    const noise = (1.0 - confidence) * 0.10
    perceived.push({
      meshName:     rootName,
      estimatedPos: [
        acc.worldPos.x + (Math.random() - 0.5) * noise,
        acc.worldPos.y,
        acc.worldPos.z + (Math.random() - 0.5) * noise,
      ],
      distance:     acc.distance,
      confidence,
      angle:        acc.rayAngle,
    })
  })

  return perceived
}

// ─── Geometric helpers ────────────────────────────────────────────────────────

/**
 * Compute angle from robot to a [x,y,z] world position array.
 */
export function angleToTarget(targetPos) {
  const rp = getRobotPos()
  return Math.atan2(targetPos[0] - rp.x, targetPos[2] - rp.z)
}

/**
 * Check if a world position is within the robot's current FOV.
 */
export function isInFOV(facingAngle, targetPos) {
  const ang  = angleToTarget(targetPos)
  const diff = Math.abs(normalizeAngle(ang - facingAngle))
  return diff < (VISION_CONFIG.fovDegrees * Math.PI) / 180 / 2
}

/**
 * Normalize an angle to [-PI, PI].
 */
export function normalizeAngle(a) {
  while (a >  Math.PI) a -= Math.PI * 2
  while (a < -Math.PI) a += Math.PI * 2
  return a
}
