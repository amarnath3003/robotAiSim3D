/**
 * env/objects.js — Environment Object System
 *
 * Creates and manages all world objects:
 *   - Boundary walls  (static, invisible physics + visible mesh)
 *   - Box/cylinder obstacles (physics + visual)
 *   - Goal marker (glowing animated sphere + PointLight)
 *   - Interactable dynamic objects (pushable boxes)
 *
 * Each object:
 *   1. Gets a Three.js mesh added to the scene
 *   2. Gets a Rapier physics body via physics/world.js
 *   3. Registers its AABB with nav/pathfinder.js for occupancy
 *   4. Optionally gets a metadata entry in the perception system
 *
 * Usage:
 *   import { initEnvironment, loadDefaultLayout } from './src/env/objects.js'
 *   initEnvironment(scene, { arenaSize: 16 })
 *   loadDefaultLayout()
 */

import * as THREE from 'three'
import { createWall, createCylinderCollider, createDynamicBody, removeBody, getRapier } from '../physics/world.js'
import { registerObstacle, unregisterObstacle, clearObstacles, rebuildGrid } from '../nav/pathfinder.js'
import { invalidateVisionCache } from '../perception/vision.js'

// ─── Color Name Helper ─────────────────────────────────────────────────────────

/**
 * Map a hex color to a human-readable color name.
 * Used to populate mesh.userData.colorName so the CV system can label objects.
 * @param {number} hex  e.g. 0xff3322
 * @returns {string}  e.g. 'red'
 */
function _hexToColorName(hex) {
  const r = (hex >> 16) & 0xff
  const g = (hex >> 8)  & 0xff
  const b =  hex        & 0xff
  if (r > 180 && g < 100 && b < 100) return 'red'
  if (r < 100 && g < 100 && b > 180) return 'blue'
  if (r < 100 && g > 150 && b < 100) return 'green'
  if (r > 200 && g > 200 && b < 100) return 'yellow'
  if (r > 200 && g > 100 && b < 60)  return 'orange'
  if (r > 200 && g < 120 && b > 150) return 'pink'
  if (r > 120 && g > 80  && b < 80)  return 'wooden'  // brown / tan
  return 'unknown'
}

// ─── Module State ──────────────────────────────────────────────────────────────

let _scene     = null
let _options   = {}
let _obstacles = new Map()   // id → { mesh, physicsBody }
let _goalMarker = null       // { mesh, light, id }
let _goalAnimId = null
let _wallMeshes = []

// Dynamic interactables: physics-driven objects the robot can push/interact with
let _interactables = new Map()  // id → { mesh, body, spawnPos, navHalfW, navHalfD, lastNavPos }

// Nav grid sync: re-register moved interactables at most once per second
let _navSyncTimer = 0
const NAV_SYNC_INTERVAL = 1000  // ms

// ─── Public API: Init ──────────────────────────────────────────────────────────

/**
 * Initialize the environment system (must call before any create* functions).
 * @param {THREE.Scene} scene
 * @param {Object} options
 * @param {number} [options.arenaSize=16]  Half-extent of the playfield in metres
 * @param {boolean} [options.showBoundary=true]  Render boundary walls
 */
export function initEnvironment(scene, options = {}) {
  _scene   = scene
  _options = {
    arenaSize:    options.arenaSize   ?? 16,
    showBoundary: options.showBoundary ?? true,
  }
  console.log(`[Env] Initialized (arena ±${_options.arenaSize} m)`)
}

// ─── Public API: Boundary Walls ───────────────────────────────────────────────

/**
 * Spawn four static boundary walls around the arena perimeter.
 * Creates both physics colliders and semi-transparent visual meshes.
 */
export function createBoundaryWalls() {
  if (!_scene) { console.error('[Env] Call initEnvironment first'); return }

  const A = _options.arenaSize
  const H = 1.5   // wall height
  const T = 0.3   // wall thickness

  const wallSpecs = [
    { id: 'wall_N', x:  0,  z: -A, w: A * 2 + T, d: T, h: H },
    { id: 'wall_S', x:  0,  z:  A, w: A * 2 + T, d: T, h: H },
    { id: 'wall_W', x: -A,  z:  0, w: T, d: A * 2 + T, h: H },
    { id: 'wall_E', x:  A,  z:  0, w: T, d: A * 2 + T, h: H },
  ]

  for (const spec of wallSpecs) {
    // Physics
    createWall({ x: spec.x, z: spec.z, w: spec.w, d: spec.d, h: spec.h }, spec.id)

    // Register with pathfinder
    registerObstacle(spec.id, {
      minX: spec.x - spec.w / 2,
      maxX: spec.x + spec.w / 2,
      minZ: spec.z - spec.d / 2,
      maxZ: spec.z + spec.d / 2,
    })

    // Visual (only if showBoundary)
    if (_options.showBoundary) {
      const mesh = _makeWallMesh(spec.w, spec.h, spec.d)
      mesh.position.set(spec.x, spec.h / 2, spec.z)
      mesh.name = spec.id
      _scene.add(mesh)
      _wallMeshes.push(mesh)
    }
  }

  console.log('[Env] Boundary walls created')
  invalidateVisionCache()
}

// ─── Public API: Obstacles ────────────────────────────────────────────────────

/**
 * Create a static box obstacle.
 * @param {Object} config
 * @param {number} config.x           World X centre
 * @param {number} config.z           World Z centre
 * @param {number} [config.w=1]       Width  (X)
 * @param {number} [config.d=1]       Depth  (Z)
 * @param {number} [config.h=1]       Height (Y)
 * @param {number} [config.color]     Hex colour
 * @param {string} [config.label]     Label for perception system
 * @returns {string} Obstacle ID
 */
export function createObstacleBox(config) {
  if (!_scene) return null

  const { x, z, w = 1, d = 1, h = 1, color = 0x334455, label } = config
  const id = `obs_box_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`

  // Physics
  createWall({ x, z, w, d, h }, id)

  // Pathfinder occupancy
  registerObstacle(id, { minX: x - w/2, maxX: x + w/2, minZ: z - d/2, maxZ: z + d/2 })

  // Visual
  const geo  = new THREE.BoxGeometry(w, h, d)
  const mat  = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.7,
    metalness: 0.3,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(x, h / 2, z)
  mesh.castShadow    = true
  mesh.receiveShadow = true
  mesh.name          = label ?? id
  mesh.userData      = { obstacleId: id, type: 'obstacle', label: label ?? 'obstacle' }
  _scene.add(mesh)

  _obstacles.set(id, { mesh, physicsBody: null, aabb: { minX: x-w/2, maxX: x+w/2, minZ: z-d/2, maxZ: z+d/2 } })
  invalidateVisionCache()
  return id
}

/**
 * Create a static cylinder obstacle.
 * Uses an exact Rapier cylinder collider (instead of a box approximation) so the
 * robot cannot clip through the visual mesh edges.
 * @param {Object} config
 * @param {number} config.x
 * @param {number} config.z
 * @param {number} [config.radius=0.5]
 * @param {number} [config.h=1]
 * @param {number} [config.color=0x445566]
 * @returns {string} Obstacle ID
 */
export function createObstacleCylinder(config) {
  if (!_scene) return null

  const { x, z, radius = 0.5, h = 1, color = 0x445566, label } = config
  const id = `obs_cyl_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`

  // Physics: exact cylinder collider — matches the visual mesh precisely
  createCylinderCollider({ x, y: 0, z, radius, height: h }, id)

  // Pathfinder: use bounding square
  registerObstacle(id, { minX: x - radius, maxX: x + radius, minZ: z - radius, maxZ: z + radius })

  // Visual
  const geo  = new THREE.CylinderGeometry(radius, radius, h, 24)
  const mat  = new THREE.MeshStandardMaterial({ color, roughness: 0.5, metalness: 0.4 })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(x, h / 2, z)
  mesh.castShadow    = true
  mesh.receiveShadow = true
  mesh.name          = label ?? id
  mesh.userData      = { obstacleId: id, type: 'obstacle', label: label ?? 'column' }
  _scene.add(mesh)

  _obstacles.set(id, { mesh, physicsBody: null })
  invalidateVisionCache()
  return id
}

/**
 * Remove an obstacle by ID.
 * @param {string} id
 */
export function removeObstacle(id) {
  const entry = _obstacles.get(id)
  if (!entry) return

  _scene.remove(entry.mesh)
  // 3J-2: dispose geometry AND material to avoid WebGL memory leaks
  entry.mesh.geometry.dispose()
  if (Array.isArray(entry.mesh.material)) {
    entry.mesh.material.forEach(m => m.dispose())
  } else if (entry.mesh.material) {
    entry.mesh.material.dispose()
  }

  // PL-1: remove the Rapier physics body to avoid an ever-growing physics world.
  // createWall() registers the body under the obstacle's ID in physics/world.js,
  // so removeBody(id) is sufficient to clean it up.
  removeBody(id)

  unregisterObstacle(id)
  _obstacles.delete(id)
}

/**
 * Remove all non-boundary obstacles.
 */
export function clearObstacleObjects() {
  for (const id of _obstacles.keys()) {
    removeObstacle(id)
  }
}

// ─── Public API: Dynamic Interactables ───────────────────────────────────────

/**
 * Create a dynamic physics-driven box that the robot can push, knock over, or
 * pick up. Unlike obstacle boxes these have a Rapier dynamic rigid body.
 *
 * @param {Object} config
 * @param {number}  config.x
 * @param {number}  config.z
 * @param {number}  [config.y]       Spawn Y (defaults to half-height, resting on floor)
 * @param {number}  [config.w=0.5]   Width (X)
 * @param {number}  [config.d=0.5]   Depth (Z)
 * @param {number}  [config.h=0.5]   Height (Y)
 * @param {number}  [config.mass=3]  Mass in kg
 * @param {number}  [config.color]   Hex colour
 * @param {string}  [config.label]   Human-readable name
 * @returns {string} Interactable ID
 */
export function createInteractableBox(config) {
  if (!_scene) return null

  const {
    x, z,
    w         = 0.8,
    d         = 0.8,
    h         = 0.8,
    mass      = 1.2,    // lightweight — easy to push/topple
    color     = 0xc8922a,
    label,
    colorName = null,   // Human color name for CV: 'red','blue','wooden', etc.
  } = config

  const id      = label ?? `ibox_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
  const spawnY  = config.y ?? h / 2

  // ── Physics — tuned for real-world lightweight crate feel ──
  const body = createDynamicBody(id, {
    position:       [x, spawnY, z],
    size:           [w, h, d],
    shape:          'box',
    mass,
    friction:       0.62,    // cardboard/wood on concrete
    restitution:    0.20,    // slight bounce on hard impact
    linearDamping:  0.18,    // slides freely once moving
    angularDamping: 0.28,    // tumbles realistically
    ccd:            false,
  })

  // ── Visual ──
  const geo = new THREE.BoxGeometry(w, h, d)
  const mat = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.85,
    metalness: 0.05,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(x, spawnY, z)
  mesh.castShadow    = true
  mesh.receiveShadow = true
  mesh.name          = id
  mesh.userData      = { type: 'interactable', subtype: 'box', id, label: label ?? 'box',
                         colorName: colorName ?? _hexToColorName(color), perceptible: true }

  // Edge highlight so crates read clearly against the floor
  const edges    = new THREE.EdgesGeometry(geo)
  const edgeMat  = new THREE.LineBasicMaterial({ color: 0xffd080, transparent: true, opacity: 0.35 })
  const edgeMesh = new THREE.LineSegments(edges, edgeMat)
  mesh.add(edgeMesh)

  _scene.add(mesh)
  _interactables.set(id, { mesh, body, spawnPos: { x, y: spawnY, z }, held: false, holdRobot: null,
    navHalfW: w / 2, navHalfD: d / 2, lastNavPos: { x, z } })

  // Register with pathfinder so A* routes around this object from the start
  registerObstacle(id, { minX: x - w/2, maxX: x + w/2, minZ: z - d/2, maxZ: z + d/2 })

  console.log(`[Env] Interactable box "${id}" spawned at (${x}, ${spawnY}, ${z})`)
  return id
}

/**
 * Create a dynamic physics-driven ball (sphere) that the robot can push or kick.
 * Uses CCD so fast balls don't tunnel through thin walls.
 *
 * @param {Object} config
 * @param {number}  config.x
 * @param {number}  config.z
 * @param {number}  [config.y]         Spawn Y (defaults to radius, resting on floor)
 * @param {number}  [config.radius=0.2] Ball radius in metres
 * @param {number}  [config.mass=1.0]  Mass in kg
 * @param {number}  [config.color]     Hex colour
 * @param {string}  [config.label]     Human-readable name
 * @returns {string} Interactable ID
 */
export function createInteractableBall(config) {
  if (!_scene) return null

  const {
    x, z,
    radius    = 0.35,   // bigger default — beach-ball scale
    mass      = 0.45,   // very light — airy and responsive
    color     = 0xff3322,
    label,
    colorName = null,   // Human color name for CV: 'red','blue','green', etc.
  } = config

  const id     = label ?? `iball_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`
  const spawnY = config.y ?? radius

  // ── Physics — tuned for real-world beach/rubber ball feel ──
  const diameter = radius * 2
  const body = createDynamicBody(id, {
    position:       [x, spawnY, z],
    size:           [diameter, diameter, diameter],
    shape:          'sphere',
    mass,
    friction:       0.32,    // rubber on concrete — rolls freely
    restitution:    0.72,    // bouncy — ball compresses and springs back
    linearDamping:  0.02,    // rolls a long way before stopping
    angularDamping: 0.03,    // spins freely like a real ball
    ccd:            true,    // prevent tunnelling at speed
  })

  // ── Visual ──
  const geo = new THREE.SphereGeometry(radius, 32, 24)   // higher segments for bigger balls
  const mat = new THREE.MeshStandardMaterial({
    color,
    roughness: 0.40,
    metalness: 0.06,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(x, spawnY, z)
  mesh.castShadow    = true
  mesh.receiveShadow = true
  mesh.name          = id
  mesh.userData      = { type: 'interactable', subtype: 'ball', id, label: label ?? 'ball',
                         colorName: colorName ?? _hexToColorName(color), perceptible: true }

  _scene.add(mesh)
  _interactables.set(id, { mesh, body, spawnPos: { x, y: spawnY, z }, held: false, holdRobot: null,
    navHalfW: radius, navHalfD: radius, lastNavPos: { x, z } })

  // Register with pathfinder so A* routes around this ball from the start
  registerObstacle(id, { minX: x - radius, maxX: x + radius, minZ: z - radius, maxZ: z + radius })

  console.log(`[Env] Interactable ball "${id}" spawned at (${x}, ${spawnY}, ${z})`)
  return id
}

/**
 * Sync all dynamic interactable meshes from Rapier physics each frame.
 * Register this as an engine system so it runs every tick after physics is stepped.
 * Also drives kinematic held-objects to follow the robot.
 */
export function updateInteractables() {
  const now = Date.now()
  const doNavSync = now - _navSyncTimer >= NAV_SYNC_INTERVAL
  let needsGridRebuild = false

  for (const [id, entry] of _interactables) {
    const body = entry.body
    if (!body) continue

    if (entry.held && entry.holdRobot) {
      // ── Held: drive the kinematic body to hover in front of the robot ──
      const rp  = entry.holdRobot.position       // THREE.Vector3
      const rot = entry.holdRobot.orientation    // THREE.Quaternion

      // Offset: 0.5 m forward along robot facing, 0.38 m up from robot origin
      const fwd = new THREE.Vector3(0, 0, 0.5).applyQuaternion(rot)
      const holdPos = {
        x: rp.x + fwd.x,
        y: rp.y + 0.38,
        z: rp.z + fwd.z,
      }

      body.setNextKinematicTranslation(holdPos)

      // Keep mesh in sync with the kinematic body
      entry.mesh.position.set(holdPos.x, holdPos.y, holdPos.z)
      // Upright orientation while held — looks intentional / stable
      entry.mesh.quaternion.set(0, 0, 0, 1)
    } else {
      // ── Free: sync mesh from physics simulation ──
      const t = body.translation()
      const r = body.rotation()

      entry.mesh.position.set(t.x, t.y, t.z)
      entry.mesh.quaternion.set(r.x, r.y, r.z, r.w)

      // Sync nav grid when this object has drifted >0.4m from last registered position
      if (doNavSync && entry.lastNavPos !== undefined && entry.navHalfW !== undefined) {
        const dx = t.x - entry.lastNavPos.x
        const dz = t.z - entry.lastNavPos.z
        if (dx * dx + dz * dz > 0.16) {  // 0.4 m threshold
          registerObstacle(id, {
            minX: t.x - entry.navHalfW, maxX: t.x + entry.navHalfW,
            minZ: t.z - entry.navHalfD, maxZ: t.z + entry.navHalfD,
          })
          entry.lastNavPos = { x: t.x, z: t.z }
          needsGridRebuild = true
        }
      }

      // Auto-respawn if object falls below the world
      if (t.y < -8) {
        const sp = entry.spawnPos
        body.setTranslation({ x: sp.x, y: sp.y + 0.5, z: sp.z }, true)
        body.setLinvel     ({ x: 0, y: 0, z: 0 }, true)
        body.setAngvel     ({ x: 0, y: 0, z: 0 }, true)
        body.wakeUp()
      }
    }
  }

  if (doNavSync) _navSyncTimer = now
  if (needsGridRebuild) rebuildGrid()
}

// ─── Public API: Interaction (grab / release / push) ─────────────────────────

/**
 * Grab an interactable object. Switches the physics body to kinematic so it can
 * be driven by the robot's position each frame via updateInteractables().
 *
 * @param {string} objectId  - Mesh name, label, or interactable map key
 * @param {Object} robot     - RobotInstance with .position (THREE.Vector3) and .orientation (THREE.Quaternion)
 * @returns {boolean} true if successfully grabbed
 */
export function grabInteractable(objectId, robot) {
  const entry = _findInteractable(objectId)
  if (!entry || !entry.body) {
    console.warn(`[Env] grabInteractable: "${objectId}" not found`)
    return false
  }
  if (entry.held) {
    console.warn(`[Env] grabInteractable: "${objectId}" already held`)
    return false
  }

  // Switch body to KinematicPositionBased so we drive its translation each frame
  const R           = getRapier()
  const kinematic   = R?.RigidBodyType?.KinematicPositionBased ?? 2
  entry.body.setBodyType(kinematic, true)

  entry.held      = true
  entry.holdRobot = robot

  // Remove from nav grid while held — it travels with the robot and must not block paths
  unregisterObstacle(entry.mesh.name)

  console.log(`[Env] Grabbed "${objectId}"`)
  return true
}

/**
 * Release the currently held object. Restores dynamic physics and applies a
 * forward placement impulse (gentle) or throw impulse (strong).
 *
 * @param {string} objectId   - Same id used in grabInteractable()
 * @param {Object} robot      - RobotInstance (for forward-direction impulse)
 * @param {number} throwForce - Impulse scale: ~1.5 = gentle place, ~8 = throw
 */
export function releaseInteractable(objectId, robot, throwForce = 1.5) {
  const entry = _findInteractable(objectId)
  if (!entry || !entry.body) {
    console.warn(`[Env] releaseInteractable: "${objectId}" not found`)
    return
  }

  // Restore dynamic simulation
  const R       = getRapier()
  const dynamic = R?.RigidBodyType?.Dynamic ?? 0
  entry.body.setBodyType(dynamic, true)

  entry.held      = false
  entry.holdRobot = null

  // Re-register at current drop position so future paths route around it
  if (entry.body && entry.navHalfW !== undefined) {
    const t = entry.body.translation()
    registerObstacle(entry.mesh.name, {
      minX: t.x - entry.navHalfW, maxX: t.x + entry.navHalfW,
      minZ: t.z - entry.navHalfD, maxZ: t.z + entry.navHalfD,
    })
    entry.lastNavPos = { x: t.x, z: t.z }
    rebuildGrid()
  }

  // Apply a forward impulse proportional to throwForce
  if (robot?.orientation) {
    const fwd = new THREE.Vector3(0, 0, 1).applyQuaternion(robot.orientation)
    entry.body.applyImpulse(
      { x: fwd.x * throwForce, y: 0.25, z: fwd.z * throwForce },
      true
    )
  }

  entry.body.setAngvel({ x: 0, y: 0, z: 0 }, true)
  entry.body.wakeUp()

  console.log(`[Env] Released "${objectId}" (force=${throwForce})`)
}

/**
 * Push an interactable object with a direct physics impulse — useful for a
 * directed push without the robot needing to walk through the object.
 *
 * @param {string} objectId      - Mesh name, label, or map key
 * @param {number} forceMagnitude - Impulse strength in N·s (default 6)
 * @param {number} [dirX=0]       - X component of push direction (auto-computed from robot if 0,0)
 * @param {number} [dirZ=0]       - Z component of push direction
 * @param {Object} [robot]        - Used to auto-compute direction if dirX/dirZ not supplied
 */
export function pushInteractable(objectId, forceMagnitude = 6, dirX = 0, dirZ = 0, robot = null) {
  const entry = _findInteractable(objectId)
  if (!entry || !entry.body) {
    console.warn(`[Env] pushInteractable: "${objectId}" not found`)
    return
  }

  let nx = dirX, nz = dirZ

  // Auto-compute direction: from robot to object (if direction not supplied)
  if (nx === 0 && nz === 0) {
    if (robot?.position) {
      const op = entry.body.translation()
      const dx = op.x - robot.position.x
      const dz = op.z - robot.position.z
      const len = Math.sqrt(dx * dx + dz * dz) || 1
      nx = dx / len
      nz = dz / len
    } else {
      nx = 0
      nz = -1  // fallback: push forward
    }
  } else {
    const len = Math.sqrt(nx * nx + nz * nz) || 1
    nx /= len
    nz /= len
  }

  entry.body.wakeUp()
  entry.body.applyImpulse(
    { x: nx * forceMagnitude, y: forceMagnitude * 0.08, z: nz * forceMagnitude },
    true
  )

  console.log(`[Env] Pushed "${objectId}" with force ${forceMagnitude}`)
}

/**
 * Find an interactable's world position for planning / LLM context.
 * Returns null if not found.
 *
 * @param {string} nameOrId
 * @returns {{x:number, y:number, z:number}|null}
 */
export function findInteractablePosition(nameOrId) {
  const entry = _findInteractable(nameOrId)
  if (!entry) return null
  if (entry.body) {
    const t = entry.body.translation()
    return { x: t.x, y: t.y, z: t.z }
  }
  return {
    x: entry.mesh.position.x,
    y: entry.mesh.position.y,
    z: entry.mesh.position.z,
  }
}

/**
 * Get all interactable names (labels) currently in the scene.
 * @returns {string[]}
 */
export function listInteractables() {
  const result = []
  for (const [id, entry] of _interactables) {
    result.push(entry.mesh.name || id)
  }
  return result
}

// ─── Internal: flexible object lookup ────────────────────────────────────────

function _findInteractable(nameOrId) {
  // 1. Direct map key
  if (_interactables.has(nameOrId)) return _interactables.get(nameOrId)

  // 2. By mesh name or label (case-insensitive partial match)
  const lower = nameOrId.toLowerCase()
  for (const [, entry] of _interactables) {
    const meshName  = (entry.mesh.name          || '').toLowerCase()
    const labelName = (entry.mesh.userData?.label || '').toLowerCase()
    if (meshName === lower || labelName === lower) return entry
    // Partial match: query inside mesh name, or generic kind ("box"/"ball")
    // of an auto-generated id (ibox_<ts>_<rand>) inside the query
    const kind = meshName.replace(/^i(box|ball)_\w+_\w+$/, '$1')
    if (meshName.includes(lower) || (kind !== meshName && lower.includes(kind))) return entry
  }

  return null
}

// ─── Public API: Goal Marker ──────────────────────────────────────────────────

/**
 * Create (or move) the goal marker — a glowing pulsing sphere.
 * @param {number} x
 * @param {number} z
 * @param {number} [y=0.5]   Height above floor
 */
export function setGoalMarker(x, z, y = 0.5) {
  if (!_scene) return

  if (_goalMarker) {
    // Move existing marker
    _goalMarker.mesh.position.set(x, y, z)
    _goalMarker.light.position.set(x, y + 0.5, z)
    return
  }

  // --- Sphere mesh ---
  const geo  = new THREE.SphereGeometry(0.18, 16, 16)
  const mat  = new THREE.MeshStandardMaterial({
    color:     0x00ffaa,
    emissive:  new THREE.Color(0x00ffaa),
    emissiveIntensity: 1.2,
    roughness: 0.3,
    metalness: 0.1,
  })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(x, y, z)
  mesh.name = 'goal_marker'
  _scene.add(mesh)

  // --- Outer glow ring ---
  const ringGeo = new THREE.RingGeometry(0.22, 0.28, 32)
  const ringMat = new THREE.MeshBasicMaterial({
    color: 0x00ffaa,
    side: THREE.DoubleSide,
    transparent: true,
    opacity: 0.5,
  })
  const ring = new THREE.Mesh(ringGeo, ringMat)
  ring.rotation.x = -Math.PI / 2
  ring.position.set(x, 0.02, z)
  ring.name = 'goal_ring'
  _scene.add(ring)

  // --- Point light ---
  const light = new THREE.PointLight(0x00ffaa, 1.2, 4)
  light.position.set(x, y + 0.5, z)
  _scene.add(light)

  _goalMarker = { mesh, ring, light, mat, baseY: y }

  // Animate
  _animateGoal()
}

/**
 * Remove the goal marker from the scene.
 */
export function removeGoalMarker() {
  if (!_goalMarker) return
  if (_goalAnimId) cancelAnimationFrame(_goalAnimId)

  _scene.remove(_goalMarker.mesh)
  _scene.remove(_goalMarker.ring)
  _scene.remove(_goalMarker.light)
  // 3J-3: dispose geometries AND materials to avoid WebGL memory leaks
  _goalMarker.mesh.geometry.dispose()
  _goalMarker.mesh.material.dispose()
  _goalMarker.ring.geometry.dispose()
  _goalMarker.ring.material.dispose()
  _goalMarker = null
}

/**
 * Get the current goal position.
 * @returns {{x:number, y:number, z:number}|null}
 */
export function getGoalPosition() {
  if (!_goalMarker) return null
  return {
    x: _goalMarker.mesh.position.x,
    y: _goalMarker.mesh.position.y,
    z: _goalMarker.mesh.position.z,
  }
}

// ─── Public API: Preset Layouts ───────────────────────────────────────────────

/**
 * Load a default test environment: boundary walls + scattered obstacles.
 * Great for testing navigation and perception out of the box.
 */
export function loadDefaultLayout() {
  createBoundaryWalls()

  // Central pillars (exact cylinder colliders — no more clipping)
  createObstacleCylinder({ x: -3, z: -3, radius: 0.4, h: 1.2, color: 0x3a4a6a, label: 'pillar_A' })
  createObstacleCylinder({ x:  3, z: -3, radius: 0.4, h: 1.2, color: 0x3a4a6a, label: 'pillar_B' })
  createObstacleCylinder({ x: -3, z:  3, radius: 0.4, h: 1.2, color: 0x3a4a6a, label: 'pillar_C' })
  createObstacleCylinder({ x:  3, z:  3, radius: 0.4, h: 1.2, color: 0x3a4a6a, label: 'pillar_D' })

  // Wall segments creating a loose maze
  createObstacleBox({ x: -6,   z: -1,   w: 0.4, d: 4,   h: 1.0, color: 0x2d3a50, label: 'wall_left'  })
  createObstacleBox({ x:  6,   z:  1,   w: 0.4, d: 4,   h: 1.0, color: 0x2d3a50, label: 'wall_right' })
  createObstacleBox({ x:  0,   z: -6,   w: 5,   d: 0.4, h: 1.0, color: 0x2d3a50, label: 'wall_back'  })
  createObstacleBox({ x:  0,   z:  0,   w: 2,   d: 0.4, h: 0.8, color: 0x2d3a50, label: 'divider'    })

  // ── Static crates (large, immovable) ──
  createObstacleBox({ x: -8,   z:  5,   w: 0.8, d: 0.8, h: 0.7, color: 0x4a3a2d, label: 'crate_1' })
  createObstacleBox({ x:  8,   z: -5,   w: 0.8, d: 0.8, h: 0.7, color: 0x4a3a2d, label: 'crate_2' })
  createObstacleBox({ x: -5,   z:  8,   w: 1.2, d: 0.6, h: 0.7, color: 0x4a3a2d, label: 'crate_3' })

  // ── Dynamic interactable BOXES ─────────────────────────────────────────────
  createInteractableBox({ x:  0.5,  z:  3.5, w: 0.80, h: 0.80, d: 0.80, mass: 1.0, color: 0xe8952a, label: 'box_A',    colorName: 'wooden' })
  createInteractableBox({ x: -1.8,  z:  4.5, w: 0.85, h: 0.85, d: 0.85, mass: 1.2, color: 0xc87525, label: 'box_B',    colorName: 'wooden' })
  createInteractableBox({ x:  2.5,  z:  3.0, w: 0.90, h: 0.90, d: 0.90, mass: 1.5, color: 0xa86022, label: 'box_C',    colorName: 'brown'  })
  createInteractableBox({ x: -4.0,  z: -2.0, w: 0.95, h: 0.95, d: 0.95, mass: 1.3, color: 0x9a5518, label: 'box_D',    colorName: 'brown'  })
  createInteractableBox({ x:  5.0,  z:  5.0, w: 0.85, h: 0.85, d: 0.85, mass: 1.1, color: 0xd47820, label: 'box_E',    colorName: 'wooden' })
  createInteractableBox({ x:  0.0,  z:  7.0, w: 1.20, h: 0.40, d: 0.90, mass: 0.9, color: 0xcc9944, label: 'box_flat', colorName: 'wooden' })
  createInteractableBox({ x: -6.0,  z:  4.0, w: 0.80, h: 0.80, d: 0.80, mass: 1.0, color: 0xb87030, label: 'box_F',    colorName: 'brown'  })

  // ── Dynamic interactable BALLS ─────────────────────────────────────────────
  createInteractableBall({ x: -0.5, z:  3.0, radius: 0.38, mass: 0.45, color: 0xff3322, label: 'ball_red',    colorName: 'red'    })
  createInteractableBall({ x: -2.5, z:  2.5, radius: 0.32, mass: 0.35, color: 0x2266ff, label: 'ball_blue',   colorName: 'blue'   })
  createInteractableBall({ x:  3.0, z:  6.0, radius: 0.45, mass: 0.60, color: 0x22cc44, label: 'ball_green',  colorName: 'green'  })
  createInteractableBall({ x:  1.5, z:  5.5, radius: 0.28, mass: 0.22, color: 0xffdd00, label: 'ball_yellow', colorName: 'yellow' })
  createInteractableBall({ x: -4.5, z:  1.5, radius: 0.35, mass: 0.40, color: 0xff66cc, label: 'ball_pink',   colorName: 'pink'   })
  createInteractableBall({ x:  6.0, z:  0.0, radius: 0.40, mass: 0.50, color: 0xff8800, label: 'ball_orange', colorName: 'orange' })

  // Goal marker at far end
  setGoalMarker(8, -8)

  console.log('[Env] Default layout loaded')
  console.log(`[Env] Obstacles: ${_obstacles.size}, interactables: ${_interactables.size}, goal at (8, -8)`)
}

/**
 * Load a maze layout for RL training (clear → recreate walls).
 * @param {Array<{x,z,w,d,h}>} wallSpecs  Wall specifications
 * @param {{x:number, z:number}} goal     Goal position
 */
export function loadRLLayout(wallSpecs, goal) {
  // Clear current non-boundary obstacles
  clearObstacleObjects()

  for (let i = 0; i < wallSpecs.length; i++) {
    const s = wallSpecs[i]
    createObstacleBox({
      x: s.x, z: s.z,
      w: s.w ?? 0.4, d: s.d ?? 2, h: s.h ?? 1,
      color: 0x2a3545,
    })
  }

  if (goal) setGoalMarker(goal.x, goal.z)
}

// ─── Public API: Robot Push ───────────────────────────────────────────────────

/**
 * Apply a proximity-based push impulse to all dynamic interactables near the robot.
 * Supplements the character controller's built-in impulse with a velocity-scaled
 * burst that gives consistent, satisfying collision feel.
 *
 * Call this once per frame (after physics is stepped) via the 'objects' engine system.
 *
 * @param {{x:number, y:number, z:number}} robotPos    - Robot world position
 * @param {{x:number, y:number, z:number}} robotVel    - Robot velocity (null = treat as stationary)
 * @param {number} dt                                   - Frame delta time
 */
export function applyRobotPush(robotPos, robotVel, dt = 1 / 60) {
  // Only push when robot is actually moving — avoids phantom shove when idle
  const speed = robotVel
    ? Math.sqrt(robotVel.x * robotVel.x + robotVel.z * robotVel.z)
    : 0
  if (speed < 0.08) return

  const PUSH_RADIUS   = 0.92   // m — slightly wider than robot capsule (r=0.25)
  const PUSH_STRENGTH = 14.0   // N — enough to visibly move a 0.5–1.5 kg object
  const speedScale    = Math.min(speed / 2.5, 1.0)  // ramp up with velocity

  for (const [, entry] of _interactables) {
    const body = entry.body
    if (!body || entry.held) continue

    const t  = body.translation()
    const dx = t.x - robotPos.x
    const dz = t.z - robotPos.z
    const distSq = dx * dx + dz * dz

    if (distSq < PUSH_RADIUS * PUSH_RADIUS) {
      body.wakeUp()

      const dist    = Math.sqrt(distSq) + 0.0001
      const falloff = 1.0 - dist / PUSH_RADIUS          // stronger when closer
      const scale   = PUSH_STRENGTH * falloff * speedScale * dt

      // Radial + small upward kick (natural collision feel)
      body.applyImpulse({
        x: (dx / dist) * scale,
        y:  0.18 * falloff * scale,
        z: (dz / dist) * scale,
      }, true)
    }
  }
}

/**
 * Remove and dispose all boundary wall meshes (including their EdgesGeometry
 * children and materials). Physics bodies are handled separately by world.js.
 * 3J-4: must call child.geometry.dispose() + child.material.dispose() on the
 *        LineSegments edge mesh; without this, EdgesGeometry leaks GPU memory.
 */
export function clearBoundaryWalls() {
  for (const mesh of _wallMeshes) {
    _scene.remove(mesh)
    // Dispose wall mesh own geometry + material
    mesh.geometry.dispose()
    if (Array.isArray(mesh.material)) {
      mesh.material.forEach(m => m.dispose())
    } else {
      mesh.material?.dispose()
    }
    // Dispose child LineSegments (EdgesGeometry + LineBasicMaterial)
    mesh.traverse((child) => {
      if (child.isLineSegments) {
        child.geometry.dispose()
        child.material?.dispose()
      }
    })
  }
  _wallMeshes.length = 0
}

function _makeWallMesh(w, h, d) {
  const geo = new THREE.BoxGeometry(w, h, d)
  const mat = new THREE.MeshStandardMaterial({
    color:       0x1a2a3a,
    transparent: true,
    opacity:     0.55,
    roughness:   0.6,
    metalness:   0.5,
    side:        THREE.DoubleSide,
  })

  // Edge highlight using EdgesGeometry + LineSegments
  const mesh  = new THREE.Mesh(geo, mat)
  mesh.castShadow    = false
  mesh.receiveShadow = true

  const edges    = new THREE.EdgesGeometry(geo)
  const edgeMat  = new THREE.LineBasicMaterial({ color: 0x4488ff, transparent: true, opacity: 0.4 })
  const edgeMesh = new THREE.LineSegments(edges, edgeMat)
  mesh.add(edgeMesh)

  return mesh
}

// ─── Internal: Goal Animation ─────────────────────────────────────────────────

function _animateGoal() {
  if (!_goalMarker) return

  const startTime = Date.now()

  function frame() {
    if (!_goalMarker) return
    _goalAnimId = requestAnimationFrame(frame)

    const t = (Date.now() - startTime) / 1000

    // Pulse scale
    const s = 1 + 0.18 * Math.sin(t * 2.8)
    _goalMarker.mesh.scale.setScalar(s)

    // Pulse light intensity
    _goalMarker.light.intensity = 0.8 + 0.6 * Math.sin(t * 2.8)

    // Pulse emissive
    _goalMarker.mat.emissiveIntensity = 0.8 + 0.5 * Math.sin(t * 2.8)

    // Float up/down — LB-5: use stored baseY, not live position.y which drifts
    _goalMarker.mesh.position.y = _goalMarker.baseY + 0.08 * Math.sin(t * 1.4)

    // Spin ring
    if (_goalMarker.ring) {
      _goalMarker.ring.rotation.z = t * 0.8
    }
  }

  frame()
}
