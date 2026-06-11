/**
 * physics/world.js — Rapier3D Physics World Management
 * 
 * Handles:
 * - Physics world initialization
 * - Fixed timestep simulation
 * - Environment body creation (floors, walls, obstacles)
 * - Collision detection and callbacks
 * - Dynamic object management
 */

import RAPIER_INIT from '@dimforge/rapier3d-compat'

// ─── State ─────────────────────────────────────────────────────────────────────

let _rapier = null
let _world = null
let _initialized = false

// Track all physics bodies for cleanup
const _bodies = new Map()      // id → {rigid, collider}
const _dynamicBodies = new Map()

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the Rapier3D physics world.
 * @returns {Promise<{world: Object, RAPIER: Object}>}
 */
export async function initPhysicsWorld() {
  if (_initialized) return { world: _world, RAPIER: _rapier }
  
  // Initialize RAPIER WASM module
  _rapier = await RAPIER_INIT.init()
    .then(() => RAPIER_INIT)
    .catch(() => RAPIER_INIT)
  
  // If init() doesn't return the module, use the import directly
  if (!_rapier.World) _rapier = RAPIER_INIT
  
  // Create world with gravity
  const gravity = { x: 0.0, y: -9.81, z: 0.0 }
  _world = new _rapier.World(gravity)

  // More solver iterations → tighter contacts (stops objects sinking into floor,
  // gives crisper collision response for lightweight physics objects).
  if (_world.integrationParameters) {
    _world.integrationParameters.numSolverIterations         = 12   // default 4
    _world.integrationParameters.numAdditionalFrictionIterations = 4  // default 1
  }
  
  _initialized = true
  console.log('[Physics] Rapier3D world initialized')
  
  return { world: _world, RAPIER: _rapier }
}

/**
 * Step the physics simulation.
 * @param {number} dt - Delta time (should be fixed timestep)
 */
export function stepPhysicsWorld(dt) {
  if (!_world) return
  // RP-2: forward the actual dt to Rapier so joint velocities and collision
  //        detection scale correctly with the real frame duration
  if (_world.integrationParameters && dt > 0) {
    _world.integrationParameters.dt = dt
  }
  _world.step()
}

/**
 * Get the Rapier world instance.
 */
export function getWorld() {
  return _world
}

/**
 * Get the RAPIER module reference.
 */
export function getRapier() {
  return _rapier
}

/**
 * Create a static floor collider.
 * @param {number} y - Floor Y position
 * @param {number} halfExtent - Half-size of the floor plane
 */
export function createFloor(y = 0, halfExtent = 50) {
  if (!_world || !_rapier) return
  
  const bodyDesc = _rapier.RigidBodyDesc.fixed()
    .setTranslation(0, y - 0.05, 0)
  const body = _world.createRigidBody(bodyDesc)
  
  const colliderDesc = _rapier.ColliderDesc.cuboid(halfExtent, 0.05, halfExtent)
    .setFriction(0.75)        // realistic concrete/tile floor — good rolling resistance
    .setRestitution(0.05)     // almost no bounce off the floor
  
  _world.createCollider(colliderDesc, body)
  _bodies.set('floor', { rigid: body, collider: null })
  
  return body
}

/**
 * Create a static wall collider.
 * @param {Object} spec - {x, z, w, d, h} wall specification
 * @param {string} id - Unique identifier
 * @returns {Object} The created rigid body
 */
export function createWall(spec, id) {
  if (!_world || !_rapier) return null
  
  const { x, z, w, d, h } = spec
  
  const bodyDesc = _rapier.RigidBodyDesc.fixed()
    .setTranslation(x, h / 2, z)
  const body = _world.createRigidBody(bodyDesc)
  
  const colliderDesc = _rapier.ColliderDesc.cuboid(w / 2, h / 2, d / 2)
    .setFriction(0.5)
    .setRestitution(0.1)
  
  const collider = _world.createCollider(colliderDesc, body)
  _bodies.set(id, { rigid: body, collider })
  
  return body
}

/**
 * Create a static cylinder collider (exact match for cylindrical pillar meshes).
 * Using a real cylinder prevents the robot from clipping through the edges of pillars.
 * @param {Object} spec - {x, y, z, radius, height} where y is the base Y (default 0)
 * @param {string} id - Unique identifier
 * @returns {Object} The created rigid body
 */
export function createCylinderCollider(spec, id) {
  if (!_world || !_rapier) return null

  const { x, y = 0, z, radius, height } = spec

  // Centre the collider vertically: Rapier's cylinder origin is at its centre
  const bodyDesc = _rapier.RigidBodyDesc.fixed()
    .setTranslation(x, y + height / 2, z)
  const body = _world.createRigidBody(bodyDesc)

  // ColliderDesc.cylinder(halfHeight, radius)
  const colliderDesc = _rapier.ColliderDesc.cylinder(height / 2, radius)
    .setFriction(0.5)
    .setRestitution(0.1)

  const collider = _world.createCollider(colliderDesc, body)
  _bodies.set(id, { rigid: body, collider })

  return body
}

/**
 * Create multiple walls at once (for maze/episode reset).
 * @param {Array<{x, z, w, d, h}>} wallSpecs
 * @param {string} prefix - ID prefix for cleanup
 */
export function createWalls(wallSpecs, prefix = 'wall_rl') {
  // Remove old walls with same prefix
  removeByPrefix(prefix)
  
  for (let i = 0; i < wallSpecs.length; i++) {
    createWall(wallSpecs[i], `${prefix}_${i}`)
  }
}

/**
 * Create a dynamic physics body for an interactable object.
 * @param {string} id - Object identifier
 * @param {Object} config - Full physics configuration
 * @param {number[]} config.position   - [x, y, z] spawn position
 * @param {number[]} config.size       - [w, h, d] for box; [diameter, h, _] for sphere/cylinder
 * @param {string}  [config.shape]     - 'box' | 'sphere' | 'cylinder'  (default 'box')
 * @param {number}  [config.mass]      - kg (default 1.0)
 * @param {number}  [config.friction]  - 0–1 (default 0.6)
 * @param {number}  [config.restitution] - 0–1 bounce (default 0.3)
 * @param {number}  [config.linearDamping]  - drag (default 0.3)
 * @param {number}  [config.angularDamping] - spin drag (default 0.5)
 * @param {boolean} [config.ccd]       - continuous collision detection for fast objects
 * @returns {Object} The created Rapier rigid body
 */
export function createDynamicBody(id, config) {
  if (!_world || !_rapier) return null

  const {
    position,
    size,
    shape          = 'box',
    mass           = 1.0,
    friction       = 0.6,
    restitution    = 0.3,
    linearDamping  = 0.3,
    angularDamping = 0.5,
    ccd            = false,
  } = config

  let bodyDesc = _rapier.RigidBodyDesc.dynamic()
    .setTranslation(position[0], position[1], position[2])
    .setLinearDamping(linearDamping)
    .setAngularDamping(angularDamping)

  if (ccd) bodyDesc = bodyDesc.setCcdEnabled(true)

  const body = _world.createRigidBody(bodyDesc)

  let colliderDesc
  switch (shape) {
    case 'sphere':
      // size[0] is the diameter, so radius = size[0] / 2
      colliderDesc = _rapier.ColliderDesc.ball(size[0] / 2)
      break
    case 'cylinder':
      // size[0] = diameter, size[1] = height
      colliderDesc = _rapier.ColliderDesc.cylinder(size[1] / 2, size[0] / 2)
      break
    default: // 'box'
      colliderDesc = _rapier.ColliderDesc.cuboid(size[0] / 2, size[1] / 2, size[2] / 2)
  }

  colliderDesc
    .setMass(mass)
    .setFriction(friction)
    .setRestitution(restitution)

  const collider = _world.createCollider(colliderDesc, body)

  // Store spawn transform for episode resets
  _dynamicBodies.set(id, {
    rigid: body,
    collider,
    spawnPos: { x: position[0], y: position[1], z: position[2] },
  })

  return body
}

/**
 * Remove a physics body by ID.
 * @param {string} id
 */
export function removeBody(id) {
  const entry = _bodies.get(id) || _dynamicBodies.get(id)
  if (!entry || !_world) return
  
  _world.removeRigidBody(entry.rigid)
  _bodies.delete(id)
  _dynamicBodies.delete(id)
}

/**
 * Remove all bodies matching a prefix.
 * @param {string} prefix
 */
export function removeByPrefix(prefix) {
  const toRemove = []
  for (const id of _bodies.keys()) {
    if (id.startsWith(prefix)) toRemove.push(id)
  }
  for (const id of toRemove) {
    removeBody(id)
  }
}

/**
 * Get the position of a dynamic body.
 * @param {string} id
 * @returns {{x: number, y: number, z: number}|null}
 */
export function getBodyPosition(id) {
  const entry = _dynamicBodies.get(id)
  if (!entry) return null
  const t = entry.rigid.translation()
  return { x: t.x, y: t.y, z: t.z }
}

/**
 * Reset the physics world (remove all non-permanent bodies).
 * Dynamic objects are teleported back to their spawn positions with zero velocity.
 */
export function resetWorld() {
  // Remove all RL walls
  removeByPrefix('wall_rl')
  
  // Restore dynamic bodies to spawn transforms and zero velocity
  for (const [id, entry] of _dynamicBodies) {
    if (!entry.spawnPos) continue
    const { x, y, z } = entry.spawnPos
    // wakeUp=true forces Rapier to re-simulate from the new position
    entry.rigid.setTranslation({ x, y, z }, true)
    entry.rigid.setLinvel({ x: 0, y: 0, z: 0 }, true)
    entry.rigid.setAngvel({ x: 0, y: 0, z: 0 }, true)
  }
}

/**
 * Cleanup: destroy the physics world entirely.
 */
export function destroyPhysicsWorld() {
  if (_world) {
    _world.free()
    _world = null
  }
  _bodies.clear()
  _dynamicBodies.clear()
  _initialized = false
}
