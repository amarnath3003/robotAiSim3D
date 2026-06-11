/**
 * core/manifest.js — Robot Manifest Loader & Validator
 * 
 * The manifest is the central abstraction of CausalBot. It describes
 * what a robot IS and what it CAN DO, enabling the LLM brain to reason
 * about feasibility without hardcoded assumptions.
 * 
 * This module:
 * - Loads manifest JSON files
 * - Validates against the schema
 * - Provides typed accessors for all consumers (brain, motor, perception, RL)
 * - Generates the LLM system prompt fragment describing the robot
 */

const REQUIRED_FIELDS = ['name', 'version', 'model', 'morphology', 'joints', 'capabilities', 'constraints', 'sensors']

/**
 * @typedef {Object} JointDef
 * @property {string} name
 * @property {'revolute'|'continuous'|'prismatic'|'fixed'|'floating'|'planar'} type
 * @property {string} axis
 * @property {{lower: number, upper: number, velocity: number, effort: number}} limits
 * @property {string} parent
 * @property {string} child
 * @property {number} restPosition
 * @property {string} group
 */

/**
 * @typedef {Object} Capability
 * @property {string} id
 * @property {string} description
 * @property {Object} parameters
 */

/**
 * @typedef {Object} SensorDef
 * @property {string} id
 * @property {'lidar'|'camera'|'depth_camera'|'imu'|'contact'|'proximity'|'joint_encoder'} type
 * @property {{x: number, y: number, z: number}} position
 * @property {Object} config
 */

// ─── Singleton manifest store ──────────────────────────────────────────────────

let _activeManifest = null
let _manifestPath = null

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Load a robot manifest from a JSON file path.
 * @param {string} path - Path to the manifest JSON file (relative to public root)
 * @returns {Promise<Object>} The validated manifest object
 */
export async function loadManifest(path) {
  console.log(`[Manifest] Loading: ${path}`)
  
  const response = await fetch(path)
  if (!response.ok) {
    throw new Error(`[Manifest] Failed to load ${path}: ${response.status} ${response.statusText}`)
  }
  
  const manifest = await response.json()
  
  // Validate required fields
  const errors = validate(manifest)
  if (errors.length > 0) {
    throw new Error(`[Manifest] Validation failed:\n  ${errors.join('\n  ')}`)
  }
  
  // Apply defaults
  applyDefaults(manifest)
  
  _activeManifest = Object.freeze(deepFreeze(manifest))
  _manifestPath = path
  
  console.log(`[Manifest] Loaded "${manifest.name}" v${manifest.version}`)
  console.log(`[Manifest]   Morphology: ${manifest.morphology}`)
  console.log(`[Manifest]   Joints: ${manifest.joints.length}`)
  console.log(`[Manifest]   Capabilities: ${manifest.capabilities.length}`)
  console.log(`[Manifest]   Sensors: ${manifest.sensors.length}`)
  
  return _activeManifest
}

/**
 * Get the currently active manifest.
 * @returns {Object|null}
 */
export function getManifest() {
  return _activeManifest
}

/**
 * Get the robot's name.
 */
export function getRobotName() {
  return _activeManifest?.name ?? 'Unknown Robot'
}

/**
 * Get all joint definitions.
 * @returns {JointDef[]}
 */
export function getJoints() {
  return _activeManifest?.joints ?? []
}

/**
 * Get joints filtered by group (e.g., 'left_arm', 'head').
 * @param {string} group
 * @returns {JointDef[]}
 */
export function getJointsByGroup(group) {
  return getJoints().filter(j => j.group === group)
}

/**
 * Get a specific joint by name.
 * @param {string} name
 * @returns {JointDef|undefined}
 */
export function getJoint(name) {
  return getJoints().find(j => j.name === name)
}

/**
 * Get all declared capabilities.
 * @returns {Capability[]}
 */
export function getCapabilities() {
  return _activeManifest?.capabilities ?? []
}

/**
 * Check if the robot has a specific capability.
 * @param {string} capabilityId - e.g., 'locomotion:ground', 'manipulation:grasp'
 * @returns {boolean}
 */
export function hasCapability(capabilityId) {
  return getCapabilities().some(c => c.id === capabilityId)
}

/**
 * Get a specific capability with its parameters.
 * @param {string} capabilityId
 * @returns {Capability|undefined}
 */
export function getCapability(capabilityId) {
  return getCapabilities().find(c => c.id === capabilityId)
}

/**
 * Get the robot's hard constraints.
 * @returns {Object}
 */
export function getConstraints() {
  return _activeManifest?.constraints ?? {}
}

/**
 * Get sensor definitions.
 * @returns {SensorDef[]}
 */
export function getSensors() {
  return _activeManifest?.sensors ?? []
}

/**
 * Get a specific sensor by ID.
 * @param {string} sensorId
 * @returns {SensorDef|undefined}
 */
export function getSensor(sensorId) {
  return getSensors().find(s => s.id === sensorId)
}

/**
 * Get the action space definition (for RL).
 * @returns {Object|null}
 */
export function getActionSpace() {
  return _activeManifest?.actionSpace ?? null
}

/**
 * Get the observation space definition (for RL).
 * @returns {Object|null}
 */
export function getObservationSpace() {
  return _activeManifest?.observationSpace ?? null
}

/**
 * Get the physics body configuration.
 * @returns {Object}
 */
export function getPhysicsConfig() {
  return _activeManifest?.physics ?? {}
}

/**
 * Get the 3D model configuration.
 * @returns {Object}
 */
export function getModelConfig() {
  return _activeManifest?.model ?? {}
}

/**
 * Generate a natural language description of the robot for the LLM system prompt.
 * This is the key bridge between the manifest and the AI brain.
 * @returns {string}
 */
export function generateLLMDescription() {
  if (!_activeManifest) return 'No robot loaded.'
  
  const m = _activeManifest
  const lines = []
  
  lines.push(`You are controlling a robot named "${m.name}".`)
  lines.push(`Morphology: ${m.morphology}`)
  lines.push(`Mass: ${m.physics?.mass ?? 'unknown'}kg`)
  lines.push('')
  
  // Capabilities
  lines.push('## What you CAN do:')
  for (const cap of m.capabilities) {
    lines.push(`- ${cap.id}: ${cap.description}`)
  }
  lines.push('')
  
  // Constraints
  lines.push('## Hard constraints (NEVER violate these):')
  const c = m.constraints
  if (c.maxSpeed) lines.push(`- Maximum speed: ${c.maxSpeed} m/s`)
  if (c.maxAngularSpeed) lines.push(`- Maximum rotation: ${c.maxAngularSpeed} rad/s`)
  if (c.maxReach) lines.push(`- Maximum reach: ${c.maxReach} m`)
  if (c.maxPayload) lines.push(`- Maximum lift capacity: ${c.maxPayload} kg`)
  if (c.maxJumpHeight) lines.push(`- Maximum jump height: ${c.maxJumpHeight} m`)
  if (c.canFly === false) lines.push('- CANNOT fly')
  if (c.canSwim === false) lines.push('- CANNOT swim')
  if (c.canClimb === false) lines.push('- CANNOT climb')
  if (c.custom) {
    for (const [key, value] of Object.entries(c.custom)) {
      lines.push(`- ${key}: ${value}`)
    }
  }
  lines.push('')
  
  // Joints
  lines.push('## Joints (movable parts):')
  for (const joint of m.joints) {
    const lim = joint.limits
    lines.push(`- ${joint.name} (${joint.type}, axis=${joint.axis}): range [${lim?.lower ?? '?'}°, ${lim?.upper ?? '?'}°], group="${joint.group}"`)
  }
  lines.push('')
  
  // Sensors
  lines.push('## Sensors (how you perceive the world):')
  for (const sensor of m.sensors) {
    const cfg = sensor.config || {}
    lines.push(`- ${sensor.id} (${sensor.type}): range=${cfg.range ?? '?'}m, fov=${cfg.fov ?? '?'}°`)
  }
  lines.push('')
  
  lines.push('## Important rules:')
  lines.push('- You must SCAN/LOOK before acting on objects you cannot currently see.')
  lines.push('- Always check if an action is within your constraints before attempting it.')
  lines.push('- If an instruction is physically impossible for your morphology, explain WHY and suggest alternatives.')
  
  return lines.join('\n')
}

/**
 * Generate a compact manifest summary for RL observation metadata.
 * @returns {Object}
 */
export function generateRLMetadata() {
  if (!_activeManifest) return null
  
  return {
    name: _activeManifest.name,
    morphology: _activeManifest.morphology,
    actionSpace: _activeManifest.actionSpace,
    observationSpace: _activeManifest.observationSpace,
    constraints: _activeManifest.constraints,
  }
}

// ─── Validation ────────────────────────────────────────────────────────────────

/**
 * Validate a manifest object against required structure.
 * @param {Object} manifest
 * @returns {string[]} Array of error messages (empty = valid)
 */
export function validate(manifest) {
  const errors = []
  
  // Required top-level fields
  for (const field of REQUIRED_FIELDS) {
    if (!(field in manifest)) {
      errors.push(`Missing required field: "${field}"`)
    }
  }
  
  if (errors.length > 0) return errors  // Can't continue without basic structure
  
  // Version format
  if (!/^\d+\.\d+\.\d+$/.test(manifest.version)) {
    errors.push(`Invalid version format: "${manifest.version}" (expected semver: X.Y.Z)`)
  }
  
  // Model
  if (!manifest.model.path) {
    errors.push('model.path is required')
  }
  if (!manifest.model.format) {
    errors.push('model.format is required')
  }
  
  // Joints
  if (!Array.isArray(manifest.joints)) {
    errors.push('joints must be an array')
  } else {
    const jointNames = new Set()
    for (let i = 0; i < manifest.joints.length; i++) {
      const j = manifest.joints[i]
      if (!j.name) errors.push(`joints[${i}]: missing "name"`)
      if (!j.type) errors.push(`joints[${i}]: missing "type"`)
      if (j.name && jointNames.has(j.name)) {
        errors.push(`joints[${i}]: duplicate name "${j.name}"`)
      }
      jointNames.add(j.name)
    }
  }
  
  // Capabilities
  if (!Array.isArray(manifest.capabilities)) {
    errors.push('capabilities must be an array')
  } else {
    for (let i = 0; i < manifest.capabilities.length; i++) {
      const cap = manifest.capabilities[i]
      if (!cap.id) errors.push(`capabilities[${i}]: missing "id"`)
      if (!cap.description) errors.push(`capabilities[${i}]: missing "description"`)
    }
  }
  
  // Sensors
  if (!Array.isArray(manifest.sensors)) {
    errors.push('sensors must be an array')
  } else {
    for (let i = 0; i < manifest.sensors.length; i++) {
      const s = manifest.sensors[i]
      if (!s.id) errors.push(`sensors[${i}]: missing "id"`)
      if (!s.type) errors.push(`sensors[${i}]: missing "type"`)
    }
  }
  
  return errors
}

// ─── Internal helpers ──────────────────────────────────────────────────────────

function applyDefaults(manifest) {
  // Model defaults
  manifest.model.scale = manifest.model.scale ?? 1.0
  manifest.model.originOffset = manifest.model.originOffset ?? { x: 0, y: 0, z: 0 }
  
  // Physics defaults
  if (!manifest.physics) manifest.physics = {}
  manifest.physics.mass = manifest.physics.mass ?? 5.0
  manifest.physics.colliderType = manifest.physics.colliderType ?? 'capsule'
  manifest.physics.friction = manifest.physics.friction ?? 0.7
  manifest.physics.restitution = manifest.physics.restitution ?? 0.1
  manifest.physics.linearDamping = manifest.physics.linearDamping ?? 0.5
  manifest.physics.angularDamping = manifest.physics.angularDamping ?? 0.8
  
  // Joint defaults
  for (const joint of manifest.joints) {
    joint.axis = joint.axis ?? 'z'
    joint.restPosition = joint.restPosition ?? 0
    joint.limits = joint.limits ?? { lower: -180, upper: 180, velocity: 3.0, effort: 10.0 }
  }
  
  // Constraint defaults
  manifest.constraints.canFly = manifest.constraints.canFly ?? false
  manifest.constraints.canSwim = manifest.constraints.canSwim ?? false
  manifest.constraints.canClimb = manifest.constraints.canClimb ?? false
  manifest.constraints.balanceRequired = manifest.constraints.balanceRequired ?? true
}

function deepFreeze(obj) {
  if (obj === null || typeof obj !== 'object') return obj
  Object.freeze(obj)
  for (const value of Object.values(obj)) {
    if (typeof value === 'object' && value !== null && !Object.isFrozen(value)) {
      deepFreeze(value)
    }
  }
  return obj
}
