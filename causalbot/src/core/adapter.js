/**
 * core/adapter.js — Universal Robot Adapter
 * 
 * This is the key abstraction that makes CausalBot robot-agnostic.
 * Given a manifest and a 3D model, it:
 * 
 * 1. Loads the GLB/GLTF model
 * 2. Discovers bones/joints in the model
 * 3. Maps manifest joint definitions to model bones
 * 4. Creates physics bodies according to manifest config
 * 5. Exposes a uniform control interface regardless of robot morphology
 * 
 * The same adapter works for a biped, quadruped, arm, drone — anything
 * described by a valid manifest.
 */

import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
// XM-1: removed dead manifest imports — manifest is passed as a parameter throughout

// Minimum Y the robot body centre may reach (capsule halfHeight + radius = 0.35 + 0.25 = 0.60)
// Adjust if the manifest changes collider dimensions.
const _FLOOR_Y = 0.60

// ─── Robot Instance Class ──────────────────────────────────────────────────────

/**
 * A loaded robot instance with mesh, physics, and joint control.
 * Created by loadRobot() — do not instantiate directly.
 */
export class RobotInstance {
  /**
   * @param {Object}  manifest
   * @param {THREE.Group} mesh
   * @param {Map}     bones
   * @param {{body, collider, controller}} physicsResult  — from createPhysicsBody
   * @param {Object}  physicsWorld  — Rapier world (for character controller)
   */
  constructor(manifest, mesh, bones, physicsResult, physicsWorld) {
    this.manifest = manifest
    this.mesh = mesh
    this.bones = bones           // Map<jointName, THREE.Bone>

    // Physics references
    const pr          = physicsResult || {}
    this.physicsBody  = pr.body       ?? null
    this._collider    = pr.collider   ?? null
    this._controller  = pr.controller ?? null
    this._world       = physicsWorld  ?? null
    
    // Joint state tracking
    this.jointStates = new Map()  // Map<jointName, {position, velocity, effort}>
    for (const joint of manifest.joints) {
      this.jointStates.set(joint.name, {
        position: joint.restPosition ?? 0,  // Current angle (degrees) or position (meters)
        velocity: 0,
        effort: 0,
        target: joint.restPosition ?? 0,
      })
    }
    
    // Robot world state
    this.position = new THREE.Vector3(0, 0, 0)
    this.orientation = new THREE.Quaternion()
    this.velocity = new THREE.Vector3(0, 0, 0)
    this.angularVelocity = 0
    
    // Status
    this.status = 'idle'  // idle | executing | planning | failed
    this.heldObjects = []
    
    // Sensor data (populated by perception system)
    this.sensorData = new Map()
  }
  
  /**
   * Get the current state of a joint.
   * @param {string} jointName
   * @returns {{position: number, velocity: number, effort: number, target: number}|null}
   */
  getJointState(jointName) {
    return this.jointStates.get(jointName) ?? null
  }
  
  /**
   * Set a joint target position (motor controller will interpolate).
   * Automatically clamps to manifest limits.
   * @param {string} jointName
   * @param {number} targetPosition - Target in degrees (revolute) or meters (prismatic)
   */
  setJointTarget(jointName, targetPosition) {
    const state = this.jointStates.get(jointName)
    if (!state) {
      console.warn(`[Adapter] Unknown joint: "${jointName}"`)
      return
    }
    
    const jointDef = this.manifest.joints.find(j => j.name === jointName)
    if (!jointDef) return
    
    // Clamp to limits
    const limits = jointDef.limits
    if (limits) {
      targetPosition = Math.max(limits.lower, Math.min(limits.upper, targetPosition))
    }
    
    state.target = targetPosition
  }
  
  /**
   * Set multiple joints at once (by group or individually).
   * @param {Object} targets - {jointName: targetPosition, ...}
   */
  setJointTargets(targets) {
    for (const [name, position] of Object.entries(targets)) {
      this.setJointTarget(name, position)
    }
  }
  
  /**
   * Set all joints in a group to the same target.
   * @param {string} group - Joint group name (e.g., 'left_arm')
   * @param {number} targetPosition
   */
  setGroupTarget(group, targetPosition) {
    for (const joint of this.manifest.joints) {
      if (joint.group === group) {
        this.setJointTarget(joint.name, targetPosition)
      }
    }
  }
  
  /**
   * Move the robot in its local forward direction.
   * Clamped to manifest maxSpeed.
   * @param {number} speed - Desired speed in m/s
   */
  moveForward(speed) {
    const maxSpeed = this.manifest.constraints.maxSpeed ?? 2.5
    // Allow negative speed for backward movement (S key / reverse skill)
    speed = Math.max(-maxSpeed, Math.min(maxSpeed, speed))
    
    // Convert local forward to world direction
    // Robot GLB faces +Z, so forward is +Z (not the Three.js camera default of -Z)
    const forward = new THREE.Vector3(0, 0, 1)
    forward.applyQuaternion(this.orientation)
    
    this.velocity.copy(forward.multiplyScalar(speed))
  }
  
  /**
   * Rotate the robot.
   * Clamped to manifest maxAngularSpeed.
   * @param {number} angularSpeed - Desired rotation in rad/s (positive = left)
   */
  rotate(angularSpeed) {
    const maxAngular = this.manifest.constraints.maxAngularSpeed ?? 4.0
    angularSpeed = Math.max(-maxAngular, Math.min(maxAngular, angularSpeed))
    this.angularVelocity = angularSpeed
  }
  
  /**
   * Stop all movement.
   */
  stop() {
    this.velocity.set(0, 0, 0)
    this.angularVelocity = 0
  }
  
  /**
   * Get all joint names grouped by their logical group.
   * @returns {Map<string, string[]>}
   */
  getJointGroups() {
    const groups = new Map()
    for (const joint of this.manifest.joints) {
      const group = joint.group ?? 'ungrouped'
      if (!groups.has(group)) groups.set(group, [])
      groups.get(group).push(joint.name)
    }
    return groups
  }
  
  /**
   * Check if the robot can perform a specific action based on manifest constraints.
   * @param {string} action - e.g., 'jump', 'fly', 'grasp'
   * @param {Object} params - Action-specific parameters
   * @returns {{feasible: boolean, reason: string}}
   */
  checkFeasibility(action, params = {}) {
    const constraints = this.manifest.constraints
    const capabilities = this.manifest.capabilities
    
    switch (action) {
      case 'jump':
        if ((constraints.maxJumpHeight ?? 0) <= 0) {
          return { feasible: false, reason: 'This robot cannot jump (maxJumpHeight = 0).' }
        }
        return { feasible: true, reason: `Can jump up to ${constraints.maxJumpHeight}m.` }
      
      case 'fly':
        if (!constraints.canFly) {
          return { feasible: false, reason: 'This robot cannot fly.' }
        }
        return { feasible: true, reason: 'Flight capable.' }
      
      case 'grasp': {
        // RC-2: wrapped in block so const is scoped correctly
        const graspCap = capabilities.find(c => c.id === 'manipulation:grasp')
        if (!graspCap) {
          return { feasible: false, reason: 'This robot has no grasping capability.' }
        }
        if (params.objectMass && params.objectMass > (constraints.maxPayload ?? 0)) {
          return { feasible: false, reason: `Object mass (${params.objectMass}kg) exceeds max payload (${constraints.maxPayload}kg).` }
        }
        if (params.distance && params.distance > (constraints.maxReach ?? 0)) {
          return { feasible: false, reason: `Object distance (${params.distance}m) exceeds max reach (${constraints.maxReach}m).` }
        }
        return { feasible: true, reason: 'Grasp is feasible.' }
      }
      
      case 'move':
        if (params.speed && params.speed > (constraints.maxSpeed ?? 0)) {
          return { feasible: false, reason: `Requested speed (${params.speed}m/s) exceeds max speed (${constraints.maxSpeed}m/s). Will clamp.` }
        }
        return { feasible: true, reason: 'Movement is feasible.' }
      
      default: {
        // RC-2: wrapped in block so const is scoped correctly
        // Check if any capability matches
        const matchingCap = capabilities.find(c => c.id.includes(action))
        if (matchingCap) {
          return { feasible: true, reason: matchingCap.description }
        }
        return { feasible: false, reason: `Unknown action "${action}" — no matching capability found in manifest.` }
      }
    }
  }
  
  /**
   * Update joint positions by interpolating toward targets (called each frame).
   * @param {number} dt - Delta time in seconds
   */
  updateJoints(dt) {
    // EC-5: guard against dt=0 to avoid divide-by-zero in velocity calc
    if (dt <= 0) return

    for (const [jointName, state] of this.jointStates) {
      const jointDef = this.manifest.joints.find(j => j.name === jointName)
      if (!jointDef) continue
      
      const maxVelocity = jointDef.limits?.velocity ?? 3.0  // rad/s (manifest convention)
      const diff = state.target - state.position
      
      if (Math.abs(diff) < 0.01) {
        state.velocity = 0
        continue
      }
      
      // LB-1: maxVelocity is in rad/s; convert to deg/step for position delta
      const maxDelta = maxVelocity * dt * (180 / Math.PI)  // deg per physics step
      const delta = Math.sign(diff) * Math.min(Math.abs(diff), maxDelta)
      
      state.position += delta
      // LB-1: store velocity in rad/s (consistent with manifest convention)
      state.velocity = (delta * Math.PI / 180) / dt
      
      // Apply to bone if mapped
      const bone = this.bones.get(jointName)
      if (bone) {
        const axis = jointDef.axis || 'z'
        const radians = state.position * (Math.PI / 180)
        
        // Reset rotation then apply
        bone.rotation.set(0, 0, 0)
        switch (axis) {
          case 'x': case '-x':
            bone.rotation.x = axis.startsWith('-') ? -radians : radians
            break
          case 'y': case '-y':
            bone.rotation.y = axis.startsWith('-') ? -radians : radians
            break
          case 'z': case '-z':
            bone.rotation.z = axis.startsWith('-') ? -radians : radians
            break
        }
      }
    }
  }
  
  /**
   * Update robot position from physics body (called after physics step).
   * Applies manifest originOffset so mesh visual centre aligns with collider.
   */
  syncFromPhysics() {
    if (!this.physicsBody) return
    
    const translation = this.physicsBody.translation()
    this.position.set(translation.x, translation.y, translation.z)
    
    const rotation = this.physicsBody.rotation()
    this.orientation.set(rotation.x, rotation.y, rotation.z, rotation.w)
    
    // Sync mesh — add originOffset so the GLB visual lines up with the capsule
    if (this.mesh) {
      const off = this.manifest.model?.originOffset ?? { x: 0, y: 0, z: 0 }
      this.mesh.position.set(
        this.position.x + off.x,
        this.position.y + off.y,
        this.position.z + off.z
      )
      this.mesh.quaternion.copy(this.orientation)
    }
  }
  
  /**
   * Apply movement commands to physics body (called before physics step).
   * Uses Rapier KinematicCharacterController when available so the robot
   * slides along walls instead of tunnelling through them.
   * @param {number} dt
   */
  applyToPhysics(dt) {
    if (!this.physicsBody) return
    // EC-5: skip step if dt is zero (avoids divide-by-zero and zero-length moves)
    if (dt <= 0) return

    // ── Rotation ────────────────────────────────────────────────────────────
    if (this.angularVelocity !== 0) {
      const euler     = new THREE.Euler(0, this.angularVelocity * dt, 0)
      const deltaQuat = new THREE.Quaternion().setFromEuler(euler)
      this.orientation.multiply(deltaQuat)

      this.physicsBody.setNextKinematicRotation({
        x: this.orientation.x,
        y: this.orientation.y,
        z: this.orientation.z,
        w: this.orientation.w,
      })
    }

    // ── Translation (with wall collision) ───────────────────────────────────
    if (this.velocity.lengthSq() > 0.0001) {
      const desired = {
        x: this.velocity.x * dt,
        y: 0,   // gravity handled by snap-to-ground; no free-fall for ground robot
        z: this.velocity.z * dt,
      }

      if (this._controller && this._collider) {
        // Collision-safe movement — character controller resolves wall contacts
        // and slides the robot along surfaces rather than stopping dead.
        try {
          this._controller.computeColliderMovement(this._collider, desired)
          const safe = this._controller.computedMovement()
          const cur  = this.physicsBody.translation()
          this.physicsBody.setNextKinematicTranslation({
            x: cur.x + safe.x,
            // Clamp Y: never let the robot sink below its spawn height
            y: Math.max(cur.y + safe.y, _FLOOR_Y),
            z: cur.z + safe.z,
          })
        } catch (e) {
          // Fallback — direct move (no collision)
          const np = this.position.clone().add(this.velocity.clone().multiplyScalar(dt))
          this.physicsBody.setNextKinematicTranslation({ x: np.x, y: np.y, z: np.z })
        }
      } else {
        // No character controller — direct move (no collision, original behaviour)
        const np = this.position.clone().add(this.velocity.clone().multiplyScalar(dt))
        this.physicsBody.setNextKinematicTranslation({ x: np.x, y: np.y, z: np.z })
      }
    }
  }
  
  /**
   * Get a serializable state snapshot (for RL observations, debugging, etc.)
   * @returns {Object}
   */
  getStateSnapshot() {
    const joints = {}
    for (const [name, state] of this.jointStates) {
      joints[name] = { ...state }
    }
    
    return {
      position: { x: this.position.x, y: this.position.y, z: this.position.z },
      orientation: { x: this.orientation.x, y: this.orientation.y, z: this.orientation.z, w: this.orientation.w },
      velocity: { x: this.velocity.x, y: this.velocity.y, z: this.velocity.z },
      angularVelocity: this.angularVelocity,
      joints,
      status: this.status,
      heldObjects: [...this.heldObjects],
    }
  }
}

// ─── Robot Loader ──────────────────────────────────────────────────────────────

/**
 * Load a robot from its manifest and create a fully functional RobotInstance.
 * This is the main entry point for the adapter system.
 * 
 * @param {Object} manifest - A loaded and validated manifest object
 * @param {THREE.Scene} scene - The Three.js scene to add the robot to
 * @param {Object} rapierWorld - The Rapier3D physics world
 * @param {Object} RAPIER - The Rapier3D module reference
 * @returns {Promise<RobotInstance>}
 */
export async function loadRobot(manifest, scene, rapierWorld, RAPIER) {
  console.log(`[Adapter] Loading robot: "${manifest.name}"`)
  
  const modelConfig = manifest.model
  const physicsConfig = manifest.physics
  
  // 1. Load 3D model
  const mesh = await loadModel(modelConfig)

  // 1b. For robots whose GLB has no armature, build joint pivot nodes from
  //     named mesh nodes so discoverBones can map them.
  setupArmPivots(mesh)

  scene.add(mesh)
  
  // 2. Discover and map bones
  const bones = discoverBones(mesh, manifest.joints)
  
  // 3. Create physics body + character controller
  const physicsResult = createPhysicsBody(rapierWorld, RAPIER, physicsConfig)
  
  // 4. Create robot instance (pass world for use in applyToPhysics)
  const robot = new RobotInstance(manifest, mesh, bones, physicsResult, rapierWorld)
  
  console.log(`[Adapter] Robot "${manifest.name}" ready.`)
  console.log(`[Adapter]   Bones mapped: ${bones.size}/${manifest.joints.length}`)
  console.log(`[Adapter]   Physics body: ${physicsConfig.colliderType}`)
  console.log(`[Adapter]   Collision controller: ${physicsResult.controller ? 'yes' : 'no (fallback)'}`)
  
  return robot
}

// ─── Internal: Arm Pivot Setup ─────────────────────────────────────────────────

/**
 * For robots whose GLB contains no armature/rig, create explicit joint pivot
 * nodes so that discoverBones() can map them and updateJoints() can rotate them.
 *
 * Strategy for the single-arm robot:
 *   - Find the `robot_arm` mesh (Cylinder) and `robot_hand` mesh (Torus).
 *   - The arm's geometry starts at Z≈0.004 (shoulder end) and extends to Z≈1.004.
 *   - Create an Object3D named "left_shoulder" positioned at the shoulder end
 *     of the arm, then use Three.js `attach()` to reparent the arm and hand
 *     meshes into it while preserving their world transforms.
 *   - After this, rotating left_shoulder.rotation.x swings the whole arm.
 *
 * Joint axis convention (see manifest: "axis": "-x"):
 *   angle=  0°  → arm points +Z (forward, rest position as modelled)
 *   angle= 90°  → arm points +Y (straight up)
 *   angle=-90°  → arm points -Y (straight down)
 */
function setupArmPivots(mesh) {
  // Locate the arm and hand meshes by their GLB node names
  let armMesh  = null
  let handMesh = null

  mesh.traverse(child => {
    if (child.name === 'robot_arm')  armMesh  = child
    if (child.name === 'robot_hand') handMesh = child
  })

  if (!armMesh) {
    console.warn('[Adapter] robot_arm mesh not found — arm pivot not created.')
    return
  }

  // Shoulder pivot = the base (near/body end) of the arm cylinder.
  // Derived from GLB bounding box: arm X centre ≈ 0.528, Y centre ≈ 0.062, Z start ≈ 0.004.
  const pivotX = (0.4528777 + 0.6023203) / 2   // ≈ 0.5276
  const pivotY = (-0.013193 + 0.136249) / 2     // ≈ 0.0615
  const pivotZ = 0.004                           // shoulder end of the arm cylinder

  // Use Object3D (not Group) so discoverBones doesn't skip it with its Group filter.
  const shoulderPivot = new THREE.Object3D()
  shoulderPivot.name = 'left_shoulder'           // exact match to manifest joint name
  shoulderPivot.position.set(pivotX, pivotY, pivotZ)

  mesh.add(shoulderPivot)

  // attach() reparents while preserving each mesh's world transform,
  // so the arm and hand don't jump when we add them to the pivot.
  shoulderPivot.attach(armMesh)
  if (handMesh) shoulderPivot.attach(handMesh)

  console.log(`[Adapter] Arm pivot "left_shoulder" created at (${pivotX.toFixed(3)}, ${pivotY.toFixed(3)}, ${pivotZ})`)
}

// ─── Internal: Model Loading ───────────────────────────────────────────────────

async function loadModel(modelConfig) {
  const loader = new GLTFLoader()
  
  return new Promise((resolve) => {
    loader.load(
      modelConfig.path,
      (gltf) => {
        const model = gltf.scene
        
        // Apply scale
        const scale = modelConfig.scale ?? 1.0
        model.scale.set(scale, scale, scale)
        
        // Enable shadows
        model.traverse((child) => {
          if (child.isMesh) {
            child.castShadow = true
            child.receiveShadow = true
          }
        })
        
        console.log(`[Adapter] Model loaded: ${modelConfig.path}`)
        resolve(model)
      },
      undefined,
      (error) => {
        console.warn(`[Adapter] GLB not found (${modelConfig.path}), using fallback mesh.`, error.message)
        resolve(buildFallbackMesh(modelConfig))
      }
    )
  })
}

/**
 * Build a simple programmatic robot shape when the GLB is unavailable.
 * Body = capsule-ish stack of geometries; serves as a visible stand-in.
 */
function buildFallbackMesh(modelConfig) {
  const group = new THREE.Group()
  group.name = 'robot_fallback'

  const scale = modelConfig.scale ?? 1.0

  // Torso
  const torsoGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.45, 12)
  const bodyMat  = new THREE.MeshStandardMaterial({ color: 0x4488ff, roughness: 0.4, metalness: 0.6 })
  const torso    = new THREE.Mesh(torsoGeo, bodyMat)
  torso.position.y = 0.22
  torso.castShadow = true

  // Head
  const headGeo  = new THREE.SphereGeometry(0.16, 12, 8)
  const headMat  = new THREE.MeshStandardMaterial({ color: 0x88ccff, roughness: 0.3, metalness: 0.5 })
  const head     = new THREE.Mesh(headGeo, headMat)
  head.position.y = 0.60
  head.castShadow = true

  // Eye glow (small emissive sphere)
  const eyeGeo  = new THREE.SphereGeometry(0.04, 8, 6)
  const eyeMat  = new THREE.MeshStandardMaterial({ color: 0x00ffff, emissive: new THREE.Color(0x00ffff), emissiveIntensity: 2.0 })
  const eye     = new THREE.Mesh(eyeGeo, eyeMat)
  eye.position.set(0.07, 0.62, 0.13)

  // Wheels (two side discs)
  const wheelGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.08, 16)
  const wheelMat = new THREE.MeshStandardMaterial({ color: 0x222222, roughness: 0.8, metalness: 0.4 })
  const wheelL   = new THREE.Mesh(wheelGeo, wheelMat)
  const wheelR   = new THREE.Mesh(wheelGeo, wheelMat)
  wheelL.rotation.z = Math.PI / 2
  wheelR.rotation.z = Math.PI / 2
  wheelL.position.set(-0.28, 0.12, 0)
  wheelR.position.set( 0.28, 0.12, 0)
  wheelL.castShadow = true
  wheelR.castShadow = true

  group.add(torso, head, eye, wheelL, wheelR)
  group.scale.setScalar(scale)

  console.log('[Adapter] Fallback robot mesh built.')
  return group
}

// ─── Internal: Bone Discovery ──────────────────────────────────────────────────

/**
 * Discover bones in the loaded model and map them to manifest joint definitions.
 * Uses fuzzy name matching to handle different naming conventions.
 */
function discoverBones(mesh, jointDefs) {
  const bones = new Map()
  const allBones = []
  
  // Collect all bones/objects in the scene graph
  // LB-2: only keep actual Bone nodes OR named Object3D leaves that aren't
  //        generic containers (Scene, Group) — avoids matching the whole tree
  mesh.traverse((child) => {
    if (child.isBone) {
      allBones.push(child)
    } else if (
      child.isObject3D &&
      child.name &&
      child.name.length > 2 &&
      child.type !== 'Scene' &&
      child.type !== 'Group'
    ) {
      allBones.push(child)
    }
  })
  
  console.log(`[Adapter] Found ${allBones.length} bones/objects in model`)
  
  for (const jointDef of jointDefs) {
    const bone = findMatchingBone(allBones, jointDef.name)
    if (bone) {
      bones.set(jointDef.name, bone)
      console.log(`[Adapter]   Mapped joint "${jointDef.name}" → bone "${bone.name}"`)
    } else {
      console.warn(`[Adapter]   No bone found for joint "${jointDef.name}" — will use virtual joint`)
    }
  }
  
  return bones
}

/**
 * Fuzzy match a joint name to a bone in the model.
 * Handles common naming differences (camelCase, snake_case, spaces, etc.)
 */
function findMatchingBone(bones, jointName) {
  const normalized = jointName.toLowerCase().replace(/[_\-\s]/g, '')
  
  // Exact match first
  let match = bones.find(b => b.name === jointName)
  if (match) return match
  
  // Case-insensitive match
  match = bones.find(b => b.name.toLowerCase() === jointName.toLowerCase())
  if (match) return match
  
  // Normalized match (strip separators)
  match = bones.find(b => b.name.toLowerCase().replace(/[_\-\s]/g, '') === normalized)
  if (match) return match
  
  // Partial match (bone name contains joint name or vice versa)
  match = bones.find(b => {
    const bNorm = b.name.toLowerCase().replace(/[_\-\s]/g, '')
    return bNorm.includes(normalized) || normalized.includes(bNorm)
  })
  if (match) return match
  
  return null
}

// ─── Internal: Physics Body Creation ───────────────────────────────────────────

/**
 * Create the kinematic rigid body, its collider, and a character controller.
 * @returns {{ body: Object, collider: Object, controller: Object|null }}
 */
function createPhysicsBody(world, RAPIER, physicsConfig) {
  if (!world || !RAPIER) {
    console.warn('[Adapter] No physics world — creating robot without physics body')
    return { body: null, collider: null, controller: null }
  }

  // Kinematic body — we drive it, physics just provides collision queries
  const bodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased()
    .setTranslation(0, _FLOOR_Y, 0)
  const body = world.createRigidBody(bodyDesc)

  // Collider shape from manifest
  let colliderDesc
  const dims = physicsConfig.colliderDimensions || {}

  switch (physicsConfig.colliderType) {
    case 'capsule':
      colliderDesc = RAPIER.ColliderDesc.capsule(
        (dims.height ?? 0.7) / 2,
        dims.radius ?? 0.25
      )
      break
    case 'box':
      colliderDesc = RAPIER.ColliderDesc.cuboid(
        (dims.width  ?? 0.5) / 2,
        (dims.height ?? 0.7) / 2,
        (dims.depth  ?? 0.5) / 2
      )
      break
    case 'cylinder':
      colliderDesc = RAPIER.ColliderDesc.cylinder(
        (dims.height ?? 0.7) / 2,
        dims.radius ?? 0.25
      )
      break
    default:
      colliderDesc = RAPIER.ColliderDesc.capsule(0.35, 0.25)
  }

  colliderDesc
    .setFriction(physicsConfig.friction ?? 0.7)
    .setRestitution(physicsConfig.restitution ?? 0.1)
    .setMass(physicsConfig.mass ?? 5.0)

  const collider = world.createCollider(colliderDesc, body)

  // ── KinematicCharacterController ─────────────────────────────────────────
  // Gives the robot proper wall-sliding collision for free.
  let controller = null
  try {
    controller = world.createCharacterController(0.02)  // 2 cm skin/gap
    controller.setSlideEnabled(true)
    controller.setMaxSlopeClimbAngle(45 * Math.PI / 180)
    controller.setMinSlopeSlideAngle(30 * Math.PI / 180)
    controller.setApplyImpulsesToDynamicBodies(true)
    controller.setSnapToGroundDistance(0.12)  // snap back onto floor after tiny jumps
    console.log('[Adapter] KinematicCharacterController ready')
  } catch (e) {
    console.warn('[Adapter] KinematicCharacterController unavailable — falling back to direct movement:', e.message)
  }

  return { body, collider, controller }
}
