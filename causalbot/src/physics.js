import RAPIER from '@dimforge/rapier3d-compat'
import { state } from './state.js'
import * as THREE from 'three'

let world
let eventQueue
let accumulator = 0
let debugBody
const FIXED_STEP = 1 / 60  // 60Hz fixed timestep

export async function initPhysics() {
  await RAPIER.init()

  world = new RAPIER.World({ x: 0.0, y: -9.81, z: 0.0 })
  world.numSolverIterations = 8  // more iterations = more accurate
  eventQueue = new RAPIER.EventQueue(true)

  // Floor — maximum friction, barely bounces
  const floor = world.createRigidBody(RAPIER.RigidBodyDesc.fixed())
  world.createCollider(
    RAPIER.ColliderDesc.cuboid(10, 0.05, 10)
      .setFriction(0.9)
      .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max)
      .setRestitution(0.05)
      .setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Min),
    floor
  )

  // Room walls — objects bounce off them
  ;[
    { pos: [0, 1.5, -3.0], size: [3.5, 1.5, 0.05] },
    { pos: [0, 1.5,  3.0], size: [3.5, 1.5, 0.05] },
    { pos: [-3.0, 1.5, 0], size: [0.05, 1.5, 3.5] },
    { pos: [ 3.0, 1.5, 0], size: [0.05, 1.5, 3.5] },
  ].forEach(({ pos, size }) => {
    const wb = world.createRigidBody(RAPIER.RigidBodyDesc.fixed().setTranslation(...pos))
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(...size).setFriction(0.3).setRestitution(0.4),
      wb
    )
  })

  function getConvexHullCollider(id, fallbackDesc) {
    const root = state.scene.three?.getObjectByName(id)
    if (!root) return fallbackDesc

    const vertices = []
    
    // Ensure all matrices are up to date
    if (state.scene.three) state.scene.three.updateMatrixWorld(true)

    root.traverse(c => {
      if (c.isMesh && c.geometry && c.geometry.attributes.position) {
        const pos = c.geometry.attributes.position
        
        let m = new THREE.Matrix4()
        if (c !== root) {
          const rootInv = new THREE.Matrix4().copy(root.matrixWorld).invert()
          m.copy(c.matrixWorld).premultiply(rootInv)
        }
        
        const v = new THREE.Vector3()
        for (let i = 0; i < pos.count; i++) {
          v.fromBufferAttribute(pos, i)
          v.applyMatrix4(m)
          vertices.push(v.x, v.y, v.z)
        }
      }
    })

    if (vertices.length > 0) {
      const f32 = new Float32Array(vertices)
      const desc = RAPIER.ColliderDesc.convexHull(f32)
      if (desc) return desc
    }
    return fallbackDesc
  }

  // Objects — each with real-world tuned physics values
  setupObject('object_glass', {
    collider: () => getConvexHullCollider('object_glass', RAPIER.ColliderDesc.cylinder(0.07, 0.04)),
    mass: 0.22,
    friction: 0.65,
    restitution: 0.05,
    linearDamping: 0.3,
    angularDamping: 0.5,
    gravityScale: 1.0,
  })

  setupObject('object_box', {
    collider: () => getConvexHullCollider('object_box', RAPIER.ColliderDesc.cuboid(0.17, 0.17, 0.17)),
    mass: 2.8,
    friction: 0.85,
    restitution: 0.08,
    linearDamping: 0.7,
    angularDamping: 0.9,
    gravityScale: 1.0,
  })

  setupObject('object_ball', {
    collider: () => getConvexHullCollider('object_ball', RAPIER.ColliderDesc.ball(0.13)),
    mass: 0.45,
    friction: 0.3,
    restitution: 0.82,
    linearDamping: 0.02,
    angularDamping: 0.05,
    gravityScale: 1.0,
  })

  // Debug Robot Physics — Dynamic with locked rotations
  const drp = state.debugRobot.position
  debugBody = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(drp[0], drp[1], drp[2])
      .setLinearDamping(0.5)
      .setAngularDamping(2.0)
  )
  debugBody.setEnabledRotations(false, true, false, true)
  
  const debugRobotColliderDesc = getConvexHullCollider('debugRobot', RAPIER.ColliderDesc.capsule(0.15, 0.1))
  debugRobotColliderDesc.setFriction(0.2).setRestitution(0.1)

  world.createCollider(debugRobotColliderDesc, debugBody)
  state.debugRobot._body = debugBody

  // AI Robot Physics — Kinematic position based
  const arp = state.robot.position
  const aiBody = world.createRigidBody(
    RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(arp[0], arp[1], arp[2])
  )
  const aiRobotColliderDesc = getConvexHullCollider('aiRobot', RAPIER.ColliderDesc.capsule(0.15, 0.1))
  aiRobotColliderDesc.setFriction(0.2).setRestitution(0.1)
  world.createCollider(aiRobotColliderDesc, aiBody)
  state.robot._body = aiBody

  state.scene.rapierWorld = world
  console.log('Physics ready — full simulation active')
}

function setupObject(id, cfg) {
  const obj = state.world.objects[id]
  if (!obj) return

  const body = world.createRigidBody(
    RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(...obj.position)
      .setLinearDamping(cfg.linearDamping)
      .setAngularDamping(cfg.angularDamping)
      .setGravityScale(cfg.gravityScale)
      .setCcdEnabled(true)   // prevents tunnelling through floor at high speed
  )
  body.wakeUp()

  world.createCollider(
    cfg.collider()
      .setMass(cfg.mass)
      .setFriction(cfg.friction)
      .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Average)
      .setRestitution(cfg.restitution)
      .setRestitutionCombineRule(RAPIER.CoefficientCombineRule.Max)
      .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS),
    body
  )

  obj._body = body
  obj._cfg = cfg
  obj._prevVel = { x: 0, y: 0, z: 0 }
}

export function stepPhysics(delta) {
  if (!world) return

  // Fixed timestep accumulator — decouples physics from frame rate
  accumulator += Math.min(delta, 0.05)  // cap at 50ms to prevent spiral of death
  while (accumulator >= FIXED_STEP) {
    world.step(eventQueue)
    accumulator -= FIXED_STEP
  }

  // Sync AI robot physics to its visual interpolation
  const aiMesh = state.scene.three?.getObjectByName('aiRobot')
  if (aiMesh && state.robot._body) {
    state.robot._body.setNextKinematicTranslation(aiMesh.position)
    state.robot._body.setNextKinematicRotation(aiMesh.quaternion)
  }

  // Process collision events — detect hard impacts
  eventQueue.drainCollisionEvents((h1, h2, started) => {
    if (!started) return
    checkImpactBreakage()
  })

  // Sync physics world → Three.js meshes
  Object.values(state.world.objects).forEach(obj => {
    if (!obj._body || obj.status === 'held') return

    const t = obj._body.translation()
    const r = obj._body.rotation()
    const v = obj._body.linvel()

    // Check impact speed for breakage (compare velocity change)
    const dvx = v.x - (obj._prevVel?.x || 0)
    const dvy = v.y - (obj._prevVel?.y || 0)
    const dvz = v.z - (obj._prevVel?.z || 0)
    const impactDeltaV = Math.sqrt(dvx*dvx + dvy*dvy + dvz*dvz)

    if (
      impactDeltaV > 4.0 &&
      obj.id === 'object_glass' &&
      obj.status === 'intact'
    ) {
      obj.status = 'broken'
      shatterGlass(obj)
    }

    obj._prevVel = { x: v.x, y: v.y, z: v.z }

    // Reset objects that fall off world
    if (t.y < -3) {
      obj._body.setTranslation({ x: obj.position[0], y: 1.5, z: obj.position[2] }, true)
      obj._body.setLinvel({ x: 0, y: 0, z: 0 }, true)
      obj._body.setAngvel({ x: 0, y: 0, z: 0 }, true)
      obj.status = 'intact'
      const mesh = state.scene.three?.getObjectByName(obj.id)
      if (mesh) mesh.visible = true
      return
    }

    obj.position[0] = t.x
    obj.position[1] = t.y
    obj.position[2] = t.z

    const mesh = state.scene.three?.getObjectByName(obj.id)
    if (mesh) {
      mesh.position.set(t.x, t.y, t.z)
      mesh.quaternion.set(r.x, r.y, r.z, r.w)
    }

    // Sync shards
    obj._shards?.forEach(({ mesh: sm, body: sb }) => {
      if (!sb) return
      const st = sb.translation()
      const sr = sb.rotation()
      sm.position.set(st.x, st.y, st.z)
      sm.quaternion.set(sr.x, sr.y, sr.z, sr.w)
      // Fade out shards over time
      if (sm.material && sm.material.opacity > 0) {
        sm.material.opacity -= 0.002
      }
    })
  })
}

function checkImpactBreakage() {
  const glass = state.world.objects['object_glass']
  if (!glass?._body || glass.status !== 'intact') return
  const v = glass._body.linvel()
  const speed = Math.sqrt(v.x**2 + v.y**2 + v.z**2)
  if (speed > 5) {
    glass.status = 'broken'
    shatterGlass(glass)
  }
}

function shatterGlass(obj) {
  const scene = state.scene.three
  const mesh = scene?.getObjectByName(obj.id)
  if (!mesh) return

  mesh.visible = false
  updateStatus('Glass shattered!')

  obj._shards = []
  for (let i = 0; i < 8; i++) {
    const geo = new (mesh.geometry.constructor)()
    const mat = mesh.material.clone()
    mat.transparent = true
    mat.opacity = 0.9
    mat.color?.set(0x88ccff)

    const shard = new (mesh.constructor)(mesh.geometry.clone(), mat)
    shard.scale.setScalar(0.08 + Math.random() * 0.12)
    shard.position.set(
      obj.position[0] + (Math.random() - 0.5) * 0.2,
      obj.position[1],
      obj.position[2] + (Math.random() - 0.5) * 0.2
    )
    scene.add(shard)

    const shardBody = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(shard.position.x, shard.position.y, shard.position.z)
        .setLinearDamping(0.5)
        .setAngularDamping(0.3)
    )
    world.createCollider(
      RAPIER.ColliderDesc.ball(0.03).setMass(0.01).setRestitution(0.2),
      shardBody
    )

    // Explode outward with random impulse
    shardBody.applyImpulse({
      x: (Math.random() - 0.5) * 2.5,
      y: 1.5 + Math.random() * 2,
      z: (Math.random() - 0.5) * 2.5
    }, true)
    shardBody.applyTorqueImpulse({
      x: Math.random() - 0.5,
      y: Math.random() - 0.5,
      z: Math.random() - 0.5
    }, true)

    obj._shards.push({ mesh: shard, body: shardBody })
  }
}

export function applyRobotCollisions() {
  // Legacy force-based repelling is no longer used.
  // AI robot now uses physical kinematic collision shapes.
}

export function releaseObjectPhysics(objectId, robotPos, forwardAngle = 0) {
  const obj = state.world.objects[objectId]
  if (!obj?._body) return

  obj._body.setBodyType(2)
  obj._body.setTranslation(
    {
      x: robotPos.x + Math.sin(forwardAngle) * 0.5,
      y: robotPos.y + 0.1,
      z: robotPos.z + Math.cos(forwardAngle) * 0.5
    },
    true
  )
  obj._body.setLinvel({ x: 0, y: -0.5, z: 0 }, true)
  obj._body.setAngvel({ x: 0, y: 0, z: 0 }, true)
  obj._body.wakeUp()
}

export function stepDebugRobotPhysics(keys, delta) {
  if (!debugBody || state.controlMode !== 'debug') return

  const vel = debugBody.linvel()
  const moveSpeed = 4.0
  const jumpSpeed = 5.0
  
  let inputX = 0
  let inputZ = 0
  if (keys.w) inputZ -= 1
  if (keys.s) inputZ += 1
  if (keys.a) inputX -= 1
  if (keys.d) inputX += 1

  // Get camera orientation
  const camera = state.scene.camera
  if (camera && (inputX !== 0 || inputZ !== 0)) {
    // Forward vector (Z axis of camera, ignoring Y pitch)
    const forward = new THREE.Vector3(0, 0, -1).applyQuaternion(camera.quaternion)
    forward.y = 0
    if (forward.lengthSq() > 0) forward.normalize()
    else forward.set(0, 0, -1)

    // Right vector (X axis of camera, ignoring Y pitch)
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(camera.quaternion)
    right.y = 0
    if (right.lengthSq() > 0) right.normalize()
    else right.set(1, 0, 0)

    // Combine based on input
    const direction = new THREE.Vector3()
    direction.addScaledVector(forward, -inputZ) // inputZ is negative for W, so we want positive forward. Wait, if W is pressed, inputZ = -1, so -inputZ = 1
    direction.addScaledVector(right, inputX)

    if (direction.lengthSq() > 0) direction.normalize()

    debugBody.setLinvel({ x: direction.x * moveSpeed, y: vel.y, z: direction.z * moveSpeed }, true)
    
    // Rotate to face movement
    const angle = Math.atan2(direction.x, direction.z)
    debugBody.setRotation(
      { x: 0, y: Math.sin(angle/2), z: 0, w: Math.cos(angle/2) },
      true
    )
  } else {
    // Friction-like stop on X/Z
    debugBody.setLinvel({ x: vel.x * 0.8, y: vel.y, z: vel.z * 0.8 }, true)
  }

  // Proper Jump check
  const ray = new RAPIER.Ray(debugBody.translation(), { x: 0, y: -1, z: 0 })
  const hit = world.castRay(ray, 0.3, true)
  const isGrounded = hit !== null

  if (keys.space && isGrounded && vel.y <= 0.1) {
    debugBody.setLinvel({ x: debugBody.linvel().x, y: jumpSpeed, z: debugBody.linvel().z }, true)
  }

  // Sync back to state
  const t = debugBody.translation()
  const r = debugBody.rotation()
  state.debugRobot.position[0] = t.x
  state.debugRobot.position[1] = t.y
  state.debugRobot.position[2] = t.z
  
  // Convert quaternion to Y angle for state (approx)
  state.debugRobot.rotation = 2 * Math.atan2(r.y, r.w)
}

export function getWorld() { return world }

function updateStatus(text) {
  const el = document.getElementById('status-bar')
  if (el) el.textContent = text
}