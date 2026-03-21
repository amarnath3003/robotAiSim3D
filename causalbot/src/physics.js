import RAPIER from '@dimforge/rapier3d-compat'
import { state } from './state.js'

let world
let eventQueue

export async function initPhysics() {
  await RAPIER.init()

  world = new RAPIER.World({ x: 0.0, y: -9.81, z: 0.0 })
  eventQueue = new RAPIER.EventQueue(true)

  // Floor — high friction, no bounce
  const floor = world.createRigidBody(RAPIER.RigidBodyDesc.fixed())
  const floorCollider = RAPIER.ColliderDesc.cuboid(10, 0.05, 10)
    .setFriction(0.9)
    .setRestitution(0.1)
    .setFrictionCombineRule(RAPIER.CoefficientCombineRule.Max)
  world.createCollider(floorCollider, floor)

  // Table surface — fixed, medium friction
  const tableBody = world.createRigidBody(
    RAPIER.RigidBodyDesc.fixed().setTranslation(0, 0.78, 0)
  )
  const tableCollider = RAPIER.ColliderDesc.cuboid(0.7, 0.04, 0.4)
    .setFriction(0.7)
    .setRestitution(0.05)
  world.createCollider(tableCollider, tableBody)

  // Table legs — so objects don't fall through sides
  const legPositions = [
    [-0.65, 0.4, -0.35], [0.65, 0.4, -0.35],
    [-0.65, 0.4,  0.35], [0.65, 0.4,  0.35]
  ]
  legPositions.forEach(([x, y, z]) => {
    const leg = world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(x, y, z)
    )
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(0.03, 0.4, 0.03).setFriction(0.5),
      leg
    )
  })

  // Walls — so objects bounce off walls
  const walls = [
    { pos: [0, 1, -3.5], size: [3.5, 2, 0.1] },
    { pos: [0, 1,  3.5], size: [3.5, 2, 0.1] },
    { pos: [-3.5, 1, 0], size: [0.1, 2, 3.5] },
    { pos: [ 3.5, 1, 0], size: [0.1, 2, 3.5] },
  ]
  walls.forEach(({ pos, size }) => {
    const wb = world.createRigidBody(
      RAPIER.RigidBodyDesc.fixed().setTranslation(...pos)
    )
    world.createCollider(
      RAPIER.ColliderDesc.cuboid(...size).setFriction(0.4).setRestitution(0.3),
      wb
    )
  })

  // Dynamic objects with realistic physics properties
  const objectConfigs = {
    object_glass: {
      shape: 'cylinder',
      halfHeight: 0.08, radius: 0.04,
      mass: 0.25,
      friction: 0.6,
      restitution: 0.05,  // glass barely bounces
      linearDamping: 0.4,
      angularDamping: 0.6,
    },
    object_box: {
      shape: 'cuboid',
      hx: 0.17, hy: 0.17, hz: 0.17,
      mass: 2.5,
      friction: 0.8,
      restitution: 0.1,
      linearDamping: 0.6,
      angularDamping: 0.8,
    },
    object_ball: {
      shape: 'ball',
      radius: 0.14,
      mass: 0.4,
      friction: 0.4,
      restitution: 0.75,  // ball bounces well
      linearDamping: 0.05,
      angularDamping: 0.1,
    },
  }

  Object.entries(objectConfigs).forEach(([id, cfg]) => {
    const obj = state.world.objects[id]
    if (!obj) return

    const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(...obj.position)
      .setLinearDamping(cfg.linearDamping)
      .setAngularDamping(cfg.angularDamping)
      .setCcdEnabled(true)  // continuous collision — no tunnelling through floor

    const body = world.createRigidBody(bodyDesc)

    let colliderDesc
    if (cfg.shape === 'cylinder') {
      colliderDesc = RAPIER.ColliderDesc.cylinder(cfg.halfHeight, cfg.radius)
    } else if (cfg.shape === 'ball') {
      colliderDesc = RAPIER.ColliderDesc.ball(cfg.radius)
    } else {
      colliderDesc = RAPIER.ColliderDesc.cuboid(cfg.hx, cfg.hy, cfg.hz)
    }

    colliderDesc
      .setMass(cfg.mass)
      .setFriction(cfg.friction)
      .setRestitution(cfg.restitution)
      .setActiveEvents(RAPIER.ActiveEvents.COLLISION_EVENTS)

    world.createCollider(colliderDesc, body)
    obj._body = body
    obj._cfg = cfg
  })

  state.scene.rapierWorld = world
  console.log('Physics ready — full simulation active')
}

export function stepPhysics() {
  if (!world) return

  world.step(eventQueue)

  // Process collision events
  eventQueue.drainCollisionEvents((handle1, handle2, started) => {
    if (!started) return

    // Find which objects collided
    Object.values(state.world.objects).forEach(obj => {
      if (!obj._body) return
      const collider = world.getCollider(obj._body.collider(0))
      if (!collider) return

      const vel = obj._body.linvel()
      const impactSpeed = Math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

      // Glass breaks on hard impact
      if (obj.id === 'object_glass' && obj.status === 'intact' && impactSpeed > 3.5) {
        obj.status = 'broken'
        shatterObject(obj)
        updateStatus(`Glass shattered!`)
      }
    })
  })

  // Sync physics → Three.js meshes
  Object.values(state.world.objects).forEach(obj => {
    if (!obj._body || obj.status === 'held') return

    const t = obj._body.translation()
    const r = obj._body.rotation()

    // Kill objects that fall off the world
    if (t.y < -5) {
      obj._body.setTranslation({ x: obj.position[0], y: 2, z: obj.position[2] }, true)
      obj._body.setLinvel({ x: 0, y: 0, z: 0 }, true)
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
  })
}

function shatterObject(obj) {
  const mesh = state.scene.three?.getObjectByName(obj.id)
  if (!mesh) return

  // Turn mesh red and shrink to simulate shattering
  if (mesh.material) {
    mesh.material = mesh.material.clone()
    mesh.material.color.set(0xff2222)
    mesh.material.opacity = 0.6
    mesh.material.transparent = true
  }

  // Spawn small shards
  const scene = state.scene.three
  for (let i = 0; i < 6; i++) {
    const shard = mesh.clone()
    shard.name = `shard_${i}`
    shard.scale.setScalar(0.15 + Math.random() * 0.15)
    shard.position.set(
      obj.position[0] + (Math.random() - 0.5) * 0.3,
      obj.position[1],
      obj.position[2] + (Math.random() - 0.5) * 0.3
    )
    scene.add(shard)

    // Give shards physics
    const shardBody = world.createRigidBody(
      RAPIER.RigidBodyDesc.dynamic()
        .setTranslation(shard.position.x, shard.position.y, shard.position.z)
        .setLinearDamping(0.8)
    )
    world.createCollider(
      RAPIER.ColliderDesc.ball(0.04).setMass(0.02).setRestitution(0.3),
      shardBody
    )

    // Explode outward
    shardBody.applyImpulse({
      x: (Math.random() - 0.5) * 1.5,
      y: Math.random() * 2,
      z: (Math.random() - 0.5) * 1.5
    }, true)

    shard._physicsBody = shardBody

    // Sync shard every frame
    obj._shards = obj._shards || []
    obj._shards.push({ mesh: shard, body: shardBody })
  }

  // Hide original mesh
  mesh.visible = false
}

export function applyRobotCollisions() {
  const rx = state.robot.position[0]
  const ry = state.robot.position[1]
  const rz = state.robot.position[2]

  Object.values(state.world.objects).forEach(obj => {
    if (!obj._body || obj.status === 'held') return

    const op = obj._body.translation()
    const dx = op.x - rx
    const dy = op.y - ry
    const dz = op.z - rz
    const dist = Math.sqrt(dx*dx + dy*dy + dz*dz)

    if (dist < 0.55 && dist > 0.001) {
      const force = (0.55 - dist) * 6
      obj._body.wakeUp()
      obj._body.applyImpulse({
        x: (dx/dist) * force,
        y: 0.3,
        z: (dz/dist) * force
      }, true)
    }

    // Sync shards if object shattered
    if (obj._shards) {
      obj._shards.forEach(({ mesh, body }) => {
        const t = body.translation()
        const r = body.rotation()
        mesh.position.set(t.x, t.y, t.z)
        mesh.quaternion.set(r.x, r.y, r.z, r.w)
      })
    }
  })
}

export function releaseObjectPhysics(objectId, robotPos, robotVelocity = { x: 0, y: 0, z: 0 }) {
  const obj = state.world.objects[objectId]
  if (!obj?._body) return

  obj._body.setBodyType(2) // dynamic
  obj._body.setTranslation(
    { x: robotPos.x + 0.4, y: robotPos.y, z: robotPos.z },
    true
  )
  // Inherit some of robot's momentum on release
  obj._body.setLinvel({
    x: robotVelocity.x * 0.5,
    y: -0.5,
    z: robotVelocity.z * 0.5
  }, true)
  obj._body.wakeUp()
}

export function getWorld() { return world }

function updateStatus(text) {
  const el = document.getElementById('status-bar')
  if (el) el.textContent = text
}