import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { state, getRobotPos, setRobotPos } from './state.js'
import { castLidar, castVision } from './perception/vision.js'
import { setStatus, setAgentStatus } from './ui.js'

const loader = new GLTFLoader()
let root, armMesh, handMesh, eyeMesh
let debugRoot, debugArmMesh, debugHandMesh, debugEyeMesh
let heldMeshParentedId = null

export async function initRobot() {
  const gltf = await loader.loadAsync('/robot1.glb')
  root = gltf.scene
  root.name = 'aiRobot'

  root.traverse(c => {
    if (!c.isMesh) return
    c.castShadow = true
    console.log('Robot mesh:', c.name)
    if (c.name === 'robot_arm')  armMesh  = c
    if (c.name === 'robot_hand') handMesh = c
    if (c.name === 'robot_eye')  eyeMesh  = c
  })

  if (armMesh && handMesh && handMesh.parent !== armMesh) {
    armMesh.attach(handMesh)
  }

  const p = getRobotPos()
  root.position.set(p.x, p.y, p.z)
  state.scene.three.add(root)

  console.log('Robot ready. arm:', !!armMesh, 'eye:', !!eyeMesh)
}

export async function initDebugRobot() {
  const gltf = await loader.loadAsync('/robot1.glb')
  debugRoot = gltf.scene

  debugRoot.name = 'debugRobot'
  debugRoot.traverse(c => {
    if (!c.isMesh) return
    c.castShadow = true
    if (c.name === 'robot_arm')  debugArmMesh  = c
    if (c.name === 'robot_hand') debugHandMesh = c
    if (c.name === 'robot_eye')  debugEyeMesh  = c
  })

  // Set distinct eye color
  if (debugEyeMesh?.material) {
    const mat = Array.isArray(debugEyeMesh.material) ? debugEyeMesh.material[0] : debugEyeMesh.material
    const newMat = mat.clone()
    newMat.color.set(state.debugRobot.eyeColor)
    if (newMat.emissive) newMat.emissive.set(state.debugRobot.eyeColor)
    debugEyeMesh.material = newMat
  }

  const p = state.debugRobot.position
  debugRoot.position.set(p[0], p[1], p[2])
  state.scene.three.add(debugRoot)
}

export function updateRobot(delta) {
  if (!root) return

  const target = getRobotPos()

  // Smooth position follow
  root.position.x = THREE.MathUtils.lerp(root.position.x, target.x, 0.08)
  root.position.y = THREE.MathUtils.lerp(root.position.y, target.y, 0.08)
  root.position.z = THREE.MathUtils.lerp(root.position.z, target.z, 0.08)

  // Face direction of travel
  const dx = target.x - root.position.x
  const dz = target.z - root.position.z
  if (Math.abs(dx) + Math.abs(dz) > 0.005) {
    const angle = Math.atan2(dx, dz)
    root.rotation.y = THREE.MathUtils.lerp(root.rotation.y, angle, 0.12)
  }

  // AI Robot Eye pulse
  if (eyeMesh?.material) {
    const targetColor = new THREE.Color(state.robot.eyeColor)
    const materials = Array.isArray(eyeMesh.material) ? eyeMesh.material : [eyeMesh.material]
    const pulse = state.robot.status === 'thinking'
      ? 1.8 + Math.sin(performance.now() * 0.01) * 0.6
      : 0.7
    materials.forEach(m => {
      if (m.emissive) {
        m.emissive.lerp(targetColor, 0.15)
        m.emissiveIntensity = THREE.MathUtils.lerp(m.emissiveIntensity || 0, pulse, 0.2)
      }
    })
  }

  // Arm follows state
  if (armMesh) {
    armMesh.rotation.x = THREE.MathUtils.lerp(
      armMesh.rotation.x, state.robot.armAngle, 0.1
    )
  }

  // Hold object — move with robot
  if (state.robot.heldObject) {
    const obj = state.world.objects[state.robot.heldObject]
    const mesh = state.scene.three?.getObjectByName(state.robot.heldObject)
    if (obj && mesh && handMesh) {
      if (mesh.parent !== handMesh) {
        handMesh.attach(mesh)
      }
      if (!handMesh.geometry.boundingBox) {
        handMesh.geometry.computeBoundingBox()
      }
      const center = new THREE.Vector3()
      handMesh.geometry.boundingBox.getCenter(center)
      mesh.position.copy(center)

      const wp = new THREE.Vector3()
      mesh.getWorldPosition(wp)
      obj.position[0] = wp.x
      obj.position[1] = wp.y
      obj.position[2] = wp.z

      heldMeshParentedId = obj.id
    }
  } else if (heldMeshParentedId && state.scene.three) {
    const releasedMesh = state.scene.three.getObjectByName(heldMeshParentedId)
    if (releasedMesh && releasedMesh.parent !== state.scene.three) {
      state.scene.three.attach(releasedMesh)
    }
    heldMeshParentedId = null
  }
}

export function updateDebugRobot(delta) {
  if (!debugRoot) return

  const p        = state.debugRobot.position
  const moveDir  = state.debugRobot._moveDir
  const speed    = state.debugRobot._speed    || 0
  const grounded = state.debugRobot._grounded ?? true
  const MAX_SPD  = 5.5  // mirror of physics MAX_SPEED

  // ── Direct position sync (physics-driven, no lag) ──
  debugRoot.position.set(p[0], p[1], p[2])

  // ── Smooth rotation: lerp toward movement direction (shortest-path) ──
  if (moveDir && speed > 0.4) {
    const targetAngle = Math.atan2(moveDir.x, moveDir.z)
    let diff = targetAngle - debugRoot.rotation.y
    // Wrap to [-PI, PI]
    while (diff >  Math.PI) diff -= Math.PI * 2
    while (diff < -Math.PI) diff += Math.PI * 2
    // Turn speed scales with movement speed for feel (fast input = fast turn)
    const turnSpeed = 10 + (speed / MAX_SPD) * 8
    debugRoot.rotation.y += diff * Math.min(1.0, delta * turnSpeed)
  }

  // ── Forward lean proportional to ground speed ──
  const leanTarget = grounded ? Math.min(speed / MAX_SPD, 1) * 0.13 : 0
  debugRoot.rotation.x = THREE.MathUtils.lerp(debugRoot.rotation.x, leanTarget, delta * 10)

  // ── Running bob (only on ground, only when moving) ──
  if (grounded && speed > 0.5) {
    const bobFreq = speed * 4.5   // faster run = faster bob
    const bobAmt  = Math.sin(performance.now() * 0.001 * bobFreq) * 0.012 * (speed / MAX_SPD)
    debugRoot.position.y = p[1] + bobAmt
  }

  // ── Debug Robot Eye pulse ──
  if (debugEyeMesh?.material) {
    const m = Array.isArray(debugEyeMesh.material) ? debugEyeMesh.material[0] : debugEyeMesh.material
    m.emissiveIntensity = 1.0 + Math.sin(performance.now() * 0.005) * 0.3
  }
}

// Navigation — Reactive Potential-Field Steering with LiDAR & CV
// The robot starts in an unknown env. No hardcoded map, no A*.
// Every tick it casts its FOV-anchored LiDAR from its eye position,
// computes an attractive force toward the goal and repulsive forces
// from any ray that hits something too close, then steps along the
// resultant vector. A tangential "wriggle" escapes local minima that
// arise when repulsive forces from a concave obstacle perfectly cancel
// the attractive force — a mathematical inevitability of pure potential
// fields regardless of whether the environment is known or not.
export function navigateTo(tx, ty, tz, onArrived, speed = 2.5, excludeIds = null) {
  const FLOOR_Y       = 0.35
  const EYE_HEIGHT    = 0.5    // lidar_front mountHeight from manifest
  const SAFE_DIST     = 0.9    // metres — repulsion activates below this
  const ARRIVE_THRESH = 0.12   // metres — arrived
  const LIDAR_FOV     = Math.PI  // 180° from manifest lidar_front
  const LIDAR_RAYS    = 11
  const TICK_DT       = 0.016

  setAgentStatus('Navigating…', 'navigating')

  // ── Scene objects for visualisers ──────────────────────────────────────────
  const lidarGeo = new THREE.BufferGeometry()
  const lidarMat = new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false })
  const lidarViz = new THREE.LineSegments(lidarGeo, lidarMat)
  lidarViz.renderOrder = 999
  state.scene.three.add(lidarViz)

  const coneGeo = new THREE.BufferGeometry()
  const coneMat = new THREE.LineBasicMaterial({ color: 0x2266ff, transparent: true, opacity: 0.25, depthTest: false })
  const coneViz = new THREE.LineSegments(coneGeo, coneMat)
  coneViz.renderOrder = 998
  state.scene.three.add(coneViz)

  // ── Local state ────────────────────────────────────────────────────────────
  let cancelled   = false
  let heading     = root ? root.rotation.y : 0  // tracked from movement, not lerped mesh
  let stuckTimer  = 0
  let escapeTimer = 0
  let escapePhase = false
  let lastX = null, lastZ = null

  const cleanup = () => {
    for (const obj of [lidarViz, coneViz]) {
      if (obj.parent) state.scene.three.remove(obj)
      obj.geometry.dispose()
      obj.material.dispose()
    }
  }

  // ── Main reactive loop (16 ms ≈ 60 fps) ───────────────────────────────────
  const interval = setInterval(() => {
    if (cancelled) return

    const p  = getRobotPos()
    const dx = tx - p.x
    const dz = tz - p.z
    const distToGoal = Math.hypot(dx, dz)

    if (distToGoal < ARRIVE_THRESH) {
      clearInterval(interval)
      cleanup()
      setAgentStatus(null)
      onArrived?.()
      return
    }

    // ── 1. Sensor origin ────────────────────────────────────────────────────
    const eyePos = new THREE.Vector3(p.x, p.y + EYE_HEIGHT, p.z)

    // ── 2. LiDAR — 11 rays in 180° arc centred on current heading ──────────
    const lidarDist = castLidar(eyePos, heading, state.scene.three)

    // ── 3. Vision / CV — 36 rays in 150° arc, object detection ─────────────
    const cvHits = castVision(eyePos, heading, state.scene.three)
    if (cvHits.length > 0) {
      // Fire-and-forget: feed detections into perceptual memory
      import('./perception/perceptualMemory.js').then(({ updatePerception }) =>
        updatePerception(cvHits.map(h => ({
          meshName:     h.id,
          estimatedPos: [h.position.x, h.position.y, h.position.z],
          distance:     h.distance,
          confidence:   h.confidence,
        })))
      ).catch(() => {})
    }

    // ── 4. Visualise ────────────────────────────────────────────────────────
    _drawLidarViz(lidarViz, eyePos, heading, lidarDist, LIDAR_FOV, LIDAR_RAYS, SAFE_DIST)
    _drawFovCone(coneViz,  eyePos, heading, 8.0, Math.PI * 0.833 /* 150° CV FOV */)

    // ── 5. Stagnation detector ──────────────────────────────────────────────
    if (lastX !== null) {
      const moved = Math.hypot(p.x - lastX, p.z - lastZ)
      if (moved < 0.005) {
        stuckTimer += TICK_DT
        if (stuckTimer > 0.6 && !escapePhase) {
          escapePhase = true
          escapeTimer = 0
        }
      } else {
        stuckTimer  = 0
        escapePhase = false
        escapeTimer = 0
      }
    }
    lastX = p.x; lastZ = p.z
    if (escapePhase) {
      escapeTimer += TICK_DT
      if (escapeTimer > 1.2) { escapePhase = false; stuckTimer = 0 }
    }

    // ── 6. Potential-field force ────────────────────────────────────────────
    // Attractive — unit vector toward goal
    let Fx = dx / distToGoal
    let Fz = dz / distToGoal

    // Repulsive — sum of per-ray pushes (sign matches castLidar: -sin/-cos)
    const halfFov    = LIDAR_FOV / 2
    const angleStep  = LIDAR_FOV / (LIDAR_RAYS - 1)
    const startAngle = heading - halfFov

    for (let i = 0; i < LIDAR_RAYS; i++) {
      const d = lidarDist[i]
      if (d < SAFE_DIST) {
        const a   = startAngle + angleStep * i
        const rdx = -Math.sin(a)   // vision.js castLidar convention
        const rdz = -Math.cos(a)
        const mag = Math.pow((SAFE_DIST - d) / SAFE_DIST, 2) * 4.5
        Fx -= rdx * mag
        Fz -= rdz * mag
      }
    }

    // Tangential escape — perpendicular push when stuck in a force-cancel dead-zone
    // This is a property of potential fields, not of knowing the env:
    // at a concave corner, forward-facing repulsive rays cancel the attractive force exactly.
    if (escapePhase) {
      // Perpendicular to current heading (rotated 90° CW)
      Fx += -Math.cos(heading) * 2.0
      Fz +=  Math.sin(heading) * 2.0
    }

    // ── 7. Normalise & step ─────────────────────────────────────────────────
    const fMag = Math.hypot(Fx, Fz)
    if (fMag > 0.001) { Fx /= fMag; Fz /= fMag }

    const step = Math.min(speed * TICK_DT, distToGoal)
    setRobotPos(p.x + Fx * step, FLOOR_Y, p.z + Fz * step)

    // ── 8. Track heading from movement direction (smooth lerp) ──────────────
    if (fMag > 0.001) {
      const wantH = Math.atan2(Fx, Fz)   // atan2(sin,cos) forward convention
      let diff = wantH - heading
      while (diff >  Math.PI) diff -= Math.PI * 2
      while (diff < -Math.PI) diff += Math.PI * 2
      heading += diff * 0.25
    }

  }, 16)

  return () => { cancelled = true; clearInterval(interval); cleanup() }
}

// ─── LiDAR ray visualiser ────────────────────────────────────────────────────
// Colours each ray green→orange→red by proximity. Same -sin/-cos convention
// as perception/vision.js castLidar so the rays point exactly where the sensor fires.
function _drawLidarViz(lineSegs, eyePos, heading, distances, fovRad, rayCount, safeDist) {
  const pts   = []
  const cols  = []
  const C_OK  = new THREE.Color(0x00ff55)
  const C_MID = new THREE.Color(0xffaa00)
  const C_BAD = new THREE.Color(0xff2222)

  const halfFov   = fovRad / 2
  const step      = fovRad / (rayCount - 1)
  const startAng  = heading - halfFov

  for (let i = 0; i < rayCount; i++) {
    const a  = startAng + step * i
    const d  = distances[i]
    const ex = eyePos.x + (-Math.sin(a)) * d
    const ez = eyePos.z + (-Math.cos(a)) * d

    pts.push(eyePos.x, eyePos.y, eyePos.z, ex, eyePos.y, ez)

    const t = 1.0 - Math.min(d / safeDist, 1.0)
    const c = t < 0.5
      ? C_OK.clone().lerp(C_MID, t * 2)
      : C_MID.clone().lerp(C_BAD, (t - 0.5) * 2)
    cols.push(c.r, c.g, c.b, c.r, c.g, c.b)
  }

  lineSegs.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  lineSegs.geometry.setAttribute('color',    new THREE.Float32BufferAttribute(cols, 3))
  lineSegs.geometry.attributes.position.needsUpdate = true
  lineSegs.geometry.attributes.color.needsUpdate    = true
}

// ─── CV FOV cone visualiser ───────────────────────────────────────────────────
// Shows the vision sensor's FOV as a blue translucent fan in the scene.
function _drawFovCone(lineSegs, eyePos, heading, range, fovRad) {
  const pts  = []
  const segs = 14
  const half = fovRad / 2

  for (let i = 0; i <= segs; i++) {
    const a  = (heading - half) + (fovRad / segs) * i
    const ex = eyePos.x + (-Math.sin(a)) * range
    const ez = eyePos.z + (-Math.cos(a)) * range
    pts.push(eyePos.x, eyePos.y, eyePos.z, ex, eyePos.y, ez)
  }
  for (let i = 0; i < segs; i++) {
    const a1 = (heading - half) + (fovRad / segs) * i
    const a2 = (heading - half) + (fovRad / segs) * (i + 1)
    pts.push(
      eyePos.x + (-Math.sin(a1)) * range, eyePos.y, eyePos.z + (-Math.cos(a1)) * range,
      eyePos.x + (-Math.sin(a2)) * range, eyePos.y, eyePos.z + (-Math.cos(a2)) * range,
    )
  }

  lineSegs.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  lineSegs.geometry.attributes.position.needsUpdate = true
}

// Instant teleport (no navigation)
export function setRobotPosition(x, y, z) {
  setRobotPos(x, y, z)
}

export function getRobotMesh() { return root }