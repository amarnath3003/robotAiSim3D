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

// Navigation — Reactive Steering with LiDAR & CV
export function navigateTo(tx, ty, tz, onArrived, speed = 2.5, excludeIds = null) {
  const FLOOR_Y = 0.35
  const targetY = FLOOR_Y

  setAgentStatus('Navigating...', 'navigating')

  // --- LiDAR VISUALIZER ---
  const lineGeo = new THREE.BufferGeometry()
  const lineMat = new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false })
  const lidarLines = new THREE.LineSegments(lineGeo, lineMat)
  lidarLines.renderOrder = 999
  state.scene.three.add(lidarLines)

  let cancelled = false
  const clearPath = () => {
    if (lidarLines.parent) {
      state.scene.three.remove(lidarLines)
      lidarLines.geometry.dispose()
      lidarLines.material.dispose()
    }
  }

  // Reactive Navigation Loop
  const interval = setInterval(() => {
    if (cancelled) return

    const p = getRobotPos()
    const dx = tx - p.x
    const dz = tz - p.z
    const distToTarget = Math.sqrt(dx * dx + dz * dz)

    if (distToTarget < 0.1) {
      clearInterval(interval)
      clearPath()
      setAgentStatus(null)
      onArrived?.()
      return
    }

    // 1. Get current facing angle
    const facingAngle = root?.rotation.y || 0

    // 2. Cast Vision (to build memory/detect objects dynamically)
    castVision(new THREE.Vector3(p.x, p.y, p.z), facingAngle, state.scene.three)

    // 3. Cast LiDAR for obstacle avoidance
    const lidarDistances = castLidar(new THREE.Vector3(p.x, p.y, p.z), facingAngle, state.scene.three)
    
    // Visualize LiDAR rays
    updateLidarVisualizer(lidarLines, p, facingAngle, lidarDistances)

    // 4. Calculate desired steering (Potential Fields / Braitenberg)
    // The target provides an attractive force.
    let forceX = (dx / distToTarget) * 1.0
    let forceZ = (dz / distToTarget) * 1.0

    // Obstacles provide repulsive forces.
    const rays = lidarDistances.length
    const fovRad = Math.PI // 180 degrees from default-bot.json config
    const halfFov = fovRad / 2
    const angleStep = fovRad / Math.max(rays - 1, 1)
    
    // The castLidar function handles ray angles as: startAngle = facingAngle - halfFov (if rays > 1)
    const startAngle = rays > 1 ? facingAngle - halfFov : facingAngle

    const safeDistance = 1.0
    for (let i = 0; i < rays; i++) {
      const d = lidarDistances[i]
      if (d < safeDistance) {
        const rayAngle = startAngle + angleStep * i
        const rayDirX = -Math.sin(rayAngle)
        const rayDirZ = -Math.cos(rayAngle)
        
        // Repulsive force is stronger when closer
        const repulseMag = Math.pow(safeDistance - d, 2) * 5.0
        
        forceX -= rayDirX * repulseMag
        forceZ -= rayDirZ * repulseMag
      }
    }

    // Move along the calculated force vector
    const forceMag = Math.sqrt(forceX * forceX + forceZ * forceZ)
    if (forceMag > 0.001) {
      forceX /= forceMag
      forceZ /= forceMag
    }

    const step = Math.min(speed * 0.016, distToTarget)
    setRobotPos(p.x + forceX * step, p.y, p.z + forceZ * step)

  }, 16)

  return () => { cancelled = true; clearInterval(interval); clearPath() }
}

function updateLidarVisualizer(lines, p, facingAngle, distances) {
  const points = []
  const colors = []
  const colorClear = new THREE.Color(0x00ff00)
  const colorBlocked = new THREE.Color(0xff0000)

  const rays = distances.length
  const fovRad = Math.PI 
  const halfFov = fovRad / 2
  const angleStep = fovRad / Math.max(rays - 1, 1)
  const startAngle = rays > 1 ? facingAngle - halfFov : facingAngle

  for (let i = 0; i < rays; i++) {
    const angle = startAngle + angleStep * i
    const d = distances[i]
    
    // Origin of ray
    const sx = p.x
    const sy = p.y + 0.5 // Lidar height
    const sz = p.z
    
    // End of ray
    const ex = sx - Math.sin(angle) * d
    const ey = sy
    const ez = sz - Math.cos(angle) * d

    points.push(sx, sy, sz, ex, ey, ez)
    
    const color = d < 1.0 ? colorBlocked : colorClear
    colors.push(color.r, color.g, color.b, color.r, color.g, color.b)
  }

  lines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(points, 3))
  lines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3))
}

// Instant teleport (for jumps/special moves)
export function setRobotPosition(x, y, z) {
  setRobotPos(x, y, z)
}

export function getRobotMesh() { return root }