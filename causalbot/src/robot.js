import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { state, getRobotPos, setRobotPos } from './state.js'
import { findPath } from './pathfinder.js'
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
      mesh.position.set(0, 0, 0)

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

// Navigation — A* pathfinding from current position to target
export function navigateTo(tx, ty, tz, onArrived, speed = 2.5, excludeIds = null) {
  const FLOOR_Y = 0.35
  const targetY = FLOOR_Y

  // Get held object ids to exclude from obstacle grid
  const finalExcludeIds = excludeIds || (state.robot.heldObject ? [state.robot.heldObject] : [])

  // Compute path
  const startPos = getRobotPos()
  setAgentStatus('Computing path...', 'thinking')
  const path = findPath(startPos.x, startPos.z, tx, tz, finalExcludeIds)

  // Fallback: straight line if pathfinder fails (open space)
  const waypoints = path
    ? path.map(wp => ({ x: wp.x, y: targetY, z: wp.z }))
    : [{ x: tx, y: targetY, z: tz }]

  // Ensure final waypoint is exactly the target
  waypoints[waypoints.length - 1] = { x: tx, y: targetY, z: tz }

  // --- PATH VISUALIZER ---
  const points = [new THREE.Vector3(startPos.x, startPos.y, startPos.z)]
  waypoints.forEach(wp => points.push(new THREE.Vector3(wp.x, wp.y, wp.z)))
  
  const pathGeo = new THREE.BufferGeometry().setFromPoints(points)
  const pathMat = new THREE.LineDashedMaterial({ color: 0x00ffff, dashSize: 0.2, gapSize: 0.1 })
  const currentPathLine = new THREE.Line(pathGeo, pathMat)
  currentPathLine.computeLineDistances()
  currentPathLine.position.y += 0.05 // Raise slightly to avoid z-fighting
  state.scene.three.add(currentPathLine)

  let wpIndex = 0
  let cancelled = false
  let stuckFrames = 0
  let lastDist = Infinity

  const clearPath = () => {
    if (currentPathLine.parent) {
      state.scene.three.remove(currentPathLine)
      currentPathLine.geometry.dispose()
      currentPathLine.material.dispose()
    }
  }

  const interval = setInterval(() => {
    if (cancelled) return

    if (wpIndex >= waypoints.length) {
      clearInterval(interval)
      clearPath()
      setAgentStatus(null)
      onArrived?.(true) // true = reached end
      return
    }

    const wp = waypoints[wpIndex]
    const p  = getRobotPos()
    const dx = wp.x - p.x
    const dy = wp.y - p.y
    const dz = wp.z - p.z
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz)

    if (dist < 0.1) {
      wpIndex++
      stuckFrames = 0
      lastDist = Infinity
      return
    }

    // Stuck detection: if distance barely changed for 30 frames (0.5s), skip waypoint
    if (Math.abs(lastDist - dist) < 0.005) {
      stuckFrames++
      if (stuckFrames > 30) {
        console.warn('Robot stuck navigating, skipping waypoint', wpIndex)
        // If this is the final waypoint and we are stuck, navigation failed
        if (wpIndex === waypoints.length - 1) {
          clearInterval(interval)
          clearPath()
          setAgentStatus(null)
          onArrived?.(false)
          return
        }
        wpIndex++
        stuckFrames = 0
        lastDist = Infinity
        return
      }
    } else {
      stuckFrames = 0
    }
    lastDist = dist

    const step = Math.min(speed * 0.016, dist)
    const n    = step / dist
    setRobotPos(p.x + dx * n, p.y + dy * n, p.z + dz * n)
  }, 16)

  return () => { cancelled = true; clearInterval(interval); clearPath(); onArrived?.(false) }
}

// Instant teleport (for jumps/special moves)
export function setRobotPosition(x, y, z) {
  setRobotPos(x, y, z)
}

export function getRobotMesh() { return root }