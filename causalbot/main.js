import { initScene, renderScene } from './src/scene.js'
import { initRobot, initDebugRobot, updateRobot, updateDebugRobot } from './src/robot.js'
import { initPhysics, stepPhysics, applyRobotCollisions, stepDebugRobotPhysics, addMazeWalls } from './src/physics.js'
import { initSkillRegistry } from './src/skillRegistry.js'
import { initUI } from './src/ui.js'
import { initControls, getKeys } from './src/controls.js'
import { getGridDebug } from './src/pathfinder.js'
import { state } from './src/state.js'
import { initRL, updateRL, flushRLState } from './src/rl.js'
import { togglePerceptionMode } from './src/perception/perceptionMode.js'
import { invalidateVisionCache } from './src/perception/visionSensor.js'
import { initMaze, updateMaze, getGoalPosition, getStartPosition } from './src/maze.js'
import * as THREE from 'three'

window.__togglePerception = togglePerceptionMode

// ─── Maze mode: activated via ?mode=maze in URL ───────────────────────────────
const MAZE_MODE = new URLSearchParams(window.location.search).get('mode') === 'maze'
window.MAZE_MODE = MAZE_MODE
window.getMazeGoal  = () => MAZE_MODE ? getGoalPosition()  : null
window.getMazeStart = () => MAZE_MODE ? getStartPosition() : null

const clock = new THREE.Clock()

async function init() {
  console.log(`Booting CausalBot... [${MAZE_MODE ? 'MAZE MODE' : 'ROOM MODE'}]`)
  await initScene(MAZE_MODE)
  await initRobot()
  await initDebugRobot()
  await initPhysics()

  if (MAZE_MODE) {
    console.log('[Main] Generating maze...')
    const wallSpecs = initMaze(state.scene.three)
    addMazeWalls(wallSpecs)
    console.log(`[Main] Maze ready — ${wallSpecs.length} wall colliders added`)

    // Override robot start position to maze center (higher up so it drops in)
    const start = getStartPosition()
    if (state.robot._body) {
      state.robot._body.setNextKinematicTranslation({ x: start.x, y: 1.5, z: start.z })
    }
    state.robot.position = [start.x, 1.5, start.z]

    // Hide debug robot in maze mode
    const debugMesh = state.scene.three.getObjectByName('debugRobot')
    if (debugMesh) debugMesh.visible = false
    if (state.debugRobot._body) {
      state.debugRobot._body.setTranslation({ x: 0, y: -100, z: 0 }, true)
    }

    // Show maze goal on HUD
    const goal = getGoalPosition()
    console.log(`[Main] Goal at (${goal.x.toFixed(2)}, ${goal.z.toFixed(2)})`)
  }

  initSkillRegistry()
  initControls()
  initUI()
  initRL()

  // After all meshes are in scene, build vision raycast cache
  invalidateVisionCache()
  console.log('All systems ready.')
  animate()
}

function animate() {
  requestAnimationFrame(animate)
  const delta = clock.getDelta()
  const keys = getKeys()
  const targetDelta = 1 / 60
  const steps = state.controlMode === 'rl' ? 10 : 1

  for (let i = 0; i < steps; i++) {
    updateRobot(targetDelta)
    updateDebugRobot(targetDelta)
    stepPhysics(targetDelta)
    stepDebugRobotPhysics(keys, targetDelta)
    updateRL(targetDelta)
    applyRobotCollisions()
  }

  flushRLState()

  if (MAZE_MODE) updateMaze(delta)

  renderScene()
}

// Optional: visualise pathfinding grid in Three.js (dev only)
function visualiseGrid() {
  const { grid, cols, rows, cellSize, halfExtent } = getGridDebug()
  const geo = new THREE.PlaneGeometry(cellSize * 0.85, cellSize * 0.85)
  geo.rotateX(-Math.PI / 2)

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const blocked = grid[r * cols + c] === 1
      if (!blocked) continue
      const mat  = new THREE.MeshBasicMaterial({ color: 0xff2222, transparent: true, opacity: 0.25 })
      const mesh = new THREE.Mesh(geo, mat)
      mesh.position.set(c * cellSize - halfExtent, 0.02, r * cellSize - halfExtent)
      state.scene.three.add(mesh)
    }
  }
}

init().catch(err => console.error('Boot failed:', err))