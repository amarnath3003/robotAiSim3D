import { initScene, renderScene } from './src/scene.js'
import { initRobot, initDebugRobot, updateRobot, updateDebugRobot } from './src/robot.js'
import { initPhysics, stepPhysics, applyRobotCollisions, stepDebugRobotPhysics, addMazeWalls } from './src/physics.js'
import { initSkillRegistry } from './src/skillRegistry.js'
import { initUI } from './src/ui.js'
import { initControls, getKeys } from './src/controls.js'
import { getGridDebug } from './src/pathfinder.js'
import { state } from './src/state.js'
import { initRL, updateRL, flushRLState, invalidateRLCache } from './src/rl.js'
import { togglePerceptionMode } from './src/perception/perceptionMode.js'
import { invalidateVisionCache } from './src/perception/visionSensor.js'
import { initMaze, updateMaze, getGoalPosition, getStartPosition } from './src/maze.js'
import * as THREE from 'three'

window.__togglePerception = togglePerceptionMode

// ─── Maze mode: activated via ?mode=maze in URL ───────────────────────────────
const MAZE_MODE = new URLSearchParams(window.location.search).get('mode') === 'maze'
window.MAZE_MODE     = MAZE_MODE
window.getMazeGoal   = () => MAZE_MODE ? getGoalPosition()  : null
window.getMazeStart  = () => MAZE_MODE ? getStartPosition() : null

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

    // Position AI robot at maze start, dropping in from above
    const start = getStartPosition()
    state.robot.position[0] = start.x
    state.robot.position[1] = 1.5
    state.robot.position[2] = start.z
    if (state.robot._body) {
      state.robot._body.setNextKinematicTranslation({ x: start.x, y: 1.5, z: start.z })
    }

    // Hide debug robot — not used in maze mode
    const debugMesh = state.scene.three.getObjectByName('debugRobot')
    if (debugMesh) debugMesh.visible = false
    if (state.debugRobot._body) {
      state.debugRobot._body.setTranslation({ x: 0, y: -100, z: 0 }, true)
    }

    const goal = getGoalPosition()
    console.log(`[Main] Goal at (${goal.x.toFixed(2)}, ${goal.z.toFixed(2)})`)

    // In maze mode, default to RL control immediately
    state.controlMode = 'rl'
  }

  initSkillRegistry()
  initControls()
  initUI()

  // Invalidate both lidar and vision caches after all meshes are in scene
  invalidateVisionCache()
  invalidateRLCache()

  // Start RL WebSocket — will switch controlMode to 'rl' only when Python connects
  initRL()

  console.log('All systems ready.')
  animate()
}

function animate() {
  requestAnimationFrame(animate)
  const delta = clock.getDelta()
  const keys  = getKeys()

  // In RL mode run multiple physics substeps per frame for faster training
  const substeps     = state.controlMode === 'rl' ? 10 : 1
  const substepDelta = 1 / 60  // fixed timestep regardless of frame rate

  for (let i = 0; i < substeps; i++) {
    updateRobot(substepDelta)
    updateDebugRobot(substepDelta)
    stepPhysics(substepDelta)
    stepDebugRobotPhysics(keys, substepDelta)
    updateRL(substepDelta)
    applyRobotCollisions()
  }

  // Send RL state to Python once per rendered frame (after all substeps)
  flushRLState()

  if (MAZE_MODE) updateMaze(delta)

  renderScene()
}

init().catch(err => console.error('Boot failed:', err))