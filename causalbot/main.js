import { initScene, renderScene } from './src/scene.js'
import { initRobot, initDebugRobot, updateRobot, updateDebugRobot } from './src/robot.js'
import { initPhysics, stepPhysics, applyRobotCollisions, stepDebugRobotPhysics } from './src/physics.js'
import { initSkillRegistry } from './src/skillRegistry.js'
import { initUI } from './src/ui.js'
import { initControls, getKeys } from './src/controls.js'
import { getGridDebug } from './src/pathfinder.js'
import { state } from './src/state.js'
import { initRL, updateRL } from './src/rl.js'
import { togglePerceptionMode } from './src/perception/perceptionMode.js'
import { invalidateVisionCache } from './src/perception/visionSensor.js'
import * as THREE from 'three'

window.__togglePerception = togglePerceptionMode

const clock = new THREE.Clock()

async function init() {
  console.log('Booting CausalBot...')
  await initScene()
  await initRobot()
  await initDebugRobot()
  await initPhysics()
  // visualiseGrid() // uncomment to see obstacle grid
  initSkillRegistry()
  initControls()
  initUI()
  initRL()
  // After all meshes are added to the scene, build the vision raycast cache
  invalidateVisionCache()
  console.log('All systems ready.')
  animate()
}

function animate() {
  requestAnimationFrame(animate)
  const delta = clock.getDelta()
  const keys = getKeys()
  const targetDelta = 1 / 60
  const steps = state.controlMode === 'rl' ? 20 : 1 // 20x speedup for RL

  for (let i = 0; i < steps; i++) {
    updateRobot(targetDelta)
    updateDebugRobot(targetDelta)

    stepPhysics(targetDelta)
    stepDebugRobotPhysics(keys, targetDelta)
    updateRL(targetDelta)
    
    applyRobotCollisions()
  }
  
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