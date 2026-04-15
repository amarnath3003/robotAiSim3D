import { initScene, renderScene } from './src/scene.js'
import { initRobot, initDebugRobot, updateRobot, updateDebugRobot } from './src/robot.js'
import { initPhysics, stepPhysics, applyRobotCollisions, stepDebugRobotPhysics } from './src/physics.js'
import { initSkillRegistry } from './src/skillRegistry.js'
import { initUI } from './src/ui.js'
import { initControls, getKeys } from './src/controls.js'
import * as THREE from 'three'

const clock = new THREE.Clock()

async function init() {
  console.log('Booting CausalBot...')
  await initScene()
  await initRobot()
  await initDebugRobot()
  await initPhysics()
  initSkillRegistry()
  initControls()
  initUI()
  console.log('All systems ready.')
  animate()
}

function animate() {
  requestAnimationFrame(animate)
  const delta = clock.getDelta()
  const keys = getKeys()

  updateRobot(delta)
  updateDebugRobot(delta)

  stepPhysics(delta)
  stepDebugRobotPhysics(keys, delta)
  
  applyRobotCollisions()
  
  renderScene()
}

init().catch(err => console.error('Boot failed:', err))