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
  await initPhysics()
  await initScene()
  await initRobot()
  await initDebugRobot()
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

  stepPhysics(delta)
  stepDebugRobotPhysics(keys, delta)
  
  applyRobotCollisions()
  
  updateRobot(delta)
  updateDebugRobot(delta)
  
  renderScene()
}

init().catch(err => console.error('Boot failed:', err))