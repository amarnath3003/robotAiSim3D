import { initScene, renderScene } from './src/scene.js'
import { initRobot } from './src/robot.js'
import { initPhysics, stepPhysics, applyRobotCollisions } from './src/physics.js'
import { initSkillRegistry } from './src/skillRegistry.js'
import { initUI } from './src/ui.js'
import { updateRobot } from './src/robot.js'
import * as THREE from 'three'

const clock = new THREE.Clock()

async function init() {
  console.log('Booting CausalBot...')
  await initPhysics()
  await initScene()
  await initRobot()
  initSkillRegistry()
  initUI()
  console.log('All systems ready.')
  animate()
}

function animate() {
  requestAnimationFrame(animate)
  const delta = clock.getDelta()
  stepPhysics(delta)
  applyRobotCollisions()
  updateRobot(delta)
  renderScene()
}

init().catch(err => console.error('Boot failed:', err))