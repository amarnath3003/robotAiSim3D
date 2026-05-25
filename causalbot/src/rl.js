import { state, getRobotPos, getObject } from './state.js'

let ws = null

export function initRL() {
  ws = new WebSocket('ws://localhost:8765')
  
  ws.onopen = () => {
    console.log('[RL] Connected to Gymnasium Server')
  }
  
  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data)
      
      if (msg.type === 'reset') {
        // Reset robot position and objects
        state.robot.position[0] = 0
        state.robot.position[1] = 0.35
        state.robot.position[2] = 2
        
        const ball = getObject('ball')
        if (ball) {
          ball.position[0] = 2.2
          ball.position[1] = 0.18
          ball.position[2] = -2.0
        }
        
        ws.send(JSON.stringify({ type: 'reset_done', observation: getObservation() }))
      } 
      else if (msg.type === 'action') {
        // action: [dx, dz]
        if (state.controlMode === 'rl') {
           const [dx, dz] = msg.action
           // Store the movement vector for updateRobot / updateDebugRobot to use
           state.rl = state.rl || {}
           state.rl.action = { dx, dz }
        }
        
        // Reply with the new state immediately (or could wait for next frame)
        ws.send(JSON.stringify({ type: 'state', observation: getObservation() }))
      }
    } catch (e) {
      console.error('[RL] Error parsing message', e)
    }
  }
  
  ws.onclose = () => {
    console.log('[RL] Disconnected')
    // Attempt reconnect after 5 seconds
    setTimeout(initRL, 5000)
  }
}

function getObservation() {
  const p = getRobotPos()
  
  const glass = getObject('glass') || { position: [0, 0, 0] }
  const box = getObject('box') || { position: [0, 0, 0] }
  const ball = getObject('ball') || { position: [0, 0, 0] }
  
  return [
    p.x, p.y, p.z, state.robot.rotation || 0,
    glass.position[0], glass.position[1], glass.position[2],
    box.position[0], box.position[1], box.position[2],
    ball.position[0], ball.position[1], ball.position[2]
  ]
}

// Function to apply the RL action, should be called in main loop
export function updateRL(delta) {
  if (state.controlMode !== 'rl' || !state.rl || !state.rl.action) return
  
  const { dx, dz } = state.rl.action
  const speed = 2.5 // Base movement speed
  
  // Apply action to robot position directly or feed into physics
  // We'll update the target position similar to how debug robot moves
  
  // Assuming debug robot's position logic is used for RL:
  state.debugRobot.position[0] += dx * speed * delta
  state.debugRobot.position[2] += dz * speed * delta
  
  // Also update standard robot position to keep them synced for observation
  state.robot.position[0] = state.debugRobot.position[0]
  state.robot.position[2] = state.debugRobot.position[2]
  
  // Reset action so it doesn't keep applying if no new action comes in
  // state.rl.action = { dx: 0, dz: 0 } // Depending on if we want continuous or step movement
}
