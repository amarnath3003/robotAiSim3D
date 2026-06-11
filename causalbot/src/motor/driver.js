/**
 * motor/driver.js — Motor Command Buffer & Application
 *
 * Sits between the brain/skills and the physics world.
 * The brain writes desired motion into the command buffer;
 * the engine calls applyMotorCommands() once per physics sub-step
 * so motion is always applied at fixed 60 Hz regardless of frame rate.
 *
 * This decoupling means skills can call moveForward() / rotate() at any
 * async point in their execution and the commands are flushed atomically
 * in the next physics tick — no dropped frames, no partial-frame drift.
 *
 * Usage (main.js):
 *   import { initMotorDriver, applyMotorCommands } from './src/motor/driver.js'
 *   initMotorDriver(robot)
 *   registerSystem('motor', applyMotorCommands)
 */

// ─── Command Buffer ────────────────────────────────────────────────────────────

let _robot = null

// Current command frame — overwritten by the latest moveForward/rotate calls
const _cmd = {
  linearSpeed:  0,   // m/s  — applied along robot's forward direction
  angularSpeed: 0,   // rad/s — applied around robot's up axis
  stop:         false,
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the motor driver with the active robot instance.
 * @param {import('../core/adapter.js').RobotInstance} robot
 */
export function initMotorDriver(robot) {
  _robot = robot
  console.log('[Motor] Driver initialized')
}

/**
 * Write a forward velocity command into the buffer.
 * Overwrites any previous command for this tick.
 * @param {number} speed  m/s (positive = forward, negative = backward)
 */
export function cmdMoveForward(speed) {
  _cmd.linearSpeed  = speed
  _cmd.stop         = false
}

/**
 * Write a rotation command into the buffer.
 * @param {number} angularSpeed  rad/s (positive = counter-clockwise when viewed from above)
 */
export function cmdRotate(angularSpeed) {
  _cmd.angularSpeed = angularSpeed
  _cmd.stop         = false
}

/**
 * Write a stop command — zeroes all motion next tick.
 */
export function cmdStop() {
  _cmd.linearSpeed  = 0
  _cmd.angularSpeed = 0
  _cmd.stop         = true
}

/**
 * Apply the current command buffer to the robot.
 * Called by the engine's motor system slot at fixed 60 Hz.
 *
 * @param {number} dt - Physics timestep (seconds)
 * @param {import('../core/adapter.js').RobotInstance} robot
 */
export function applyMotorCommands(dt, robot) {
  const r = robot || _robot
  if (!r) return

  if (_cmd.stop) {
    r.stop()
  } else {
    if (_cmd.linearSpeed  !== 0) r.moveForward(_cmd.linearSpeed)
    if (_cmd.angularSpeed !== 0) r.rotate(_cmd.angularSpeed)
  }

  // Commands are consumed — reset for next tick so motion doesn't persist
  // beyond the tick in which it was requested.
  _cmd.linearSpeed  = 0
  _cmd.angularSpeed = 0
  _cmd.stop         = false
}
