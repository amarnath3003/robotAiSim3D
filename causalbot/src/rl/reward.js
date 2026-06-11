/**
 * rl/reward.js — Modular Reward Signal Computation
 * 
 * Defines reward functions for different training objectives.
 * Rewards are manifest-aware: they respect robot constraints
 * and can be configured per-task.
 * 
 * Design philosophy:
 * - Dense rewards for stable training (progress toward goal)
 * - Large sparse rewards for terminal events (success/death)
 * - Penalties for constraint violations (teaches the robot its limits)
 * - Modular: add new reward components for new tasks
 */

import { getConstraints } from '../core/manifest.js'

// ─── Configuration ─────────────────────────────────────────────────────────────

const REWARD_CONFIG = {
  // Navigation rewards
  progressScale: 5.0,       // Reward per meter of progress toward goal
  stepPenalty: -0.005,      // Small penalty per step (encourages efficiency)
  
  // Terminal rewards
  goalReward: 20.0,         // Reaching the goal
  deathPenalty: -20.0,      // Dying (collision)
  
  // Thresholds
  goalRadius: 0.45,         // Distance to goal = success
  deathDistance: 0.28,      // LiDAR min distance = death
  maxSteps: 500,            // Truncation limit
  
  // Constraint violation penalties
  speedViolationPenalty: -0.1,
  boundaryPenalty: -1.0,
}

// ─── State ─────────────────────────────────────────────────────────────────────

let _prevDist = 0

/**
 * Reset reward tracking for a new episode.
 * @param {number} initialDist - Initial distance to goal
 */
export function resetRewardState(initialDist) {
  _prevDist = initialDist
}

// ─── Curriculum Learning ───────────────────────────────────────────────────────

/**
 * Curriculum levels — indexed 0 (easiest) → 4 (hardest).
 * Each level defines:
 *   goalRadius     — how close counts as "reached" (smaller = harder)
 *   goalDistance   — how far from the robot to spawn the goal
 *   maxSteps       — episode length limit
 *   progressScale  — denser reward for tight goals
 */
const CURRICULUM_LEVELS = [
  { goalRadius: 1.20, goalDistance: 1.5,  maxSteps: 600, progressScale: 4.0 },
  { goalRadius: 0.80, goalDistance: 2.5,  maxSteps: 550, progressScale: 4.5 },
  { goalRadius: 0.55, goalDistance: 3.5,  maxSteps: 500, progressScale: 5.0 },
  { goalRadius: 0.40, goalDistance: 5.0,  maxSteps: 450, progressScale: 5.5 },
  { goalRadius: 0.28, goalDistance: 7.0,  maxSteps: 400, progressScale: 6.0 },
]

// Rolling history of the last N episode outcomes ('success' | 'death' | 'timeout')
const CURRICULUM_WINDOW    = 20    // How many recent episodes to consider
const ADVANCE_THRESHOLD    = 0.75  // Success rate needed to advance a level
const RETREAT_THRESHOLD    = 0.35  // Success rate below which we retreat a level

let _curriculumLevel    = 0
let _episodeOutcomes    = []   // Array of 'success' | 'death' | 'timeout'
let _totalEpisodes      = 0

/**
 * Record the outcome of a completed episode and update the curriculum level.
 * Call from the RL bridge (bridge.js) after each episode reset.
 *
 * @param {'success'|'death'|'timeout'} outcome
 */
export function updateCurriculum(outcome) {
  _episodeOutcomes.push(outcome)
  _totalEpisodes++

  // Keep rolling window
  if (_episodeOutcomes.length > CURRICULUM_WINDOW) {
    _episodeOutcomes.shift()
  }

  // Only adjust after we have enough data
  if (_episodeOutcomes.length < Math.min(10, CURRICULUM_WINDOW)) return

  const successRate = _episodeOutcomes.filter(o => o === 'success').length / _episodeOutcomes.length

  const prevLevel = _curriculumLevel

  if (successRate >= ADVANCE_THRESHOLD && _curriculumLevel < CURRICULUM_LEVELS.length - 1) {
    _curriculumLevel++
    _episodeOutcomes = []   // Reset window after level change so stats are fresh
    console.log(`[Curriculum] Level UP → ${_curriculumLevel} (success rate ${(successRate * 100).toFixed(0)}%)`)
  } else if (successRate <= RETREAT_THRESHOLD && _curriculumLevel > 0) {
    _curriculumLevel--
    _episodeOutcomes = []
    console.log(`[Curriculum] Level DOWN → ${_curriculumLevel} (success rate ${(successRate * 100).toFixed(0)}%)`)
  }

  // Sync REWARD_CONFIG to current level
  if (_curriculumLevel !== prevLevel) _syncRewardConfig()
}

/**
 * Get information about the current curriculum state.
 * @returns {{level: number, maxLevel: number, successRate: number, totalEpisodes: number, goalRadius: number, goalDistance: number}}
 */
export function getCurriculumInfo() {
  const lvl = CURRICULUM_LEVELS[_curriculumLevel]
  const successRate = _episodeOutcomes.length > 0
    ? _episodeOutcomes.filter(o => o === 'success').length / _episodeOutcomes.length
    : 0

  return {
    level:         _curriculumLevel,
    maxLevel:      CURRICULUM_LEVELS.length - 1,
    successRate,
    totalEpisodes: _totalEpisodes,
    goalRadius:    lvl.goalRadius,
    goalDistance:  lvl.goalDistance,
  }
}

/**
 * Get the goal spawn distance for the current curriculum level.
 * Use this in the RL bridge when choosing where to place the goal each episode.
 * @returns {number} metres
 */
export function getCurriculumGoalDistance() {
  return CURRICULUM_LEVELS[_curriculumLevel].goalDistance
}

/**
 * Reset the curriculum back to level 0 (call on full training restart).
 */
export function resetCurriculum() {
  _curriculumLevel = 0
  _episodeOutcomes = []
  _totalEpisodes   = 0
  _syncRewardConfig()
}

/** Apply the current curriculum level parameters to REWARD_CONFIG. */
function _syncRewardConfig() {
  const lvl = CURRICULUM_LEVELS[_curriculumLevel]
  REWARD_CONFIG.goalRadius    = lvl.goalRadius
  REWARD_CONFIG.maxSteps      = lvl.maxSteps
  REWARD_CONFIG.progressScale = lvl.progressScale
}

// ─── Main Reward Function ──────────────────────────────────────────────────────

/**
 * Compute reward, termination, and outcome for the current step.
 * 
 * @param {import('../core/adapter.js').RobotInstance} robot
 * @param {{x: number, z: number}} goalPosition
 * @param {Float32Array} lidarDistances
 * @param {number} stepCount
 * @param {number} dt
 * @returns {{reward: number, terminated: boolean, truncated: boolean, outcome: string|null}}
 */
export function computeReward(robot, goalPosition, lidarDistances, stepCount, dt) {
  const config = REWARD_CONFIG
  const constraints = getConstraints()
  
  // Compute distance to goal
  const dx = goalPosition.x - robot.position.x
  const dz = goalPosition.z - robot.position.z
  const distToGoal = Math.sqrt(dx * dx + dz * dz)
  
  // Compute min LiDAR reading
  // EC-3: guard against null/empty lidar array (can happen before first cast)
  let lidarMin = Infinity
  if (lidarDistances && lidarDistances.length > 0) {
    for (let i = 0; i < lidarDistances.length; i++) {
      if (lidarDistances[i] < lidarMin) lidarMin = lidarDistances[i]
    }
  }
  
  let reward = 0
  let terminated = false
  let truncated = false
  let outcome = null
  
  // ─── Dense progress reward ────────────────────────────────────────
  const progress = (_prevDist - distToGoal) * config.progressScale
  reward += progress
  _prevDist = distToGoal
  
  // ─── Step penalty (encourages efficiency) ─────────────────────────
  reward += config.stepPenalty
  
  // ─── Terminal: Death (collision) ──────────────────────────────────
  if (lidarMin < config.deathDistance) {
    reward = config.deathPenalty
    terminated = true
    outcome = 'death'
  }
  
  // ─── Terminal: Goal reached ───────────────────────────────────────
  else if (distToGoal < config.goalRadius) {
    reward = config.goalReward
    terminated = true
    outcome = 'success'
  }
  
  // ─── Truncation: Max steps ────────────────────────────────────────
  else if (stepCount >= config.maxSteps) {
    truncated = true
    outcome = 'timeout'
  }
  
  // ─── Constraint violation penalties ───────────────────────────────
  if (!terminated && !truncated) {
    // Speed violation
    const speed = robot.velocity.length()
    if (speed > (constraints.maxSpeed || 2.5) * 1.1) {
      reward += config.speedViolationPenalty
    }
    
    // Boundary penalty (approaching world edges)
    const boundaryMargin = 0.3
    const maxBound = 2.5  // Room half-size
    if (Math.abs(robot.position.x) > maxBound - boundaryMargin ||
        Math.abs(robot.position.z) > maxBound - boundaryMargin) {
      reward += config.boundaryPenalty * dt
    }
  }

  // ─── Update curriculum when episode ends ──────────────────────────
  if (outcome !== null) {
    updateCurriculum(outcome)
  }
  
  return { reward, terminated, truncated, outcome }
}

// ─── Reward Presets (for different training tasks) ─────────────────────────────

/**
 * Configure reward for navigation training.
 */
export function setNavigationRewards() {
  REWARD_CONFIG.progressScale = 5.0
  REWARD_CONFIG.goalReward = 20.0
  REWARD_CONFIG.deathPenalty = -20.0
  REWARD_CONFIG.stepPenalty = -0.005
}

/**
 * Configure reward for manipulation training.
 */
export function setManipulationRewards() {
  REWARD_CONFIG.progressScale = 3.0
  REWARD_CONFIG.goalReward = 30.0
  REWARD_CONFIG.deathPenalty = -5.0   // Less harsh — manipulation is fiddly
  REWARD_CONFIG.stepPenalty = -0.002
}

/**
 * Configure reward for exploration training.
 */
export function setExplorationRewards() {
  REWARD_CONFIG.progressScale = 0.5   // Less goal-directed
  REWARD_CONFIG.goalReward = 10.0
  REWARD_CONFIG.deathPenalty = -15.0
  REWARD_CONFIG.stepPenalty = -0.001  // Encourage longer episodes
}

/**
 * Get current reward configuration (for debugging/display).
 */
export function getRewardConfig() {
  return { ...REWARD_CONFIG }
}
