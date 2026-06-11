/**
 * brain/feasibility.js — Constraint Checking Against Manifest
 * 
 * Before any action is attempted, this module checks whether the robot
 * CAN physically perform it based on the manifest constraints.
 * 
 * This prevents the LLM from generating impossible plans and provides
 * clear feedback about WHY something can't be done + alternatives.
 * 
 * Used by:
 * - LLM Planner (before committing to a plan)
 * - Skill Verifier (before accepting an invented skill)
 * - Motor Controller (as a safety layer)
 */

import { getManifest, getCapabilities, getConstraints, getJoints, hasCapability, getCapability } from '../core/manifest.js'

/**
 * @typedef {Object} FeasibilityResult
 * @property {boolean} feasible - Whether the action is feasible
 * @property {string} reason - Explanation of why/why not
 * @property {string[]} alternatives - Suggested alternatives if not feasible
 * @property {Object} adjustedParams - Parameters adjusted to be within limits (if feasible with modification)
 */

/**
 * Check if an instruction/action is feasible for the current robot.
 * Returns detailed reasoning suitable for feeding back to the LLM.
 * 
 * @param {string} action - The action to check (e.g., 'jump', 'grasp', 'fly', 'backflip')
 * @param {Object} params - Action-specific parameters
 * @param {Object} worldState - Current world state (object positions, etc.)
 * @returns {FeasibilityResult}
 */
export function checkFeasibility(action, params = {}, worldState = {}) {
  const manifest = getManifest()
  if (!manifest) {
    return { feasible: false, reason: 'No robot manifest loaded.', alternatives: [], adjustedParams: null }
  }
  
  const constraints = manifest.constraints
  const capabilities = manifest.capabilities
  
  // Normalize action name
  const normalizedAction = action.toLowerCase().replace(/[_\-\s]/g, '')
  
  // Route to specific checkers
  switch (normalizedAction) {
    case 'jump':
    case 'hop':
      return checkJump(params, constraints, capabilities)
    
    case 'fly':
    case 'hover':
    case 'takeoff':
      return checkFlight(constraints)
    
    case 'swim':
    case 'dive':
      return checkSwim(constraints)
    
    case 'climb':
    case 'scale':
      return checkClimb(constraints)
    
    case 'grasp':
    case 'grab':
    case 'pickup':
    case 'pick':
      return checkGrasp(params, constraints, capabilities, worldState)
    
    case 'push':
    case 'shove':
      return checkPush(params, constraints, capabilities)
    
    case 'move':
    case 'walk':
    case 'run':
    case 'navigate':
    case 'goto':
      return checkLocomotion(params, constraints, capabilities)
    
    case 'rotate':
    case 'turn':
    case 'spin':
      return checkRotation(params, constraints)
    
    case 'backflip':
    case 'frontflip':
    case 'flip':
    case 'cartwheel':
    case 'somersault':
      return checkAcrobatic(action, params, constraints, capabilities, manifest)
    
    case 'scan':
    case 'look':
    case 'observe':
      return checkPerception(params, manifest)
    
    case 'reach':
    case 'extend':
      return checkReach(params, constraints)
    
    default:
      return checkGeneric(action, params, constraints, capabilities)
  }
}

/**
 * Check a full plan (array of actions) for feasibility.
 * Returns the first infeasible action found, or all-clear.
 * 
 * @param {Array<{action: string, params: Object}>} plan
 * @param {Object} worldState
 * @returns {{feasible: boolean, failedStep: number|null, results: FeasibilityResult[]}}
 */
export function checkPlanFeasibility(plan, worldState = {}) {
  const results = []
  
  for (let i = 0; i < plan.length; i++) {
    const { action, params } = plan[i]
    const result = checkFeasibility(action, params, worldState)
    results.push(result)
    
    if (!result.feasible) {
      return {
        feasible: false,
        failedStep: i,
        results,
        summary: `Plan fails at step ${i + 1} ("${action}"): ${result.reason}`,
      }
    }
  }
  
  return { feasible: true, failedStep: null, results, summary: 'All steps feasible.' }
}

/**
 * Generate a feasibility prompt for the LLM (helps it self-check before planning).
 * @returns {string}
 */
export function generateFeasibilityPrompt() {
  const manifest = getManifest()
  if (!manifest) return ''
  
  const c = manifest.constraints
  const lines = [
    '## Feasibility Rules (check BEFORE planning):',
    '',
    `1. Speed: Never exceed ${c.maxSpeed ?? '?'} m/s`,
    `2. Rotation: Never exceed ${c.maxAngularSpeed ?? '?'} rad/s`,
    `3. Reach: Arms reach max ${c.maxReach ?? '?'} m from body center`,
    `4. Payload: Can lift max ${c.maxPayload ?? '?'} kg`,
    `5. Jump: Max height ${c.maxJumpHeight ?? 0} m${(c.maxJumpHeight ?? 0) <= 0 ? ' (CANNOT jump)' : ''}`,
    `6. Flight: ${c.canFly ? 'YES' : 'IMPOSSIBLE'}`,
    `7. Swimming: ${c.canSwim ? 'YES' : 'IMPOSSIBLE'}`,
    `8. Climbing: ${c.canClimb ? 'YES' : 'IMPOSSIBLE'}`,
    '',
    '## Before each action ask:',
    '- Is this within my physical capabilities?',
    '- Do I need to perceive/scan first?',
    '- Is the object within reach, or do I need to navigate first?',
    '- Would this violate any joint limits?',
  ]
  
  if (c.custom) {
    lines.push('', '## Special constraints:')
    for (const [key, value] of Object.entries(c.custom)) {
      lines.push(`- ${value}`)
    }
  }
  
  return lines.join('\n')
}

// ─── Specific Checkers ─────────────────────────────────────────────────────────

function checkJump(params, constraints, capabilities) {
  const maxHeight = constraints.maxJumpHeight ?? 0
  
  if (maxHeight <= 0) {
    return {
      feasible: false,
      reason: 'This robot cannot jump (no jump capability, maxJumpHeight=0).',
      alternatives: ['If trying to reach something high, try extending arms instead.'],
      adjustedParams: null,
    }
  }
  
  const requestedHeight = params.height ?? maxHeight
  if (requestedHeight > maxHeight) {
    return {
      feasible: true,  // Can still jump, just not that high
      reason: `Can jump, but only up to ${maxHeight}m (requested ${requestedHeight}m). Will jump at max height.`,
      alternatives: [],
      adjustedParams: { ...params, height: maxHeight },
    }
  }
  
  return {
    feasible: true,
    reason: `Jump feasible (max ${maxHeight}m).`,
    alternatives: [],
    adjustedParams: params,
  }
}

function checkFlight(constraints) {
  if (constraints.canFly) {
    return { feasible: true, reason: 'Flight capable.', alternatives: [], adjustedParams: null }
  }
  return {
    feasible: false,
    reason: 'This robot cannot fly. It is a ground-based robot.',
    alternatives: ['Jump (if available)', 'Navigate around obstacles on the ground', 'Climb (if available)'],
    adjustedParams: null,
  }
}

function checkSwim(constraints) {
  if (constraints.canSwim) {
    return { feasible: true, reason: 'Swimming capable.', alternatives: [], adjustedParams: null }
  }
  return {
    feasible: false,
    reason: 'This robot cannot swim. Water would damage electronics.',
    alternatives: ['Navigate around water obstacles', 'Find a bridge or alternate path'],
    adjustedParams: null,
  }
}

function checkClimb(constraints) {
  if (constraints.canClimb) {
    return { feasible: true, reason: 'Climbing capable.', alternatives: [], adjustedParams: null }
  }
  return {
    feasible: false,
    reason: 'This robot cannot climb. Requires flat ground for locomotion.',
    alternatives: ['Navigate to a ramp if available', 'Find an alternate flat path'],
    adjustedParams: null,
  }
}

function checkGrasp(params, constraints, capabilities, worldState) {
  const graspCap = capabilities.find(c => c.id === 'manipulation:grasp')
  if (!graspCap) {
    return {
      feasible: false,
      reason: 'This robot has no grasping capability (no gripper/hand).',
      alternatives: ['Push the object instead (if push capability exists)', 'Request a different robot for this task'],
      adjustedParams: null,
    }
  }
  
  // Check payload
  if (params.objectMass && params.objectMass > (constraints.maxPayload ?? Infinity)) {
    return {
      feasible: false,
      reason: `Object is too heavy (${params.objectMass}kg > max payload ${constraints.maxPayload}kg).`,
      alternatives: ['Push instead of lift', 'Find a lighter alternative'],
      adjustedParams: null,
    }
  }
  
  // Check reach
  if (params.distance && params.distance > (constraints.maxReach ?? Infinity)) {
    return {
      feasible: false,
      reason: `Object is out of reach (${params.distance.toFixed(2)}m away, max reach ${constraints.maxReach}m). Navigate closer first.`,
      alternatives: ['Move closer to the object, then grasp'],
      adjustedParams: null,
    }
  }
  
  return {
    feasible: true,
    reason: `Grasp feasible (payload: ${params.objectMass ?? '?'}kg / ${constraints.maxPayload}kg max).`,
    alternatives: [],
    adjustedParams: params,
  }
}

function checkPush(params, constraints, capabilities) {
  const pushCap = capabilities.find(c => c.id === 'manipulation:push')
  if (!pushCap) {
    return {
      feasible: false,
      reason: 'This robot has no push capability.',
      alternatives: ['Try grasping and carrying instead'],
      adjustedParams: null,
    }
  }
  
  return {
    feasible: true,
    reason: 'Push feasible.',
    alternatives: [],
    adjustedParams: params,
  }
}

function checkLocomotion(params, constraints, capabilities) {
  const moveCap = capabilities.find(c => c.id.startsWith('locomotion:'))
  if (!moveCap) {
    return {
      feasible: false,
      reason: 'This robot has no locomotion capability (fixed base?).',
      alternatives: ['This robot can only manipulate within its workspace, not move.'],
      adjustedParams: null,
    }
  }
  
  // LB-8: constraints.maxSpeed may be undefined — default to 2.5 m/s
  const maxSpeed = constraints.maxSpeed ?? 2.5
  const requestedSpeed = params.speed ?? maxSpeed
  if (requestedSpeed > maxSpeed) {
    return {
      feasible: true,
      reason: `Can move, but speed clamped to ${maxSpeed} m/s (requested ${requestedSpeed}).`,
      alternatives: [],
      adjustedParams: { ...params, speed: maxSpeed },
    }
  }
  
  return {
    feasible: true,
    reason: `Movement feasible at ${requestedSpeed} m/s.`,
    alternatives: [],
    adjustedParams: params,
  }
}

function checkRotation(params, constraints) {
  const maxAngular = constraints.maxAngularSpeed ?? 4.0
  const requestedSpeed = params.speed ?? maxAngular
  
  return {
    feasible: true,
    reason: `Rotation feasible (max ${maxAngular} rad/s).`,
    alternatives: [],
    adjustedParams: { ...params, speed: Math.min(requestedSpeed, maxAngular) },
  }
}

function checkAcrobatic(action, params, constraints, capabilities, manifest) {
  // Acrobatic moves require: jumping + rotation + enough joint freedom
  const canJump = (constraints.maxJumpHeight ?? 0) > 0.2
  const joints = manifest.joints
  
  if (!canJump) {
    return {
      feasible: false,
      reason: `"${action}" requires jumping ability (need > 0.2m jump height, have ${constraints.maxJumpHeight ?? 0}m).`,
      alternatives: ['Spin in place (no jump required)', 'Dance/wave instead'],
      adjustedParams: null,
    }
  }
  
  // Check if the robot has enough rotational freedom
  // A backflip needs the body to rotate 360° — for a simulated robot this means
  // we need to synthesize a skill that uses jump + body rotation
  const hasTorsoFreedom = joints.some(j => 
    j.group?.includes('torso') || j.group?.includes('body')
  )
  
  // In simulation, even without dedicated torso joints, we can rotate the whole body
  // during a jump (the physics allows it). So we'll mark it as feasible but risky.
  return {
    feasible: true,
    reason: `"${action}" is possible in simulation: jump (${constraints.maxJumpHeight}m) + full body rotation. Note: this is a synthesized skill — physics verification required.`,
    alternatives: [],
    adjustedParams: { ...params, requiresVerification: true },
  }
}

function checkPerception(params, manifest) {
  const sensors = manifest.sensors
  if (sensors.length === 0) {
    return {
      feasible: false,
      reason: 'This robot has no sensors configured.',
      alternatives: [],
      adjustedParams: null,
    }
  }
  
  const scanCap = manifest.capabilities.find(c => c.id === 'perception:scan')
  return {
    feasible: true,
    reason: `Can observe using ${sensors.map(s => s.type).join(', ')} sensors.${scanCap ? ` Full scan takes ~${scanCap.parameters?.sweepTime ?? 2}s.` : ''}`,
    alternatives: [],
    adjustedParams: params,
  }
}

function checkReach(params, constraints) {
  const maxReach = constraints.maxReach ?? 0
  const requestedReach = params.distance ?? 0
  
  if (requestedReach > maxReach) {
    return {
      feasible: false,
      reason: `Target is beyond max reach (${requestedReach.toFixed(2)}m > ${maxReach}m). Navigate closer first.`,
      alternatives: ['Move closer to target, then extend'],
      adjustedParams: null,
    }
  }
  
  return {
    feasible: true,
    reason: `Reach feasible (${requestedReach.toFixed(2)}m / ${maxReach}m max).`,
    alternatives: [],
    adjustedParams: params,
  }
}

function checkGeneric(action, params, constraints, capabilities) {
  // Try to match against any capability
  const matchingCap = capabilities.find(c => {
    const capNorm = c.id.toLowerCase().replace(/[_\-:\s]/g, '')
    const actionNorm = action.toLowerCase().replace(/[_\-:\s]/g, '')
    return capNorm.includes(actionNorm) || actionNorm.includes(capNorm)
  })
  
  if (matchingCap) {
    return {
      feasible: true,
      reason: `Matched capability "${matchingCap.id}": ${matchingCap.description}`,
      alternatives: [],
      adjustedParams: params,
    }
  }
  
  // Unknown action — don't block it, but flag it for skill synthesis
  return {
    feasible: true,  // Optimistic — let the skill synthesizer and physics verifier decide
    reason: `No exact capability match for "${action}". Will attempt via skill synthesis + physics verification.`,
    alternatives: [],
    adjustedParams: { ...params, requiresSynthesis: true, requiresVerification: true },
  }
}
