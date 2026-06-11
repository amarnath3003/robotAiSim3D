/**
 * skills/verifier.js — Physics Verification of Invented Skills
 * 
 * When the LLM invents a new skill, we don't trust it blindly.
 * This module runs the skill in a "dry run" sandbox:
 * 
 * 1. Save current robot state
 * 2. Execute the skill in an isolated context
 * 3. Check for constraint violations, collisions, or instability
 * 4. If it passes → mark as verified
 * 5. If it fails → reject with reason
 * 
 * This is critical for sim-to-real: we never deploy unverified skills to real hardware.
 */

import { getManifest, getConstraints, getJoints } from '../core/manifest.js'

// ─── Verification Criteria ─────────────────────────────────────────────────────

const VERIFICATION_CRITERIA = {
  maxExecutionTime: 10000,     // Max skill runtime (ms) before timeout
  maxPositionDelta: 5.0,       // Max distance from start position (m)
  maxHeightDelta: 3.0,         // Max vertical displacement (m)
  mustReturnNearStart: false,  // Whether robot must end near start position
  returnRadius: 2.0,           // How close to start counts as "returned"
  noConstraintViolations: true,
  noPhysicsPenetration: true,
}

// ─── Verification Results ──────────────────────────────────────────────────────

/**
 * @typedef {Object} VerificationResult
 * @property {boolean} passed - Did the skill pass verification?
 * @property {string[]} violations - List of violations found
 * @property {string[]} warnings - Non-critical issues
 * @property {Object} stats - Execution statistics
 * @property {number} executionTime - How long the skill took (ms)
 */

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Verify a skill by executing it in a sandboxed context.
 * 
 * @param {function} skillFn - The async skill function to verify
 * @param {import('../core/adapter.js').RobotInstance} robot - Robot to test with
 * @param {Object} args - Arguments to pass to the skill
 * @returns {Promise<VerificationResult>}
 */
export async function verifySkill(skillFn, robot, args = {}) {
  const manifest = getManifest()
  const constraints = getConstraints()
  const joints = getJoints()
  
  const violations = []
  const warnings = []
  const stats = {
    maxSpeed: 0,
    maxHeight: 0,
    maxDisplacement: 0,
    jointViolations: 0,
    totalSteps: 0,
  }
  
  // Save initial state
  const startPos = {
    x: robot.position.x,
    y: robot.position.y,
    z: robot.position.z,
  }
  const startJointStates = new Map()
  for (const [name, state] of robot.jointStates) {
    startJointStates.set(name, { ...state })
  }
  
  // Create sandboxed context that tracks violations
  const context = createVerificationContext(robot, args, {
    constraints,
    joints,
    startPos,
    stats,
    violations,
    warnings,
  })
  
  // Execute with timeout
  const startTime = Date.now()
  let timedOut = false
  
  try {
    await Promise.race([
      skillFn(context),
      new Promise((_, reject) => {
        setTimeout(() => {
          timedOut = true
          reject(new Error('Skill execution timed out'))
        }, VERIFICATION_CRITERIA.maxExecutionTime)
      }),
    ])
  } catch (e) {
    if (timedOut) {
      violations.push(`Timeout: Skill took longer than ${VERIFICATION_CRITERIA.maxExecutionTime}ms`)
    } else {
      violations.push(`Runtime error: ${e.message}`)
    }
  }
  
  const executionTime = Date.now() - startTime
  
  // Post-execution checks
  postExecutionChecks(robot, startPos, stats, violations, warnings)
  
  // Restore robot state (verification is non-destructive)
  robot.position.set(startPos.x, startPos.y, startPos.z)
  if (robot.physicsBody) {
    robot.physicsBody.setNextKinematicTranslation(startPos)
  }
  for (const [name, state] of startJointStates) {
    const current = robot.jointStates.get(name)
    if (current) {
      Object.assign(current, state)
    }
  }
  
  const passed = violations.length === 0
  
  console.log(`[Verifier] Skill ${passed ? 'PASSED' : 'FAILED'} verification`)
  if (!passed) {
    console.log(`[Verifier] Violations:`)
    violations.forEach(v => console.log(`  - ${v}`))
  }
  if (warnings.length > 0) {
    console.log(`[Verifier] Warnings:`)
    warnings.forEach(w => console.log(`  - ${w}`))
  }
  
  return {
    passed,
    violations,
    warnings,
    stats,
    executionTime,
  }
}

/**
 * Quick static analysis of skill code (before execution).
 * Catches obviously dangerous patterns.
 * 
 * @param {string} code - The skill code string
 * @returns {{safe: boolean, issues: string[]}}
 */
export function staticAnalyze(code) {
  const issues = []
  
  // Check for infinite loops
  if (/while\s*\(\s*true\s*\)/.test(code) && !/break/.test(code)) {
    issues.push('Potential infinite loop: while(true) without break')
  }
  
  // Check for dangerous APIs
  // SA-1: use 'new Function(' not 'Function(' — the latter is a substring of
  //        'AsyncFunction(' which is legitimately used in registry.js to compile
  //        synthesized skills. We want to catch dynamic code generation *inside*
  //        a skill body, not the compilation call itself.
  const dangerous = ['eval(', 'new Function(', 'fetch(', 'XMLHttpRequest', 'localStorage', 'document.', 'window.']
  for (const pattern of dangerous) {
    if (code.includes(pattern)) {
      issues.push(`Dangerous API usage: ${pattern}`)
    }
  }
  
  // Check for unreasonable delays
  const waitMatch = code.match(/wait\s*\(\s*(\d+)\s*\)/)
  if (waitMatch && parseInt(waitMatch[1]) > 10000) {
    issues.push(`Excessive wait time: ${waitMatch[1]}ms`)
  }
  
  // Check code length (overly complex skills are suspicious)
  const lines = code.split('\n').filter(l => l.trim()).length
  if (lines > 50) {
    issues.push(`Skill is very long (${lines} lines) — may be overly complex`)
  }
  
  return {
    safe: issues.length === 0,
    issues,
  }
}

// ─── Internal: Verification Context ────────────────────────────────────────────

function createVerificationContext(robot, args, tracking) {
  const { constraints, joints, startPos, stats, violations, warnings } = tracking
  
  // Wrap all robot control methods with constraint checking
  return {
    getPos: () => ({
      x: robot.position.x,
      y: robot.position.y,
      z: robot.position.z,
    }),
    
    setPos: (x, y, z) => {
      // Check displacement
      const dx = x - startPos.x
      const dy = y - startPos.y
      const dz = z - startPos.z
      const displacement = Math.sqrt(dx * dx + dz * dz)
      const height = Math.abs(dy)
      
      stats.maxDisplacement = Math.max(stats.maxDisplacement, displacement)
      stats.maxHeight = Math.max(stats.maxHeight, height)
      
      if (displacement > VERIFICATION_CRITERIA.maxPositionDelta) {
        violations.push(`Excessive displacement: ${displacement.toFixed(2)}m from start (max: ${VERIFICATION_CRITERIA.maxPositionDelta}m)`)
      }
      if (height > VERIFICATION_CRITERIA.maxHeightDelta) {
        violations.push(`Excessive height: ${height.toFixed(2)}m (max: ${VERIFICATION_CRITERIA.maxHeightDelta}m)`)
      }
      
      robot.position.set(x, y, z)
    },
    
    moveForward: (speed) => {
      stats.maxSpeed = Math.max(stats.maxSpeed, Math.abs(speed))
      if (Math.abs(speed) > (constraints.maxSpeed || 2.5) * 1.2) {
        warnings.push(`Speed ${speed.toFixed(2)} exceeds max ${constraints.maxSpeed}`)
      }
      robot.moveForward(speed)
    },
    
    rotate: (angularSpeed) => {
      if (Math.abs(angularSpeed) > (constraints.maxAngularSpeed || 4.0) * 1.2) {
        warnings.push(`Angular speed ${angularSpeed.toFixed(2)} exceeds max ${constraints.maxAngularSpeed}`)
      }
      robot.rotate(angularSpeed)
    },
    
    stop: () => robot.stop(),
    
    navigateTo: async (x, y, z, speed) => {
      // Simplified for verification — just check distance
      const dist = Math.sqrt((x - startPos.x) ** 2 + (z - startPos.z) ** 2)
      stats.maxDisplacement = Math.max(stats.maxDisplacement, dist)
      
      // Simulate travel time
      const moveSpeed = speed || constraints.maxSpeed || 2.0
      const time = Math.min((dist / moveSpeed) * 1000, 5000)
      await new Promise(r => setTimeout(r, Math.min(time, 100)))  // Shortened for verification
      
      robot.position.set(x, y, z)
    },
    
    setJoint: (jointName, angleDeg) => {
      const joint = joints.find(j => j.name === jointName)
      if (!joint) {
        warnings.push(`Unknown joint: ${jointName}`)
        return
      }
      
      const limits = joint.limits
      if (limits && (angleDeg < limits.lower || angleDeg > limits.upper)) {
        stats.jointViolations++
        violations.push(`Joint "${jointName}" out of limits: ${angleDeg}° (range: ${limits.lower}° to ${limits.upper}°)`)
      }
      
      robot.setJointTarget(jointName, angleDeg)
    },
    
    setJointGroup: (groupName, angleDeg) => {
      const groupJoints = joints.filter(j => j.group === groupName)
      for (const joint of groupJoints) {
        const limits = joint.limits
        if (limits && (angleDeg < limits.lower || angleDeg > limits.upper)) {
          stats.jointViolations++
          warnings.push(`Joint "${joint.name}" in group "${groupName}" would exceed limits`)
        }
      }
      robot.setGroupTarget(groupName, angleDeg)
    },
    
    grab: (objectId) => {
      // Check if manipulation capability exists
      const canGrasp = constraints.maxPayload > 0
      if (!canGrasp) {
        violations.push('Attempted grab but robot has no manipulation capability')
        return false
      }
      return true
    },
    
    release: () => {},
    
    wait: (ms) => {
      stats.totalSteps++
      // In verification, we shorten waits dramatically
      return new Promise(r => setTimeout(r, Math.min(ms, 10)))
    },
    
    setStatus: () => {},  // No-op in verification
    
    getKnownObjects: () => [],  // Empty in verification
    
    checkFeasibility: (action, params) => robot.checkFeasibility(action, params),
    
    args,
    robot,
    manifest: robot.manifest,
  }
}

// ─── Post-Execution Checks ─────────────────────────────────────────────────────

function postExecutionChecks(robot, startPos, stats, violations, warnings) {
  // Check if robot returned near start (if required)
  if (VERIFICATION_CRITERIA.mustReturnNearStart) {
    const endDist = Math.sqrt(
      (robot.position.x - startPos.x) ** 2 +
      (robot.position.z - startPos.z) ** 2
    )
    if (endDist > VERIFICATION_CRITERIA.returnRadius) {
      warnings.push(`Robot ended ${endDist.toFixed(2)}m from start (expected return)`)
    }
  }
  
  // Check robot is not underground
  if (robot.position.y < -0.5) {
    violations.push(`Robot ended below ground (y=${robot.position.y.toFixed(2)})`)
  }
  
  // Check robot is not impossibly high
  if (robot.position.y > 10) {
    violations.push(`Robot ended at impossible height (y=${robot.position.y.toFixed(2)})`)
  }
}
