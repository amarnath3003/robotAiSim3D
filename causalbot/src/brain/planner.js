/**
 * brain/planner.js — High-Level Task Decomposition
 * 
 * The planner is the orchestrator between the user's instruction and skill execution.
 * It:
 * 1. Receives a natural language instruction
 * 2. Checks feasibility against the manifest
 * 3. Calls the LLM for task decomposition
 * 4. Validates the plan steps
 * 5. Triggers skill synthesis if needed
 * 6. Executes the plan step by step
 * 7. Handles failures with re-planning
 * 
 * This replaces the old executor.js with a more robust pipeline.
 */

import { planWithLLM, synthesizeSkill, clearConversationHistory, reflectWithLLM } from './llm.js'
// XM-4: removed checkPlanFeasibility — it's not called directly in planner.js
import { checkFeasibility } from './feasibility.js'
import { getRobot, setRobotStatus, logExecution, getKnownObjects, getRecentHistory } from '../core/state.js'
// XM-4: removed dead manifest imports (getManifest, getCapabilities, hasCapability)
//        — manifest decisions are delegated to feasibility.js and adapter.js
import { navigateTo as pathNavigateTo, abortNavigation } from '../nav/pathfinder.js'
import { BTRunner, Blackboard, planToBehaviorTree } from './behavior_tree.js'
import { reflect } from './reflection.js'
import {
  grabInteractable,
  releaseInteractable,
  pushInteractable,
  findInteractablePosition,
  listInteractables,
} from '../env/objects.js'

// ─── Plan Execution State ──────────────────────────────────────────────────────

let _executing = false
let _currentPlan = null
let _currentStep = 0
let _abortRequested = false

const MAX_REPLAN_ATTEMPTS = 2   // How many times we'll ask the LLM to recover

// Callbacks for UI integration
let _onThinking = null
let _onExecuting = null
let _onComplete = null
let _onError = null
let _onSkillApproval = null

/**
 * Register event callbacks for the planner lifecycle.
 */
export function onPlannerEvent(event, callback) {
  switch (event) {
    case 'thinking': _onThinking = callback; break
    case 'executing': _onExecuting = callback; break
    case 'complete': _onComplete = callback; break
    case 'error': _onError = callback; break
    case 'skillApproval': _onSkillApproval = callback; break
  }
}

/**
 * Is the planner currently executing?
 */
export function isExecuting() {
  return _executing
}

/**
 * Abort current execution.
 */
export function abortExecution() {
  _abortRequested = true
  // EC-1: also cancel any in-progress A* navigation
  abortNavigation()
}

// ─── Main Entry Point ──────────────────────────────────────────────────────────

/**
 * Handle a user instruction end-to-end.
 * This is the main entry point — replaces the old handleInstruction().
 * 
 * @param {string} instruction - Natural language instruction from user
 * @param {Object} skillRegistry - The skill registry instance
 * @returns {Promise<{success: boolean, reason: string}>}
 */
export async function handleInstruction(instruction, skillRegistry) {
  if (_executing) {
    return { success: false, reason: 'Already executing a plan. Wait or abort.' }
  }
  
  const robot = getRobot()
  if (!robot) {
    return { success: false, reason: 'No robot loaded.' }
  }
  
  _executing = true
  _abortRequested = false
  setRobotStatus('planning')

  // Clear prior conversation so each new user instruction starts a fresh LLM context
  clearConversationHistory()

  _onThinking?.(`Planning: "${instruction}"`)
  
  try {    // ─── Step 1: Quick feasibility pre-check ─────────────────────────────
    const quickCheck = quickFeasibilityCheck(instruction)
    if (quickCheck && !quickCheck.feasible) {
      setRobotStatus('idle')
      _executing = false
      _onError?.(quickCheck.reason)
      logExecution(instruction, 'failure', quickCheck.reason)
      return { success: false, reason: quickCheck.reason }
    }
    
    // ─── Step 2: Try direct skill match (skip LLM for simple commands) ───
    const directMatch = tryDirectMatch(instruction, skillRegistry)
    if (directMatch) {
      const result = await executePlan(
        [{ skill: directMatch.skill, args: directMatch.args, description: instruction }],
        skillRegistry,
        robot,
        instruction
      )
      return result
    }
    
    // ─── Step 3: Call LLM for task decomposition ─────────────────────────
    const robotState = robot.getStateSnapshot()
    const knownObjects = getKnownObjects(0.2)
    const history = getRecentHistory(5)
    const availableSkills = skillRegistry.getAllForLLM()   // full objects: name, description, args
    
    const llmResponse = await planWithLLM(instruction, {
      knownObjects,
      history,
      availableSkills,
      robotState,
    })
    
    if (!llmResponse) {
      throw new Error('LLM returned no response')
    }
    
    // ─── Step 4: Handle infeasible response ──────────────────────────────
    if (llmResponse.infeasible) {
      const reason = llmResponse.infeasibleReason || 'The LLM determined this is not possible.'
      setRobotStatus('idle')
      _executing = false
      _onError?.(reason)
      logExecution(instruction, 'failure', reason)
      return { success: false, reason }
    }
    
    // ─── Step 5: Handle skill synthesis request ──────────────────────────
    let plan = llmResponse.plan || []
    
    if (llmResponse.needsSynthesis && llmResponse.newSkillSpec) {
      const synthesized = await handleSkillSynthesis(
        llmResponse.newSkillSpec,
        skillRegistry,
        robot
      )
      
      if (synthesized) {
        // Replace plan with the synthesized skill
        plan = [{ 
          skill: synthesized.name, 
          args: {}, 
          description: llmResponse.newSkillSpec.description || instruction 
        }]
      } else {
        throw new Error('Skill synthesis failed')
      }
    }
    
    // ─── Step 6: Show reasoning ──────────────────────────────────────────
    if (llmResponse.reasoning) {
      _onThinking?.(llmResponse.reasoning)
    }
    
    // ─── Step 7: Execute the plan ────────────────────────────────────────
    const result = await executePlan(plan, skillRegistry, robot, instruction)
    return result
    
  } catch (e) {
    console.error('[Planner] Error:', e)
    setRobotStatus('failed')

    // Surface a helpful message when the LLM key is missing
    const isNoKey = e.message?.includes('No API key') || e.message?.includes('VITE_LLM_API_KEY')
    const userMsg = isNoKey
      ? 'LLM not configured — copy .env.example to .env and set VITE_NVIDIA_API_KEY, then restart'
      : e.message

    _onError?.(userMsg)
    logExecution(instruction, 'failure', userMsg)
    return { success: false, reason: userMsg }
  } finally {
    // AP-2: guarantee _executing is cleared even if catch itself throws
    _executing = false
  }
}

// ─── Plan Execution ────────────────────────────────────────────────────────────

/**
 * Execute a plan via the Behavior Tree runtime.
 * Each step is wrapped with retry / optional semantics from the LLM response.
 * On tree-level failure the planner attempts LLM replanning (depth-limited).
 */
async function executePlan(plan, skillRegistry, robot, instruction, replanDepth = 0) {
  if (!plan || plan.length === 0) {
    setRobotStatus('idle')
    _executing = false
    return { success: false, reason: 'Empty plan — no actions to execute.' }
  }

  setRobotStatus('executing')
  _currentPlan = plan
  _currentStep = 0

  // Blackboard carries the abort flag so every ActionNode can check it
  const blackboard = new Blackboard({ abortFlag: () => _abortRequested })
  const contextFn  = (args) => buildExecutionContext(robot, args, skillRegistry)

  // Build the behavior tree from the plan steps
  const tree = planToBehaviorTree(plan, skillRegistry, contextFn, {
    onStepStart: (i, step, total) => {
      _currentStep = i
      _onExecuting?.({
        step: i + 1,
        total,
        skill: step.skill,
        description: step.description,
      })
    },
    onStepDone: (i, step) => {
      logExecution(step.skill, 'success', step.description)
      skillRegistry.recordExecution?.(step.skill, true)
    },
    onStepFail: (i, step, errMsg) => {
      console.error(`[Planner] Skill "${step.skill}" failed:`, errMsg)
      logExecution(step.skill, 'failure', errMsg)
      skillRegistry.recordExecution?.(step.skill, false)
    },
  })

  const runner = new BTRunner()
  const { success, blackboard: bb } = await runner.start(tree, blackboard)

  // ── Abort path ──────────────────────────────────────────────────────────
  if (_abortRequested) {
    setRobotStatus('idle')
    _executing = false
    _currentPlan = null
    _onError?.('Execution aborted by user.')
    return { success: false, reason: 'Aborted' }
  }

  // ── Success path ─────────────────────────────────────────────────────────
  if (success) {
    setRobotStatus('idle')
    _executing = false
    _currentPlan = null
    _onComplete?.(instruction)
    logExecution(instruction, 'success', `Completed ${plan.length} steps`)

    // Fire-and-forget reflection — records episode + extracts lessons (non-blocking)
    reflect({
      instruction,
      plan,
      success: true,
      robotPos: { x: robot.position.x, y: robot.position.y, z: robot.position.z },
      knownObjects: getKnownObjects(0.2),
    }, reflectWithLLM).catch(() => {})

    return { success: true, reason: 'Plan executed successfully.' }
  }

  // ── Failure path + optional replanning ───────────────────────────────────
  const failedStep = bb.get('failedStep')
  const failReason = failedStep
    ? `Skill "${failedStep.step.skill}" failed: ${failedStep.error}`
    : 'Plan execution failed'

  if (replanDepth < MAX_REPLAN_ATTEMPTS && failedStep && !_abortRequested) {
    _onThinking?.(`Step ${failedStep.index + 1} failed ("${failedStep.error}"). Replanning...`)

    const replanResult = await _replanAfterFailure(
      instruction,
      failReason,
      failedStep.remaining,
      skillRegistry,
      robot,
      replanDepth + 1,
    )

    if (replanResult !== null) {
      setRobotStatus('idle')
      _executing = false
      _currentPlan = null
      return replanResult
    }

    _onThinking?.('Replan unsuccessful, continuing with graceful degradation.')
  }

  setRobotStatus('idle')
  _executing = false
  _currentPlan = null
  _onError?.(failReason)

  // Fire-and-forget reflection — records failure episode + lessons
  reflect({
    instruction,
    plan,
    success: false,
    failReason,
    robotPos: { x: robot.position.x, y: robot.position.y, z: robot.position.z },
    knownObjects: getKnownObjects(0.2),
  }, reflectWithLLM).catch(() => {})

  return { success: false, reason: failReason }
}

// ─── Replanning ────────────────────────────────────────────────────────────────

/**
 * Ask the LLM to produce an alternative plan after a skill failure.
 * Returns the outcome of executing that new plan, or null if replanning fails.
 *
 * @param {string} instruction        - Original user instruction
 * @param {string} failReason         - Why the previous step failed
 * @param {Array}  remainingSteps     - Steps that hadn't run yet
 * @param {Object} skillRegistry
 * @param {Object} robot
 * @param {number} replanDepth        - Depth counter (prevents infinite loops)
 * @returns {Promise<{success,reason}|null>}
 */
async function _replanAfterFailure(instruction, failReason, remainingSteps, skillRegistry, robot, replanDepth) {
  try {
    const robotState     = robot.getStateSnapshot()
    const knownObjects   = getKnownObjects(0.2)
    const history        = getRecentHistory(5)
    const availableSkills = skillRegistry.getAllForLLM()   // full objects: name, description, args

    const remainingDesc = remainingSteps.length > 0
      ? `Remaining unexecuted steps were: ${remainingSteps.map(s => s.skill).join(', ')}.`
      : 'No remaining steps.'

    const llmResponse = await planWithLLM(instruction, {
      knownObjects,
      history,
      availableSkills,
      robotState,
      failureContext: `${failReason}. ${remainingDesc} Produce a new recovery plan.`,
    })

    if (!llmResponse || llmResponse.infeasible || !llmResponse.plan?.length) {
      console.warn('[Planner] Replan returned infeasible or empty plan')
      return null
    }

    _onThinking?.(`Replan (attempt ${replanDepth}): ${llmResponse.plan.length} recovery steps`)

    // Execute the recovery plan (recursive, depth-limited by MAX_REPLAN_ATTEMPTS)
    return await executePlan(llmResponse.plan, skillRegistry, robot, instruction, replanDepth)

  } catch (e) {
    console.warn('[Planner] Replan call failed:', e.message)
    return null
  }
}

// ─── Execution Context Builder ─────────────────────────────────────────────────

/**
 * Build the context object passed to skill functions.
 * This is the API that skills use to control the robot.
 * Exported so background_agent.js and main.js can build contexts directly.
 */
export function buildExecutionContext(robot, args, skillRegistry) {
  return {
    // ─── Robot Control ────────────────────────────────────────────
    getPos: () => ({
      x: robot.position.x,
      y: robot.position.y,
      z: robot.position.z,
    }),
    
    setPos: (x, y, z) => {
      robot.position.set(x, y, z)
      if (robot.physicsBody) {
        robot.physicsBody.setNextKinematicTranslation({ x, y, z })
      }
    },
    
    moveForward: (speed) => robot.moveForward(speed),
    rotate: (angularSpeed) => robot.rotate(angularSpeed),
    stop: () => robot.stop(),
    
    navigateTo: async (x, y, z, speed) => {
      // Real A* navigation — pathfinder handles obstacle avoidance
      _onExecuting?.({ description: `Navigating to (${x.toFixed(1)}, ${z.toFixed(1)})` })
      await pathNavigateTo(robot, x, z, speed)
    },
    
    // ─── Joint Control ───────────────────────────────────────────
    setJoint: (jointName, angleDeg) => robot.setJointTarget(jointName, angleDeg),
    setJointGroup: (groupName, angleDeg) => robot.setGroupTarget(groupName, angleDeg),
    getJointState: (jointName) => robot.getJointState(jointName),
    
    // ─── Manipulation ────────────────────────────────────────────
    grab: (objectId) => {
      // Switch physics body to kinematic and track robot each frame
      const success = grabInteractable(objectId, robot)
      if (success) {
        robot.heldObjects.push(objectId)
        return true
      }
      return false
    },

    release: () => {
      const heldId = robot.heldObjects[robot.heldObjects.length - 1]
      if (!heldId) return
      // Gentle forward placement
      releaseInteractable(heldId, robot, 1.5)
      robot.heldObjects.pop()
    },

    // Throw the held object with significant forward force
    throwObject: (force = 9.0) => {
      const heldId = robot.heldObjects[robot.heldObjects.length - 1]
      if (!heldId) return
      releaseInteractable(heldId, robot, force)
      robot.heldObjects.pop()
    },

    // Direct position lookup for objects — works even before vision scan
    findObject: (nameOrId) => findInteractablePosition(nameOrId),
    listObjects: () => listInteractables(),
    pushObject: (objectId, force = 6) => pushInteractable(objectId, force, 0, 0, robot),
    
    // ─── Perception ──────────────────────────────────────────────
    getKnownObjects: () => getKnownObjects(0.2),
    
    checkFeasibility: (action, params) => {
      return robot.checkFeasibility(action, params)
    },
    
    // ─── Utility ─────────────────────────────────────────────────
    wait: (ms) => new Promise(r => setTimeout(r, ms)),
    
    setStatus: (text) => {
      _onExecuting?.({ description: text })
    },
    
    // ─── Arguments ───────────────────────────────────────────────
    args,
    
    // ─── Robot reference (for advanced skills) ───────────────────
    robot,
    manifest: robot.manifest,
  }
}

// ─── Skill Synthesis ───────────────────────────────────────────────────────────

async function handleSkillSynthesis(spec, skillRegistry, robot) {
  const { name, description } = spec
  
  _onThinking?.(`Inventing new skill: "${name}"...`)
  
  const code = await synthesizeSkill(name, description, {
    robotPos: robot.position,
    existingSkills: skillRegistry.getAllNames(),
  })
  
  if (!code) return null
  
  // Register as pending approval
  const skill = skillRegistry.registerSynthesized(name, code, description)
  
  if (!skill) {
    console.error(`[Planner] Failed to register synthesized skill "${name}"`)
    return null
  }
  
  // Request approval from user (if callback registered)
  if (_onSkillApproval) {
    _onSkillApproval({ name, code, description })
  }
  
  return skill
}

// ─── Quick Checks ──────────────────────────────────────────────────────────────

/**
 * Fast keyword-based feasibility check before hitting the LLM.
 * Catches obviously impossible requests immediately.
 */
function quickFeasibilityCheck(instruction) {
  const lower = instruction.toLowerCase()
  
  // Check for flight-related words
  if (/\b(fly|hover|take\s?off|soar|glide)\b/.test(lower)) {
    const result = checkFeasibility('fly')
    if (!result.feasible) return result
  }
  
  // Check for swimming
  if (/\b(swim|dive|underwater)\b/.test(lower)) {
    const result = checkFeasibility('swim')
    if (!result.feasible) return result
  }
  
  // Check for climbing
  if (/\b(climb|scale|ascend)\b/.test(lower)) {
    const result = checkFeasibility('climb')
    if (!result.feasible) return result
  }
  
  return null  // No obvious infeasibility detected
}

/**
 * Try to match instruction directly to a known skill without calling the LLM.
 * Returns {skill, args} on a match, or null if the LLM is needed.
 *
 * Handles common phrasing so users get instant responses for basic commands:
 *   "move forward", "go forward 3 seconds", "turn left 90 degrees", "stop", etc.
 */
function tryDirectMatch(instruction, skillRegistry) {
  const lower = instruction.toLowerCase().trim()

  // ── Parse optional numeric qualifiers from the instruction ──────────────
  // Duration: "3 seconds", "2 sec", "1.5s"
  const secMatch = lower.match(/\b(\d+\.?\d*)\s*(?:second|sec|s)\b/)
  const durationMs = secMatch ? Math.round(parseFloat(secMatch[1]) * 1000) : null

  // Angle: "90 degrees", "45 deg"
  const degMatch = lower.match(/\b(\d+\.?\d*)\s*(?:degree|deg)\b/)
  const degrees = degMatch ? parseFloat(degMatch[1]) : null

  // ── Match table: ordered most-specific → least-specific ─────────────────
  const entries = [
    // ── Scanning ──
    { pat: /\bscan\s+room\b/,             skill: 'scan_room',     args: {} },
    { pat: /\blook\s+around\b/,           skill: 'scan_room',     args: {} },
    { pat: /\bscan\s+for\b/,              skill: 'scan_for',      args: {} },
    { pat: /\bscan\b/,                    skill: 'scan_room',     args: {} },

    // ── Forward movement ──
    { pat: /\b(move|go|walk|drive|run|travel)\s+(forward|straight|ahead)\b/,
                                           skill: 'move_forward',  args: durationMs ? { duration: durationMs } : {} },
    { pat: /\b(move|go|walk)\s+forward\b/, skill: 'move_forward',  args: durationMs ? { duration: durationMs } : {} },
    { pat: /\bgo\s+ahead\b/,              skill: 'move_forward',  args: durationMs ? { duration: durationMs } : {} },
    { pat: /\bwalk\b/,                    skill: 'move_forward',  args: durationMs ? { duration: durationMs } : {} },
    { pat: /\bgo\s+forward\b/,            skill: 'move_forward',  args: durationMs ? { duration: durationMs } : {} },
    { pat: /\bmove\s+forward\b/,          skill: 'move_forward',  args: durationMs ? { duration: durationMs } : {} },

    // ── Backward movement ──
    { pat: /\b(move|go|walk|drive|back\s*up)\s+(back|backward|backwards|reverse)\b/,
                                           skill: 'move_backward', args: durationMs ? { duration: durationMs } : {} },
    { pat: /\bgo\s+back\b/,              skill: 'move_backward',  args: durationMs ? { duration: durationMs } : {} },
    { pat: /\breverse\b/,                skill: 'move_backward',  args: durationMs ? { duration: durationMs } : {} },
    { pat: /\bback\s*up\b/,              skill: 'move_backward',  args: durationMs ? { duration: durationMs } : {} },

    // ── Turning ──
    { pat: /\bturn\s+left\b/,            skill: 'turn_left',      args: degrees ? { degrees } : {} },
    { pat: /\bturn\s+right\b/,           skill: 'turn_right',     args: degrees ? { degrees } : {} },
    { pat: /\brotate\s+left\b/,          skill: 'turn_left',      args: degrees ? { degrees } : {} },
    { pat: /\brotate\s+right\b/,         skill: 'turn_right',     args: degrees ? { degrees } : {} },
    { pat: /\bface\s+left\b/,            skill: 'turn_left',      args: { degrees: 90 } },
    { pat: /\bface\s+right\b/,           skill: 'turn_right',     args: { degrees: 90 } },

    // ── Arm position (robot has one physical arm — left arm) ──
    // Most-specific patterns first so "arm up" doesn't accidentally match "rotate"
    { pat: /\b(raise|lift|put)\s+(the\s+)?arm\s+up\b/,      skill: 'arm_up',      args: {} },
    { pat: /\barm\s+(up|raised|overhead|above)\b/,           skill: 'arm_up',      args: {} },
    { pat: /\b(reach|point)\s+up\b/,                         skill: 'arm_up',      args: {} },
    { pat: /\b(lower|drop|bring)\s+(the\s+)?arm\s+down\b/,  skill: 'arm_down',    args: {} },
    { pat: /\barm\s+(down|lowered)\b/,                       skill: 'arm_down',    args: {} },
    { pat: /\b(extend|stretch|reach|point)\s+(the\s+)?arm\s+(forward|out|ahead)\b/,
                                                              skill: 'arm_forward', args: {} },
    { pat: /\barm\s+(forward|horizontal|out|extended)\b/,   skill: 'arm_forward', args: {} },
    { pat: /\b(rest|relax|neutral|reset)\s+(the\s+)?arm\b/, skill: 'arm_rest',    args: {} },
    { pat: /\barm\s+(rest|neutral|down\s+at\s+side|at\s+side)\b/,
                                                              skill: 'arm_rest',    args: {} },

    // ── Fixed-action skills ──
    { pat: /\bpatrol\b/,                 skill: 'patrol',         args: {} },
    { pat: /\bspin\b/,                   skill: 'spin',           args: {} },
    { pat: /\bjump\b/,                   skill: 'jump',           args: {} },
    { pat: /\brotate\b/,                 skill: 'rotate',         args: degrees ? { degrees } : {} },
    { pat: /\bwave\b/,                   skill: 'wave',           args: {} },
    { pat: /\bdance\b/,                  skill: 'spin',           args: {} },
    { pat: /\bsurvey\b/,                 skill: 'survey_grid',    args: {} },
    { pat: /\b(go\s+home|return\s+home|return\s+to\s+(origin|start))\b/,
                                          skill: 'return_home',   args: {} },

    // ── Stop ──
    { pat: /\b(stop|halt|freeze|standby|stand\s+still)\b/,
                                          skill: 'stop',           args: {} },
  ]

  for (const { pat, skill, args } of entries) {
    if (pat.test(lower) && skillRegistry.has(skill)) {
      return { skill, args }
    }
  }

  return null  // No direct match — fall through to LLM
}
