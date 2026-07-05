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

import { planWithLLM, synthesizeSkill, clearConversationHistory, reflectWithLLM, getTokenUsage } from './llm.js'
// XM-4: removed checkPlanFeasibility — it's not called directly in planner.js
import { checkFeasibility } from './feasibility.js'
import { getRobot, setRobotStatus, logExecution, getKnownObjects, getRecentHistory } from '../core/state.js'
// XM-4: removed dead manifest imports (getManifest, getCapabilities, hasCapability)
//        — manifest decisions are delegated to feasibility.js and adapter.js
import { navigateTo as pathNavigateTo, navigatePath as pathNavigatePath, abortNavigation } from '../nav/pathfinder.js'
import { BTRunner, Blackboard, planToBehaviorTree } from './behavior_tree.js'
import { reflect } from './reflection.js'
import { runReActLoop } from './react_loop.js'
import { cotTrace, getCoTMode } from './cot_trace.js'
import {
  grabInteractable,
  releaseInteractable,
  pushInteractable,
} from '../env/objects.js'

// ─── Plan Execution State ──────────────────────────────────────────────────────

let _executing = false
let _currentPlan = null
let _currentStep = 0
let _abortRequested = false
let _activeMode = 'adaptive'     // CoT mode for the current instruction (off|prompt|react|adaptive)
let _reactEscalated = false      // Guard: at most one plan→ReAct escalation per instruction

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
 * Wraps the core pipeline with CoT trace bookkeeping: every instruction
 * becomes one trace episode (mode, routing tier, thoughts, actions, outcome,
 * token cost) exportable via window.exportCoTTraces() for experiments.
 *
 * @param {string} instruction - Natural language instruction from user
 * @param {Object} skillRegistry - The skill registry instance
 * @returns {Promise<{success: boolean, reason: string}>}
 */
export async function handleInstruction(instruction, skillRegistry) {
  if (_executing) {
    return { success: false, reason: 'Already executing a plan. Wait or abort.' }
  }

  _activeMode = getCoTMode()
  _reactEscalated = false
  cotTrace.startEpisode(instruction, _activeMode, 'unrouted')
  const tokensBefore = getTokenUsage()

  let result
  try {
    result = await _handleInstructionCore(instruction, skillRegistry)
  } catch (e) {
    result = { success: false, reason: e.message }
  }

  const tokensAfter = getTokenUsage()
  cotTrace.endEpisode({
    success: result.success,
    reason: result.reason,
    tokens: {
      prompt: tokensAfter.prompt - tokensBefore.prompt,
      completion: tokensAfter.completion - tokensBefore.completion,
    },
  })

  return result
}

async function _handleInstructionCore(instruction, skillRegistry) {
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
      cotTrace.record('route', { tier: 'direct', skill: directMatch.skill })
      const result = await executePlan(
        [{ skill: directMatch.skill, args: directMatch.args, description: instruction }],
        skillRegistry,
        robot,
        instruction
      )
      return result
    }

    // ─── Step 2.5: Adaptive CoT routing ──────────────────────────────────
    // 'react' mode (ablation arm): every LLM-planned instruction runs the
    // ReAct loop. 'adaptive' mode ALWAYS tries the cheap single-call plan
    // first — closed-loop ReAct is an escalation path, entered when the
    // planner LLM itself declares the task unplannable upfront
    // (needsStepByStep) or when the batch plan fails. Upfront ReAct costs
    // 4-10x latency with the robot frozen between calls, and the LLM with
    // full scene context is a better judge of plannability than a regex.
    // The local classifier only picks reasoning DEPTH: complex instructions
    // get the full structured CoT, simple ones a brief rationale.
    const complexity = classifyComplexity(instruction)
    if (_activeMode === 'react') {
      cotTrace.record('route', { tier: 'react', complexity })
      _onThinking?.(`Reasoning step-by-step (ReAct).`)
      return await runReActPath(instruction, skillRegistry, robot)
    }
    cotTrace.record('route', { tier: 'plan', complexity })

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
      cotStyle: _activeMode === 'off' ? 'off'
        : _activeMode === 'prompt' ? 'structured'
        : (complexity === 'complex' ? 'structured' : 'brief'),
    })

    if (!llmResponse) {
      throw new Error('LLM returned no response')
    }

    // ─── Step 3.5: LLM self-escalation to ReAct ──────────────────────────
    // The planner LLM can declare the task unplannable-upfront (later steps
    // depend on what earlier steps discover) — switch to the ReAct loop.
    if (llmResponse.needsStepByStep && !llmResponse.plan?.length &&
        (_activeMode === 'adaptive' || _activeMode === 'react')) {
      cotTrace.record('escalation', { from: 'plan', reason: 'needsStepByStep' })
      _onThinking?.('Plan depends on discoveries along the way — switching to step-by-step reasoning.')
      return await runReActPath(instruction, skillRegistry, robot)
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
    if (llmResponse.cot) {
      const c = llmResponse.cot
      cotTrace.record('plan', { cot: c, skills: plan.map(s => s.skill) })
      const parts = []
      if (c.situation)   parts.push(`Situation: ${c.situation}`)
      if (c.unknowns)    parts.push(`Unknowns: ${c.unknowns}`)
      if (c.feasibility) parts.push(`Feasibility: ${c.feasibility}`)
      if (c.strategy)    parts.push(`Strategy: ${c.strategy}`)
      if (c.risks)       parts.push(`Risks: ${c.risks}`)
      if (parts.length) _onThinking?.(`🧠 ${parts.join('\n')}`)
    } else if (llmResponse.reasoning) {
      cotTrace.record('plan', { reasoning: llmResponse.reasoning, skills: plan.map(s => s.skill) })
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

  // ── Last resort: escalate batch failure to closed-loop ReAct recovery ────
  // Batch replans reason about a stale world snapshot; the ReAct loop
  // re-observes after every action, so it can recover from failures whose
  // cause the snapshot can't see (moved objects, decayed perception).
  if (!_abortRequested && !_reactEscalated && _activeMode === 'adaptive') {
    _reactEscalated = true
    cotTrace.record('escalation', { from: 'replan', reason: failReason })
    _onThinking?.('Batch replanning exhausted — escalating to step-by-step ReAct recovery.')
    _currentPlan = null
    return await runReActPath(instruction, skillRegistry, robot, failReason)
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
      cotStyle: _activeMode === 'off' ? 'off' : 'structured',
    })

    if (!llmResponse || llmResponse.infeasible || !llmResponse.plan?.length) {
      console.warn('[Planner] Replan returned infeasible or empty plan')
      return null
    }

    // Structured failure-analysis CoT: root cause + what the failure reveals
    // about the world + why the new strategy should work
    if (llmResponse.failureAnalysis) {
      const fa = llmResponse.failureAnalysis
      cotTrace.record('failure_analysis', fa)
      const faParts = []
      if (fa.rootCause)    faParts.push(`Root cause: ${fa.rootCause}`)
      if (fa.worldChanged) faParts.push(`Learned: ${fa.worldChanged}`)
      if (fa.newStrategy)  faParts.push(`New strategy: ${fa.newStrategy}`)
      if (faParts.length) _onThinking?.(`🧠 ${faParts.join('\n')}`)
    }

    _onThinking?.(`Replan (attempt ${replanDepth}): ${llmResponse.plan.length} recovery steps`)

    // Execute the recovery plan (recursive, depth-limited by MAX_REPLAN_ATTEMPTS)
    return await executePlan(llmResponse.plan, skillRegistry, robot, instruction, replanDepth)

  } catch (e) {
    console.warn('[Planner] Replan call failed:', e.message)
    return null
  }
}

// ─── ReAct Path ────────────────────────────────────────────────────────────────

/**
 * Execute an instruction via the ReAct closed loop (think → act → observe).
 * Used for complex instructions (adaptive routing), for everything when
 * cot_mode='react', and as last-resort escalation after batch replans fail.
 *
 * @param {string} instruction
 * @param {Object} skillRegistry
 * @param {Object} robot
 * @param {string|null} [failureContext] - Set when escalating from a failed batch plan
 * @returns {Promise<{success: boolean, reason: string}>}
 */
async function runReActPath(instruction, skillRegistry, robot, failureContext = null) {
  setRobotStatus('executing')

  const task = failureContext
    ? `${instruction}\n(NOTE: a previous batch plan already failed: ${failureContext}. Recover and finish the task.)`
    : instruction

  const result = await runReActLoop(task, skillRegistry, robot, {
    contextFn: (args) => buildExecutionContext(robot, args, skillRegistry),
    abortFlag: () => _abortRequested,
    onThought: (t) => _onThinking?.(t),
    onExecuting: (e) => _onExecuting?.(e),
  })

  setRobotStatus(result.success ? 'idle' : (_abortRequested ? 'idle' : 'failed'))
  _executing = false

  // Reconstruct a plan-shaped step list from the trace so reflection,
  // episodic memory and affordance recording stay consistent with the batch path
  const executedSteps = result.trace
    .filter(s => s.action)
    .map(s => ({
      skill: s.action.skill,
      args: s.action.args || {},
      description: s.action.description || s.thought?.slice(0, 80) || s.action.skill,
    }))

  if (result.success) {
    _onComplete?.(instruction)
    logExecution(instruction, 'success', `ReAct: ${result.steps} steps, ${result.llmCalls} LLM calls`)
  } else {
    _onError?.(result.reason)
    logExecution(instruction, 'failure', result.reason)
  }

  reflect({
    instruction,
    plan: executedSteps,
    success: result.success,
    failReason: result.success ? undefined : result.reason,
    robotPos: { x: robot.position.x, y: robot.position.y, z: robot.position.z },
    knownObjects: getKnownObjects(0.2),
  }, reflectWithLLM).catch(() => {})

  return { success: result.success, reason: result.reason }
}

// ─── Complexity Classifier ─────────────────────────────────────────────────────

/**
 * Cheap local heuristic estimating instruction complexity. No LLM call — pure
 * regex. Used ONLY to pick the reasoning depth of the single planning call
 * (brief rationale vs full structured CoT) — never to route execution mode;
 * the planner LLM's own needsStepByStep signal decides ReAct escalation.
 *
 * Complex signals: sequenced clauses, discovery/search, iteration over sets,
 * conditionals, many distinct action verbs.
 *
 * @param {string} instruction
 * @returns {'simple'|'complex'}
 */
function classifyComplexity(instruction) {
  const lower = instruction.toLowerCase()
  let score = 0

  // Sequenced clauses — later steps depend on earlier outcomes
  if (/\b(then|after that|and then|before|while|until|once|first|next|finally)\b/.test(lower)) score += 2

  // Conditionals — plan branches on runtime state
  if (/\b(if|unless|whenever|depending|in case)\b/.test(lower)) score += 2

  // Iteration over an unknown-sized set
  if (/\b(all|each|every|both|remaining)\b/.test(lower)) score += 2

  // Discovery — target state unknown until sensed
  if (/\b(find|search|look for|locate|explore|bring|fetch|collect|gather|count|check)\b/.test(lower)) score += 1

  // Many distinct action verbs — long multi-skill task
  const verbs = lower.match(/\b(go|move|turn|pick|grab|push|place|put|bring|scan|find|drop|throw|navigate|explore|search|carry|fetch|release|jump|wave|patrol)\b/g)
  if (verbs && new Set(verbs).size >= 3) score += 1

  return score >= 2 ? 'complex' : 'simple'
}

// ─── Execution Context Builder ─────────────────────────────────────────────────

/**
 * Build the context object passed to skill functions.
 * This is the API that skills use to control the robot.
 * Exported so background_agent.js and main.js can build contexts directly.
 */
export function buildExecutionContext(robot, args, skillRegistry) {
  const ctx = {
    // ─── Robot Control ────────────────────────────────────────────
    getPos: () => ({
      x: robot.position.x,
      y: robot.position.y,
      z: robot.position.z,
    }),

    // Heading in radians: 0 = facing +Z, forward vector = (sin h, 0, cos h)
    getHeading: () => {
      const q = robot.orientation
      return Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z))
    },

    setPos: (x, y, z) => {
      robot.position.set(x, y, z)
      if (robot.physicsBody) {
        robot.physicsBody.setNextKinematicTranslation({ x, y, z })
      }
    },

    moveForward: (speed) => robot.moveForward(speed),
    rotate: (angularSpeed) => robot.rotate(angularSpeed),
    stop: () => robot.stop(),

    navigateTo: async (x, y, z, opts) => {
      // Real A* navigation — pathfinder handles obstacle avoidance, replanning
      // and reactive LiDAR steering. opts: number (speed) or
      // {speed, approach, arrive, face:{x,z}, timeout}. Throws when unreachable.
      _onExecuting?.({ description: `Navigating to (${x.toFixed(1)}, ${z.toFixed(1)})` })
      return await pathNavigateTo(robot, x, z, opts)
    },

    // Follow an ordered list of world waypoints (any geometry — shapes,
    // patrol routes, LLM-computed trajectories). Obstacles are detoured
    // around automatically; unreachable vertices are skipped and counted.
    navigatePath: async (points, opts) => {
      _onExecuting?.({ description: `Following ${points.length}-point path` })
      return await pathNavigatePath(robot, points, opts)
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

    pushObject: (objectId, force = 6) => pushInteractable(objectId, force, 0, 0, robot),

    // ─── Perception (perception-memory only — NO direct physics lookup) ───
    // Skills must use getPerceivedObjects() to find object positions.
    // This enforces "environment unknowness" — the robot only knows what it has seen.
    getKnownObjects:     () => getKnownObjects(0.2),
    getPerceivedObjects: () => getKnownObjects(0.1),

    // Scored fuzzy match against perception memory: matches ids ("ball_green"),
    // colour metadata ("green"), labels ("sports ball") and free descriptions
    // ("the big green ball"). Highest-scoring object wins; ties break toward
    // higher confidence.
    //
    // Attributes are DISCRIMINATIVE, not just additive: "blue ball" must never
    // settle for ball_orange just because "ball" matched — a known-but-different
    // colour is strong evidence of the WRONG object, so it outweighs a generic
    // type-word match. (CV colour labels are noisy, so contradiction is a heavy
    // penalty rather than a hard reject.) A colour alone is never enough either:
    // "blue ball" must not match a blue stop sign.
    findPerceivedObject: (nameOrDesc) => {
      const desc = String(nameOrDesc || '').toLowerCase().trim()
      if (!desc) return null
      const known = getKnownObjects(0.08)
      if (!known.length) return null

      const COLORS = new Set([
        'red', 'blue', 'green', 'yellow', 'orange', 'pink', 'purple',
        'brown', 'wooden', 'white', 'black', 'gray', 'grey',
      ])
      const dWords = desc.split(/[^a-z0-9]+/).filter(w => w.length > 1 || /\d/.test(w))
      const dColors = dWords.filter(w => COLORS.has(w))
      const dRest = dWords.filter(w => !COLORS.has(w) && w !== 'the')

      let best = null
      let bestScore = 0

      for (const o of known) {
        const idL = o.id.toLowerCase()
        const oWords = new Set([
          ...idL.split(/[^a-z0-9]+/),
          ...String(o.meta?.colorName || '').toLowerCase().split(/[^a-z0-9]+/),
          ...String(o.meta?.label || '').toLowerCase().split(/[^a-z0-9]+/),
        ].filter(Boolean))
        const oColors = [...oWords].filter(w => COLORS.has(w))

        let s = 0
        if (idL === desc) s += 6

        // Colour evidence: match rewards, contradiction heavily penalizes
        if (dColors.length && oColors.length) {
          s += dColors.some(c => oColors.includes(c)) ? 3 : -4
        }

        // Non-colour words (type/name): at least one must land, else no match —
        // colour similarity alone must not pick an object of the wrong kind
        let restScore = 0
        for (const w of dRest) {
          if (oWords.has(w)) restScore += 2
          else {
            for (const ow of oWords) {
              if (ow.length > 2 && (ow.includes(w) || w.includes(ow))) { restScore += 1; break }
            }
          }
        }
        if (dRest.length && restScore === 0 && idL !== desc) continue
        s += restScore

        s += Math.min(o.confidence, 1) * 0.8

        if (s > bestScore) { bestScore = s; best = o }
      }

      return bestScore >= 2 ? best : null
    },

    // Rotate in place through a full circle, capturing CV frames, until the
    // target is perceived (or the sweep completes). Returns the match or null.
    // With no target, completes the sweep and returns everything known.
    spinScan: async (target = null) => {
      const speed = 1.0
      const stepMs = 400
      const steps = Math.ceil(((2 * Math.PI) / speed) * 1000 / stepMs)

      for (let i = 0; i < steps; i++) {
        if (_abortRequested) break
        robot.rotate(speed)
        await ctx.wait(stepMs)
        robot.stop()
        await ctx.wait(90)
        await ctx.captureCV()
        await ctx.wait(160)

        if (target) {
          const m = ctx.findPerceivedObject(target)
          if (m) { robot.stop(); return m }
        }
      }
      robot.stop()
      return target ? ctx.findPerceivedObject(target) : ctx.getPerceivedObjects()
    },

    // Actively acquire an object: perception memory → in-place spin scan →
    // drive to spread-out vantage points and scan from each. This is what
    // makes "go to X" work even when X has never been seen.
    acquireObject: async (target, { fullSearch = true } = {}) => {
      let m = ctx.findPerceivedObject(target)
      if (m) return m

      ctx.setStatus(`Searching for "${target}" — scanning...`)
      m = await ctx.spinScan(target)
      if (m && !Array.isArray(m)) return m
      if (!fullSearch) return null

      // Vantage exploration: visit well-spread points across the arena,
      // nearest first, scanning at each stop.
      const pos = ctx.getPos()
      const vantages = [
        { x: 0, z: 0 }, { x: 7, z: 7 }, { x: -7, z: 7 },
        { x: 7, z: -7 }, { x: -7, z: -7 }, { x: 0, z: 10 }, { x: 10, z: 0 },
      ]
        .map(v => ({ ...v, d: Math.hypot(v.x - pos.x, v.z - pos.z) }))
        .filter(v => v.d > 2.5)
        .sort((a, b) => a.d - b.d)
        .slice(0, 5)

      for (const v of vantages) {
        if (_abortRequested) break
        ctx.setStatus(`Searching for "${target}" — moving to vantage (${v.x}, ${v.z})...`)
        try {
          await ctx.navigateTo(v.x, 0, v.z, { arrive: 0.8 })
        } catch { continue }   // vantage unreachable — try the next one

        m = await ctx.spinScan(target)
        if (m && !Array.isArray(m)) return m
      }
      return null
    },

    // Trigger an immediate CV capture (for scan skills that need live results)
    captureCV: async () => {
      try {
        const { captureAndAnalyze } = await import('../perception/cv_camera.js')
        return await captureAndAnalyze()
      } catch { return [] }
    },

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

  return ctx
}

// ─── Skill Synthesis ───────────────────────────────────────────────────────────

async function handleSkillSynthesis(spec, skillRegistry, robot) {
  const { name, description } = spec

  _onThinking?.(`Inventing new skill: "${name}"...`)

  const synth = await synthesizeSkill(name, description, {
    robotPos: robot.position,
    existingSkills: skillRegistry.getAllNames(),
  })

  if (!synth?.code) return null

  // Surface the reason-before-code CoT: approach, physics checks, declared risks
  if (synth.reasoning) {
    cotTrace.record('synthesis_reasoning', { name, ...synth.reasoning })
    const r = synth.reasoning
    const parts = []
    if (r.approach)      parts.push(`Approach: ${r.approach}`)
    if (r.physicsChecks) parts.push(`Physics: ${r.physicsChecks}`)
    if (r.risks?.length) parts.push(`Risks: ${[].concat(r.risks).join('; ')}`)
    if (parts.length) _onThinking?.(`🧠 ${parts.join('\n')}`)
  }

  // Register as pending approval
  const skill = skillRegistry.registerSynthesized(name, synth.code, description)

  if (!skill) {
    console.error(`[Planner] Failed to register synthesized skill "${name}"`)
    return null
  }

  // Request approval from user (if callback registered)
  if (_onSkillApproval) {
    _onSkillApproval({ name, code: synth.code, description })
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

  // Fast path is for SHORT imperative commands only ("turn left 90 degrees",
  // "scan the room"). Longer sentences must go to the LLM — the loose substring
  // patterns below (\bwalk\b, \bscan\b, \bstop\b, "go ahead"…) would otherwise
  // hijack them, run one trivial skill, and report success without doing the
  // actual task ("walk to the red ball" → bare move_forward → "Task complete").
  if (lower.split(/\s+/).length > 4) return null

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
