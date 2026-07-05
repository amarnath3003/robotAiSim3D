/**
 * brain/react_loop.js — ReAct Closed-Loop Reasoning Engine
 *
 * Implements the ReAct pattern (Yao et al. 2023) adapted for an embodied,
 * perception-limited robot: instead of committing to a full plan upfront,
 * the LLM reasons one step at a time —
 *
 *   THOUGHT  (reason about goal + current knowledge)
 *   → ACTION (execute exactly ONE skill)
 *   → OBSERVATION (fresh perception + action result)
 *   → repeat until done / infeasible / step budget exhausted
 *
 * This grounds every reasoning step in up-to-date sensor state, which matters
 * here because perception memory decays and physics objects move. A batch
 * plan reasons about a world snapshot; ReAct reasons about the world as it
 * actually unfolds.
 *
 * Used by planner.js for complex instructions (adaptive routing) or for all
 * LLM-planned instructions when cot_mode = 'react'.
 */

import { reactStepWithLLM } from './llm.js'
import { getKnownObjects, logExecution } from '../core/state.js'
import { cotTrace } from './cot_trace.js'

const DEFAULT_MAX_STEPS = 12
const MAX_OBSERVATION_CHARS = 400
const MAX_CONSECUTIVE_FAILURES = 3

/**
 * Run the ReAct loop for one instruction.
 *
 * @param {string} instruction - Natural language instruction
 * @param {Object} skillRegistry
 * @param {Object} robot
 * @param {Object} opts
 * @param {function(args): Object} opts.contextFn - Builds skill execution context
 * @param {function(): boolean}    opts.abortFlag - Returns true when user aborted
 * @param {function(string): void} [opts.onThought]   - UI: streamed thought text
 * @param {function(Object): void} [opts.onExecuting] - UI: {step, skill, description}
 * @param {number} [opts.maxSteps]
 * @returns {Promise<{success: boolean, reason: string, steps: number, llmCalls: number, trace: Array}>}
 */
export async function runReActLoop(instruction, skillRegistry, robot, opts = {}) {
  const {
    contextFn,
    abortFlag = () => false,
    onThought,
    onExecuting,
    maxSteps = DEFAULT_MAX_STEPS,
  } = opts

  const scratchpad = []   // [{thought, action, observation}]
  let llmCalls = 0
  let consecutiveFailures = 0
  let lastFailedActionKey = null

  for (let step = 0; step < maxSteps; step++) {
    if (abortFlag()) {
      return { success: false, reason: 'Aborted by user.', steps: step, llmCalls, trace: scratchpad }
    }

    // ── OBSERVE: fresh world state every iteration ─────────────────────────
    const observation = buildObservation(robot)

    // ── THINK: one LLM call → thought + single action (or done/infeasible) ─
    let decision
    try {
      llmCalls++
      decision = await reactStepWithLLM(instruction, {
        scratchpad,
        observation,
        availableSkills: skillRegistry.getAllForLLM(),
        stepNumber: step + 1,
        maxSteps,
      })
    } catch (e) {
      cotTrace.record('error', { step: step + 1, message: e.message })
      return { success: false, reason: `Reasoning failed: ${e.message}`, steps: step, llmCalls, trace: scratchpad }
    }

    const thought = String(decision.thought || '').trim()
    if (thought) {
      onThought?.(`🧠 Thought ${step + 1}: ${thought}`)
      cotTrace.record('thought', { step: step + 1, thought })
    }

    // ── Terminal states ────────────────────────────────────────────────────
    if (decision.infeasible) {
      const reason = decision.infeasibleReason || 'The task is not feasible for this robot.'
      cotTrace.record('observation', { step: step + 1, terminal: 'infeasible', reason })
      return { success: false, reason, steps: step + 1, llmCalls, trace: scratchpad }
    }

    if (decision.done) {
      const reason = decision.doneReason || 'Task complete.'
      cotTrace.record('observation', { step: step + 1, terminal: 'done', reason })
      return { success: true, reason, steps: step + 1, llmCalls, trace: scratchpad }
    }

    // ── ACT: execute exactly one skill ─────────────────────────────────────
    const action = decision.action
    if (!action?.skill) {
      scratchpad.push({
        thought,
        action: null,
        observation: 'ERROR: You must provide either an action, done:true, or infeasible:true.',
      })
      continue
    }

    cotTrace.record('action', { step: step + 1, skill: action.skill, args: action.args })
    onExecuting?.({
      step: step + 1,
      total: maxSteps,
      skill: action.skill,
      description: action.description || `${action.skill}(${JSON.stringify(action.args || {})})`,
    })

    const result = await executeAction(action, skillRegistry, contextFn, abortFlag)
    cotTrace.record('observation', { step: step + 1, ok: result.ok, text: result.observation })

    // Bookkeeping mirrors the batch-plan path so history/affordances stay consistent
    logExecution(action.skill, result.ok ? 'success' : 'failure', action.description || result.observation)
    skillRegistry.recordExecution?.(action.skill, result.ok)

    scratchpad.push({ thought, action, observation: result.observation })

    // ── Loop-guard: stop burning steps on a repeating failure ──────────────
    const actionKey = `${action.skill}:${JSON.stringify(action.args || {})}`
    if (!result.ok) {
      consecutiveFailures = actionKey === lastFailedActionKey ? consecutiveFailures + 1 : 1
      lastFailedActionKey = actionKey
      if (consecutiveFailures >= MAX_CONSECUTIVE_FAILURES) {
        return {
          success: false,
          reason: `Stuck: "${action.skill}" failed ${consecutiveFailures} times in a row (${result.observation})`,
          steps: step + 1,
          llmCalls,
          trace: scratchpad,
        }
      }
      // Nudge the next thought away from verbatim repetition
      scratchpad[scratchpad.length - 1].observation +=
        ' — Do NOT repeat this exact action. Try a different skill, different args, or scan first.'
    } else {
      consecutiveFailures = 0
      lastFailedActionKey = null
    }
  }

  return {
    success: false,
    reason: `Step budget exhausted (${maxSteps} steps) without completing the task.`,
    steps: maxSteps,
    llmCalls,
    trace: scratchpad,
  }
}

// ─── Observation Builder ───────────────────────────────────────────────────────

/**
 * Snapshot the world as the robot currently knows it.
 * Perception-memory only — same "environment unknowness" rule as the planner.
 */
function buildObservation(robot) {
  const lines = []
  const p = robot.position
  lines.push(`Position: (${p.x.toFixed(1)}, ${p.z.toFixed(1)})`)

  const q = robot.orientation
  if (q) {
    const headingDeg = Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z)) * (180 / Math.PI)
    lines.push(`Heading: ${headingDeg.toFixed(0)}°`)
  }

  lines.push(`Holding: ${robot.heldObjects?.length ? robot.heldObjects.join(', ') : 'nothing'}`)

  const perceived = getKnownObjects(0.1)
  if (perceived.length > 0) {
    lines.push('Perceived objects:')
    for (const o of perceived.slice(0, 12)) {
      const ageMs = Date.now() - (o.lastSeen || Date.now())
      const stale = ageMs > 5000 ? ` [stale ${(ageMs / 1000).toFixed(0)}s]` : ''
      lines.push(`- ${o.id} conf=${o.confidence.toFixed(2)} pos=(${o.position.x.toFixed(1)}, ${o.position.z.toFixed(1)})${stale}`)
    }
  } else {
    lines.push('Perceived objects: none — not scanned yet.')
  }

  return lines.join('\n')
}

// ─── Action Executor ───────────────────────────────────────────────────────────

/**
 * Execute one skill and convert the outcome into an observation string.
 * Never throws — failures become observations the LLM reasons about next step.
 */
async function executeAction(action, skillRegistry, contextFn, abortFlag) {
  const skill = skillRegistry.get(action.skill)
  if (!skill) {
    return {
      ok: false,
      observation: `ERROR: skill "${action.skill}" does not exist. Use exact names from Available Skills.`,
    }
  }

  if (abortFlag()) {
    return { ok: false, observation: 'Aborted by user.' }
  }

  try {
    await skill.execute(contextFn(action.args || {}))
    return { ok: true, observation: `${action.skill} succeeded.` }
  } catch (e) {
    const msg = String(e.message || e).slice(0, MAX_OBSERVATION_CHARS)
    return { ok: false, observation: `${action.skill} FAILED: ${msg}` }
  }
}
