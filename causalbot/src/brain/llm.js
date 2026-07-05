/**
 * brain/llm.js — Provider-Agnostic LLM Communication Layer
 * 
 * Handles all LLM API calls for CausalBot. Provider-agnostic:
 * supports NVIDIA, OpenAI, Anthropic, or local models.
 * 
 * Key improvements over old llm.js:
 * - Injects robot manifest into system prompt automatically
 * - Structured message history for multi-turn context
 * - Retry logic with exponential backoff
 * - Token usage tracking
 * - Response validation
 */

import { generateLLMDescription, getManifest } from '../core/manifest.js'
import { generateFeasibilityPrompt } from './feasibility.js'
import { getRecentHistory, getKnownObjects } from '../core/state.js'
import { sceneGraph } from '../perception/scene_graph.js'
import { episodicMemory } from './episodic_memory.js'
import { worldModel } from './world_model.js'
// NOTE: listInteractables intentionally NOT imported — LLM must not have omniscient object knowledge

// ─── Configuration ─────────────────────────────────────────────────────────────

const config = {
  apiKey: import.meta.env.VITE_LLM_API_KEY || import.meta.env.VITE_NVIDIA_API_KEY || '',
  apiUrl: import.meta.env.VITE_LLM_API_URL || '/api/gemini/chat/completions',
  model: import.meta.env.VITE_LLM_MODEL || import.meta.env.VITE_NVIDIA_MODEL || 'gemma-4-31b-it',
  maxRetries: 2,
  retryDelay: 500,
  defaultMaxTokens: 2048,   // Increased from 1024 — Gemini thinking output can be verbose
  temperature: 0.1,
}

// ─── Token Usage Tracking ──────────────────────────────────────────────────────

let _totalTokensUsed = { prompt: 0, completion: 0 }

export function getTokenUsage() {
  return { ..._totalTokensUsed }
}

// ─── Multi-Turn Conversation History ───────────────────────────────────────────

const MAX_HISTORY_TURNS = 6   // Keep last N user+assistant turn pairs

/**
 * Stored as flat [{role, content}, ...] — user and assistant alternate.
 * Only the last MAX_HISTORY_TURNS*2 entries are sent to the LLM.
 */
let _conversationHistory = []

/**
 * Append a message to the conversation history.
 * Automatically prunes to MAX_HISTORY_TURNS pairs.
 */
function _addToHistory(role, content) {
  _conversationHistory.push({ role, content })
  // Keep at most MAX_HISTORY_TURNS full turns (2 messages per turn)
  const maxEntries = MAX_HISTORY_TURNS * 2
  if (_conversationHistory.length > maxEntries) {
    _conversationHistory.splice(0, _conversationHistory.length - maxEntries)
  }
}

/**
 * Clear conversation history (call between unrelated tasks or on session reset).
 */
export function clearConversationHistory() {
  _conversationHistory = []
}

/**
 * Get the current conversation history.
 * @returns {Array<{role:string, content:string}>}
 */
export function getConversationHistory() {
  return [..._conversationHistory]
}

// ─── System Prompt Generation ──────────────────────────────────────────────────

/**
 * Build the system prompt. Robot has NO omniscient world knowledge.
 * It must use sensors (CV camera, LiDAR) to discover the environment.
 */
function buildSystemPrompt() {
  const manifest = getManifest()
  if (!manifest) return 'You are a robot controller. No robot manifest loaded.'

  const robotDesc = generateLLMDescription()
  const feasibilityRules = generateFeasibilityPrompt()

  return `You are the brain of a fully autonomous robot. You have NO built-in knowledge of the world.
You discover your environment exclusively through your sensors: a camera (CV) and LiDAR.

# Robot Profile
${robotDesc}

# ${feasibilityRules}

# PERCEPTION (Read carefully)
You are BLIND to the world by default. The only things you know are:
1. Your current position/heading and what you are holding.
2. Objects listed in "Perceived Objects" below — detected by your camera/sensors.
3. Your recent action history.

You do NOT have a map of objects. You do NOT know where anything is unless it appears in "Perceived Objects".
Object positions are APPROXIMATE (plus or minus 0.3m) — sensors are not perfectly accurate.
Physics-driven objects (balls, boxes) can MOVE — skills re-verify positions automatically.

# NAVIGATION — AUTOMATIC OBSTACLE AVOIDANCE
All movement skills (navigate_to, move_to_position, follow_path, patrol, explore…)
use LiDAR + A* pathfinding underneath: they automatically avoid ALL walls, pillars
and objects, replan when something blocks the way, and stop at safe distances.
NEVER add manual obstacle-avoidance steps to a plan — it is built in.

navigate_to(target) is FULLY self-sufficient: if the target is not yet perceived it
automatically rotate-scans, then explores vantage points until it finds the object,
then drives to it and stops at a safe distance facing it. One step does it all.
scan_for is only needed when you want to LOOK without moving to the object.

If a skill fails it throws a clear error and you will be asked to replan — trust
the error text, do not repeat the same step unchanged.

# GEOMETRIC / PATTERN MOVEMENT — COMPUTE THE WAYPOINTS YOURSELF
For ANY shape, letter, curve or custom trajectory, compute the vertex list and call
follow_path. There are no per-shape skills — you are the geometry engine.
frame "robot": each point {x, y} is metres RIGHT(+x)/LEFT(−x) and FORWARD(+y) from
the robot's current pose (start point (0,0) is implicit — do not include it).
frame "world": absolute {x, z} coordinates. closed:true returns to the start.
Walls in the way are fine — the path detours around them and resumes.

Example — "move in a triangle with 2 m sides":
{"skill":"follow_path","args":{"points":[{"x":0,"y":2},{"x":-1.73,"y":1}],"frame":"robot","closed":true}}
Example — "walk in a circle of radius 2":
points = 12 samples of {x: 2*cos(t)-2, y: 2*sin(t)} for t=30°,60°,…,360° (circle through the start, centre 2 m to the left), closed:true.
Same method for squares, stars, zigzags, letters, spirals — sample the geometry, keep points within ±14 world metres.

# TASK DECOMPOSITION
"go to X / reach X / find X"  → [navigate_to(target:"X")]          (one step — it self-searches)
"pick up X"        → [pick_up(target:"X")]                          (self-searching too)
"push X"           → [push(target:"X")]
"bring X to Y"     → [pick_up(target:"X"), place_near(target:"Y")]
"move in a <shape>" → [follow_path(points: <computed vertices>, frame:"robot", closed:true)]
"scan the room"    → [scan_room]
"go forward 2s"    → [move_forward(duration:2000)]
"turn left 90deg"  → [turn_left(degrees:90)]
"explore"          → [explore(steps:4)]

# OBJECT IDENTITY
- Use the EXACT id from "Perceived Objects" (e.g. "ball_red", "ball_blue", "box_A").
- CV labels objects by color/type: "red ball" maps to id "ball_red".
- If unsure of exact id, use the color description — skills will fuzzy-match.
- Object IDs are case-sensitive: box_A not box_a.

# Execution Rules
- Break complex tasks into atomic skill steps.
- If physically impossible given robot constraints, set infeasible:true and explain why.
- After failure, analyze what went wrong and produce a DIFFERENT recovery plan.
- Never repeat the same failed plan.

# Response Format
Always respond with ONLY valid JSON — no markdown, no text outside the JSON object.`
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Call the LLM with a structured prompt for task planning.
 * Automatically includes robot manifest context and perception state.
 * 
 * @param {string} instruction - The user's instruction
 * @param {Object} context - Additional context (known objects, history, etc.)
 * @returns {Promise<Object>} Parsed JSON response from LLM
 */
export async function planWithLLM(instruction, context = {}) {
  const systemPrompt = buildSystemPrompt()
  
  // Build the user message with current state
  const knownObjects = context.knownObjects || getKnownObjects(0.2)
  const history = context.history || getRecentHistory(5)
  const availableSkills = context.availableSkills || []
  const robotState = context.robotState || {}
  
  const userMessage = buildPlanningPrompt(instruction, {
    knownObjects,
    history,
    availableSkills,
    robotState,
    failureContext: context.failureContext || null,
    cotStyle: context.cotStyle || 'structured',
  })

  // Include prior conversation turns so the LLM has multi-turn context
  const messages = [
    { role: 'system', content: systemPrompt },
    ..._conversationHistory,
    { role: 'user', content: userMessage },
  ]
  
  const raw = await callLLM(messages, config.defaultMaxTokens, true /* forceJSON */)

  // Save this turn to history so future calls have context
  _addToHistory('user', userMessage)
  _addToHistory('assistant', raw)

  return parseJSONResponse(raw)
}

/**
 * Stream the LLM planning response, calling onChunk for each text delta.
 * Returns the full parsed JSON plan once streaming completes.
 *
 * @param {string} instruction
 * @param {Object} context
 * @param {function(string):void} onChunk - Called with each incremental token
 * @returns {Promise<Object>} Parsed JSON plan
 */
export async function streamPlanWithLLM(instruction, context = {}, onChunk) {
  const systemPrompt = buildSystemPrompt()

  const knownObjects   = context.knownObjects   || getKnownObjects(0.2)
  const history        = context.history        || getRecentHistory(5)
  const availableSkills = context.availableSkills || []
  const robotState     = context.robotState     || {}

  const userMessage = buildPlanningPrompt(instruction, {
    knownObjects,
    history,
    availableSkills,
    robotState,
    failureContext: context.failureContext || null,
    cotStyle: context.cotStyle || 'structured',
  })

  const messages = [
    { role: 'system', content: systemPrompt },
    ..._conversationHistory,
    { role: 'user', content: userMessage },
  ]

  const raw = await callLLMStream(messages, config.defaultMaxTokens, onChunk)

  _addToHistory('user', userMessage)
  _addToHistory('assistant', raw)

  return parseJSONResponse(raw)
}

/**
 * One ReAct reasoning step: given the instruction, the scratchpad of prior
 * thought/action/observation triplets, and a FRESH observation of the world,
 * return the next thought plus exactly one action (or done / infeasible).
 *
 * Stateless per call — the scratchpad IS the context, so reasoning is always
 * grounded in the observation built this step (perception decays, objects move).
 *
 * @param {string} instruction
 * @param {Object} params
 * @param {Array<{thought, action, observation}>} params.scratchpad
 * @param {string} params.observation - Fresh world snapshot (perception-memory only)
 * @param {Array}  params.availableSkills - Full skill objects {name, description, args}
 * @param {number} params.stepNumber
 * @param {number} params.maxSteps
 * @returns {Promise<{thought, action, done, doneReason, infeasible, infeasibleReason}>}
 */
export async function reactStepWithLLM(instruction, { scratchpad, observation, availableSkills, stepNumber, maxSteps }) {
  const systemPrompt = buildSystemPrompt() + `

# ReAct MODE — ONE STEP AT A TIME
You are reasoning step-by-step in a closed loop. Each turn you produce ONE thought
and ONE action. After the action executes you receive a fresh observation and think again.
- Base decisions on the CURRENT observation, not assumptions. Perception decays; objects move.
- If the target is not in perceived objects, remember navigate_to/pick_up self-search —
  or scan first if you only need to look.
- If the previous action FAILED, the observation says why. Change strategy — never repeat
  a failed action unchanged.
- Declare done:true as soon as the goal is achieved. Do not add unnecessary steps.
- Declare infeasible:true if the task violates robot constraints or is impossible.
- You have ${maxSteps} steps total. Be economical.`

  const lines = []
  lines.push(`# Task: "${instruction}"`)
  lines.push('')

  if (scratchpad.length > 0) {
    lines.push(`# Previous Steps:`)
    scratchpad.forEach((s, i) => {
      lines.push(`Thought ${i + 1}: ${s.thought || '(none)'}`)
      lines.push(`Action ${i + 1}: ${s.action ? `${s.action.skill}(${JSON.stringify(s.action.args || {})})` : '(none)'}`)
      lines.push(`Observation ${i + 1}: ${s.observation}`)
    })
    lines.push('')
  }

  lines.push(`# Current Observation (step ${stepNumber}/${maxSteps}):`)
  lines.push(observation)
  lines.push('')

  lines.push(`# Available Skills (use EXACT skill names and arg keys):`)
  for (const s of availableSkills) {
    const argStr = s.args && Object.keys(s.args).length > 0
      ? Object.entries(s.args).map(([k, v]) => `${k}: ${v}`).join(', ')
      : 'no args'
    lines.push(`- ${s.name}(${argStr}) — "${s.description}"`)
  }
  lines.push('')

  lines.push(`# Respond with JSON (ONE action per step):`)
  lines.push(`{`)
  lines.push(`  "thought": "reason about the goal, what you know now, and the best next move",`)
  lines.push(`  "action": {"skill": "skill_name", "args": {"target": "object_id"}, "description": "what this does"},`)
  lines.push(`  "done": false,`)
  lines.push(`  "doneReason": null,`)
  lines.push(`  "infeasible": false,`)
  lines.push(`  "infeasibleReason": null`)
  lines.push(`}`)
  lines.push(`If done or infeasible, set action to null.`)

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: lines.join('\n') },
  ]

  const raw = await callLLM(messages, 1024, true /* forceJSON */)
  return parseJSONResponse(raw)
}

/**
 * Call the LLM for skill synthesis (code generation) with reason-before-code CoT.
 *
 * The LLM must first reason about approach, physics constraints, and risks,
 * THEN write the code — reducing constraint violations and giving the verifier
 * (and trace log) a machine-readable risk declaration.
 *
 * @param {string} skillName - Name for the new skill
 * @param {string} description - What the skill should do
 * @param {Object} context - Robot state, available primitives, etc.
 * @returns {Promise<{code: string, reasoning: Object|null}|null>} Code + reasoning, or null on failure
 */
export async function synthesizeSkill(skillName, description, context = {}) {
  const manifest = getManifest()
  const constraints = manifest?.constraints || {}
  const joints = manifest?.joints || []
  
  const systemPrompt = `You are a robot skill programmer. You write async JavaScript function bodies
that control a robot through a provided context API. The code runs in a physics simulation.

# Robot Constraints (MUST respect these):
- Max speed: ${constraints.maxSpeed || '?'} m/s
- Max reach: ${constraints.maxReach || '?'} m
- Max jump height: ${constraints.maxJumpHeight || 0} m
- Joint groups: ${joints.map(j => j.group).filter(Boolean).filter((v, i, a) => a.indexOf(v) === i).join(', ')}

# Available Context API:
- context.getPos() → {x, y, z}
- context.getHeading() → radians (0 = +Z; forward = (sin h, cos h))
- context.setPos(x, y, z) — teleport (use sparingly, prefer smooth movement)
- context.navigateTo(x, y, z, opts?) → Promise<{arrived, finalDist}> — A* + LiDAR
  obstacle avoidance built in; opts = speed number or {speed, approach, face:{x,z}}.
  THROWS if the destination is unreachable.
- context.navigatePath([{x,z},...], opts?) → Promise<{reached, skipped, total}> —
  follow world waypoints in order with automatic detours around obstacles
- context.acquireObject(desc) → Promise<match|null> — find an object by description,
  auto-scanning and exploring if it has not been seen yet
- context.spinScan(desc?) → Promise — 360° camera sweep in place
- context.moveForward(speed) — set forward velocity
- context.rotate(angularSpeed) — set rotation speed
- context.stop() — stop all movement
- context.setJoint(jointName, angleDeg) — set joint target angle
- context.setJointGroup(groupName, angleDeg) — set all joints in group
- context.grab(objectId) → boolean
- context.release()
- context.wait(ms) → Promise
- context.setStatus(text) — update UI status
- context.getKnownObjects() → [{id, position, confidence, meta:{radius, colorName}}]
- context.findPerceivedObject(desc) → match|null (memory only, no search)
- context.checkFeasibility(action, params) → {feasible, reason}

# Rules:
- Use smooth interpolation for ALL movement (loops with await context.wait(16))
- For jumps: use sinusoidal arc (Math.sin(t * Math.PI) * height)
- Respect joint limits from the manifest
- Return to neutral state after skill completes
- Max 30 lines, efficient, no comments

# THINK BEFORE YOU CODE
First reason about the approach, the physics constraints involved, and what could
go wrong. THEN write the code. Respond with JSON:
{
  "reasoning": {
    "approach": "how the skill will work, step by step",
    "physicsChecks": "which constraints (speed/reach/joints) apply and how the code respects them",
    "risks": ["what could fail or violate constraints"]
  },
  "code": "the function body as a single string (no \`\`\`, no function declaration)"
}`

  const userMessage = `Write skill "${skillName}": ${description}

Robot position: [${context.robotPos?.x?.toFixed(1) || 0}, ${context.robotPos?.y?.toFixed(1) || 0}, ${context.robotPos?.z?.toFixed(1) || 0}]
Available joints: ${joints.map(j => `${j.name}(${j.limits?.lower}° to ${j.limits?.upper}°)`).join(', ')}
Existing skills (don't duplicate): ${context.existingSkills?.join(', ') || 'none'}`

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userMessage },
  ]

  try {
    const raw = await callLLM(messages, 1200, true /* forceJSON */)

    // Preferred path: structured {reasoning, code} response
    try {
      const parsed = parseJSONResponse(raw)
      if (parsed?.code && typeof parsed.code === 'string') {
        return { code: parsed.code.trim(), reasoning: parsed.reasoning || null }
      }
    } catch { /* fall through to legacy extraction */ }

    // Fallback: model ignored the JSON format and returned bare code
    const code = raw.replace(/```(?:javascript|js|json)?\n?/gi, '').replace(/```/g, '').trim()
    return code ? { code, reasoning: null } : null
  } catch (e) {
    console.error(`[LLM] Skill synthesis failed for "${skillName}":`, e)
    return null
  }
}

/**
 * Ask the LLM a simple question (for debugging, explanations, etc.)
 * @param {string} question
 * @returns {Promise<string>}
 */
export async function askLLM(question) {
  const messages = [
    { role: 'system', content: buildSystemPrompt() },
    { role: 'user', content: question },
  ]
  return await callLLM(messages, 512)
}

/**
 * Expose the raw callLLM function for use by reflection.js (passed as llmFn).
 * This avoids circular imports — reflection.js never imports llm.js directly.
 *
 * @param {Array<{role:string, content:string}>} messages
 * @param {number} [maxTokens=400]
 * @returns {Promise<string>}
 */
export async function reflectWithLLM(messages, maxTokens = 400) {
  return callLLM(messages, maxTokens)
}

// ─── Internal: Prompt Building ─────────────────────────────────────────────────

function buildPlanningPrompt(instruction, { knownObjects, history, availableSkills, robotState, failureContext, cotStyle = 'structured' }) {
  const lines = []
  
  lines.push(`# Current State`)
  lines.push(`Robot position: [${robotState.position?.x?.toFixed(2) || 0}, ${robotState.position?.y?.toFixed(2) || 0}, ${robotState.position?.z?.toFixed(2) || 0}]`)
  const q = robotState.orientation
  if (q) {
    const headingDeg = Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z)) * (180 / Math.PI)
    lines.push(`Robot heading: ${headingDeg.toFixed(0)}° (0° = +Z axis; forward = (sin h, cos h))`)
  }
  lines.push(`Robot status: ${robotState.status || 'idle'}`)
  lines.push(`Holding: ${robotState.heldObjects?.length ? robotState.heldObjects.join(', ') : 'nothing'}`)
  lines.push(`Arena: 32m x 32m, boundary walls at x=±16 and z=±16. Keep goals within ±15.`)
  lines.push('')

  // Semantic scene graph — richer than flat object list
  lines.push(`# Scene (semantic, from sensors):`)
  lines.push(sceneGraph.getLLMContext(robotState.position || { x: 0, y: 0, z: 0 }))
  lines.push('')

  // Episodic memory — relevant past experiences
  const memCtx = episodicMemory.getRelevantContext(instruction)
  if (memCtx) {
    lines.push(memCtx)
    lines.push('')
  }

  // World model — navigation coverage + affordances + beliefs
  const worldCtx = worldModel.getWorldModelContext(robotState.position || { x: 0, y: 0, z: 0 })
  if (worldCtx) {
    lines.push(worldCtx)
    lines.push('')
  }
  
  // Available skills — with descriptions and arg formats so the LLM knows exactly what to pass
  lines.push(`# Available Skills (use EXACT skill names and arg keys):`)
  if (availableSkills.length > 0 && typeof availableSkills[0] === 'object') {
    for (const s of availableSkills) {
      const argStr = s.args && Object.keys(s.args).length > 0
        ? Object.entries(s.args).map(([k, v]) => `${k}: ${v}`).join(', ')
        : 'no args'
      lines.push(`- ${s.name}(${argStr}) — "${s.description}"`)
    }
  } else {
    // Fallback: plain name list (shouldn't happen after planner.js fix)
    lines.push(availableSkills.join(', ') || 'none loaded')
  }
  lines.push('')

  // Perceived Objects — ONLY from sensor memory, never from physics ground truth.
  // If empty: robot has not scanned. LLM must plan to scan first.
  lines.push(`# Perceived Objects (detected by camera/sensors — positions are APPROXIMATE):`)
  const perceivedObjects = getKnownObjects(0.1)
  if (perceivedObjects.length > 0) {
    for (const o of perceivedObjects) {
      const ageMs  = Date.now() - (o.lastSeen || Date.now())
      const stale  = ageMs > 5000 ? ` [STALE ${(ageMs / 1000).toFixed(0)}s ago — may have moved]` : ''
      lines.push(`- ${o.id}  conf=${o.confidence.toFixed(2)}  pos=(${o.position.x.toFixed(1)}, ${o.position.z.toFixed(1)})${stale}`)
    }
  } else {
    lines.push(`  (none — robot has not scanned yet. Use scan_for or scan_room to discover objects.)`)
  }
  lines.push('')
  
  // Recent history
  if (history.length > 0) {
    lines.push(`# Recent Actions:`)
    for (const h of history.slice(-5)) {
      lines.push(`- ${h.action} → ${h.result}${h.details ? ` (${h.details})` : ''}`)
    }
    lines.push('')
  }

  // Failure context (for replanning calls)
  if (failureContext) {
    lines.push(`# Previous Attempt Failed:`)
    lines.push(failureContext)
    lines.push(`Devise an ALTERNATIVE plan that avoids repeating the same failure.`)
    lines.push('')
  }

  // The instruction
  lines.push(`# Instruction: "${instruction}"`)
  lines.push('')

  // Response format — includes retries and optional per-step fields
  lines.push(`# Respond with JSON:`)
  lines.push(`{`)
  if (cotStyle === 'structured') {
    // Chain-of-thought: force explicit reasoning BEFORE the plan fields.
    // Field order matters — autoregressive generation means the plan tokens
    // are conditioned on the reasoning tokens.
    lines.push(`  "cot": {`)
    lines.push(`    "situation": "what I know right now from perception, memory and history",`)
    lines.push(`    "unknowns": "what I do NOT know yet and whether the plan must discover it",`)
    lines.push(`    "feasibility": "check the task against robot constraints (speed, reach, capabilities)",`)
    lines.push(`    "strategy": "chosen approach and WHY it beats the alternatives",`)
    lines.push(`    "risks": "most likely failure point of this plan and its mitigation"`)
    lines.push(`  },`)
    if (failureContext) {
      lines.push(`  "failureAnalysis": {`)
      lines.push(`    "whatHappened": "factual description of the failure",`)
      lines.push(`    "rootCause": "the underlying reason, not the symptom",`)
      lines.push(`    "worldChanged": "what the failure reveals about the world (blocked path, missing object...)",`)
      lines.push(`    "newStrategy": "how the recovery plan differs and why it will work"`)
      lines.push(`  },`)
    }
  } else {
    lines.push(`  "reasoning": "step-by-step thinking about feasibility and approach",`)
  }
  lines.push(`  "plan": [`)
  lines.push(`    {`)
  lines.push(`      "skill": "skill_name",`)
  lines.push(`      "args": {"target": "object_id"},`)
  lines.push(`      "description": "what this step does",`)
  lines.push(`      "retries": 0,`)
  lines.push(`      "optional": false`)
  lines.push(`    }`)
  lines.push(`  ],`)
  lines.push(`  "needsSynthesis": false,`)
  lines.push(`  "newSkillSpec": null,`)
  lines.push(`  "infeasible": false,`)
  lines.push(`  "infeasibleReason": null,`)
  lines.push(`  "needsStepByStep": false`)
  lines.push(`}`)
  lines.push(``)
  lines.push(`Notes on plan fields:`)
  lines.push(`  retries: number of automatic retries on failure (0 = no retry, 1-3 for flaky steps)`)
  lines.push(`  optional: true if this step can fail without aborting the whole plan`)
  lines.push(`  needsStepByStep: set true (with empty plan) ONLY if the task cannot be planned upfront`)
  lines.push(`    because later steps depend on what earlier steps discover — you will then be run in`)
  lines.push(`    an interactive think-act-observe loop instead.`)

  return lines.join('\n')
}

// ─── Internal: LLM API Call ────────────────────────────────────────────────────

/**
 * Make the actual API call with retry logic.
 * @param {Array<{role: string, content: string}>} messages
 * @param {number} maxTokens
 * @param {boolean} [forceJSON=false] - Request JSON-only output (planning calls)
 * @returns {Promise<string>}
 */
async function callLLM(messages, maxTokens = 512, forceJSON = false) {
  if (!config.apiKey) {
    throw new Error('[LLM] No API key configured. Set VITE_LLM_API_KEY in .env')
  }
  
  let lastError = null
  
  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      const body = {
        model: config.model,
        messages,
        temperature: config.temperature,
        max_completion_tokens: maxTokens,
      }

      // Force JSON-only output when calling for structured plan responses.
      // Google Gemini OpenAI-compat endpoint supports this parameter.
      if (forceJSON) {
        body.response_format = { type: 'json_object' }
      }

      const res = await fetch(config.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify(body),
      })
      
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.error?.message || `HTTP ${res.status}`)
      }
      
      const data = await res.json()
      
      // Track token usage
      if (data.usage) {
        _totalTokensUsed.prompt += data.usage.prompt_tokens || 0
        _totalTokensUsed.completion += data.usage.completion_tokens || 0
      }
      
      const text = data.choices?.[0]?.message?.content || ''
      
      if (!text) {
        throw new Error('Empty response from LLM')
      }
      
      return text
      
    } catch (e) {
      lastError = e
      console.warn(`[LLM] Attempt ${attempt + 1}/${config.maxRetries + 1} failed:`, e.message)
      
      if (attempt < config.maxRetries) {
        await new Promise(r => setTimeout(r, config.retryDelay * (attempt + 1)))
      }
    }
  }
  
  throw lastError
}

/**
 * Streaming LLM call. Parses Server-Sent Events and calls onChunk for each token.
 * Falls back to non-streaming if the endpoint doesn't support SSE.
 *
 * @param {Array<{role: string, content: string}>} messages
 * @param {number} maxTokens
 * @param {function(string):void} [onChunk] - Called with each text delta
 * @returns {Promise<string>} Full accumulated response text
 */
async function callLLMStream(messages, maxTokens = 512, onChunk) {
  if (!config.apiKey) {
    throw new Error('[LLM] No API key configured. Set VITE_LLM_API_KEY in .env')
  }

  let lastError = null

  for (let attempt = 0; attempt <= config.maxRetries; attempt++) {
    try {
      const res = await fetch(config.apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${config.apiKey}`,
        },
        body: JSON.stringify({
          model: config.model,
          messages,
          temperature: config.temperature,
          max_completion_tokens: maxTokens,
          stream: true,
        }),
      })

      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}))
        throw new Error(errBody.error?.message || `HTTP ${res.status}`)
      }

      // Read SSE stream
      const reader  = res.body.getReader()
      const decoder = new TextDecoder()
      let accumulated = ''
      let buffer      = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''   // Keep the incomplete last line

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed || trimmed === 'data: [DONE]') continue
          if (!trimmed.startsWith('data: ')) continue

          try {
            const json  = JSON.parse(trimmed.slice(6))
            const delta = json.choices?.[0]?.delta?.content ?? ''
            if (delta) {
              accumulated += delta
              onChunk?.(delta)
            }
            // Track usage if included in the final SSE chunk
            if (json.usage) {
              _totalTokensUsed.prompt     += json.usage.prompt_tokens     || 0
              _totalTokensUsed.completion += json.usage.completion_tokens || 0
            }
          } catch {
            // Ignore malformed SSE lines
          }
        }
      }

      if (!accumulated) throw new Error('Empty streaming response from LLM')
      return accumulated

    } catch (e) {
      lastError = e
      console.warn(`[LLM] Stream attempt ${attempt + 1}/${config.maxRetries + 1} failed:`, e.message)

      if (attempt < config.maxRetries) {
        // On retry, fall back to non-streaming for reliability
        console.info('[LLM] Retrying with non-streaming fallback')
        return callLLM(messages, maxTokens)
      }
    }
  }

  throw lastError
}

// ─── Internal: Response Parsing ────────────────────────────────────────────────

function parseJSONResponse(raw) {
  // Strip markdown code fencing if present
  let clean = raw.replace(/```json\n?/gi, '').replace(/```\n?/g, '').trim()
  
  // Find JSON object boundaries
  const start = clean.indexOf('{')
  const end = clean.lastIndexOf('}')
  
  if (start === -1 || end === -1) {
    throw new Error(`[LLM] No JSON object found in response: "${raw.slice(0, 100)}..."`)
  }
  
  const jsonStr = clean.slice(start, end + 1)
  
  try {
    return JSON.parse(jsonStr)
  } catch (e) {
    throw new Error(`[LLM] Invalid JSON in response: ${e.message}\nRaw: "${jsonStr.slice(0, 200)}..."`)
  }
}
