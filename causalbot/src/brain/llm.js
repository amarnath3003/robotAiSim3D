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
import { listInteractables } from '../env/objects.js'

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
 * Build the system prompt that gives the LLM full context about the robot.
 * This is the bridge between the manifest and the AI brain.
 */
function buildSystemPrompt() {
  const manifest = getManifest()
  if (!manifest) return 'You are a robot controller. No robot manifest loaded.'
  
  const robotDesc = generateLLMDescription()
  const feasibilityRules = generateFeasibilityPrompt()
  
  return `You are the brain of an autonomous robot operating in a physics simulation.
Your decisions control a real physical body with real constraints.

# Robot Profile
${robotDesc}

# ${feasibilityRules}

# Perception
- "All Objects in Scene" lists every object that EXISTS in the world (names are always accurate).
- Objects in "Scene" have been sensor-observed — they have known positions and confidence scores.
- pick_up, push, and navigate_to all use direct physics lookup internally — NO scan required.
- scan_room is only needed when you need exact coordinates for place(x, z) or distance math.

# Task Decomposition Guide
"pick up X"                    → [pick_up(target: "X")]
"place near Y"                 → [place_near(target: "Y")]
"pick up X and place near Y"   → [pick_up(target: "X"), place_near(target: "Y")]
"navigate to X"                → [navigate_to(target: "X")]
"scan the room"                → [scan_room]
"find X and go to it"          → [find_goal(target: "X")]
"push X"                       → [push(target: "X")]
"go forward 2 seconds"         → [move_forward(duration: 2000)]
"turn left 90 degrees"         → [turn_left(degrees: 90)]
"patrol"                       → [patrol]
"go home / return to start"    → [return_home]
"survey the area"              → [survey_grid]
For multi-step tasks like "pick up X and put it near Y": do NOT add scan_room unless truly needed.

# Execution Rules
- Break complex tasks into the minimum number of atomic skill steps.
- If something is physically impossible given the robot constraints, set infeasible: true and explain why.
- After failure, analyze what went wrong and produce a different recovery plan.

# Critical Object ID Rules
- Use EXACT names from "All Objects in Scene" (e.g. ball_red, ball_blue, box_A, box_B).
- Do NOT say "the ball" — use the exact ID "ball_red".
- Object names are case-sensitive: box_A not box_a.

# Response Format
Always respond with ONLY valid JSON — no markdown code blocks, no text outside the JSON object.`
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
 * Call the LLM for skill synthesis (code generation).
 * 
 * @param {string} skillName - Name for the new skill
 * @param {string} description - What the skill should do
 * @param {Object} context - Robot state, available primitives, etc.
 * @returns {Promise<string|null>} Generated JavaScript code or null on failure
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
- context.setPos(x, y, z) — teleport (use sparingly, prefer smooth movement)
- context.navigateTo(x, y, z, speed?) → Promise (A* pathfinding, resolves on arrival)
- context.moveForward(speed) — set forward velocity
- context.rotate(angularSpeed) — set rotation speed
- context.stop() — stop all movement
- context.setJoint(jointName, angleDeg) — set joint target angle
- context.setJointGroup(groupName, angleDeg) — set all joints in group
- context.grab(objectId) → boolean
- context.release()
- context.wait(ms) → Promise
- context.setStatus(text) — update UI status
- context.getKnownObjects() → [{id, position, confidence}]
- context.checkFeasibility(action, params) → {feasible, reason}

# Rules:
- Use smooth interpolation for ALL movement (loops with await context.wait(16))
- For jumps: use sinusoidal arc (Math.sin(t * Math.PI) * height)
- Respect joint limits from the manifest
- Return to neutral state after skill completes
- Max 30 lines, efficient, no comments
- Output ONLY the function body (no \`\`\`, no function declaration)`

  const userMessage = `Write skill "${skillName}": ${description}

Robot position: [${context.robotPos?.x?.toFixed(1) || 0}, ${context.robotPos?.y?.toFixed(1) || 0}, ${context.robotPos?.z?.toFixed(1) || 0}]
Available joints: ${joints.map(j => `${j.name}(${j.limits?.lower}° to ${j.limits?.upper}°)`).join(', ')}
Existing skills (don't duplicate): ${context.existingSkills?.join(', ') || 'none'}`

  const messages = [
    { role: 'system', content: systemPrompt },
    { role: 'user', content: userMessage },
  ]
  
  try {
    const raw = await callLLM(messages, 600)
    // Strip any markdown code fencing
    const code = raw.replace(/```(?:javascript|js)?\n?/gi, '').replace(/```/g, '').trim()
    return code
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

function buildPlanningPrompt(instruction, { knownObjects, history, availableSkills, robotState, failureContext }) {
  const lines = []
  
  lines.push(`# Current State`)
  lines.push(`Robot position: [${robotState.position?.x?.toFixed(2) || 0}, ${robotState.position?.y?.toFixed(2) || 0}, ${robotState.position?.z?.toFixed(2) || 0}]`)
  lines.push(`Robot status: ${robotState.status || 'idle'}`)
  lines.push(`Holding: ${robotState.heldObjects?.length ? robotState.heldObjects.join(', ') : 'nothing'}`)
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

  // Always show all interactable objects so LLM can plan without requiring a scan first
  try {
    const allObjects = listInteractables()
    if (allObjects && allObjects.length > 0) {
      lines.push(`# All Objects in Scene (use EXACT names as skill arg values):`)
      lines.push(allObjects.join(', '))
      lines.push('Note: pick_up, push, navigate_to, place_near all do direct physics lookup — no scan needed.')
      lines.push('')
    }
  } catch (_) { /* env not yet initialized */ }
  
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
  lines.push(`  "reasoning": "step-by-step thinking about feasibility and approach",`)
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
  lines.push(`  "infeasibleReason": null`)
  lines.push(`}`)
  lines.push(``)
  lines.push(`Notes on plan fields:`)
  lines.push(`  retries: number of automatic retries on failure (0 = no retry, 1-3 for flaky steps)`)
  lines.push(`  optional: true if this step can fail without aborting the whole plan`)
  
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
        max_tokens: maxTokens,
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
          max_tokens: maxTokens,
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
