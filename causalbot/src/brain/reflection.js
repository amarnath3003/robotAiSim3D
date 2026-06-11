/**
 * brain/reflection.js — Post-Task LLM Reflection
 *
 * After each task completes (success or failure), performs asynchronous
 * reflection that extracts generalizable lessons and spatial facts, then
 * writes them to:
 *   - episodicMemory  (task episode, lessons, spatial facts)
 *   - worldModel      (affordances, visited cell, named beliefs)
 *
 * Usage (fire-and-forget from planner.js — do NOT await):
 *   reflect({ instruction, plan, success, failReason, robotPos }, reflectWithLLM)
 *     .catch(() => {})
 *
 * The `llmFn` parameter is passed in directly to avoid circular imports.
 * Signature: async (messages: [{role, content}], maxTokens: number) => string
 */

import { episodicMemory } from './episodic_memory.js'
import { worldModel }     from './world_model.js'

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Reflect on a completed task.
 *
 * @param {Object}   params
 * @param {string}   params.instruction    - The user's original instruction
 * @param {Array}    params.plan           - Executed plan steps [{skill, args, description}]
 * @param {boolean}  params.success        - Overall task outcome
 * @param {string}   [params.failReason]   - Failure reason if success=false
 * @param {Object}   [params.robotPos]     - Robot world position {x,y,z} at task end
 * @param {Array}    [params.knownObjects] - Known objects at task time [{id, type?, position}]
 * @param {Function} llmFn                - async (messages, maxTokens) => string
 */
export async function reflect(
  { instruction, plan = [], success, failReason, robotPos, knownObjects = [] },
  llmFn
) {
  // ── 1. Always record the task episode (no LLM needed) ──────────────────────
  const skillsUsed = plan.map(s => s.skill).filter(Boolean)
  const summary = success
    ? `Completed via: ${skillsUsed.join(' → ') || 'direct execution'}`
    : `Failed: ${failReason || 'unknown error'}`

  episodicMemory.recordTask({
    instruction,
    success,
    summary,
    skillsUsed,
    robotPos,
  })

  // ── 2. Mark robot's current cell as visited ─────────────────────────────────
  if (robotPos) {
    worldModel.markVisited(robotPos.x, robotPos.z)
  }

  // ── 3. Record skill→object affordances from executed plan ───────────────────
  for (const step of plan) {
    if (!step.skill) continue
    const objectRef = step.args?.target || step.args?.object || step.args?.id
    if (objectRef) {
      const known   = knownObjects.find(o => o.id === objectRef)
      const objType = known?.type || _inferTypeFromId(objectRef)
      if (objType) {
        worldModel.recordAffordance(objType, step.skill, success)
      }
    }
  }

  // ── 4. LLM reflection (lessons + spatial facts + beliefs) ──────────────────
  // Skip if no LLM function or trivial single-step plan
  if (!llmFn || plan.length === 0) return

  try {
    const prompt = _buildReflectionPrompt({
      instruction, plan, success, failReason, robotPos, knownObjects,
    })

    const raw = await llmFn(
      [
        {
          role: 'system',
          content:
            'You are a reflective AI that analyzes robot task outcomes. ' +
            'Be concise. Output only valid JSON.',
        },
        { role: 'user', content: prompt },
      ],
      400    // maxTokens — keep reflection lightweight
    )

    const parsed = _parseReflectionResponse(raw)
    if (!parsed) return

    // Write lessons to episodic memory
    for (const lesson of (parsed.lessons || [])) {
      if (typeof lesson === 'string' && lesson.trim().length > 5) {
        const kw = _extractKeywords(lesson + ' ' + instruction)
        episodicMemory.recordLesson(lesson.trim(), kw)
      }
    }

    // Write spatial facts to episodic memory
    for (const fact of (parsed.spatialFacts || [])) {
      if (typeof fact === 'string' && fact.trim().length > 5) {
        const kw = _extractKeywords(fact)
        episodicMemory.recordSpatialFact(fact.trim(), kw)
      }
    }

    // Write named beliefs to world model
    for (const belief of (parsed.beliefs || [])) {
      if (belief?.key && belief?.value != null) {
        worldModel.setBelief(String(belief.key), belief.value)
      }
    }

  } catch (e) {
    // Reflection is non-critical — swallow errors silently
    console.warn('[Reflection] LLM reflection failed:', e.message)
  }
}

// ─── Prompt Building ──────────────────────────────────────────────────────────

function _buildReflectionPrompt({ instruction, plan, success, failReason, robotPos, knownObjects }) {
  const outcome = success
    ? 'SUCCEEDED'
    : `FAILED (${failReason || 'unknown error'})`

  const steps = plan.map((s, i) => {
    const argStr = s.args && Object.keys(s.args).length
      ? JSON.stringify(s.args)
      : '{}'
    return `  ${i + 1}. ${s.skill}(${argStr})${s.description ? ` — ${s.description}` : ''}`
  }).join('\n')

  const posStr = robotPos
    ? `(${robotPos.x?.toFixed(1)}, ${robotPos.z?.toFixed(1)})`
    : 'unknown'

  return `Robot task outcome:

Instruction: "${instruction}"
Outcome: ${outcome}
Steps:
${steps}

Robot position at end: ${posStr}

Respond with JSON:
{
  "lessons": ["generalizable rule 1", "rule 2"],
  "spatialFacts": ["location/layout observation"],
  "beliefs": [{"key": "belief_name", "value": "value"}]
}

Guidelines:
- lessons: rules that would help with similar future tasks (max 2; leave [] if nothing new)
- spatialFacts: discovered locations, layouts, or object positions (max 2; leave [] if none)
- beliefs: named world-state facts to remember (max 2; leave [] if none)

Output ONLY the JSON object. No explanation.`
}

// ─── Response Parsing ─────────────────────────────────────────────────────────

function _parseReflectionResponse(raw) {
  try {
    const clean = raw
      .replace(/```json\n?/gi, '')
      .replace(/```\n?/g, '')
      .trim()
    const start = clean.indexOf('{')
    const end   = clean.lastIndexOf('}')
    if (start === -1 || end === -1) return null
    return JSON.parse(clean.slice(start, end + 1))
  } catch {
    return null
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Infer an object type from its ID string by matching common keywords.
 * Falls back to the first word of the ID if no keyword matches.
 */
function _inferTypeFromId(id) {
  if (!id) return null
  const lower = id.toLowerCase()
  const typeMap = {
    box: 'box', crate: 'box', cube: 'box',
    ball: 'ball', sphere: 'ball',
    door: 'door', gate: 'door',
    button: 'button', switch: 'button', lever: 'button',
    station: 'station', dock: 'station', charger: 'station',
    wall: 'wall', obstacle: 'obstacle', barrier: 'obstacle',
    table: 'table', desk: 'table',
    chair: 'chair',
  }
  for (const [keyword, type] of Object.entries(typeMap)) {
    if (lower.includes(keyword)) return type
  }
  // Fall back to the first word of the ID
  return lower.split(/[_\-\s]/)[0] || null
}

/**
 * Extract keyword tokens from free text for episodic memory tagging.
 * @param {string} text
 * @returns {string[]}
 */
function _extractKeywords(text) {
  return [
    ...new Set(
      text.toLowerCase()
        .replace(/[^a-z0-9\s]/g, ' ')
        .split(/\s+/)
        .filter(w => w.length > 3)
        .slice(0, 10)
    ),
  ]
}
