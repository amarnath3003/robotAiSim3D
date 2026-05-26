import { state, getObject, getRobotPos } from './state.js'
import { getMemorySummary } from './memory.js'
import { getAllSkillNames, hasSkill } from './skillRegistry.js'
import { getObjectsForPlanner } from './perception/perceptionMode.js'
import { setStatus, setAgentStatus } from './ui.js'

const API_KEY = import.meta.env.VITE_NVIDIA_API_KEY
const API_URL = '/api/nvidia/chat/completions'
const MODEL   = import.meta.env.VITE_NVIDIA_MODEL || 'google/gemma-4-31b-it'

// ─── Training intent keywords ─────────────────────────────────────────────────
// Checked BEFORE hitting the planner so we never waste a token on non-training tasks.
// IMPORTANT: 'run', 'execute', 'do' are NOT training commands — they INVOKE learned skills.

const TRAIN_KEYWORDS = [
  'train',
  'teach the robot',
  'teach it to',
  'learn to',
  'practice',
  'reinforce',
  'use rl',
  'rl train',
  'train the robot',
  'train robot',
]

/**
 * Returns true if the instruction is asking for RL training (not skill execution).
 * "run pick_up_ball_rl" → false (invoke skill)
 * "train the robot to pick up the ball" → true (start training)
 */
export function isTrainingInstruction(instruction) {
  const lower = instruction.toLowerCase().trim()
  // Explicitly exclude "run X" / "execute X" / "do X" — those invoke existing skills
  if (/^(run|execute|do|perform|start|activate)\s/.test(lower)) return false
  return TRAIN_KEYWORDS.some(k => lower.includes(k))
}

// ─── Training planner ─────────────────────────────────────────────────────────

/**
 * Ask the LLM to parse a training instruction into structured task parameters.
 * Returns a task object ready for rl.trainPolicy().
 *
 * Example input:  "train the robot to pick up the box"
 * Example output: { name: "pick up the box", targetObjectId: "object_box",
 *                   goalType: "pick_up", skillName: "pick_up_box_rl" }
 */
export async function planTraining(instruction) {
  setStatus('Parsing training task...')
  setAgentStatus('Understanding training goal', 'thinking')
  state.robot.eyeColor = 0xffaa00

  const objects = Object.values(state.world.objects).map(o => `${o.name}(id:${o.id})`).join(', ')

  const prompt = `You are a robotics trainer. Parse this training instruction into a JSON task object.

INSTRUCTION: "${instruction}"

AVAILABLE OBJECTS: ${objects}

GOAL TYPES:
- "pick_up"       — robot must navigate to object and grab it
- "navigate_to"   — robot must reach the object's location
- "push_to_corner" — robot must push object to the far corner

Respond ONLY with valid JSON (no markdown):
{
  "name": "<short human label for the task>",
  "targetObjectId": "<exact object id from the list above, or null>",
  "goalType": "<pick_up | navigate_to | push_to_corner>",
  "skillName": "<snake_case name for the learned skill, end with _rl>",
  "impossible": false,
  "reason": null
}`

  try {
    const raw  = await callLLM(prompt, 256)
    const json = extractJSON(raw)
    const task = JSON.parse(json)

    if (task.impossible) {
      setStatus(task.reason || 'Cannot train that.')
      setAgentStatus(task.reason || 'Impossible training goal', 'error')
      state.robot.eyeColor = 0x4488ff
      return null
    }

    // Validate object exists
    if (task.targetObjectId) {
      const obj = getObject(task.targetObjectId)
      if (!obj) {
        // Try fuzzy match
        const fuzzy = Object.values(state.world.objects).find(o =>
          task.targetObjectId.toLowerCase().includes(o.name.toLowerCase()) ||
          o.name.toLowerCase().includes(task.targetObjectId.toLowerCase())
        )
        if (fuzzy) task.targetObjectId = fuzzy.id
      }
      // Store object origin for episode resets
      const obj2 = getObject(task.targetObjectId)
      if (obj2) {
        task._objOrigin = [...obj2.position]
      }
    }

    state.robot.eyeColor = 0x4488ff
    return task

  } catch (e) {
    console.error('[LLM] planTraining failed:', e)
    setStatus('Could not parse training task.')
    setAgentStatus('Training parse error', 'error')
    state.robot.eyeColor = 0x4488ff
    return null
  }
}

// ─── Minimal world snapshot — only what the LLM needs ────────────────────────

function getWorldSnapshot() {
  const rp = getRobotPos()

  return {
    r: {
      p: [+rp.x.toFixed(1), +rp.y.toFixed(1), +rp.z.toFixed(1)],
      h: state.robot.heldObject || null,
    },
    o: getObjectsForPlanner(),
    b: state.world.roomBounds,
    sk: getAllSkillNames(),
    mem: getMemorySummary(),
    perceptionMode: state.perceptionMode,
  }
}

// ─── Classify instruction cheaply before hitting planner ─────────────────────

const DIRECT_SKILL_MAP = {
  jump:         'jump',
  backflip:     'backflip',
  frontflip:    'frontflip',
  flip:         'backflip',
  spin:         'spin',
  rotate:       'spin',
  dance:        'dance',
  celebrate:    'celebrate',
  cheer:        'celebrate',
  scared:       'scared',
  fly:          'fly',
  patrol:       'patrol',
  scan:         'scan_room',
  'look around':'scan_room',
  'scan room':  'scan_room',
}

function tryDirectMatch(instruction) {
  const lower = instruction.toLowerCase().trim()
  for (const [keyword, skill] of Object.entries(DIRECT_SKILL_MAP)) {
    if (lower.includes(keyword) && hasSkill(skill)) {
      return skill
    }
  }
  return null
}

// ─── Planner ─────────────────────────────────────────────────────────────────

export async function planInstruction(instruction) {
  setStatus('Thinking...')
  setAgentStatus('Thinking about ' + instruction, 'thinking')
  state.robot.eyeColor = 0xffaa00
  state.robot.status   = 'thinking'

  // Fast path — skip LLM entirely for known single skills
  const direct = tryDirectMatch(instruction)
  if (direct) {
    state.robot.status   = 'idle'
    state.robot.eyeColor = 0x4488ff
    return {
      reasoning:          `Direct match: ${direct}`,
      goal:               instruction,
      thoughts:           [`Plan: ${direct}`],
      actions:            [{ skill: direct, args: {}, description: instruction }],
      needsNewSkill:      false,
      newSkillName:       null,
      newSkillDescription:null,
      impossible:         false,
      impossibleReason:   null,
    }
  }

  const snap     = getWorldSnapshot()
  const isVision = state.perceptionMode === 'vision'

  const objectsLabel = isVision
    ? `PERCEIVED OBJECTS (vision mode — only what robot has seen):`
    : `WORLD OBJECTS (omniscient — ground truth, always accurate):`

  const visionRules = isVision ? `
PERCEPTION MODE: VISION ONLY.
- The object list above is what the robot has PHYSICALLY SEEN so far.
- If a needed object is NOT in the list, you MUST call scanforobject to find it first.
- After scanforobject, follow with go_to in the same actions array.
- NEVER assume you know where something is if it is not in the list.
` : `
PERCEPTION MODE: OMNISCIENT — all positions above are ground truth, always use them directly.
- Do NOT scan. Just use go_to, pick_up, etc. directly with the object id.
`

  const prompt = `You are a robot brain. Execute the instruction using the available skills.

ROBOT STATE:
- Position: [${snap.r.p.join(', ')}]
- Holding: ${state.robot.heldObject || 'nothing'}
- Skills available: ${snap.sk.join(', ')}
- Recent memory: ${snap.mem || 'none'}

${objectsLabel}
${JSON.stringify(snap.o, null, 0)}

Room bounds: x[${snap.b.minX} to ${snap.b.maxX}] z[${snap.b.minZ} to ${snap.b.maxZ}]

INSTRUCTION: "${instruction}"
${visionRules}
SKILL USAGE (how to call each skill):
- go_to: args { "target": "<object id or name>" }
- pick_up: args { "target": "<object id or name>" }
- release: args {}
- scanforobject: args { "target": "<object id or name>" }
- scan_room: args {}
- patrol: args {}
- spin: args {}
- dance: args {}

Primitives for NEW skills only: navigateTo(x,y,z) setPos(x,y,z) getPos() setArm(rad) grab(id) release() wait(ms) setStatus(text)

RULES:
- Output ALL actions needed to complete the goal in one response.
- Example: "go to the ball" → [{skill:"go_to", args:{target:"object_ball"}}]
- Example (vision mode, ball not seen): [{skill:"scanforobject", args:{target:"ball"}}, {skill:"go_to", args:{target:"object_ball"}}]
- In OMNISCIENT mode, objects are always known — use go_to directly, never scan.
- One hand. If holding something, skip pick_up.
- Fragility>0.6 means handle carefully. Mass>2 means heavy.
- If truly impossible, set impossible=true.

Respond ONLY with valid JSON (no markdown, no code blocks):
{"reasoning":"<why>","goal":"<goal>","thoughts":["..."],"actions":[{"skill":"<name>","args":{"target":"<id_or_null>"},"description":"<brief>"}],"needsNewSkill":false,"newSkillName":null,"newSkillDescription":null,"impossible":false,"impossibleReason":null}`

  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const raw  = await callLLM(prompt, 512)
      const json = extractJSON(raw)
      const plan = JSON.parse(json)

      if (plan.impossible) {
        setStatus(plan.impossibleReason || 'Cannot do that.')
        setAgentStatus(plan.impossibleReason || 'Impossible goal', 'error')
        state.robot.status   = 'idle'
        state.robot.eyeColor = 0x4488ff
        return null
      }

      state.robot.status = 'idle'
      return plan

    } catch (e) {
      console.warn(`Plan attempt ${attempt + 1} failed:`, e.message)
      if (attempt === 1) {
        state.robot.status   = 'failed'
        state.robot.eyeColor = 0xff3333
        setStatus('Could not understand.')
        setAgentStatus('LLM processing error', 'error')
        return null
      }
      await sleep(200)
    }
  }
}

// ─── Skill inventor ───────────────────────────────────────────────────────────

export async function inventSkill(skillName, instruction, existingSkills) {
  setStatus(`Inventing: ${skillName}...`)
  setAgentStatus(`Inventing skill: ${skillName}`, 'thinking')

  const rp = getRobotPos()
  const objects = Object.values(state.world.objects)
    .map(o => `${o.name}(${o.id})@[${o.position.map(v => v.toFixed(1))}]`)
    .join(' ')

  const prompt = `Write JS async function body for robot skill "${skillName}": "${instruction}"

API: context.getPos() context.setPos(x,y,z) context.navigateTo(x,y,z,speed?) context.setArm(-1.5to1.5) context.grab(id) context.release() context.setEye(0xRRGGBB) context.wait(ms) context.getObject(id) context.target context.setStatus(text) context.getWorldBounds()

Robot@[${rp.x.toFixed(1)},${rp.y.toFixed(1)},${rp.z.toFixed(1)}] holding:${state.robot.heldObject || 'none'}
Objects: ${objects}
Existing skills (don't duplicate): ${existingSkills.join(',')}

Animation rules:
- Use loops+interpolation for ALL movement (never raw setPos jumps)
- Arc: arcY=Math.sin(t*Math.PI)*HEIGHT for jumps/flips
- Always: const p=context.getPos() at start
- Always end: arm=0, eye=0x4488ff, near start pos
- Max 20 lines, no comments, pure JS body only`

  try {
    const raw  = await callLLM(prompt, 400)
    const code = raw.replace(/```javascript|```js|```/gi, '').trim()
    return code
  } catch (e) {
    console.error('Skill invention failed:', e)
    return null
  }
}

// ─── LLM caller ──────────────────────────────────────────────────────────────

async function callLLM(prompt, maxTokens = 512) {
  if (!API_KEY) {
    throw new Error('VITE_NVIDIA_API_KEY is not set in .env')
  }

  const res = await fetch(API_URL, {
    method:  'POST',
    headers: {
      'Content-Type':  'application/json',
      'Authorization': `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model:       MODEL,
      messages:    [{ role: 'user', content: prompt }],
      temperature: 0.1,
      max_tokens:  maxTokens,
    }),
  })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.error?.message || `HTTP ${res.status}`)
  }

  const data = await res.json()
  const text = data.choices?.[0]?.message?.content || ''
  if (data.usage?.finish_reason === 'length') console.warn('Response cut off')
  console.log('LLM raw:', text)
  return text
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function extractJSON(text) {
  const clean = text.replace(/```json|```/gi, '').trim()
  const start = clean.indexOf('{')
  const end   = clean.lastIndexOf('}')
  if (start === -1 || end === -1) throw new Error('No JSON in response')
  return clean.slice(start, end + 1)
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms))
}