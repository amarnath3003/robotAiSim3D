import { state, getObject, getRobotPos } from './state.js'
import { getMemorySummary } from './memory.js'
import { getAllSkillNames, hasSkill } from './skillRegistry.js'

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY
const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent`

function buildWorldContext() {
  const objs = Object.values(state.world.objects)
    .map(o => `  ${o.name} (id:${o.id}) at [${o.position.map(v => v.toFixed(2)).join(', ')}] status:${o.status}`)
    .join('\n')

  const rp = getRobotPos()
  const skills = getAllSkillNames()

  return `ROBOT STATE:
  position: [${rp.x.toFixed(2)}, ${rp.y.toFixed(2)}, ${rp.z.toFixed(2)}]
  holding: ${state.robot.heldObject || 'nothing'}
  status: ${state.robot.status}

WORLD OBJECTS:
${objs}

ROOM BOUNDS: x[-3,3] z[-3,3] floor_y:0

AVAILABLE SKILLS: ${skills.length ? skills.join(', ') : 'none yet'}

MEMORY:
${getMemorySummary()}`
}

// Ask LLM to decide what to do — returns a plan
function buildPlanPrompt(instruction) {
  const rp = getRobotPos()
  return `Robot brain. Pick skills for instruction.

ROBOT: [${rp.x.toFixed(1)},${rp.y.toFixed(1)},${rp.z.toFixed(1)}] holding:${state.robot.heldObject || 'nothing'}
OBJECTS: ${Object.values(state.world.objects).map(o => `${o.name}(${o.id})@[${o.position.map(v => v.toFixed(1)).join(',')}]`).join(' ')}
SKILLS: ${getAllSkillNames().join(',')}

INSTRUCTION: "${instruction}"

Return compact JSON only, no spaces, no markdown:
{"thoughts":["step1","step2","step3"],"plan":"x","actions":[{"skill":"s","args":{"target":"id"},"description":"x"}],"needsNewSkill":false,"newSkillName":null}

In "thoughts", provide 3-4 distinct reasoning steps as an array of strings.
Keep "plan" value under 5 words. Keep "description" under 5 words.`
}

export async function planInstruction(instruction) {
  setStatus('Thinking...')
  state.robot.eyeColor = 0xffaa00
  state.robot.status = 'thinking'

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const res = await callGemini(buildPlanPrompt(instruction))
      const json = extractJSON(res)
      const plan = JSON.parse(json)
      state.robot.status = 'idle'
      return plan
    } catch (e) {
      console.warn(`Plan attempt ${attempt + 1} failed:`, e.message)
      if (attempt === 2) {
        state.robot.status = 'failed'
        state.robot.eyeColor = 0xff3333
        setStatus('Could not understand. Try again.')
        return null
      }
    }
  }
}

// Ask LLM to invent a new skill as JS code
export async function inventSkill(skillName, instruction, worldContext) {
  setStatus(`Inventing skill: ${skillName}...`)

  const prompt = `Write a JavaScript skill body for a robot.

Skill name: "${skillName}"
Task: "${instruction}"

RULES:
- Use context.getObject(name) to find object positions — NEVER hardcode coordinates
- context.getObject() returns {position: [x,y,z]} 
- For navigation: const obj = context.getObject('ball'); await context.navigateTo(obj.position[0], 1.2, obj.position[2]);
- Always use context.getPos() for current robot position
- Max 6 lines, no markdown, no comments, just code

EXAMPLES BY SKILL TYPE:

jump:
const p = context.getPos();
context.setPos(p.x, p.y + 0.8, p.z);
await context.wait(400);
context.setPos(p.x, p.y, p.z);
context.setStatus('Jumped!');

go_to (going to an object):
const obj = context.getObject('${instruction.toLowerCase().includes('ball') ? 'ball' : instruction.toLowerCase().includes('glass') ? 'glass' : instruction.toLowerCase().includes('box') ? 'box' : 'ball'}');
if (!obj) { context.setStatus('Object not found'); return; }
await context.navigateTo(obj.position[0], 1.2, obj.position[2]);
context.setStatus('Arrived!');

pick_up:
const obj = context.getObject('${instruction.toLowerCase().includes('ball') ? 'object_ball' : instruction.toLowerCase().includes('glass') ? 'object_glass' : 'object_box'}');
if (!obj) { context.setStatus('Object not found'); return; }
await context.navigateTo(obj.position[0], 1.2, obj.position[2]);
context.setArm(-1.2);
await context.wait(300);
context.grab(obj.id);
context.setArm(0);
context.setStatus('Picked up!');

spin:
const p = context.getPos();
for (let i = 0; i < 8; i++) {
  const angle = (i / 8) * Math.PI * 2;
  context.setPos(p.x + Math.cos(angle) * 0.3, p.y, p.z + Math.sin(angle) * 0.3);
  await context.wait(100);
}
context.setStatus('Spun!');

Now write the skill body for "${skillName}". Return ONLY the code lines, nothing else.`

  try {
    const code = await callGemini(prompt)
    console.log('Invented skill code:', code)
    // Clean any accidental markdown
    return code
      .replace(/```javascript|```js|```/gi, '')
      .trim()
  } catch (e) {
    console.error('Skill invention failed:', e)
    return null
  }
}

async function callGemini(prompt) {
  const res = await fetch(`${API_URL}?key=${API_KEY}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: {
        temperature: 0.1,
        maxOutputTokens: 2048,
      }
    })
  })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.error?.message || 'API error')
  }

  const data = await res.json()
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text || ''
  console.log('LLM raw:', text)
  return text
}

function extractJSON(text) {
  const clean = text.replace(/```json|```/gi, '').trim()
  const start = clean.indexOf('{')
  const end = clean.lastIndexOf('}')
  if (start === -1 || end === -1) throw new Error('No JSON in response')
  return clean.slice(start, end + 1)
}

function setStatus(text) {
  const el = document.getElementById('status-bar')
  if (el) el.textContent = text
}