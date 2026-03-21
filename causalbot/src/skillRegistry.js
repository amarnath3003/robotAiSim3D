import { state } from './state.js'

// Built-in permanent skills that always exist
const BUILTIN_SKILLS = {
  jump: {
    name: 'jump',
    code: `const p = context.getPos(); context.setPos(p.x, p.y + 0.8, p.z); await context.wait(400); context.setPos(p.x, p.y, p.z); context.setStatus('Jumped!');`,
    fn: (context) => { const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor; return new AsyncFunction('context', `const p = context.getPos(); context.setPos(p.x, p.y + 0.8, p.z); await context.wait(400); context.setPos(p.x, p.y, p.z); context.setStatus('Jumped!');`)(context) }
  },
  go_to: {
    name: 'go_to',
    code: `if (!context.target) { context.setStatus('No target'); return; } await context.navigateTo(context.target.position[0], 1.2, context.target.position[2]); context.setStatus('Arrived!');`,
    fn: (ctx) => { const F = Object.getPrototypeOf(async function(){}).constructor; return new F('context', `if (!context.target) { context.setStatus('No target'); return; } await context.navigateTo(context.target.position[0], 1.2, context.target.position[2]); context.setStatus('Arrived!');`)(ctx) }
  },
  go_to_object: {
    name: 'go_to_object',
    code: `if (!context.target) { context.setStatus('No target'); return; } await context.navigateTo(context.target.position[0], 1.2, context.target.position[2]); context.setStatus('Arrived!');`,
    fn: (ctx) => { const F = Object.getPrototypeOf(async function(){}).constructor; return new F('context', `if (!context.target) { context.setStatus('No target'); return; } await context.navigateTo(context.target.position[0], 1.2, context.target.position[2]); context.setStatus('Arrived!');`)(ctx) }
  },
  pick_up: {
    name: 'pick_up',
    code: `if (!context.target) { context.setStatus('No target'); return; } const tp = context.target.position; const rp = context.getPos(); const dx = rp.x - tp[0]; const dz = rp.z - tp[2]; const dist = Math.sqrt(dx*dx+dz*dz) || 1; await context.navigateTo(tp[0] + (dx/dist)*0.25, 1.2, tp[2] + (dz/dist)*0.25); context.setArm(-1.2); await context.wait(300); context.grab(context.target.id); context.setArm(0); context.setStatus('Picked up!');`,
    fn: (ctx) => { const F = Object.getPrototypeOf(async function(){}).constructor; return new F('context', `if (!context.target) { context.setStatus('No target'); return; } const tp = context.target.position; const rp = context.getPos(); const dx = rp.x - tp[0]; const dz = rp.z - tp[2]; const dist = Math.sqrt(dx*dx+dz*dz) || 1; await context.navigateTo(tp[0] + (dx/dist)*0.25, 1.2, tp[2] + (dz/dist)*0.25); context.setArm(-1.2); await context.wait(300); context.grab(context.target.id); context.setArm(0); context.setStatus('Picked up!');`)(ctx) }
  },
  pick_up_object: {
    name: 'pick_up_object',
    code: `if (!context.target) { context.setStatus('No target'); return; } const tp = context.target.position; const rp = context.getPos(); const dx = rp.x - tp[0]; const dz = rp.z - tp[2]; const dist = Math.sqrt(dx*dx+dz*dz) || 1; await context.navigateTo(tp[0] + (dx/dist)*0.25, 1.2, tp[2] + (dz/dist)*0.25); context.setArm(-1.2); await context.wait(300); context.grab(context.target.id); context.setArm(0); context.setStatus('Picked up!');`,
    fn: (ctx) => { const F = Object.getPrototypeOf(async function(){}).constructor; return new F('context', `if (!context.target) { context.setStatus('No target'); return; } const tp = context.target.position; const rp = context.getPos(); const dx = rp.x - tp[0]; const dz = rp.z - tp[2]; const dist = Math.sqrt(dx*dx+dz*dz) || 1; await context.navigateTo(tp[0] + (dx/dist)*0.25, 1.2, tp[2] + (dz/dist)*0.25); context.setArm(-1.2); await context.wait(300); context.grab(context.target.id); context.setArm(0); context.setStatus('Picked up!');`)(ctx) }
  },
  release: {
    name: 'release',
    code: `context.release(); context.setArm(0); context.setStatus('Released!');`,
    fn: (context) => { const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor; return new AsyncFunction('context', `context.release(); context.setArm(0); context.setStatus('Released!');`)(context) }
  },
  spin: {
    name: 'spin',
    code: `const p = context.getPos(); for (let i = 0; i <= 8; i++) { const a = (i/8)*Math.PI*2; context.setPos(p.x + Math.cos(a)*0.3, p.y, p.z + Math.sin(a)*0.3); await context.wait(80); } context.setPos(p.x, p.y, p.z); context.setStatus('Spun!');`,
    fn: (context) => { const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor; return new AsyncFunction('context', `const p = context.getPos(); for (let i = 0; i <= 8; i++) { const a = (i/8)*Math.PI*2; context.setPos(p.x + Math.cos(a)*0.3, p.y, p.z + Math.sin(a)*0.3); await context.wait(80); } context.setPos(p.x, p.y, p.z); context.setStatus('Spun!');`)(context) }
  },
  patrol: {
    name: 'patrol',
    code: `const corners = [[-2,1.2,-2],[-2,1.2,2],[2,1.2,2],[2,1.2,-2]]; for (const c of corners) { await context.navigateTo(c[0],c[1],c[2]); await context.wait(200); } context.setStatus('Patrol done!');`,
    fn: (context) => { const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor; return new AsyncFunction('context', `const corners = [[-2,1.2,-2],[-2,1.2,2],[2,1.2,2],[2,1.2,-2]]; for (const c of corners) { await context.navigateTo(c[0],c[1],c[2]); await context.wait(200); } context.setStatus('Patrol done!');`)(context) }
  },
  fly: {
    name: 'fly',
    code: `const p = context.getPos(); context.setPos(p.x, p.y + 1.5, p.z); await context.wait(1000); context.setPos(p.x, 1.2, p.z); context.setStatus('Landed!');`,
    fn: (context) => { const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor; return new AsyncFunction('context', `const p = context.getPos(); context.setPos(p.x, p.y + 1.5, p.z); await context.wait(1000); context.setPos(p.x, 1.2, p.z); context.setStatus('Landed!');`)(context) }
  },
  dance: {
    name: 'dance',
    code: `for (let i = 0; i < 4; i++) { const p = context.getPos(); context.setPos(p.x+0.3, p.y+0.3, p.z); await context.wait(150); context.setPos(p.x-0.3, p.y, p.z); await context.wait(150); context.setPos(p.x, p.y, p.z); await context.wait(150); } context.setStatus('Danced!');`,
    fn: (context) => { const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor; return new AsyncFunction('context', `for (let i = 0; i < 4; i++) { const p = context.getPos(); context.setPos(p.x+0.3, p.y+0.3, p.z); await context.wait(150); context.setPos(p.x-0.3, p.y, p.z); await context.wait(150); context.setPos(p.x, p.y, p.z); await context.wait(150); } context.setStatus('Danced!');`)(context) }
  },
}

// Session skills — invented this session, not yet approved
const SESSION_SKILLS = {}

// Approved persistent skills — saved to localStorage
let SAVED_SKILLS = {}

const STORAGE_KEY = 'causalbot_skills'

export function initSkillRegistry() {
  // Load saved skills from localStorage
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      const parsed = JSON.parse(raw)
      // Reconstruct functions from stored code strings
      Object.entries(parsed).forEach(([name, code]) => {
        try {
          SAVED_SKILLS[name] = {
            name,
            code,
            fn: (context) => {
              const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor
              return new AsyncFunction('context', code)(context)
            }
          }
        } catch (e) {
          console.warn(`Failed to restore skill "${name}":`, e)
        }
      })
    }
  } catch (e) {
    console.warn('Could not load saved skills:', e)
  }

  renderSkillLog()
  console.log('Skill registry ready. Saved skills:', Object.keys(SAVED_SKILLS))
}

export function hasSkill(name) {
  const n = name.toLowerCase().replace(/\s+/g, '_')
  return !!(BUILTIN_SKILLS[n] || SESSION_SKILLS[n] || SAVED_SKILLS[n])
}

export function getSkill(name) {
  const n = name.toLowerCase().replace(/\s+/g, '_')
  return BUILTIN_SKILLS[n] || SAVED_SKILLS[n] || SESSION_SKILLS[n] || null
}

export function getAllSkillNames() {
  return [
    ...Object.keys(BUILTIN_SKILLS),
    ...Object.keys(SAVED_SKILLS),
    ...Object.keys(SESSION_SKILLS),
  ]
}

export function registerSessionSkill(name, code) {
  const n = name.toLowerCase().replace(/\s+/g, '_')
  try {
    const fn = (context) => {
      const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor
      return new AsyncFunction('context', code)(context)
    }
    SESSION_SKILLS[n] = { name: n, code, fn }
    renderSkillLog()
    return true
  } catch (e) {
    console.error('Skill code invalid:', e)
    return false
  }
}

export function approveSkill(name) {
  const n = name.toLowerCase().replace(/\s+/g, '_')
  const skill = SESSION_SKILLS[n]
  if (!skill) return
  SAVED_SKILLS[n] = skill
  delete SESSION_SKILLS[n]

  // Persist to localStorage
  const toStore = {}
  Object.entries(SAVED_SKILLS).forEach(([k, v]) => {
    toStore[k] = v.code
  })
  localStorage.setItem(STORAGE_KEY, JSON.stringify(toStore))
  renderSkillLog()
  console.log(`Skill "${n}" approved and saved.`)
}

export function rejectSkill(name) {
  const n = name.toLowerCase().replace(/\s+/g, '_')
  delete SESSION_SKILLS[n]
  renderSkillLog()
}

function renderSkillLog() {
  const el = document.getElementById('skill-log')
  if (!el) return
  const saved = Object.keys(SAVED_SKILLS).map(n => `
    <div class="skill-item">${n}</div>
  `).join('')
  const session = Object.keys(SESSION_SKILLS).map(n => `
    <div class="skill-item new">${n} (new)</div>
  `).join('')
  el.innerHTML = saved + session
}