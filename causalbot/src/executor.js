import { state, getObject, getRobotPos, setRobotPos } from './state.js'
import { navigateTo } from './robot.js'
import { releaseObjectPhysics } from './physics.js'
import { remember } from './memory.js'
import { getSkill, registerSessionSkill, approveSkill, rejectSkill } from './skillRegistry.js'
import { planInstruction, inventSkill } from './llm.js'

// The context object passed to every skill function
function buildContext(instruction) {
  return {
    navigateTo: (x, y, z, speed) => new Promise(resolve => {
      navigateTo(x, y, z, resolve, speed)
    }),

    setPos: (x, y, z) => setRobotPos(x, y, z),

    getPos: () => getRobotPos(),

    setArm: (angle) => { state.robot.armAngle = angle },

    extendArm: (amount) => { state.robot.armExtend = Math.max(0, Math.min(1, amount)) },

    grab: (objectId) => {
      const obj = getObject(objectId)
      if (!obj || !obj.snapable) return false
      if (obj._body) obj._body.setBodyType(0) // freeze physics
      obj.status = 'held'
      state.robot.heldObject = obj.id
      state.robot.eyeColor = 0x00ff88
      return true
    },

    release: () => {
      if (!state.robot.heldObject) return
      const obj = getObject(state.robot.heldObject)
      if (obj) {
        obj.status = 'intact'
        const rp = getRobotPos()
        releaseObjectPhysics(obj.id, rp)
      }
      state.robot.heldObject = null
      state.robot.eyeColor = 0x4488ff
    },

    setEye: (hex) => { state.robot.eyeColor = hex },

    wait: (ms) => new Promise(resolve => setTimeout(resolve, ms)),

    getObject: (nameOrId) => getObject(nameOrId),

    getWorldBounds: () => state.world.roomBounds,

    remember: (outcome, detail) => remember(instruction, outcome, detail),

    setStatus: (text) => {
      const el = document.getElementById('status-bar')
      if (el) el.textContent = text
    }
  }
}

export async function handleInstruction(instruction) {
  if (state.execution.running) return
  state.execution.running = true
  state.robot.status = 'thinking'

  try {
    // Step 1 — LLM plans what to do
    const plan = await planInstruction(instruction)
    if (!plan) { state.execution.running = false; return }

    console.log('Plan:', plan)

    // Step 2 — if new skill needed, invent it
    if (plan.needsNewSkill && plan.newSkillName) {
      const worldCtx = JSON.stringify(
        Object.values(state.world.objects).map(o => ({
          id: o.id, name: o.name, pos: o.position, status: o.status
        })), null, 2
      )

      const code = await inventSkill(plan.newSkillName, instruction, worldCtx)
      console.log('Raw skill code:', code)
      if (!code) {
        setStatus('Could not invent skill.')
        state.execution.running = false
        return
      }

      const ok = registerSessionSkill(plan.newSkillName, code)
      if (!ok) {
        setStatus('Skill code was invalid.')
        state.execution.running = false
        return
      }

      // Inject the new skill into the plan
      plan.actions = [{
        skill: plan.newSkillName,
        args: {},
        description: instruction
      }]

      // Show approval UI after execution
      state.execution.pendingApproval = plan.newSkillName
    }

    // Step 3 — execute each action
    state.robot.status = 'executing'
    const ctx = buildContext(instruction)

    for (const action of plan.actions) {
      const skill = getSkill(action.skill)
      if (!skill) {
        console.warn(`Skill not found: ${action.skill}`)
        continue
      }

      setStatus(`${action.description || action.skill}...`)
      console.log('Running skill:', action.skill, action.args)

      // Merge args into context
      const enrichedCtx = {
        ...ctx,
        args: action.args || {},
        target: action.args?.target ? getObject(action.args.target) : null,
      }

      try {
        await skill.fn(enrichedCtx)
      } catch (e) {
        console.error(`Skill "${action.skill}" threw:`, e)
        remember(instruction, 'fail', e.message)
      }
    }

    // Step 4 — show approval if new skill
    if (state.execution.pendingApproval) {
      showApprovalUI(state.execution.pendingApproval)
      state.execution.pendingApproval = null
    }

    state.robot.status = 'idle'
    state.robot.eyeColor = 0x4488ff
    remember(instruction, 'success', plan.plan)
    setStatus('Done.')

  } catch (e) {
    console.error('Execution error:', e)
    state.robot.status = 'failed'
    state.robot.eyeColor = 0xff3333
    remember(instruction, 'fail', e.message)
    setStatus('Something went wrong.')
  }

  state.execution.running = false
}

function showApprovalUI(skillName) {
  const panel = document.getElementById('approve-panel')
  const nameEl = document.getElementById('approve-skill-name')
  if (!panel || !nameEl) return

  nameEl.textContent = skillName
  panel.classList.add('visible')

  document.getElementById('btn-approve').onclick = () => {
    approveSkill(skillName)
    panel.classList.remove('visible')
    setStatus(`Skill "${skillName}" saved permanently.`)
  }

  document.getElementById('btn-reject').onclick = () => {
    rejectSkill(skillName)
    panel.classList.remove('visible')
    setStatus(`Skill "${skillName}" discarded.`)
  }
}

function setStatus(text) {
  const el = document.getElementById('status-bar')
  if (el) el.textContent = text
}