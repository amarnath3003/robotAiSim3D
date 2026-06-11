import { state, getObject, getRobotPos, setRobotPos } from './state.js'
import { navigateTo } from './robot.js'
import { releaseObjectPhysics } from './physics.js'
import { remember } from './memory.js'
import { getSkill, registerSessionSkill, approveSkill, rejectSkill, getAllSkillNames } from './skillRegistry.js'
import { planInstruction, inventSkill } from './llm.js'
import { showThoughts, clearThoughts, setStatus, setAgentStatus } from './ui.js'
import { ensureWorldModel, getVisionContextAdditions, resolveObject } from './perception/perceptionMode.js'

// ─── Context builder ──────────────────────────────────────────────────────────

function buildContext(instruction) {
  return {
    navigateTo: (x, y, z, speed) => new Promise(resolve => {
      const excludeIds = state.robot.heldObject ? [state.robot.heldObject] : []
      navigateTo(x, y, z, resolve, speed, excludeIds)
    }),

    setPos: (x, y, z) => setRobotPos(x, y, z),

    getPos: () => getRobotPos(),

    setArm: (angle) => { state.robot.armAngle = angle },

    extendArm: (amount) => { state.robot.armExtend = Math.max(0, Math.min(1, amount)) },

    grab: (objectId) => {
      const obj = getObject(objectId)
      if (!obj || !obj.snapable) return false
      if (obj._body) obj._body.setBodyType(1)  // Fixed — disable physics while held
      obj.status             = 'held'
      state.robot.heldObject = obj.id
      state.robot.eyeColor   = 0x00ff88
      return true
    },

    release: () => {
      if (!state.robot.heldObject) return
      const obj = getObject(state.robot.heldObject)
      if (obj) {
        obj.status = 'intact'
        const rp    = getRobotPos()
        const angle = state.scene.three
          ?.getObjectByName('robot_body')
          ?.parent?.rotation?.y || 0
        releaseObjectPhysics(obj.id, rp, angle)
      }
      state.robot.heldObject = null
      state.robot.eyeColor   = 0x4488ff
    },

    setEye:         (hex)       => { state.robot.eyeColor = hex },
    wait:           (ms)        => new Promise(r => setTimeout(r, ms)),
    getObject:      (nameOrId)  => getObject(nameOrId),
    getWorldBounds: ()          => state.world.roomBounds,
    remember:       (outcome, detail) => remember(instruction, outcome, detail),

    setStatus: (text) => {
      const el = document.getElementById('status-bar')
      if (el) el.textContent = text
    },
  }
}

// ─── Main entry point ─────────────────────────────────────────────────────────

export async function handleInstruction(instruction) {
  // Guard: don't run LLM in RL mode (input is routed to Python instead)
  if (state.controlMode === 'rl') return
  if (state.execution.running) return

  state.execution.running = true
  state.robot.status      = 'thinking'

  try {
    await ensureWorldModel()

    // Step 1 — LLM plans
    const plan = await planInstruction(instruction)
    if (!plan) { state.execution.running = false; return }

    console.log('Plan:', plan)

    const thoughts = []
    if (plan.reasoning)       thoughts.push(`🧠 ${plan.reasoning}`)
    if (plan.thoughts?.length) thoughts.push(...plan.thoughts)
    if (plan.goal)             thoughts.push(`🎯 Goal: ${plan.goal}`)
    if (thoughts.length)       showThoughts(thoughts)

    // Step 2 — invent skill if needed
    if (plan.needsNewSkill && plan.newSkillName) {
      const code = await inventSkill(
        plan.newSkillName,
        plan.newSkillDescription || instruction,
        getAllSkillNames()
      )
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

      plan.actions = [{
        skill:       plan.newSkillName,
        args:        {},
        description: plan.newSkillDescription || instruction,
      }]

      state.execution.pendingApproval = plan.newSkillName
    }

    // Step 3 — execute actions
    state.robot.status = 'executing'
    const ctx = {
      ...buildContext(instruction),
      ...getVisionContextAdditions(),
    }

    for (const action of plan.actions) {
      const skill = getSkill(action.skill)
      if (!skill) {
        console.warn(`Skill not found: ${action.skill}`)
        continue
      }

      setStatus(`${action.description || action.skill}...`)
      setAgentStatus(`${action.description || action.skill}...`, 'navigating')

      const SCAN_SKILLS = ['scanforobject', 'scan_room']
      let target = null
      if (action.args?.target && !SCAN_SKILLS.includes(action.skill.toLowerCase())) {
        target = await resolveObject(action.args.target)
      }

      const enrichedCtx = { ...ctx, args: action.args || {}, target }

      try {
        await skill.fn(enrichedCtx)
      } catch (e) {
        console.error(`Skill "${action.skill}" threw:`, e)
        remember(instruction, 'fail', e.message)
      }
    }

    // Step 4 — approval UI for new skills
    if (state.execution.pendingApproval) {
      showApprovalUI(state.execution.pendingApproval)
      state.execution.pendingApproval = null
    }

    state.robot.status   = 'idle'
    state.robot.eyeColor = 0x4488ff
    remember(instruction, 'success', plan.plan || plan.goal)
    setStatus('Done.')
    setAgentStatus('Goal completed', 'success')
    setTimeout(() => setAgentStatus(null), 3000)

  } catch (e) {
    console.error('Execution error:', e)
    state.robot.status   = 'failed'
    state.robot.eyeColor = 0xff3333
    remember(instruction, 'fail', e.message)
    setStatus('Something went wrong.')
    setAgentStatus('Execution error', 'error')
    setTimeout(() => setAgentStatus(null), 5000)
  }

  state.execution.running = false
}

// ─── Approval UI ──────────────────────────────────────────────────────────────

function showApprovalUI(skillName) {
  const panel  = document.getElementById('approve-panel')
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
    setAgentStatus('Skill discarded', 'error')
    setTimeout(() => setAgentStatus(null), 3000)
  }
}