import { state, getObject, getRobotPos, setRobotPos } from './state.js'
import { navigateTo } from './robot.js'
import { releaseObjectPhysics } from './physics.js'
import { remember } from './memory.js'
import { getSkill, registerSessionSkill, approveSkill, rejectSkill, getAllSkillNames } from './skillRegistry.js'
import { planInstruction, inventSkill, isTrainingInstruction, planTraining } from './llm.js'
import { showThoughts, clearThoughts, setStatus, setAgentStatus } from './ui.js'
import { ensureWorldModel, getVisionContextAdditions, resolveObject } from './perception/perceptionMode.js'
import { trainPolicy, getTrainingProgress, trainingState } from './rl.js'
import { showRLPanel, hideRLPanel, updateRLPanel, showRLResult } from './rlPanel.js'

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

      // Enforce physical proximity
      const rp = getRobotPos()
      const op = obj.position
      const dist = Math.sqrt((rp.x - op[0])**2 + (rp.y - op[1])**2 + (rp.z - op[2])**2)
      if (dist > 1.2) {
        console.warn(`[Executor] Grab failed: object too far (${dist.toFixed(2)}m)`)
        return false
      }

      if (obj._body) {
        obj._body.setBodyType(2) // 2 = KinematicPositionBased
      }
      obj.status            = 'held'
      state.robot.heldObject = obj.id
      state.robot.eyeColor  = 0x00ff88
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

    setEye:   (hex) => { state.robot.eyeColor = hex },
    wait:     (ms)  => new Promise(resolve => setTimeout(resolve, ms)),
    getObject:(nameOrId) => getObject(nameOrId),
    getWorldBounds: () => state.world.roomBounds,
    remember: (outcome, detail) => remember(instruction, outcome, detail),
    setStatus: (text) => {
      const el = document.getElementById('status-bar')
      if (el) el.textContent = text
    },
  }
}

// ─── Main instruction handler ─────────────────────────────────────────────────

export async function handleInstruction(instruction) {
  if (state.execution.running) return
  state.execution.running = true
  state.robot.status      = 'thinking'

  try {
    // ── Route 1: RL training ──
    if (isTrainingInstruction(instruction)) {
      await handleTraining(instruction)
      return
    }

    // ── Route 2: Direct skill invocation ("run pick_up_ball_rl") ──
    const directSkill = _resolveDirectSkillInvocation(instruction)
    if (directSkill) {
      await handleDirectSkillRun(instruction, directSkill)
      return
    }

    // ── Route 3: Normal LLM-planned execution ──
    await handleNormalInstruction(instruction)

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

// ─── Direct skill invocation resolver ────────────────────────────────────────
// Recognises patterns like "run pick_up_ball_rl", "do pick up ball",
// "execute the learned skill", "pick_up_ball_rl" (bare skill name), etc.

function _resolveDirectSkillInvocation(instruction) {
  const lower = instruction.toLowerCase().trim()

  // Pattern 1: "run <skillName>" / "execute <skillName>" / "do <skillName>"
  const runMatch = lower.match(/^(?:run|execute|do|perform|start|activate)\s+(.+)$/)
  if (runMatch) {
    const candidate = runMatch[1].replace(/\s+/g, '_').replace(/-/g, '_')
    const skill = getSkill(candidate) || getSkill(candidate + '_rl') || getSkill(candidate.replace(/_rl$/, ''))
    if (skill) return skill
    // Also try partial match against all known skill names
    const allNames = getAllSkillNames()
    const match = allNames.find(n => n.includes(candidate) || candidate.includes(n))
    if (match) return getSkill(match)
  }

  // Pattern 2: Instruction IS exactly a skill name (with or without _rl suffix)
  const exactCandidate = lower.replace(/\s+/g, '_')
  if (getSkill(exactCandidate)) return getSkill(exactCandidate)
  if (getSkill(exactCandidate + '_rl')) return getSkill(exactCandidate + '_rl')

  return null
}

// ─── Direct skill run ─────────────────────────────────────────────────────────
// Executes a pre-learned skill directly — used when user says "run pick_up_ball_rl"

async function handleDirectSkillRun(instruction, skill) {
  clearThoughts()
  showThoughts([
    `🎯 Direct skill invocation: "${skill.name}"`,
    '🤖 Executing learned policy...',
  ])

  setStatus(`🎯 Running "${skill.name}"...`)
  setAgentStatus(`Running: ${skill.name}`, 'navigating')
  state.robot.status = 'executing'

  const ctx = {
    ...buildContext(instruction),
    args:   {},
    target: null,  // skill has the target embedded
  }

  try {
    await skill.fn(ctx)
    remember(instruction, 'success', `Executed learned skill "${skill.name}"`)
    setStatus(`✅ "${skill.name}" completed.`)
    setAgentStatus('Skill executed', 'success')
    setTimeout(() => setAgentStatus(null), 3000)
  } catch (e) {
    console.error(`[RL] Direct skill run error:`, e)
    remember(instruction, 'fail', e.message)
    setStatus(`❌ Skill "${skill.name}" failed: ${e.message}`)
    setAgentStatus('Skill error', 'error')
    setTimeout(() => setAgentStatus(null), 5000)
  }

  state.robot.status      = 'idle'
  state.robot.eyeColor    = 0x4488ff
  state.execution.running = false
}

// ─── Training handler ─────────────────────────────────────────────────────────

async function handleTraining(instruction) {
  clearThoughts()
  showThoughts([
    '🎓 Training mode detected',
    '🧠 Parsing task from instruction...',
    '⚙️  Q-learning in pure simulation (no physics touched)',
    '💾 Learned policy will be baked as a runnable skill',
  ])

  // Step 1 — LLM parses the instruction into a structured task
  const task = await planTraining(instruction)
  if (!task) {
    state.execution.running = false
    state.robot.status      = 'idle'
    return
  }

  console.log('[Training] Task:', task)
  setStatus(`🎓 Starting RL training: "${task.name}"`)

  // Step 2 — Show training progress panel
  showRLPanel(task.name)

  // Step 3 — Run training loop (pure simulation, no Three.js/physics)
  const result = await trainPolicy(task, (episode, total, reward, epsilon) => {
    const progress = getTrainingProgress()
    if (progress) {
      updateRLPanel(progress, trainingState.rewardHistory, trainingState.successHistory)
    }
  })

  // Step 4 — Show result & run demo
  if (result) {
    showRLResult(result.successRate, result.converged)

    // Normalize skill name the same way registerSessionSkill does
    const rawSkillName = task.skillName || (task.name.replace(/\s+/g, '_').toLowerCase() + '_rl')
    const skillName    = rawSkillName.toLowerCase().replace(/\s+/g, '_')

    if (result.successRate >= 2 && result.skill) {
      remember(
        instruction,
        'success',
        `RL training done — ${result.successRate}% success${result.converged ? ' (converged early)' : ''}. ` +
        `Skill "${skillName}" is now available. Say "run ${skillName}" to execute it.`
      )

      // ── Brief pause so user can see the result panel ──
      await new Promise(r => setTimeout(r, 1500))

      // ── Demo run: execute the baked skill in the real environment ──
      const skill = getSkill(skillName)
      if (skill) {
        setStatus(`🎬 Demo: running "${skillName}"...`)
        setAgentStatus('Demonstrating learned skill', 'navigating')
        const ctx = {
          ...buildContext(instruction),
          args:   {},
          target: getObject(task.targetObjectId),
        }
        try {
          await skill.fn(ctx)
        } catch (demoErr) {
          console.warn('[RL] Demo run error:', demoErr)
          setStatus(`Demo encountered an issue: ${demoErr.message}`)
        }
      } else {
        console.warn(`[RL] Could not find skill "${skillName}" for demo`)
      }

      // Show save/discard approval UI
      showApprovalUI(skillName)
      setStatus(`✅ Training complete! Say "${skillName.replace(/_/g,' ')}" or "run ${skillName.replace(/_/g,' ')}" to execute.`)

    } else if (result.successRate === 0) {
      remember(instruction, 'fail', 'RL training produced 0% success — object may be unreachable or task too complex')
      setStatus('❌ Training did not converge. Try a simpler goal or check the object is accessible.')
    } else {
      // Partial success — still save the skill
      setStatus(`⚠️ Training partial (${result.successRate}%). Skill saved — may work imperfectly. Say "${skillName.replace(/_/g,' ')}" to try.`)
      showApprovalUI(skillName)
    }
  }

  hideRLPanel()
  state.robot.status      = 'idle'
  state.robot.eyeColor    = 0x4488ff
  state.execution.running = false
}

// ─── Normal instruction handler ───────────────────────────────────────────────

async function handleNormalInstruction(instruction) {
  // Vision mode: scan before planning if world model empty
  await ensureWorldModel()

  // Step 1 — LLM plans what to do
  const plan = await planInstruction(instruction)
  if (!plan) { state.execution.running = false; return }

  console.log('Plan:', plan)
  const thoughts = []
  if (plan.reasoning)     thoughts.push(`🧠 ${plan.reasoning}`)
  if (plan.thoughts?.length) thoughts.push(...plan.thoughts)
  if (plan.goal)          thoughts.push(`🎯 Goal: ${plan.goal}`)
  if (thoughts.length)    showThoughts(thoughts)

  // Step 2 — if new skill needed, invent it
  if (plan.needsNewSkill && plan.newSkillName) {
    const code = await inventSkill(
      plan.newSkillName,
      plan.newSkillDescription || instruction,
      getAllSkillNames()
    )
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

    plan.actions = [{
      skill:       plan.newSkillName,
      args:        {},
      description: plan.newSkillDescription || instruction,
    }]

    state.execution.pendingApproval = plan.newSkillName
  }

  // Step 3 — execute each action
  state.robot.status = 'executing'
  const ctx = {
    ...buildContext(instruction),
    ...getVisionContextAdditions(),
  }

  console.log('Starting action execution loop. Actions:', plan.actions)

  for (const action of plan.actions) {
    console.log('Action starting:', action)
    const skill = getSkill(action.skill)
    if (!skill) {
      console.warn(`Skill not found: ${action.skill}`)
      continue
    }

    setStatus(`${action.description || action.skill}...`)
    setAgentStatus(`${action.description || action.skill}...`, 'navigating')
    console.log('Running skill:', action.skill, action.args)

    const SCAN_SKILLS = ['scanforobject', 'scan_room']
    let target = null
    if (action.args?.target && !SCAN_SKILLS.includes(action.skill.toLowerCase())) {
      console.log('Resolving target:', action.args.target)
      target = await resolveObject(action.args.target)
      console.log('Target resolved:', target)
    }

    const enrichedCtx = {
      ...ctx,
      args:   action.args || {},
      target,
    }

    try {
      console.log('Invoking skill function')
      await skill.fn(enrichedCtx)
      console.log('Skill finished:', action.skill)
    } catch (e) {
      console.error(`Skill "${action.skill}" threw:`, e)
      remember(instruction, 'fail', e.message)
    }
  }

  console.log('Action execution loop finished')

  // Step 4 — show approval if new skill
  if (state.execution.pendingApproval) {
    showApprovalUI(state.execution.pendingApproval)
    state.execution.pendingApproval = null
  }

  state.robot.status   = 'idle'
  state.robot.eyeColor = 0x4488ff
  remember(instruction, 'success', plan.plan)
  setStatus('Done.')
  setAgentStatus('Goal completed', 'success')
  setTimeout(() => setAgentStatus(null), 3000)

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