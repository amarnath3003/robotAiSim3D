/**
 * rl.js — Reinforcement Learning trainer for CausalBot
 *
 * Design: lightweight Q-table RL that runs entirely in-browser.
 * The LLM defines the reward function and task parameters.
 * After training, the learned policy is registered as a callable skill.
 *
 * Flow:
 *   1. LLM calls planTraining() with task description
 *   2. RL agent runs N episodes against the physics sim
 *   3. Best policy exported as a skill function
 */

import { state, getRobotPos, setRobotPos, getObject } from './state.js'
import { setStatus, setAgentStatus } from './ui.js'
import { registerSessionSkill } from './skillRegistry.js'

// ─── Config ───────────────────────────────────────────────────────────────────

const RL_CONFIG = {
  episodes:       80,       // total training episodes
  maxStepsPerEp:  60,       // max actions per episode before timeout
  learningRate:   0.18,     // α — how fast Q-values update
  discount:       0.92,     // γ — future reward weight
  epsilonStart:   1.0,      // exploration at episode 0
  epsilonEnd:     0.08,     // minimum exploration
  epsilonDecay:   0.96,     // multiply epsilon each episode
  stepDelayMs:    0,        // 0 = max speed; raise to animate training
}

// ─── State discretisation ─────────────────────────────────────────────────────
// We bucket continuous positions into a coarse grid so the Q-table stays small.

const BUCKET = 0.6          // metres per cell
const ROOM   = 3.0          // room half-extent

function discretise(x, z) {
  const bx = Math.round((x + ROOM) / BUCKET)
  const bz = Math.round((z + ROOM) / BUCKET)
  return `${Math.max(0, Math.min(10, bx))}_${Math.max(0, Math.min(10, bz))}`
}

// Actions the robot can take each step
const ACTIONS = [
  { id: 'north',  dx:  0,  dz: -BUCKET },
  { id: 'south',  dx:  0,  dz:  BUCKET },
  { id: 'east',   dx:  BUCKET, dz: 0   },
  { id: 'west',   dx: -BUCKET, dz: 0   },
  { id: 'grab',   dx:  0,  dz:  0      },
]

// ─── Q-table ──────────────────────────────────────────────────────────────────

class QTable {
  constructor() {
    this._table = {}
  }

  _key(state, actionIdx) {
    return `${state}__${actionIdx}`
  }

  get(state, actionIdx) {
    return this._table[this._key(state, actionIdx)] ?? 0
  }

  set(state, actionIdx, value) {
    this._table[this._key(state, actionIdx)] = value
  }

  bestAction(state) {
    let best = 0, bestQ = -Infinity
    for (let i = 0; i < ACTIONS.length; i++) {
      const q = this.get(state, i)
      if (q > bestQ) { bestQ = q; best = i }
    }
    return best
  }

  // Serialise to compact JSON for skill code embedding
  serialise() {
    return JSON.stringify(this._table)
  }
}

// ─── Training session state ────────────────────────────────────────────────────

export const trainingState = {
  active:        false,
  task:          null,       // { name, targetObjectId, goalType }
  episode:       0,
  totalEpisodes: RL_CONFIG.episodes,
  successCount:  0,
  rewardHistory: [],         // reward per episode
  epsilon:       RL_CONFIG.epsilonStart,
  qtable:        null,
  onProgress:    null,       // UI callback
}

// ─── Main training entry point ────────────────────────────────────────────────

/**
 * Start a training run.
 * Called by executor when LLM identifies a "train" instruction.
 *
 * @param {object} task
 *   task.name           - human label e.g. "pick up the box"
 *   task.targetObjectId - object id to interact with e.g. "object_box"
 *   task.goalType       - 'pick_up' | 'navigate_to' | 'push_to_corner'
 *   task.rewardFn       - optional JS string override for reward function
 * @param {function} onProgress - (episode, total, reward, epsilon) => void
 */
export async function trainPolicy(task, onProgress) {
  if (trainingState.active) {
    console.warn('[RL] Training already running')
    return null
  }

  trainingState.active        = true
  trainingState.task          = task
  trainingState.episode       = 0
  trainingState.successCount  = 0
  trainingState.rewardHistory = []
  trainingState.epsilon       = RL_CONFIG.epsilonStart
  trainingState.qtable        = new QTable()
  trainingState.onProgress    = onProgress || null

  const q       = trainingState.qtable
  let   epsilon = trainingState.epsilon

  setStatus(`🎓 Training: ${task.name} (0/${RL_CONFIG.episodes})`)
  setAgentStatus('RL training started', 'thinking')

  // Save robot start position so we can reset each episode
  const origin = { ...getRobotPos() }

  for (let ep = 0; ep < RL_CONFIG.episodes; ep++) {
    trainingState.episode = ep

    // ── Reset environment for this episode ──
    _resetEpisode(origin, task)

    let totalReward = 0
    let done        = false

    for (let step = 0; step < RL_CONFIG.maxStepsPerEp && !done; step++) {
      const rp  = getRobotPos()
      const s   = _buildState(rp, task)

      // ε-greedy action selection
      const actionIdx = (Math.random() < epsilon)
        ? Math.floor(Math.random() * ACTIONS.length)
        : q.bestAction(s)

      // Execute action
      const { nextPos, grabbed } = _applyAction(ACTIONS[actionIdx], rp, task)
      setRobotPos(nextPos.x, nextPos.y, nextPos.z)

      // Compute reward
      const reward = _computeReward(nextPos, grabbed, task)
      totalReward += reward

      // Q-update
      const s2     = _buildState(nextPos, task)
      const maxQ2  = Math.max(...ACTIONS.map((_, i) => q.get(s2, i)))
      const oldQ   = q.get(s, actionIdx)
      const newQ   = oldQ + RL_CONFIG.learningRate * (reward + RL_CONFIG.discount * maxQ2 - oldQ)
      q.set(s, actionIdx, newQ)

      // Terminal conditions
      if (reward >= 8) {
        done = true
        trainingState.successCount++
      }

      if (RL_CONFIG.stepDelayMs > 0) {
        await _sleep(RL_CONFIG.stepDelayMs)
      }
    }

    trainingState.rewardHistory.push(totalReward)
    epsilon = Math.max(RL_CONFIG.epsilonEnd, epsilon * RL_CONFIG.epsilonDecay)
    trainingState.epsilon = epsilon

    // Progress callback every 10 episodes
    if (ep % 10 === 0 || ep === RL_CONFIG.episodes - 1) {
      const pct = Math.round((ep / RL_CONFIG.episodes) * 100)
      setStatus(`🎓 Training: ${task.name} — ${pct}% (${trainingState.successCount} successes)`)
      setAgentStatus(`Training... ${pct}%`, 'scanning')
      trainingState.onProgress?.(ep, RL_CONFIG.episodes, totalReward, epsilon)
      // Yield to browser so UI can update
      await _sleep(0)
    }
  }

  // ── Reset robot to origin ──
  setRobotPos(origin.x, origin.y, origin.z)
  _resetEpisode(origin, task)

  trainingState.active = false

  const successRate = (trainingState.successCount / RL_CONFIG.episodes * 100).toFixed(0)
  setStatus(`✅ Training done! ${successRate}% success rate`)
  setAgentStatus(`Training complete: ${successRate}% success`, 'success')
  setTimeout(() => setAgentStatus(null), 4000)

  // ── Bake policy into a skill ──
  if (trainingState.successCount > 0) {
    const skill = _bakeSkill(task, q)
    const registered = registerSessionSkill(task.skillName || task.name.replace(/\s+/g, '_'), skill)
    if (registered) {
      console.log('[RL] Policy baked into skill:', task.skillName)
    }
    return { successRate: parseInt(successRate), skill, qtable: q }
  }

  return { successRate: 0, skill: null, qtable: q }
}

// ─── Episode helpers ──────────────────────────────────────────────────────────

function _resetEpisode(origin, task) {
  // Robot back to start
  setRobotPos(origin.x, origin.y, origin.z)
  state.robot.heldObject = null
  state.robot.armAngle   = 0
  state.robot.eyeColor   = 0x4488ff

  // Reset target object position if it was moved
  const obj = getObject(task.targetObjectId)
  if (obj && task._objOrigin) {
    obj.position[0] = task._objOrigin[0]
    obj.position[1] = task._objOrigin[1]
    obj.position[2] = task._objOrigin[2]
    if (obj._body) {
      obj._body.setTranslation(
        { x: task._objOrigin[0], y: task._objOrigin[1], z: task._objOrigin[2] },
        true
      )
      obj._body.setLinvel({ x: 0, y: 0, z: 0 }, true)
      obj._body.setAngvel({ x: 0, y: 0, z: 0 }, true)
    }
    obj.status = 'intact'
  }
}

// ─── State builder ────────────────────────────────────────────────────────────

function _buildState(robotPos, task) {
  const rDisc = discretise(robotPos.x, robotPos.z)
  const obj   = getObject(task.targetObjectId)
  const holding = state.robot.heldObject === task.targetObjectId ? '1' : '0'

  if (!obj) return `${rDisc}_none_${holding}`

  const oDisc = discretise(obj.position[0], obj.position[2])
  return `${rDisc}_${oDisc}_${holding}`
}

// ─── Action executor ──────────────────────────────────────────────────────────

function _applyAction(action, robotPos, task) {
  const bounds = state.world.roomBounds
  const newX   = Math.max(bounds.minX + 0.3, Math.min(bounds.maxX - 0.3, robotPos.x + action.dx))
  const newZ   = Math.max(bounds.minZ + 0.3, Math.min(bounds.maxZ - 0.3, robotPos.z + action.dz))
  const nextPos = { x: newX, y: robotPos.y, z: newZ }

  let grabbed = false

  if (action.id === 'grab') {
    const obj = getObject(task.targetObjectId)
    if (obj) {
      const dist = Math.sqrt(
        (obj.position[0] - robotPos.x) ** 2 +
        (obj.position[2] - robotPos.z) ** 2
      )
      if (dist < 0.8 && obj.snapable && !state.robot.heldObject) {
        state.robot.heldObject = obj.id
        obj.status = 'held'
        if (obj._body) obj._body.setBodyType(0)
        grabbed = true
      }
    }
  }

  return { nextPos, grabbed }
}

// ─── Reward function ──────────────────────────────────────────────────────────

function _computeReward(robotPos, grabbed, task) {
  const obj = getObject(task.targetObjectId)
  if (!obj) return -0.1

  const distToObj = Math.sqrt(
    (obj.position[0] - robotPos.x) ** 2 +
    (obj.position[2] - robotPos.z) ** 2
  )

  if (task.goalType === 'pick_up') {
    if (grabbed)                          return 10.0    // terminal success
    if (state.robot.heldObject === obj.id) return 10.0   // already holding
    if (distToObj < 0.5)                  return  2.0    // close to object
    if (distToObj < 1.2)                  return  0.5    // nearby
    return -0.05 * distToObj                             // small distance penalty
  }

  if (task.goalType === 'navigate_to') {
    if (distToObj < 0.4)  return 10.0
    if (distToObj < 1.0)  return  1.5
    return -0.05 * distToObj
  }

  if (task.goalType === 'push_to_corner') {
    // Goal: push object to room corner (maxX, maxZ)
    const cornerDist = Math.sqrt(
      (obj.position[0] - (state.world.roomBounds.maxX - 0.5)) ** 2 +
      (obj.position[2] - (state.world.roomBounds.maxZ - 0.5)) ** 2
    )
    if (cornerDist < 0.4) return 10.0
    return -0.05 * cornerDist
  }

  return 0
}

// ─── Policy baking ────────────────────────────────────────────────────────────

/**
 * Convert the trained Q-table into an executable skill JS string.
 * The skill embeds the Q-table and runs a greedy policy at runtime.
 */
function _bakeSkill(task, qtable) {
  const tableJSON = qtable.serialise()
  const targetId  = task.targetObjectId

  // Compact greedy executor — embedded Q-table, no external deps
  return `
const _qt = ${tableJSON};
const _BUCKET = ${BUCKET};
const _ROOM   = ${ROOM};
const _ACTIONS = [
  { id:'north', dx:0,       dz:-_BUCKET },
  { id:'south', dx:0,       dz: _BUCKET },
  { id:'east',  dx:_BUCKET, dz:0        },
  { id:'west',  dx:-_BUCKET,dz:0        },
  { id:'grab',  dx:0,       dz:0        },
];
function _disc(x,z){
  const bx=Math.round((x+_ROOM)/_BUCKET);
  const bz=Math.round((z+_ROOM)/_BUCKET);
  return Math.max(0,Math.min(10,bx))+'_'+Math.max(0,Math.min(10,bz));
}
function _bestAction(s){
  let best=0,bestQ=-Infinity;
  for(let i=0;i<_ACTIONS.length;i++){
    const q=_qt[s+'__'+i]??0;
    if(q>bestQ){bestQ=q;best=i;}
  }
  return best;
}
const target = context.getObject('${targetId}');
if(!target){context.setStatus('Target not found');return;}
context.setEye(0xffaa00);
context.setStatus('Running learned policy...');
for(let step=0;step<50;step++){
  const rp=context.getPos();
  const holding=context.args?._held==='${targetId}'?'1':'0';
  const rDisc=_disc(rp.x,rp.z);
  const oDisc=_disc(target.position[0],target.position[2]);
  const s=rDisc+'_'+oDisc+'_'+holding;
  const ai=_bestAction(s);
  const act=_ACTIONS[ai];
  if(act.id==='grab'){
    const grabbed=context.grab('${targetId}');
    if(grabbed){context.setEye(0x00ff88);context.setStatus('Picked up via learned policy!');return;}
  } else {
    const b=state?.world?.roomBounds||{minX:-3,maxX:3,minZ:-3,maxZ:3};
    const nx=Math.max(b.minX+0.3,Math.min(b.maxX-0.3,rp.x+act.dx));
    const nz=Math.max(b.minZ+0.3,Math.min(b.maxZ-0.3,rp.z+act.dz));
    context.setPos(nx,rp.y,nz);
    await context.wait(80);
  }
}
context.setEye(0x4488ff);
context.setStatus('Policy run complete.');
`.trim()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function _sleep(ms) {
  return new Promise(r => setTimeout(r, ms))
}

// ─── Public: get training progress for UI ─────────────────────────────────────

export function getTrainingProgress() {
  if (!trainingState.active && trainingState.rewardHistory.length === 0) return null
  const hist = trainingState.rewardHistory
  const avg5 = hist.length >= 5
    ? (hist.slice(-5).reduce((a, b) => a + b, 0) / 5).toFixed(1)
    : '—'
  return {
    episode:      trainingState.episode,
    total:        trainingState.totalEpisodes,
    successCount: trainingState.successCount,
    epsilon:      trainingState.epsilon.toFixed(2),
    avgReward:    avg5,
    active:       trainingState.active,
  }
}
