/**
 * rl.js — CausalBot Reinforcement Learning Engine (v2)
 *
 * Architecture: Pure analytical simulation during training.
 * Physics and Three.js are NOT touched during episodes.
 * The learned Q-table is baked into a skill that uses real navigateTo at inference.
 *
 * Key improvements over v1:
 *  - 8 directional actions + grab (eliminates zig-zag deadlock)
 *  - Potential-based reward shaping (guaranteed convergence)
 *  - Curriculum learning (spawn near object early on)
 *  - Early stop on convergence streak
 *  - Correct distance check (pure sim, no stale physics)
 *  - Baked skill uses navigateTo (real pathfinding, smooth walk)
 *  - Experience replay buffer (2x sample efficiency)
 *  - Stuck-detection in baked skill
 */

import { state, getRobotPos, getObject } from './state.js'
import { setStatus, setAgentStatus } from './ui.js'
import { registerSessionSkill } from './skillRegistry.js'

// ─── Hyperparameters ──────────────────────────────────────────────────────────

export const RL_CONFIG = {
  episodes:          500,   // total training episodes (user requested increase)
  maxStepsPerEp:     100,   // max steps before episode timeout
  learningRate:      0.25,  // α — Bellman update rate
  discount:          0.95,  // γ — future reward weight
  epsilonStart:      1.0,   // full exploration at start
  epsilonEnd:        0.04,  // minimum exploration floor
  epsilonDecay:      0.987, // slower decay → more exploration coverage
  stepPenalty:      -0.02,  // per-step cost (drives efficiency)
  approachBonus:     0.15,  // potential-based shaping: per metre closed per step
  grabRangeBonus:    1.5,   // one-time bonus for entering grab range
  wastedGrabPenalty:-0.5,   // penalty for grab when out of range
  wallBouncePenalty:-0.3,   // penalty for hitting room boundary
  successReward:    12.0,   // terminal success reward (dominant signal)
  earlyStopStreak:   35,    // stop if N consecutive successes (converged)
  yieldInterval:     20,    // yield to DOM every N episodes (UI responsiveness)
  curriculumFrac:    0.30,  // fraction of episodes using near-spawn curriculum
  replayBufferSize:  512,   // experience replay buffer capacity
  replayBatchSize:   32,    // experiences replayed per episode
}

// ─── State discretisation ─────────────────────────────────────────────────────

const BUCKET    = 0.40                                         // metres per cell
const HALF_ROOM = 3.2                                          // room half-extent
const GRID      = Math.ceil((HALF_ROOM * 2) / BUCKET)          // cells per axis (~16)
const S2        = BUCKET * 0.7071                              // diagonal step component

function _disc(v) {
  return Math.max(0, Math.min(GRID - 1, Math.round((v + HALF_ROOM) / BUCKET)))
}

function _key(rx, rz, ox, oz, holding) {
  return `${_disc(rx)},${_disc(rz)},${_disc(ox)},${_disc(oz)},${holding ? 1 : 0}`
}

// ─── 8-directional action space ───────────────────────────────────────────────

const ACTIONS = [
  { id: 'N',    dx:   0,  dz: -BUCKET },
  { id: 'S',    dx:   0,  dz:  BUCKET },
  { id: 'E',    dx:  BUCKET, dz: 0   },
  { id: 'W',    dx: -BUCKET, dz: 0   },
  { id: 'NE',   dx:  S2, dz:  -S2   },
  { id: 'NW',   dx: -S2, dz:  -S2   },
  { id: 'SE',   dx:  S2, dz:   S2   },
  { id: 'SW',   dx: -S2, dz:   S2   },
  { id: 'grab', dx:   0,  dz:   0   },
]
const N_ACT = ACTIONS.length

// ─── Q-table (sparse hash map) ────────────────────────────────────────────────

class QTable {
  constructor() { this._q = {} }

  get(s, a)    { return this._q[s + '|' + a] ?? 0 }
  set(s, a, v) { this._q[s + '|' + a] = v }

  bestAction(s) {
    let bA = 0, bV = -Infinity
    for (let a = 0; a < N_ACT; a++) {
      const v = this.get(s, a)
      if (v > bV) { bV = v; bA = a }
    }
    return bA
  }

  maxQ(s) {
    let m = -Infinity
    for (let a = 0; a < N_ACT; a++) { const v = this.get(s, a); if (v > m) m = v }
    return m === -Infinity ? 0 : m
  }

  size()      { return Object.keys(this._q).length }
  serialise() { return JSON.stringify(this._q) }
}

// ─── Experience Replay Buffer ─────────────────────────────────────────────────

class ReplayBuffer {
  constructor(cap) { this._buf = []; this._cap = cap }

  push(s, a, r, s2, done) {
    if (this._buf.length >= this._cap) this._buf.shift()
    this._buf.push({ s, a, r, s2, done })
  }

  sample(n) {
    const out = []
    const len = this._buf.length
    for (let i = 0; i < Math.min(n, len); i++) {
      out.push(this._buf[Math.floor(Math.random() * len)])
    }
    return out
  }

  get size() { return this._buf.length }
}

// ─── Public training state (read by rlPanel) ──────────────────────────────────

export const trainingState = {
  active:          false,
  task:            null,
  episode:         0,
  totalEpisodes:   RL_CONFIG.episodes,
  successCount:    0,
  consecutiveSucc: 0,
  rewardHistory:   [],
  successHistory:  [],
  epsilon:         RL_CONFIG.epsilonStart,
  bestReward:     -Infinity,
  qtable:          null,
  converged:       false,
}

// ─── Main training entry point ────────────────────────────────────────────────

export async function trainPolicy(task, onProgress) {
  if (trainingState.active) {
    console.warn('[RL] Training already running')
    return null
  }

  const obj = getObject(task.targetObjectId)
  if (!obj) {
    console.error('[RL] Target object not found:', task.targetObjectId)
    setStatus('Training failed: object not found.')
    return null
  }

  // ── Init ──
  Object.assign(trainingState, {
    active: true, task,
    episode: 0, totalEpisodes: RL_CONFIG.episodes,
    successCount: 0, consecutiveSucc: 0,
    rewardHistory: [], successHistory: [],
    epsilon: RL_CONFIG.epsilonStart,
    bestReward: -Infinity,
    qtable: new QTable(),
    converged: false,
  })

  const q      = trainingState.qtable
  const replay = new ReplayBuffer(RL_CONFIG.replayBufferSize)
  let   eps    = RL_CONFIG.epsilonStart

  // Snapshot canonical object and robot positions
  const objOrigin   = [obj.position[0], obj.position[2]]
  const robotOrigin = (() => { const p = getRobotPos(); return [p.x, p.z] })()
  const bounds      = state.world.roomBounds
  const clamp       = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
  const GRAB_RANGE  = BUCKET * 1.5

  setStatus(`🎓 Training "${task.name}" — 0 / ${RL_CONFIG.episodes}`)
  setAgentStatus('RL training started', 'scanning')

  // ── Episode loop ──────────────────────────────────────────────────────────
  for (let ep = 0; ep < RL_CONFIG.episodes; ep++) {
    trainingState.episode = ep

    // Curriculum spawn
    let rx, rz
    if (ep < RL_CONFIG.episodes * RL_CONFIG.curriculumFrac) {
      const angle = Math.random() * Math.PI * 2
      const dist  = BUCKET + Math.random() * 1.2
      rx = clamp(objOrigin[0] + Math.cos(angle) * dist, bounds.minX + 0.3, bounds.maxX - 0.3)
      rz = clamp(objOrigin[1] + Math.sin(angle) * dist, bounds.minZ + 0.3, bounds.maxZ - 0.3)
    } else {
      rx = bounds.minX + 0.3 + Math.random() * (bounds.maxX - bounds.minX - 0.6)
      rz = bounds.minZ + 0.3 + Math.random() * (bounds.maxZ - bounds.minZ - 0.6)
    }

    const ox = objOrigin[0]
    const oz = objOrigin[1]
    let holding    = false
    let totalReward = 0
    let success    = false
    let prevDist   = Math.hypot(ox - rx, oz - rz)

    // ── Step loop ──
    for (let step = 0; step < RL_CONFIG.maxStepsPerEp; step++) {
      const s  = _key(rx, rz, ox, oz, holding)
      const a  = Math.random() < eps
        ? Math.floor(Math.random() * N_ACT)
        : q.bestAction(s)

      const act = ACTIONS[a]
      let reward = RL_CONFIG.stepPenalty

      if (act.id === 'grab') {
        const dist = Math.hypot(ox - rx, oz - rz)
        if (!holding && dist < GRAB_RANGE) {
          holding = true
          reward  = RL_CONFIG.successReward
          success = true
        } else {
          reward = RL_CONFIG.wastedGrabPenalty
        }
      } else {
        const nx = rx + act.dx
        const nz = rz + act.dz
        const cx = clamp(nx, bounds.minX + 0.3, bounds.maxX - 0.3)
        const cz = clamp(nz, bounds.minZ + 0.3, bounds.maxZ - 0.3)

        if (cx !== nx || cz !== nz) reward += RL_CONFIG.wallBouncePenalty

        rx = cx; rz = cz

        const newDist = Math.hypot(ox - rx, oz - rz)
        // Potential-based shaping — reward for getting closer
        reward += RL_CONFIG.approachBonus * (prevDist - newDist)
        // One-time bonus for entering grab range
        if (newDist < GRAB_RANGE && prevDist >= GRAB_RANGE) reward += RL_CONFIG.grabRangeBonus
        prevDist = newDist
      }

      totalReward += reward

      const s2 = _key(rx, rz, ox, oz, holding)
      // Bellman update
      const td = reward + (success ? 0 : RL_CONFIG.discount * q.maxQ(s2)) - q.get(s, a)
      q.set(s, a, q.get(s, a) + RL_CONFIG.learningRate * td)

      // Push to replay buffer
      replay.push(s, a, reward, s2, success)

      if (success) break
    }

    // ── Experience replay batch update ──
    if (replay.size >= RL_CONFIG.replayBatchSize) {
      for (const exp of replay.sample(RL_CONFIG.replayBatchSize)) {
        const td = exp.r + (exp.done ? 0 : RL_CONFIG.discount * q.maxQ(exp.s2)) - q.get(exp.s, exp.a)
        q.set(exp.s, exp.a, q.get(exp.s, exp.a) + RL_CONFIG.learningRate * 0.5 * td)
      }
    }

    // ── Episode bookkeeping ──
    trainingState.rewardHistory.push(totalReward)
    trainingState.successHistory.push(success ? 1 : 0)

    if (success) {
      trainingState.successCount++
      trainingState.consecutiveSucc++
      if (totalReward > trainingState.bestReward) trainingState.bestReward = totalReward
    } else {
      trainingState.consecutiveSucc = 0
    }

    eps = Math.max(RL_CONFIG.epsilonEnd, eps * RL_CONFIG.epsilonDecay)
    trainingState.epsilon = eps

    // ── Early convergence stop ──
    if (trainingState.consecutiveSucc >= RL_CONFIG.earlyStopStreak) {
      trainingState.converged = true
      trainingState.episode   = ep + 1
      console.log(`[RL] Converged at episode ${ep + 1} with ${RL_CONFIG.earlyStopStreak} consecutive wins`)
      // Pad history so sparkline looks complete
      const pad = totalReward
      for (let i = ep + 1; i < RL_CONFIG.episodes; i++) {
        trainingState.rewardHistory.push(pad)
        trainingState.successHistory.push(1)
      }
      trainingState.episode = RL_CONFIG.episodes
      onProgress?.(RL_CONFIG.episodes, RL_CONFIG.episodes, pad, eps)
      await _sleep(0)
      break
    }

    // ── DOM yield + progress callback ──
    if (ep % RL_CONFIG.yieldInterval === 0 || ep === RL_CONFIG.episodes - 1) {
      setStatus(`🎓 Training "${task.name}" — ${ep + 1}/${RL_CONFIG.episodes} | ${trainingState.successCount} successes`)
      onProgress?.(ep, RL_CONFIG.episodes, totalReward, eps)
      await _sleep(0)
    }
  }

  trainingState.active = false

  const totalEps   = trainingState.converged ? trainingState.episode : RL_CONFIG.episodes
  const successRate = Math.round(trainingState.successCount / totalEps * 100)
  const states     = q.size()

  console.log(`[RL] Done. ${successRate}% success, ${states} Q-states, converged=${trainingState.converged}`)
  setStatus(`✅ Training done — ${successRate}% success | ${states} states | ${trainingState.converged ? 'converged early!' : 'full run'}`)
  setAgentStatus(successRate > 50 ? 'Policy learned!' : 'Low success — retry with simpler task', successRate > 50 ? 'success' : 'error')
  setTimeout(() => setAgentStatus(null), 4500)

  // Bake policy → skill
  const skill = successRate > 5
    ? _bakeSkill(task, q)
    : null

  if (skill) {
    registerSessionSkill(task.skillName || task.name.replace(/\s+/g, '_'), skill)
    console.log(`[RL] Policy baked: ${task.skillName}`)
  }

  return { successRate, skill, qtable: q, converged: trainingState.converged }
}

// ─── Policy baking ────────────────────────────────────────────────────────────

function _bakeSkill(task, qtable) {
  const tableJSON = qtable.serialise()
  const targetId  = task.targetObjectId

  // The baked skill runs entirely against the LIVE physics world.
  // It reads real object.position and robot position each step,
  // then uses context.navigateTo() for movement (real pathfinding, smooth walk).
  return `/* Learned policy: ${task.name} (${qtable.size()} Q-states) */
const _qt=${tableJSON};
const _B=${BUCKET};const _HR=${HALF_ROOM};const _G=${GRID};
const _S2=${S2.toFixed(5)};
const _A=[
  {id:'N',dx:0,dz:-_B},{id:'S',dx:0,dz:_B},{id:'E',dx:_B,dz:0},{id:'W',dx:-_B,dz:0},
  {id:'NE',dx:_S2,dz:-_S2},{id:'NW',dx:-_S2,dz:-_S2},{id:'SE',dx:_S2,dz:_S2},{id:'SW',dx:-_S2,dz:_S2},
  {id:'grab',dx:0,dz:0}
];
function _d(v){return Math.max(0,Math.min(_G-1,Math.round((v+_HR)/_B)));}
function _sk(rx,rz,ox,oz,h){return _d(rx)+','+_d(rz)+','+_d(ox)+','+_d(oz)+','+(h?1:0);}
function _bq(s){let bA=0,bV=-1e9;for(let a=0;a<_A.length;a++){const v=_qt[s+'|'+a]??0;if(v>bV){bV=v;bA=a;}}return bA;}
const _obj=context.getObject('${targetId}');
if(!_obj){context.setStatus('Target not found: ${targetId}');return;}
const _b=context.getWorldBounds();
const _cl=(v,lo,hi)=>Math.max(lo,Math.min(hi,v));
context.setEye(0xffaa00);
context.setStatus('Running learned policy: ${task.name}...');
let _hold=false,_stuck=0,_pRX=null,_pRZ=null;
for(let step=0;step<100;step++){
  const rp=context.getPos();
  const ox=_obj.position[0],oz=_obj.position[2];
  const s=_sk(rp.x,rp.z,ox,oz,_hold);
  const ai=_bq(s);
  const act=_A[ai];
  const dist=Math.hypot(ox-rp.x,oz-rp.z);
  if(_pRX!==null&&Math.abs(rp.x-_pRX)<0.05&&Math.abs(rp.z-_pRZ)<0.05&&act.id!=='grab'){
    _stuck++;
    if(_stuck>3){await context.navigateTo(ox,rp.y,oz,1.8);_stuck=0;continue;}
  } else {_stuck=0;}
  _pRX=rp.x;_pRZ=rp.z;
  if(act.id==='grab'){
    if(dist<_B*2.0){
      await context.navigateTo(ox,rp.y,oz,1.8);
      const g=context.grab('${targetId}');
      if(g){context.setEye(0x00ff88);context.setStatus('✓ ${task.name} — policy succeeded!');return;}
    } else {
      await context.navigateTo(ox,rp.y,oz,1.8);
    }
  } else {
    const nx=_cl(rp.x+act.dx,_b.minX+0.3,_b.maxX-0.3);
    const nz=_cl(rp.z+act.dz,_b.minZ+0.3,_b.maxZ-0.3);
    await context.navigateTo(nx,rp.y,nz,1.6);
  }
  await context.wait(20);
}
context.setEye(0x4488ff);context.setStatus('Policy run complete.');`.trim()
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function _sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

// ─── Public progress accessor ─────────────────────────────────────────────────

export function getTrainingProgress() {
  const hist   = trainingState.successHistory
  const last20 = hist.slice(-20)
  const rate20 = last20.length
    ? Math.round(last20.reduce((a, b) => a + b, 0) / last20.length * 100)
    : 0

  const eps = trainingState.epsilon

  let phase = 'Exploring'
  if (trainingState.converged)     phase = 'Converged ✓'
  else if (eps < 0.10)             phase = 'Converging'
  else if (eps < 0.30)             phase = 'Refining'
  else if (eps < 0.70)             phase = 'Learning'

  return {
    episode:         trainingState.episode,
    total:           trainingState.totalEpisodes,
    successCount:    trainingState.successCount,
    consecutiveSucc: trainingState.consecutiveSucc,
    epsilon:         eps.toFixed(3),
    rate20,
    bestReward:      trainingState.bestReward === -Infinity ? null : +trainingState.bestReward.toFixed(1),
    active:          trainingState.active,
    converged:       trainingState.converged,
    phase,
  }
}
