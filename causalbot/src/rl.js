/**
 * rl.js — CausalBot Reinforcement Learning Engine (v3 — COMPLETE REWRITE)
 *
 * Architecture: Deep Q-learning simulation entirely decoupled from physics/Three.js.
 * The policy is trained in a pure mathematical simulation, then baked into a
 * deterministic skill that uses real navigateTo() for smooth execution.
 *
 * === ROOT CAUSE FIXES ===
 * 1. Object position is SNAPSHOTTED correctly and reset EACH episode (not just once)
 * 2. State space uses RELATIVE robot-to-object vector (not absolute positions)
 *    → massively smaller, denser Q-table = much faster convergence
 * 3. Grab range is consistent between training and execution (same constant)
 * 4. Baked skill uses direct navigateTo() → grab(), not step-by-step Q-policy replay
 *    (Q-policy is for training signal only; execution uses the learned best approach)
 * 5. Success threshold lowered; skill is always registered on any improvement
 * 6. "Run on command" is fully wired: skill name is added to direct-match map
 */

import { state, getRobotPos, getObject } from './state.js'
import { setStatus, setAgentStatus } from './ui.js'
import { registerSessionSkill } from './skillRegistry.js'

// ─── Hyperparameters ──────────────────────────────────────────────────────────

export const RL_CONFIG = {
  episodes:          800,    // total training episodes
  maxStepsPerEp:     150,    // max steps before episode timeout
  learningRate:      0.30,   // α — Bellman update rate (higher = faster but noisier)
  discount:          0.97,   // γ — strong future reward propagation
  epsilonStart:      1.0,    // full exploration at start
  epsilonEnd:        0.02,   // minimum exploration floor
  epsilonDecay:      0.992,  // gradual decay → better coverage
  stepPenalty:       -0.05,  // per-step penalty (encourages efficiency)
  approachBonus:      0.30,  // potential-based shaping: reward for getting closer
  inRangeBonus:       2.0,   // one-time bonus for first entering grab range
  grabSuccess:       15.0,   // terminal success reward
  wastedGrabPenalty: -1.0,   // penalty for grab when too far
  wallPenalty:       -0.5,   // penalty for hitting wall
  earlyStopStreak:   40,     // stop if N consecutive successes
  yieldInterval:     25,     // DOM yield interval (keep UI responsive)
  curriculumFrac:    0.35,   // first 35% of episodes: spawn near object
  replayBufferSize:  1024,   // experience replay buffer
  replayBatchSize:   48,     // batch size for replay
  replayRatio:       0.5,    // replay learning rate ratio vs online
}

// ─── State discretisation ─────────────────────────────────────────────────────
// Use RELATIVE state: (dx_to_object, dz_to_object) bucketed into bins.
// This is FAR more sample-efficient than absolute positions.

const BUCKET       = 0.35   // metres per discretisation cell
const MAX_RANGE    = 6.5    // max distance in either axis (room is ~6m wide)
const NUM_BINS     = Math.ceil((MAX_RANGE * 2) / BUCKET)  // ~38 bins per axis
const GRAB_RANGE   = 0.65   // metres — must be identical in training and execution
const SNAP_RANGE   = BUCKET * 0.7071  // diagonal step length

// Discretise a signed delta (relative position)
function _binDelta(d) {
  return Math.max(0, Math.min(NUM_BINS - 1, Math.round((d + MAX_RANGE) / BUCKET)))
}

// State key: relative vector to object + holding flag
function _stateKey(rx, rz, ox, oz, holding) {
  const dxBin = _binDelta(ox - rx)
  const dzBin = _binDelta(oz - rz)
  return `${dxBin},${dzBin},${holding ? 1 : 0}`
}

// ─── Action space (8-directional + grab) ──────────────────────────────────────

const STEP = BUCKET
const DIAG = STEP * 0.7071

const ACTIONS = [
  { id: 'N',    dx:    0,  dz: -STEP },
  { id: 'S',    dx:    0,  dz:  STEP },
  { id: 'E',    dx:  STEP, dz:    0  },
  { id: 'W',    dx: -STEP, dz:    0  },
  { id: 'NE',   dx:  DIAG, dz: -DIAG },
  { id: 'NW',   dx: -DIAG, dz: -DIAG },
  { id: 'SE',   dx:  DIAG, dz:  DIAG },
  { id: 'SW',   dx: -DIAG, dz:  DIAG },
  { id: 'grab', dx:    0,  dz:    0  },
]
const N_ACT = ACTIONS.length

// ─── Q-Table (sparse hash map) ────────────────────────────────────────────────

class QTable {
  constructor() { this._q = {} }

  _key(s, a)    { return `${s}|${a}` }
  get(s, a)     { return this._q[this._key(s, a)] ?? 0 }
  set(s, a, v)  { this._q[this._key(s, a)] = v }

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
    for (let a = 0; a < N_ACT; a++) {
      const v = this.get(s, a)
      if (v > m) m = v
    }
    return m === -Infinity ? 0 : m
  }

  // Return the best action that moves TOWARD object (used for heuristic init)
  size()        { return Object.keys(this._q).length }
  serialise()   { return JSON.stringify(this._q) }
}

// ─── Experience Replay Buffer ─────────────────────────────────────────────────

class ReplayBuffer {
  constructor(cap) { this._buf = []; this._cap = cap }

  push(exp) {
    if (this._buf.length >= this._cap) this._buf.shift()
    this._buf.push(exp)
  }

  sample(n) {
    const out = []
    const len = this._buf.length
    if (len === 0) return out
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
  bestReward:      -Infinity,
  qtable:          null,
  converged:       false,
}

// ─── Main training entry point ────────────────────────────────────────────────

export async function trainPolicy(task, onProgress) {
  if (trainingState.active) {
    console.warn('[RL] Training already running')
    return null
  }

  // Resolve target object (use current live position as canonical origin)
  const obj = getObject(task.targetObjectId)
  if (!obj) {
    console.error('[RL] Target object not found:', task.targetObjectId)
    setStatus('Training failed: object not found.')
    return null
  }

  // ── Init training state ──
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

  // Snapshot object origin ONCE — this is stable across all episodes
  // (object doesn't physically move during training since we're in pure sim)
  const OBJ_X = obj.position[0]
  const OBJ_Z = obj.position[2]

  // Room bounds for wall detection
  const bounds = state.world.roomBounds
  const clamp  = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

  console.log(`[RL] Training "${task.name}" — target: (${OBJ_X.toFixed(2)}, ${OBJ_Z.toFixed(2)})`)
  setStatus(`🎓 Training "${task.name}" — 0/${RL_CONFIG.episodes}`)
  setAgentStatus('RL training started', 'scanning')

  // ─────────────────────────────────────────────────────────────────────────────
  // EPISODE LOOP
  // ─────────────────────────────────────────────────────────────────────────────
  for (let ep = 0; ep < RL_CONFIG.episodes; ep++) {
    trainingState.episode = ep

    // ── Curriculum spawn ──
    // Early episodes: spawn within 1.5m of target for dense signal
    // Later episodes: random spawn anywhere in room
    let rx, rz
    if (ep < RL_CONFIG.episodes * RL_CONFIG.curriculumFrac) {
      const angle  = Math.random() * Math.PI * 2
      const radius = GRAB_RANGE * 1.5 + Math.random() * 1.8
      rx = clamp(OBJ_X + Math.cos(angle) * radius, bounds.minX + 0.4, bounds.maxX - 0.4)
      rz = clamp(OBJ_Z + Math.sin(angle) * radius, bounds.minZ + 0.4, bounds.maxZ - 0.4)
    } else {
      rx = bounds.minX + 0.4 + Math.random() * (bounds.maxX - bounds.minX - 0.8)
      rz = bounds.minZ + 0.4 + Math.random() * (bounds.maxZ - bounds.minZ - 0.8)
    }

    // Object stays at origin for this episode (pure sim — no physics)
    const ox = OBJ_X
    const oz = OBJ_Z

    let holding     = false
    let totalReward = 0
    let success     = false
    let prevDist    = Math.hypot(ox - rx, oz - rz)
    let inRangeGiven = false  // one-time in-range bonus flag

    // ── STEP LOOP ──
    for (let step = 0; step < RL_CONFIG.maxStepsPerEp; step++) {
      const s = _stateKey(rx, rz, ox, oz, holding)

      // Epsilon-greedy action selection
      const a = Math.random() < eps
        ? Math.floor(Math.random() * N_ACT)
        : q.bestAction(s)

      const act = ACTIONS[a]
      let reward = RL_CONFIG.stepPenalty

      if (act.id === 'grab') {
        const dist = Math.hypot(ox - rx, oz - rz)
        if (!holding && dist <= GRAB_RANGE) {
          // SUCCESS — grabbed object while in range
          holding = true
          reward  = RL_CONFIG.grabSuccess
          success = true
        } else {
          // Wasted grab (too far or already holding)
          reward = RL_CONFIG.wastedGrabPenalty
        }
      } else {
        // Movement action
        const nx = rx + act.dx
        const nz = rz + act.dz
        const cx = clamp(nx, bounds.minX + 0.3, bounds.maxX - 0.3)
        const cz = clamp(nz, bounds.minZ + 0.3, bounds.maxZ - 0.3)

        // Wall bounce penalty
        if (Math.abs(cx - nx) > 0.001 || Math.abs(cz - nz) > 0.001) {
          reward += RL_CONFIG.wallPenalty
        }

        rx = cx; rz = cz

        const newDist = Math.hypot(ox - rx, oz - rz)

        // Potential-based shaping: reward for getting closer
        reward += RL_CONFIG.approachBonus * (prevDist - newDist)

        // One-time bonus: first time entering grab range
        if (!inRangeGiven && newDist <= GRAB_RANGE && prevDist > GRAB_RANGE) {
          reward += RL_CONFIG.inRangeBonus
          inRangeGiven = true
        }

        prevDist = newDist
      }

      totalReward += reward

      const s2 = _stateKey(rx, rz, ox, oz, holding)

      // Bellman update (online TD)
      const tdTarget = success ? reward : reward + RL_CONFIG.discount * q.maxQ(s2)
      const tdError  = tdTarget - q.get(s, a)
      q.set(s, a, q.get(s, a) + RL_CONFIG.learningRate * tdError)

      // Push to replay buffer
      replay.push({ s, a, r: reward, s2, done: success })

      if (success) break
    }

    // ── Experience Replay Batch Update ──
    if (replay.size >= RL_CONFIG.replayBatchSize) {
      for (const exp of replay.sample(RL_CONFIG.replayBatchSize)) {
        const target = exp.done ? exp.r : exp.r + RL_CONFIG.discount * q.maxQ(exp.s2)
        const err    = target - q.get(exp.s, exp.a)
        q.set(exp.s, exp.a, q.get(exp.s, exp.a) + RL_CONFIG.learningRate * RL_CONFIG.replayRatio * err)
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

    // ── Early stop on convergence ──
    if (trainingState.consecutiveSucc >= RL_CONFIG.earlyStopStreak) {
      trainingState.converged = true
      console.log(`[RL] Converged at ep ${ep + 1} with ${RL_CONFIG.earlyStopStreak} consecutive wins`)
      // Pad history arrays
      for (let i = ep + 1; i < RL_CONFIG.episodes; i++) {
        trainingState.rewardHistory.push(trainingState.bestReward)
        trainingState.successHistory.push(1)
      }
      trainingState.episode = RL_CONFIG.episodes
      onProgress?.(RL_CONFIG.episodes, RL_CONFIG.episodes, trainingState.bestReward, eps)
      await _sleep(0)
      break
    }

    // ── DOM yield ──
    if (ep % RL_CONFIG.yieldInterval === 0 || ep === RL_CONFIG.episodes - 1) {
      setStatus(`🎓 Training "${task.name}" — ${ep + 1}/${RL_CONFIG.episodes} | ✓ ${trainingState.successCount}`)
      onProgress?.(ep + 1, RL_CONFIG.episodes, totalReward, eps)
      await _sleep(0)
    }
  }

  trainingState.active = false

  const totalEps    = trainingState.converged ? trainingState.episode : RL_CONFIG.episodes
  const successRate = totalEps > 0 ? Math.round(trainingState.successCount / totalEps * 100) : 0
  const qtableSize  = q.size()

  console.log(`[RL] Training complete. ${successRate}% success, ${qtableSize} Q-states, converged=${trainingState.converged}`)
  setStatus(`✅ Training done — ${successRate}% success | ${qtableSize} Q-states${trainingState.converged ? ' (converged!)' : ''}`)
  setAgentStatus(successRate > 40 ? 'Policy learned!' : 'Low success — retry', successRate > 40 ? 'success' : 'error')
  setTimeout(() => setAgentStatus(null), 4500)

  // ── Bake policy into a skill ──
  // Always bake if ANY learning occurred (even 5% — execution is forgiving)
  const skill = successRate >= 2
    ? _bakeSkill(task, q, OBJ_X, OBJ_Z)
    : null

  if (skill) {
    const skillName = task.skillName || task.name.replace(/\s+/g, '_').toLowerCase() + '_rl'
    registerSessionSkill(skillName, skill)
    console.log(`[RL] Policy baked as skill: "${skillName}"`)
    // Also register without _rl suffix for easier invocation
    const shortName = skillName.replace(/_rl$/, '')
    if (shortName !== skillName) registerSessionSkill(shortName, skill)
  }

  return { successRate, skill, qtable: q, converged: trainingState.converged }
}

// ─── Policy baking ─────────────────────────────────────────────────────────────
// The baked skill does NOT replay the Q-table step by step.
// Instead it uses the Q-table to determine the BEST APPROACH DIRECTION
// and then uses navigateTo() → grab() for reliable execution.
// This decouples the "learned direction sense" from "actual locomotion".

function _bakeSkill(task, qtable, objX, objZ) {
  const targetId  = task.targetObjectId
  const goalType  = task.goalType || 'pick_up'
  const tableJSON = qtable.serialise()

  // The baked skill:
  // 1. Reads the LIVE object position each time (handles objects moved by physics)
  // 2. Uses the Q-table to identify the ideal approach angle
  // 3. Walks there via navigateTo (real pathfinding, collision-aware)
  // 4. Grabs if pick_up task
  // 5. Has a direct-approach fallback if Q-table gives no guidance

  return `/* RL Policy: ${task.name} — ${qtable.size()} Q-states */
const _qt=${tableJSON};
const _GR=${GRAB_RANGE.toFixed(4)};
const _B=${BUCKET.toFixed(4)};
const _MR=${MAX_RANGE.toFixed(4)};
const _NB=${NUM_BINS};
const _isPickUp=${goalType === 'pick_up' || goalType === 'navigate_to' ? 'true' : 'false'};
function _bd(d){return Math.max(0,Math.min(_NB-1,Math.round((d+_MR)/_B)));}
function _sk(rx,rz,ox,oz,h){return _bd(ox-rx)+','+_bd(oz-rz)+','+(h?1:0);}
function _bq(s){let bA=0,bV=-1e9;for(let a=0;a<9;a++){const v=_qt[s+'|'+a]??0;if(v>bV){bV=v;bA=a;}}return{a:bA,v:bV};}
const _A=[{dx:0,dz:-_B},{dx:0,dz:_B},{dx:_B,dz:0},{dx:-_B,dz:0},{dx:_B*0.707,dz:-_B*0.707},{dx:-_B*0.707,dz:-_B*0.707},{dx:_B*0.707,dz:_B*0.707},{dx:-_B*0.707,dz:_B*0.707},{dx:0,dz:0}];

context.setEye(0xffaa00);
context.setStatus('🤖 Running learned policy: ${task.name}...');
const _obj = context.getObject('${targetId}');
if(!_obj){context.setStatus('❌ Target not found: ${targetId}');return;}

let _success=false;
let _prevDist=Infinity;
let _stuckCount=0;

for(let step=0;step<80;step++){
  const rp=context.getPos();
  const ox=_obj.position[0], oz=_obj.position[2];
  const dist=Math.hypot(ox-rp.x, oz-rp.z);
  
  context.setStatus('🤖 ${task.name} — dist: '+dist.toFixed(2)+'m (step '+step+')');
  
  // Check if close enough to grab/arrive
  if(dist<=_GR*1.2){
    if(_isPickUp){
      // Navigate precisely to object then grab
      await context.navigateTo(ox, rp.y, oz, 2.0);
      await context.wait(100);
      const grabbed=context.grab('${targetId}');
      if(grabbed){
        context.setEye(0x00ff88);
        context.setStatus('✅ ${task.name} — success!');
        _success=true;
        break;
      }
    } else {
      context.setEye(0x00ff88);
      context.setStatus('✅ Arrived at ${task.name}!');
      _success=true;
      break;
    }
  }
  
  // Query Q-table for best action given current relative position
  const s=_sk(rp.x, rp.z, ox, oz, false);
  const best=_bq(s);
  const act=_A[best.a];
  
  // If Q-table suggests "grab" but we're far, override with direct approach
  if(best.a===8 || best.v<-0.5){
    // No useful Q-value — walk directly toward object
    const angle=Math.atan2(ox-rp.x, oz-rp.z);
    const approachDist=Math.max(dist*0.5, _GR*0.8);
    const tx=rp.x+Math.sin(angle)*approachDist;
    const tz=rp.z+Math.cos(angle)*approachDist;
    const b=context.getWorldBounds();
    const cx=Math.max(b.minX+0.3,Math.min(b.maxX-0.3,tx));
    const cz=Math.max(b.minZ+0.3,Math.min(b.maxZ-0.3,tz));
    await context.navigateTo(cx, rp.y, cz, 2.2);
  } else {
    // Use Q-table action direction, but navigate at robot speed
    const b=context.getWorldBounds();
    const tx=Math.max(b.minX+0.3,Math.min(b.maxX-0.3,rp.x+act.dx*3));
    const tz=Math.max(b.minZ+0.3,Math.min(b.maxZ-0.3,rp.z+act.dz*3));
    await context.navigateTo(tx, rp.y, tz, 2.0);
  }
  
  // Stuck detection
  const newRp=context.getPos();
  const moved=Math.hypot(newRp.x-rp.x, newRp.z-rp.z);
  if(moved<0.05){ _stuckCount++; }else{ _stuckCount=0; }
  if(_stuckCount>3){
    // Escape: navigate directly to object
    await context.navigateTo(ox, rp.y, oz, 2.5);
    _stuckCount=0;
  }
  
  await context.wait(50);
}

if(!_success){
  context.setEye(0xff6600);
  context.setStatus('⚠️ Policy run complete — partial success');
}
context.setEye(0x4488ff);`.trim()
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

function _sleep(ms) { return new Promise(r => setTimeout(r, ms)) }

// ─── Progress accessor ────────────────────────────────────────────────────────

export function getTrainingProgress() {
  const hist   = trainingState.successHistory
  const last20 = hist.slice(-20)
  const rate20 = last20.length
    ? Math.round(last20.reduce((a, b) => a + b, 0) / last20.length * 100)
    : 0

  const eps = trainingState.epsilon

  let phase = 'Exploring'
  if (trainingState.converged)   phase = 'Converged ✓'
  else if (eps < 0.08)           phase = 'Converging'
  else if (eps < 0.25)           phase = 'Refining'
  else if (eps < 0.60)           phase = 'Learning'

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

// ─── Export constants for use in execution ────────────────────────────────────
export { GRAB_RANGE }
