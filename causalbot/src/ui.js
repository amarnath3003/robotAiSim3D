import { state } from './state.js'
import { handleInstruction } from './executor.js'
import { isRLConnected, sendPromptRL, getRLTelemetry, setRLGoalOverride, setRLParams } from './rl.js'

export function initUI() {
  const input = document.getElementById('instruction')

  input.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return
    const text = input.value.trim()
    if (!text) return
    input.value = ''

    const mode = state.controlMode

    if (mode === 'rl') {
      sendPromptRL(text)
      return
    }

    if (state.execution.running) return
    input.disabled = true
    await handleInstruction(text)
    input.disabled = false
    input.focus()
  })

  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      state.execution.running = false
      state.robot.status      = 'idle'
      setStatus('Cancelled.')
    }
  })

  // Refresh input placeholder each second
  const _refreshHint = () => {
    if (!input) return
    switch (state.controlMode) {
      case 'rl':
        input.placeholder = isRLConnected()
          ? 'send goal to RL agent (e.g. "go to the ball")...'
          : 'waiting for Python agent (run run_agent.py)...'
        break
      case 'debug':
        input.placeholder = 'WASD to drive · press 1 for AI mode'
        break
      default:
        input.placeholder = 'tell the robot what to do...'
    }
  }
  _refreshHint()
  setInterval(_refreshHint, 1000)

  // Build + start the RL dashboard
  _buildRLDashboard()
  setInterval(_tickRLDashboard, 100)

  console.log('UI ready')
}

// ─── Thought sidebar ──────────────────────────────────────────────────────────

export function showThoughts(thoughtArray) {
  const sidebar = document.getElementById('thought-sidebar')
  const list    = document.getElementById('thought-list')
  if (!sidebar || !list) return
  clearThoughts()
  sidebar.classList.add('visible')
  thoughtArray.forEach((text, i) => {
    const el       = document.createElement('div')
    el.className   = 'thought-step'
    el.textContent = text
    list.appendChild(el)
    setTimeout(() => el.classList.add('active'), i * 600)
  })
}

export function clearThoughts() {
  const sidebar = document.getElementById('thought-sidebar')
  const list    = document.getElementById('thought-list')
  if (sidebar) sidebar.classList.remove('visible')
  if (list)    list.innerHTML = ''
}

export function setStatus(text) {
  const el = document.getElementById('status-bar')
  if (el) el.textContent = text
}

export function setAgentStatus(text, type = 'thinking') {
  const el     = document.getElementById('agent-status')
  const textEl = document.getElementById('agent-status-text')
  if (!el || !textEl) return
  if (!text) { el.classList.remove('visible'); return }
  el.classList.remove('thinking', 'navigating', 'scanning', 'error', 'success')
  el.classList.add(type)
  textEl.textContent = text
  el.classList.add('visible')
}

// ─── RL Dashboard ─────────────────────────────────────────────────────────────

function _buildRLDashboard() {
  const panel = document.getElementById('rl-dashboard')
  if (!panel) return

  panel.innerHTML = `
    <div class="rld-header">
      <div class="rld-title-row">
        <span class="rld-icon">🤖</span>
        <span class="rld-title">RL Agent</span>
        <div class="rld-conn-badge" id="rld-conn-badge">OFFLINE</div>
      </div>
      <div class="rld-mode-row">
        <span class="rld-mode-label">MODE</span>
        <span class="rld-mode-val" id="rld-mode">IDLE</span>
      </div>
    </div>

    <div class="rld-section">
      <div class="rld-section-title">ROBOT POSITION</div>
      <div class="rld-kv-row">
        <span class="rld-key">X</span><span class="rld-val" id="rld-rx">—</span>
        <span class="rld-key" style="margin-left:12px">Z</span><span class="rld-val" id="rld-rz">—</span>
      </div>
    </div>

    <div class="rld-section">
      <div class="rld-section-title">GOAL</div>
      <div class="rld-kv-row">
        <span class="rld-key">X</span><span class="rld-val" id="rld-gx">—</span>
        <span class="rld-key" style="margin-left:12px">Z</span><span class="rld-val" id="rld-gz">—</span>
        <span class="rld-key" style="margin-left:12px">DIST</span><span class="rld-val" id="rld-dist">—</span>
      </div>
      <div class="rld-goal-inputs">
        <input class="rld-coord-input" id="rld-goal-x" type="number" step="0.1" min="-2.8" max="2.8" placeholder="x" />
        <input class="rld-coord-input" id="rld-goal-z" type="number" step="0.1" min="-2.8" max="2.8" placeholder="z" />
        <button class="rld-btn" id="rld-set-goal">Set Goal</button>
      </div>
    </div>

    <div class="rld-section">
      <div class="rld-section-title">LAST ACTION</div>
      <div class="rld-kv-row">
        <span class="rld-key">LIN</span><span class="rld-val" id="rld-lin">0.00</span>
        <span class="rld-key" style="margin-left:8px">ROT</span><span class="rld-val" id="rld-ang">0.00</span>
        <span class="rld-key" style="margin-left:8px">ARM</span><span class="rld-val" id="rld-arm">0.00</span>
        <span class="rld-key" style="margin-left:8px">JMP</span><span class="rld-val" id="rld-jmp">0</span>
      </div>
      <div class="rld-action-bars">
        <div class="rld-bar-wrap">
          <div class="rld-bar-label">fwd</div>
          <div class="rld-bar-track"><div class="rld-bar-fill rld-bar-linear" id="rld-bar-lin"></div></div>
        </div>
        <div class="rld-bar-wrap">
          <div class="rld-bar-label">rot</div>
          <div class="rld-bar-track rld-bar-center">
            <div class="rld-bar-fill rld-bar-angular" id="rld-bar-ang"></div>
          </div>
        </div>
        <div class="rld-bar-wrap">
          <div class="rld-bar-label">arm</div>
          <div class="rld-bar-track rld-bar-center">
            <div class="rld-bar-fill" id="rld-bar-arm" style="background:#c084fc"></div>
          </div>
        </div>
      </div>
    </div>

    <div class="rld-section">
      <div class="rld-section-title">LIDAR (11 rays · 165° FOV)</div>
      <canvas id="rld-lidar-canvas" width="220" height="80"></canvas>
    </div>

    <div class="rld-section">
      <div class="rld-section-title">EPISODE</div>
      <div class="rld-kv-row">
        <span class="rld-key">EP #</span><span class="rld-val" id="rld-ep">0</span>
        <span class="rld-key" style="margin-left:8px">STEP</span><span class="rld-val" id="rld-step">0</span>
        <span class="rld-key" style="margin-left:8px">REWARD</span><span class="rld-val" id="rld-reward">—</span>
      </div>
      <div class="rld-step-bar-wrap">
        <div class="rld-step-bar-track"><div class="rld-step-bar-fill" id="rld-step-fill"></div></div>
      </div>
    </div>

    <div class="rld-section">
      <div class="rld-section-title">SIMULATION OF DEATH</div>
      <div class="rld-kv-row">
        <span class="rld-key">☠ DEATHS</span><span class="rld-val" id="rld-deaths" style="color:#f87171">0</span>
        <span class="rld-key" style="margin-left:8px">✓ WIN</span><span class="rld-val" id="rld-succ" style="color:#4ade80">0</span>
        <span class="rld-key" style="margin-left:8px">RATE</span><span class="rld-val" id="rld-rate">—</span>
      </div>
      <div class="rld-kv-row" style="margin-top:4px">
        <span class="rld-key">EP REWARD</span><span class="rld-val" id="rld-ep-reward">0.000</span>
        <span class="rld-key" style="margin-left:8px">TOTAL STEPS</span><span class="rld-val" id="rld-total">0</span>
      </div>
    </div>

    <div class="rld-section">
      <div class="rld-section-title">ENV PARAMS</div>
      <div class="rld-param-row">
        <label class="rld-param-label">Max Steps</label>
        <input class="rld-param-input" id="rld-param-maxsteps" type="number" value="1000" min="50" max="5000" step="50" />
        <button class="rld-btn rld-btn-sm" id="rld-apply-params">Apply</button>
      </div>
    </div>

    <div class="rld-section rld-controls">
      <div class="rld-section-title">QUICK CONTROLS</div>
      <div class="rld-btn-row">
        <button class="rld-btn" id="rld-btn-reset">⟳ Reset</button>
        <button class="rld-btn rld-btn-rl" id="rld-btn-mode-rl">→ RL Mode</button>
        <button class="rld-btn" id="rld-btn-mode-ai">→ AI Mode</button>
      </div>
    </div>
  `

  // Wire up controls
  document.getElementById('rld-set-goal').addEventListener('click', () => {
    const x = parseFloat(document.getElementById('rld-goal-x').value)
    const z = parseFloat(document.getElementById('rld-goal-z').value)
    if (!isNaN(x) && !isNaN(z)) {
      setRLGoalOverride(x, z)
      setStatus(`🎯 Goal override → (${x.toFixed(2)}, ${z.toFixed(2)})`)
    }
  })

  document.getElementById('rld-apply-params').addEventListener('click', () => {
    const ms = parseInt(document.getElementById('rld-param-maxsteps').value)
    if (!isNaN(ms) && ms > 0) {
      setRLParams({ max_steps: ms })
      setStatus(`⚙ Max steps set to ${ms}`)
    }
  })

  document.getElementById('rld-btn-reset').addEventListener('click', () => {
    import('./rl.js').then(m => {
      if (typeof m.sendPromptRL === 'function') {
        // signal reset via WebSocket param
        setRLParams({ reset: true })
        setStatus('↺ Reset sent to agent')
      }
    })
  })

  document.getElementById('rld-btn-mode-rl').addEventListener('click', () => {
    state.controlMode = 'rl'
    setStatus('🤖 Switched to RL mode')
  })

  document.getElementById('rld-btn-mode-ai').addEventListener('click', () => {
    state.controlMode = 'ai'
    setStatus('🧠 Switched to AI mode')
  })
}

let _lastLidar = Array(11).fill(5.0)

function _tickRLDashboard() {
  const t = getRLTelemetry()
  const connected = isRLConnected()

  // Connection badge
  const badge = document.getElementById('rld-conn-badge')
  if (badge) {
    badge.textContent  = connected ? 'ONLINE' : 'OFFLINE'
    badge.className    = 'rld-conn-badge ' + (connected ? 'rld-conn-online' : '')
  }

  // Mode
  const modeEl = document.getElementById('rld-mode')
  if (modeEl) {
    modeEl.textContent = t.mode
    modeEl.style.color = t.mode === 'EXECUTING' ? '#4ade80' : '#888'
  }

  // Robot pos
  _setText('rld-rx', t.robotPos.x?.toFixed(3) ?? '—')
  _setText('rld-rz', t.robotPos.z?.toFixed(3) ?? '—')

  // Goal
  _setText('rld-gx', t.goal.x !== null ? t.goal.x.toFixed(3) : '—')
  _setText('rld-gz', t.goal.z !== null ? t.goal.z.toFixed(3) : '—')
  _setText('rld-dist', t.distToGoal !== null ? t.distToGoal.toFixed(2) + ' m' : '—')

  // Last action — 4-dim
  const lin = t.lastAction.linear ?? 0
  const ang = t.lastAction.angular ?? 0
  const arm = t.lastAction.armRot ?? 0
  const jmp = t.lastAction.jump ?? 0
  _setText('rld-lin', lin.toFixed(2))
  _setText('rld-ang', ang.toFixed(2))
  _setText('rld-arm', arm.toFixed(2))
  _setText('rld-jmp', jmp > 0.5 ? '↑' : '—')

  const barLin = document.getElementById('rld-bar-lin')
  if (barLin) barLin.style.width = (Math.abs(lin) / 2.5 * 100).toFixed(1) + '%'

  const barAng = document.getElementById('rld-bar-ang')
  if (barAng) {
    const pct = (ang / 2.0 + 1) / 2
    barAng.style.left  = pct >= 0.5 ? '50%' : (pct * 100).toFixed(1) + '%'
    barAng.style.width = Math.abs(pct - 0.5) * 100 + '%'
    barAng.style.background = ang >= 0 ? '#f59e0b' : '#818cf8'
  }

  const barArm = document.getElementById('rld-bar-arm')
  if (barArm) {
    const pct = (arm / Math.PI + 1) / 2
    barArm.style.left  = pct >= 0.5 ? '50%' : (pct * 100).toFixed(1) + '%'
    barArm.style.width = Math.abs(pct - 0.5) * 100 + '%'
  }

  // Episode stats
  _setText('rld-ep',    t.episode)
  _setText('rld-step',  t.stepCount)
  _setText('rld-total', t.totalSteps)

  const rewardEl = document.getElementById('rld-reward')
  if (rewardEl) {
    const r = t.lastReward
    rewardEl.textContent = r !== undefined ? (r > 0 ? '+' : '') + r.toFixed(3) : '—'
    rewardEl.style.color = r > 0 ? '#4ade80' : r < 0 ? '#f87171' : '#888'
  }

  // Simulation of Death stats
  _setText('rld-deaths', t.deaths)
  _setText('rld-succ',   t.successes)
  const total = t.deaths + t.successes
  const rateEl = document.getElementById('rld-rate')
  if (rateEl) {
    if (total > 0) {
      const rate = (t.successes / total * 100).toFixed(0) + '%'
      rateEl.textContent  = rate
      rateEl.style.color  = t.successes / total > 0.5 ? '#4ade80' : t.successes / total > 0.2 ? '#f59e0b' : '#f87171'
    } else {
      rateEl.textContent = '—'
    }
  }

  const epRewEl = document.getElementById('rld-ep-reward')
  if (epRewEl) {
    const er = t.epReward ?? 0
    epRewEl.textContent = (er > 0 ? '+' : '') + er.toFixed(2)
    epRewEl.style.color = er > 0 ? '#4ade80' : er < -5 ? '#f87171' : '#888'
  }

  // Step progress bar
  const fill = document.getElementById('rld-step-fill')
  if (fill) {
    const pct = Math.min(t.stepCount / t.maxSteps, 1) * 100
    fill.style.width = pct.toFixed(1) + '%'
    fill.style.background = pct > 80 ? '#f87171' : pct > 50 ? '#f59e0b' : '#4ade80'
  }

  // Lidar radar
  _drawLidar(t.lastLidar)

  // Sync max_steps input if changed externally
  const msEl = document.getElementById('rld-param-maxsteps')
  if (msEl && document.activeElement !== msEl) {
    msEl.value = t.maxSteps
  }
}

function _setText(id, val) {
  const el = document.getElementById(id)
  if (el) el.textContent = val
}

function _drawLidar(readings) {
  const canvas = document.getElementById('rld-lidar-canvas')
  if (!canvas) return
  const ctx = canvas.getContext('2d')
  const W = canvas.width, H = canvas.height
  ctx.clearRect(0, 0, W, H)

  const N        = readings.length
  const MAX_DIST = 5.0
  const FOV      = 165 * Math.PI / 180
  const cx       = W / 2
  const cy       = H - 6
  const R        = H - 12
  const halfFov  = FOV / 2

  // Background arc
  ctx.beginPath()
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'
  ctx.lineWidth   = 1
  for (let r = 0.25; r <= 1; r += 0.25) {
    ctx.beginPath()
    ctx.arc(cx, cy, R * r, Math.PI + halfFov, -halfFov, false)
    ctx.stroke()
  }

  // Spokes
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'
  for (let i = 0; i < N; i++) {
    const angle = Math.PI + halfFov - (i / (N - 1)) * FOV
    ctx.beginPath()
    ctx.moveTo(cx, cy)
    ctx.lineTo(cx + Math.cos(angle) * R, cy + Math.sin(angle) * R)
    ctx.stroke()
  }

  // Lidar fill polygon
  ctx.beginPath()
  for (let i = 0; i < N; i++) {
    const d     = Math.min(readings[i] ?? MAX_DIST, MAX_DIST)
    const frac  = d / MAX_DIST
    const angle = Math.PI + halfFov - (i / (N - 1)) * FOV
    const px    = cx + Math.cos(angle) * R * frac
    const py    = cy + Math.sin(angle) * R * frac
    i === 0 ? ctx.moveTo(px, py) : ctx.lineTo(px, py)
  }
  ctx.closePath()
  const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, R)
  grad.addColorStop(0, 'rgba(56, 189, 248, 0.35)')
  grad.addColorStop(1, 'rgba(56, 189, 248, 0.05)')
  ctx.fillStyle = grad
  ctx.fill()
  ctx.strokeStyle = '#38bdf8'
  ctx.lineWidth   = 1.5
  ctx.stroke()

  // Dots at each reading
  for (let i = 0; i < N; i++) {
    const d     = Math.min(readings[i] ?? MAX_DIST, MAX_DIST)
    const frac  = d / MAX_DIST
    const angle = Math.PI + halfFov - (i / (N - 1)) * FOV
    const px    = cx + Math.cos(angle) * R * frac
    const py    = cy + Math.sin(angle) * R * frac
    ctx.beginPath()
    ctx.arc(px, py, 2, 0, Math.PI * 2)
    const danger = d < 0.5 ? '#f87171' : d < 1.5 ? '#fbbf24' : '#38bdf8'
    ctx.fillStyle = danger
    ctx.fill()
  }

  // Robot dot
  ctx.beginPath()
  ctx.arc(cx, cy, 4, 0, Math.PI * 2)
  ctx.fillStyle = '#a78bfa'
  ctx.fill()
}