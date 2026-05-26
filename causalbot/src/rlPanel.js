/**
 * rlPanel.js — Training progress UI v2
 *
 * Improvements:
 *  - Progress bar (fills as episodes run)
 *  - Phase labels: Exploring → Learning → Refining → Converging → Converged
 *  - Dual sparkline: cumulative reward + binary success rate
 *  - Rolling 20-episode success % shown live
 *  - Consecutive success streak bar
 *  - Best episode reward display
 *  - Smooth slide-in animation
 */

let _panel = null
let _rewardCanvas = null
let _successCanvas = null
let _rCtx = null
let _sCtx = null

// ─── Init ─────────────────────────────────────────────────────────────────────

function _init() {
  if (_panel) return

  _panel = document.createElement('div')
  _panel.id = 'rl-panel'
  _panel.style.cssText = `
    position:absolute; bottom:90px; right:16px; width:240px;
    background:rgba(8,8,12,0.92);
    border:1px solid rgba(250,199,117,0.25);
    border-radius:12px; padding:14px 16px 12px;
    font-family:'Courier New',monospace; font-size:11px; color:#888;
    display:none; flex-direction:column; gap:8px;
    pointer-events:none; backdrop-filter:blur(10px);
    box-shadow:0 8px 32px rgba(0,0,0,0.6);
    z-index:200; opacity:0;
    transition:opacity 0.3s ease, transform 0.3s ease;
    transform:translateX(16px);
  `

  _panel.innerHTML = `
    <div style="display:flex;align-items:center;gap:7px;">
      <span id="rl-dot" style="width:7px;height:7px;border-radius:50%;background:#FAC775;
        box-shadow:0 0 6px #FAC775;display:inline-block;flex-shrink:0;"></span>
      <span style="color:#FAC775;font-size:9.5px;letter-spacing:0.12em;text-transform:uppercase;font-weight:600;">
        RL Training
      </span>
      <span id="rl-phase" style="margin-left:auto;color:#534AB7;font-size:9px;letter-spacing:0.05em;"></span>
    </div>

    <div id="rl-task" style="color:#eee;font-size:12px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;"></div>

    <!-- Progress bar -->
    <div style="height:3px;background:rgba(255,255,255,0.06);border-radius:2px;overflow:hidden;">
      <div id="rl-progressbar" style="height:100%;width:0%;background:linear-gradient(90deg,#534AB7,#FAC775);border-radius:2px;transition:width 0.15s ease;"></div>
    </div>

    <!-- Stats grid -->
    <div id="rl-stats" style="display:grid;grid-template-columns:1fr 1fr;gap:3px 14px;"></div>

    <!-- Streak bar -->
    <div style="display:flex;align-items:center;gap:6px;">
      <span style="color:#555;font-size:9px;min-width:38px;">Streak</span>
      <div style="flex:1;height:4px;background:rgba(255,255,255,0.05);border-radius:2px;overflow:hidden;">
        <div id="rl-streak" style="height:100%;width:0%;background:#1D9E75;border-radius:2px;transition:width 0.2s ease;"></div>
      </div>
      <span id="rl-streak-num" style="color:#1D9E75;font-size:9px;min-width:18px;text-align:right;">0</span>
    </div>

    <!-- Reward sparkline label -->
    <div style="color:#444;font-size:9px;margin-bottom:-4px;">Episode reward</div>
    <canvas id="rl-reward-canvas" width="208" height="40"
      style="border-radius:4px;background:rgba(255,255,255,0.02);"></canvas>

    <!-- Success rate sparkline label -->
    <div style="color:#444;font-size:9px;margin-bottom:-4px;">Rolling success rate (%)</div>
    <canvas id="rl-success-canvas" width="208" height="32"
      style="border-radius:4px;background:rgba(255,255,255,0.02);"></canvas>

    <div id="rl-result" style="display:none;padding:6px 8px;border-radius:6px;font-size:11px;font-weight:600;text-align:center;margin-top:2px;"></div>
  `

  // Inject keyframe styles
  if (!document.getElementById('rl-styles')) {
    const s = document.createElement('style')
    s.id = 'rl-styles'
    s.textContent = `
      @keyframes rl-pulse {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:0.35; transform:scale(1.5); }
      }
      @keyframes rl-slide-in {
        from { opacity:0; transform:translateX(16px); }
        to   { opacity:1; transform:translateX(0); }
      }
    `
    document.head.appendChild(s)
  }

  const ui = document.getElementById('ui')
  if (ui) ui.appendChild(_panel)

  _rewardCanvas  = document.getElementById('rl-reward-canvas')
  _successCanvas = document.getElementById('rl-success-canvas')
  _rCtx = _rewardCanvas?.getContext('2d')
  _sCtx = _successCanvas?.getContext('2d')
}

// ─── Show / Hide ──────────────────────────────────────────────────────────────

export function showRLPanel(taskName) {
  _init()
  _panel.style.display = 'flex'
  // Force reflow then animate
  requestAnimationFrame(() => {
    _panel.style.animation = 'rl-slide-in 0.3s ease forwards'
    _panel.style.opacity   = '1'
    _panel.style.transform = 'translateX(0)'
  })
  const taskEl = document.getElementById('rl-task')
  if (taskEl) taskEl.textContent = taskName
  const resultEl = document.getElementById('rl-result')
  if (resultEl) resultEl.style.display = 'none'
  _startPulse()
}

export function hideRLPanel() {
  if (!_panel) return
  _stopPulse()
  setTimeout(() => {
    if (_panel) {
      _panel.style.opacity   = '0'
      _panel.style.transform = 'translateX(16px)'
      setTimeout(() => { if (_panel) _panel.style.display = 'none' }, 300)
    }
  }, 3500)
}

// ─── Update ───────────────────────────────────────────────────────────────────

export function updateRLPanel(progress, rewardHistory, successHistory) {
  if (!_panel) return

  // Progress bar
  const pct = Math.round((progress.episode / progress.total) * 100)
  const bar = document.getElementById('rl-progressbar')
  if (bar) bar.style.width = pct + '%'

  // Phase label + colour
  const phaseEl = document.getElementById('rl-phase')
  if (phaseEl) {
    phaseEl.textContent = progress.phase
    const phaseColors = {
      'Exploring':   '#666',
      'Learning':    '#FAC775',
      'Refining':    '#AFA9EC',
      'Converging':  '#1D9E75',
      'Converged ✓': '#1D9E75',
    }
    phaseEl.style.color = phaseColors[progress.phase] || '#666'
  }

  // Stats grid
  const stats = document.getElementById('rl-stats')
  if (stats) {
    stats.innerHTML = `
      <span style="color:#444">Episode</span>
      <span style="color:#ccc">${progress.episode}/${progress.total}</span>
      <span style="color:#444">Success</span>
      <span style="color:#1D9E75">${progress.successCount}</span>
      <span style="color:#444">Last 20</span>
      <span style="color:#FAC775">${progress.rate20}%</span>
      <span style="color:#444">Explore ε</span>
      <span style="color:#AFA9EC">${progress.epsilon}</span>
      <span style="color:#444">Best R</span>
      <span style="color:#534AB7">${progress.bestReward ?? '—'}</span>
    `
  }

  // Streak bar
  const streakBar = document.getElementById('rl-streak')
  const streakNum = document.getElementById('rl-streak-num')
  if (streakBar && streakNum) {
    const streakPct = Math.min(100, (progress.consecutiveSucc / 35) * 100)
    streakBar.style.width = streakPct + '%'
    streakNum.textContent = progress.consecutiveSucc
  }

  // Sparklines
  _drawRewardSparkline(rewardHistory)
  _drawSuccessSparkline(successHistory)
}

export function showRLResult(successRate, converged) {
  if (!_panel) return
  const el = document.getElementById('rl-result')
  if (!el) return

  _stopPulse()

  const color = successRate > 65 ? '#1D9E75' : successRate > 30 ? '#FAC775' : '#E24B4A'
  const icon  = successRate > 65 ? '✓' : successRate > 30 ? '⚠' : '✕'
  const label = converged ? ' (early convergence!)' : ''

  el.style.display    = 'block'
  el.style.color      = color
  el.style.background = color + '18'
  el.style.border     = `1px solid ${color}44`
  el.textContent      = `${icon} ${successRate}% success rate${label}`

  const dot = document.getElementById('rl-dot')
  if (dot) {
    dot.style.background  = color
    dot.style.boxShadow   = `0 0 6px ${color}`
    dot.style.animation   = 'none'
  }

  // Fill progress bar to 100%
  const bar = document.getElementById('rl-progressbar')
  if (bar) bar.style.width = '100%'
}

// ─── Sparklines ───────────────────────────────────────────────────────────────

function _drawRewardSparkline(history) {
  if (!_rCtx || !history || history.length < 2) return
  const w = _rewardCanvas.width, h = _rewardCanvas.height
  _rCtx.clearRect(0, 0, w, h)

  const vals  = history.slice(-80)
  const min   = Math.min(...vals)
  const max   = Math.max(...vals)
  const range = (max - min) || 1

  // Gradient fill
  const grad = _rCtx.createLinearGradient(0, 0, 0, h)
  grad.addColorStop(0, 'rgba(83,74,183,0.4)')
  grad.addColorStop(1, 'rgba(83,74,183,0.0)')

  _rCtx.beginPath()
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * w
    const y = h - ((v - min) / range) * (h - 4) - 2
    i === 0 ? _rCtx.moveTo(x, y) : _rCtx.lineTo(x, y)
  })
  // Close path for fill
  _rCtx.lineTo(w, h); _rCtx.lineTo(0, h); _rCtx.closePath()
  _rCtx.fillStyle = grad
  _rCtx.fill()

  // Line on top
  _rCtx.beginPath()
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * w
    const y = h - ((v - min) / range) * (h - 4) - 2
    i === 0 ? _rCtx.moveTo(x, y) : _rCtx.lineTo(x, y)
  })
  _rCtx.strokeStyle = '#7B71E8'
  _rCtx.lineWidth = 1.5
  _rCtx.stroke()
}

function _drawSuccessSparkline(successHistory) {
  if (!_sCtx || !successHistory || successHistory.length < 2) return
  const w = _successCanvas.width, h = _successCanvas.height
  _sCtx.clearRect(0, 0, w, h)

  // Compute rolling 10-ep success rate
  const window = 10
  const rolling = []
  for (let i = 0; i < successHistory.length; i++) {
    const slice = successHistory.slice(Math.max(0, i - window + 1), i + 1)
    rolling.push(slice.reduce((a, b) => a + b, 0) / slice.length * 100)
  }
  const vals = rolling.slice(-80)

  const grad = _sCtx.createLinearGradient(0, 0, 0, h)
  grad.addColorStop(0, 'rgba(29,158,117,0.35)')
  grad.addColorStop(1, 'rgba(29,158,117,0.0)')

  _sCtx.beginPath()
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * w
    const y = h - (v / 100) * (h - 4) - 2
    i === 0 ? _sCtx.moveTo(x, y) : _sCtx.lineTo(x, y)
  })
  _sCtx.lineTo(w, h); _sCtx.lineTo(0, h); _sCtx.closePath()
  _sCtx.fillStyle = grad
  _sCtx.fill()

  _sCtx.beginPath()
  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * w
    const y = h - (v / 100) * (h - 4) - 2
    i === 0 ? _sCtx.moveTo(x, y) : _sCtx.lineTo(x, y)
  })
  _sCtx.strokeStyle = '#1D9E75'
  _sCtx.lineWidth = 1.5
  _sCtx.stroke()

  // 50% reference line
  _sCtx.setLineDash([2, 3])
  _sCtx.strokeStyle = 'rgba(255,255,255,0.08)'
  _sCtx.lineWidth = 1
  _sCtx.beginPath()
  _sCtx.moveTo(0, h / 2)
  _sCtx.lineTo(w, h / 2)
  _sCtx.stroke()
  _sCtx.setLineDash([])
}

// ─── Pulse animation ──────────────────────────────────────────────────────────

function _startPulse() {
  const dot = document.getElementById('rl-dot')
  if (dot) {
    dot.style.background  = '#FAC775'
    dot.style.boxShadow   = '0 0 6px #FAC775'
    dot.style.animation   = 'rl-pulse 1s ease-in-out infinite'
  }
}

function _stopPulse() {
  const dot = document.getElementById('rl-dot')
  if (dot) dot.style.animation = 'none'
}
