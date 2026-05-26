/**
 * rlPanel.js — Training progress UI
 *
 * Injects a floating panel into the existing #ui div.
 * Shows: episode counter, epsilon, success rate, mini reward sparkline.
 * No external deps — pure DOM + Canvas.
 */

let _panel     = null
let _canvas    = null
let _ctx       = null
let _animFrame = null

// ─── Init ─────────────────────────────────────────────────────────────────────

export function initRLPanel() {
  if (_panel) return

  _panel = document.createElement('div')
  _panel.id = 'rl-panel'
  _panel.style.cssText = `
    position: absolute;
    bottom: 80px;
    right: 16px;
    width: 220px;
    background: rgba(0,0,0,0.82);
    border: 1px solid rgba(250,199,117,0.3);
    border-radius: 10px;
    padding: 12px 14px 10px;
    font-family: monospace;
    font-size: 11px;
    color: #aaa;
    display: none;
    flex-direction: column;
    gap: 7px;
    pointer-events: none;
    backdrop-filter: blur(6px);
    z-index: 200;
  `

  _panel.innerHTML = `
    <div style="color:#FAC775;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;
                display:flex;align-items:center;gap:6px;">
      <span id="rl-dot" style="width:6px;height:6px;border-radius:50%;background:#FAC775;
            box-shadow:0 0 5px #FAC775;display:inline-block;"></span>
      RL Training
    </div>
    <div id="rl-task"  style="color:#fff;font-size:12px;"></div>
    <div id="rl-stats" style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;color:#888;"></div>
    <canvas id="rl-canvas" width="192" height="48"
      style="border-radius:4px;background:rgba(255,255,255,0.03);margin-top:2px;"></canvas>
    <div id="rl-result" style="color:#1D9E75;font-size:11px;display:none;"></div>
  `

  document.getElementById('ui').appendChild(_panel)

  _canvas = document.getElementById('rl-canvas')
  _ctx    = _canvas.getContext('2d')
}

// ─── Show / hide ──────────────────────────────────────────────────────────────

export function showRLPanel(taskName) {
  initRLPanel()
  _panel.style.display = 'flex'
  document.getElementById('rl-task').textContent = taskName
  document.getElementById('rl-result').style.display = 'none'
  _startPulse()
}

export function hideRLPanel() {
  if (!_panel) return
  _stopPulse()
  setTimeout(() => {
    if (_panel) _panel.style.display = 'none'
  }, 3000)
}

// ─── Update ───────────────────────────────────────────────────────────────────

/**
 * Call this on each progress tick.
 * @param {object} progress - from getTrainingProgress()
 * @param {number[]} rewardHistory
 */
export function updateRLPanel(progress, rewardHistory) {
  if (!_panel) return

  const stats = document.getElementById('rl-stats')
  const pct   = Math.round((progress.episode / progress.total) * 100)

  stats.innerHTML = `
    <span style="color:#666">Episode</span>
    <span style="color:#ccc">${progress.episode}/${progress.total}</span>
    <span style="color:#666">Success</span>
    <span style="color:#1D9E75">${progress.successCount}</span>
    <span style="color:#666">Explore ε</span>
    <span style="color:#FAC775">${progress.epsilon}</span>
    <span style="color:#666">Avg reward</span>
    <span style="color:#AFA9EC">${progress.avgReward}</span>
  `

  _drawSparkline(rewardHistory)
}

export function showRLResult(successRate) {
  if (!_panel) return
  const el = document.getElementById('rl-result')
  if (!el) return
  const color = successRate > 60 ? '#1D9E75' : successRate > 30 ? '#FAC775' : '#E24B4A'
  el.style.color   = color
  el.style.display = 'block'
  el.textContent   = `✓ Done — ${successRate}% success rate. Skill saved.`
  _stopPulse()
  document.getElementById('rl-dot').style.background    = color
  document.getElementById('rl-dot').style.boxShadow     = `0 0 5px ${color}`
  document.getElementById('rl-dot').style.animation     = 'none'
}

// ─── Sparkline ────────────────────────────────────────────────────────────────

function _drawSparkline(history) {
  if (!_ctx || history.length < 2) return
  const w = _canvas.width, h = _canvas.height
  _ctx.clearRect(0, 0, w, h)

  const vals = history.slice(-60)
  const min  = Math.min(...vals)
  const max  = Math.max(...vals) || 1
  const range = max - min || 1

  _ctx.beginPath()
  _ctx.strokeStyle = '#534AB7'
  _ctx.lineWidth   = 1.5

  vals.forEach((v, i) => {
    const x = (i / (vals.length - 1)) * w
    const y = h - ((v - min) / range) * (h - 4) - 2
    i === 0 ? _ctx.moveTo(x, y) : _ctx.lineTo(x, y)
  })
  _ctx.stroke()

  // Fill under curve
  _ctx.lineTo(w, h)
  _ctx.lineTo(0, h)
  _ctx.closePath()
  _ctx.fillStyle = 'rgba(83,74,183,0.15)'
  _ctx.fill()
}

// ─── Dot pulse animation ──────────────────────────────────────────────────────

function _startPulse() {
  const dot = document.getElementById('rl-dot')
  if (dot) {
    dot.style.animation = 'none'
    dot.style.background = '#FAC775'
    dot.style.boxShadow  = '0 0 5px #FAC775'
    // Inject keyframe if not already there
    if (!document.getElementById('rl-pulse-style')) {
      const s = document.createElement('style')
      s.id = 'rl-pulse-style'
      s.textContent = `
        @keyframes rl-pulse {
          0%,100% { opacity:1; transform:scale(1); }
          50%      { opacity:0.4; transform:scale(1.4); }
        }
      `
      document.head.appendChild(s)
    }
    dot.style.animation = 'rl-pulse 1s ease-in-out infinite'
  }
}

// ─── Dot pulse helper ─────────────────────────────────────────────────────────

function _stopPulse() {
  const dot = document.getElementById('rl-dot')
  if (dot) dot.style.animation = 'none'
  cancelAnimationFrame(_animFrame)
}
