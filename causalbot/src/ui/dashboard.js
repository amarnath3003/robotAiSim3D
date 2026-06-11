/**
 * ui/dashboard.js — HUD Overlay for CausalBot
 * 
 * Creates and manages the in-browser UI overlay:
 * - Robot status (idle, executing, thinking)
 * - Perception readout (objects detected, confidence)
 * - Current plan / skill being executed
 * - Skill inventory panel
 * - FPS / physics stats
 * - RL episode info (if connected)
 * 
 * All DOM is created programmatically — no external HTML dependencies.
 * The dashboard subscribes to state changes reactively.
 */

// XM-3: removed getRobotStatus (not used — status comes via subscribe callback)
//        removed isRLConnected (not used — RL status comes via subscribe callback)
import { subscribe, getState, getKnownObjects } from '../core/state.js'
import { getFPS } from '../core/engine.js'
import { skillRegistry } from '../skills/registry.js'
import { episodicMemory } from '../brain/episodic_memory.js'
import { worldModel } from '../brain/world_model.js'

// ─── State ─────────────────────────────────────────────────────────────────────

let dashboardEl = null
let panels = {}
let updateInterval = null
// UI-2: store unsubscribe functions so destroyDashboard() can remove them
//        and re-init doesn't accumulate duplicate subscriptions
let _unsubscribers = []

// Execution log ring buffer (max 10 entries shown)
const MAX_LOG_ENTRIES = 10
let _executionLog = []

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the dashboard overlay.
 * Creates all DOM elements and starts reactive updates.
 * 
 * @param {Object} options
 * @param {string} [options.position='top-right'] - Where to anchor the dashboard
 * @param {boolean} [options.collapsed=false] - Start collapsed
 * @param {boolean} [options.showPerception=true] - Show perception panel
 * @param {boolean} [options.showSkills=true] - Show skills panel
 * @param {boolean} [options.showStats=true] - Show stats panel
 */
export function initDashboard(options = {}) {
  const {
    position = 'top-right',
    collapsed = false,
    showPerception = true,
    showSkills = true,
    showStats = true,
    showBrain = true,
    showMemory = true,
  } = options

  // Create root container
  dashboardEl = document.createElement('div')
  dashboardEl.id = 'cb-dashboard'
  dashboardEl.className = `cb-dashboard cb-pos-${position}`
  if (collapsed) dashboardEl.classList.add('cb-collapsed')

  // Inject styles
  injectStyles()

  // Build panels
  panels.status = createPanel('Status', buildStatusPanel())
  dashboardEl.appendChild(panels.status.el)

  if (showBrain) {
    panels.brain = createPanel('Brain', buildBrainPanel())
    dashboardEl.appendChild(panels.brain.el)
  }

  if (showMemory) {
    panels.memory = createPanel('Memory', buildMemoryPanel())
    dashboardEl.appendChild(panels.memory.el)
  }

  if (showPerception) {
    panels.perception = createPanel('Perception', buildPerceptionPanel())
    dashboardEl.appendChild(panels.perception.el)
  }

  if (showSkills) {
    panels.skills = createPanel('Skills', buildSkillsPanel())
    dashboardEl.appendChild(panels.skills.el)
  }

  if (showStats) {
    panels.stats = createPanel('Stats', buildStatsPanel())
    dashboardEl.appendChild(panels.stats.el)
  }

  // Collapse toggle button
  const toggle = document.createElement('button')
  toggle.className = 'cb-toggle'
  toggle.textContent = '◀'
  toggle.onclick = () => {
    dashboardEl.classList.toggle('cb-collapsed')
    toggle.textContent = dashboardEl.classList.contains('cb-collapsed') ? '▶' : '◀'
  }
  dashboardEl.prepend(toggle)

  document.body.appendChild(dashboardEl)

  // Start update loop (for non-reactive data like FPS)
  updateInterval = setInterval(tickUpdate, 250)

  // UI-2: store unsubscribe handles; re-init won't duplicate subscriptions
  _unsubscribers.push(subscribe('robot.status', (status) => {
    updateStatusDisplay(status)
  }))

  _unsubscribers.push(subscribe('robot.task', (task) => {
    const el = document.getElementById('cb-status-task')
    if (el) el.textContent = task || '—'
  }))

  _unsubscribers.push(subscribe('robot.step', (step) => {
    const el = document.getElementById('cb-status-step')
    if (el) el.textContent = step || '—'
  }))

  _unsubscribers.push(subscribe('perception.memory', () => {
    updatePerceptionDisplay()
  }))

  _unsubscribers.push(subscribe('rl.connected', (connected) => {
    updateRLDisplay(connected)
  }))

  // Populate skills panel once (registry is already init'd before dashboard)
  updateSkillsDisplay()

  console.log('[Dashboard] Initialized')
}

/**
 * Destroy the dashboard and clean up.
 */
export function destroyDashboard() {
  if (updateInterval) clearInterval(updateInterval)
  // UI-2: call stored unsubscribers so re-init starts with a clean slate
  _unsubscribers.forEach(fn => fn?.())
  _unsubscribers = []
  if (dashboardEl) dashboardEl.remove()
  dashboardEl = null
  panels = {}
}

/**
 * Programmatically show a temporary notification on the HUD.
 * 
 * @param {string} message - Text to display
 * @param {'info'|'success'|'warning'|'error'} type - Visual style
 * @param {number} duration - How long to show (ms)
 */
export function showNotification(message, type = 'info', duration = 3000) {
  if (!dashboardEl) return

  const note = document.createElement('div')
  note.className = `cb-notification cb-note-${type}`
  note.textContent = message
  dashboardEl.appendChild(note)

  // Animate in
  requestAnimationFrame(() => note.classList.add('cb-note-visible'))

  setTimeout(() => {
    note.classList.remove('cb-note-visible')
    setTimeout(() => note.remove(), 300)
  }, duration)
}

/**
 * Display streaming LLM reasoning text in the Brain panel.
 * Call from planner's `_onThinking` callback.
 * Appends a single character at a time if the text ends with a new token.
 *
 * @param {string} text - Latest thinking text (full string, not delta)
 */
export function updateThinkingPanel(text) {
  const el = document.getElementById('cb-brain-thinking')
  if (!el) return
  el.textContent = text || ''
  // Auto-scroll to bottom so latest tokens are visible
  el.scrollTop = el.scrollHeight
}

/**
 * Add an entry to the execution log in the Brain panel.
 * @param {string} skill   - Skill name that was executed
 * @param {'success'|'failure'|'skip'} status
 * @param {string} [detail] - Optional detail text
 */
export function addExecutionLog(skill, status, detail = '') {
  const timestamp = new Date().toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  _executionLog.push({ skill, status, detail, timestamp })

  // Keep ring buffer
  if (_executionLog.length > MAX_LOG_ENTRIES) {
    _executionLog.shift()
  }

  _renderExecutionLog()
}

// ─── Panel Builders ────────────────────────────────────────────────────────────

function buildStatusPanel() {
  const container = document.createElement('div')
  container.innerHTML = `
    <div class="cb-row">
      <span class="cb-label">State:</span>
      <span class="cb-value" id="cb-status-state">Initializing</span>
    </div>
    <div class="cb-row">
      <span class="cb-label">Task:</span>
      <span class="cb-value cb-truncate" id="cb-status-task">—</span>
    </div>
    <div class="cb-row">
      <span class="cb-label">Step:</span>
      <span class="cb-value" id="cb-status-step">—</span>
    </div>
    <div class="cb-row cb-rl-row" style="display:none">
      <span class="cb-label">RL:</span>
      <span class="cb-value" id="cb-status-rl">Disconnected</span>
    </div>
  `
  return container
}

function buildBrainPanel() {
  const container = document.createElement('div')
  container.innerHTML = `
    <div class="cb-brain-thinking" id="cb-brain-thinking">—</div>
    <div class="cb-brain-log-header">Execution Log</div>
    <div id="cb-brain-log" class="cb-brain-log"></div>
  `
  return container
}

function buildMemoryPanel() {
  const container = document.createElement('div')
  container.innerHTML = `
    <div class="cb-row">
      <span class="cb-label">Episodes:</span>
      <span class="cb-value" id="cb-mem-count">0</span>
    </div>
    <div class="cb-row">
      <span class="cb-label">Explored:</span>
      <span class="cb-value" id="cb-mem-explored">0%</span>
    </div>
    <div class="cb-row">
      <span class="cb-label">Affordances:</span>
      <span class="cb-value" id="cb-mem-afford">0</span>
    </div>
    <div class="cb-mem-lessons-header">Recent Lessons</div>
    <div id="cb-mem-lessons" class="cb-mem-lessons"></div>
  `
  return container
}

function buildPerceptionPanel() {
  const container = document.createElement('div')
  container.innerHTML = `
    <div class="cb-row">
      <span class="cb-label">Objects:</span>
      <span class="cb-value" id="cb-perc-count">0</span>
    </div>
    <div id="cb-perc-list" class="cb-list"></div>
  `
  return container
}

function buildSkillsPanel() {
  const container = document.createElement('div')
  container.innerHTML = `
    <div class="cb-row">
      <span class="cb-label">Loaded:</span>
      <span class="cb-value" id="cb-skills-count">0</span>
    </div>
    <div id="cb-skills-list" class="cb-list"></div>
  `
  return container
}

function buildStatsPanel() {
  const container = document.createElement('div')
  container.innerHTML = `
    <div class="cb-row">
      <span class="cb-label">FPS:</span>
      <span class="cb-value" id="cb-stats-fps">—</span>
    </div>
    <div class="cb-row">
      <span class="cb-label">Pos:</span>
      <span class="cb-value" id="cb-stats-pos">—</span>
    </div>
    <div class="cb-row">
      <span class="cb-label">Heading:</span>
      <span class="cb-value" id="cb-stats-heading">—</span>
    </div>
  `
  return container
}

// ─── Update Functions ──────────────────────────────────────────────────────────

function tickUpdate() {
  // FPS
  const fpsEl = document.getElementById('cb-stats-fps')
  if (fpsEl) fpsEl.textContent = getFPS().toFixed(0)

  // Position
  const pos = getState('robot.position')
  const posEl = document.getElementById('cb-stats-pos')
  if (posEl && pos) {
    posEl.textContent = `(${pos.x?.toFixed(1)}, ${pos.y?.toFixed(1)}, ${pos.z?.toFixed(1)})`
  }

  // Heading
  const heading = getState('robot.heading')
  const headEl = document.getElementById('cb-stats-heading')
  if (headEl && heading != null) {
    headEl.textContent = `${(heading * 180 / Math.PI).toFixed(0)}°`
  }

  // Skills (refreshes whenever new skills are synthesized or approved)
  updateSkillsDisplay()

  // Memory panel (episode count, exploration %, affordances, recent lessons)
  updateMemoryDisplay()
}

function updateStatusDisplay(status) {
  const el = document.getElementById('cb-status-state')
  if (el) {
    el.textContent = status || 'Idle'
    el.className = `cb-value cb-state-${(status || 'idle').toLowerCase()}`
  }
}

function updatePerceptionDisplay() {
  const objects = getKnownObjects()
  const countEl = document.getElementById('cb-perc-count')
  const listEl = document.getElementById('cb-perc-list')

  if (countEl) countEl.textContent = objects.length

  if (listEl) {
    listEl.innerHTML = objects.slice(0, 8).map(obj => `
      <div class="cb-list-item">
        <span class="cb-obj-type">${obj.id || '?'}</span>
        <span class="cb-obj-conf">${(obj.confidence * 100).toFixed(0)}%</span>
      </div>
    `).join('')
    // NG-4: getKnownObjects() returns {id, position, confidence} — no `type` field

    if (objects.length > 8) {
      listEl.innerHTML += `<div class="cb-list-more">+${objects.length - 8} more</div>`
    }
  }
}

function updateRLDisplay(connected) {
  const row = document.querySelector('.cb-rl-row')
  const el = document.getElementById('cb-status-rl')
  if (row) row.style.display = connected ? 'flex' : 'none'
  if (el) {
    el.textContent = connected ? 'Connected' : 'Disconnected'
    el.className = `cb-value ${connected ? 'cb-state-executing' : ''}`
  }
}

function _renderExecutionLog() {
  const logEl = document.getElementById('cb-brain-log')
  if (!logEl) return

  logEl.innerHTML = _executionLog.map(entry => {
    const cls = entry.status === 'success' ? 'cb-log-success'
               : entry.status === 'failure' ? 'cb-log-failure'
               : 'cb-log-skip'
    const icon = entry.status === 'success' ? '✓' : entry.status === 'failure' ? '✗' : '↷'
    const detail = entry.detail ? ` — ${entry.detail.slice(0, 30)}` : ''
    return `<div class="cb-log-entry ${cls}">
      <span class="cb-log-icon">${icon}</span>
      <span class="cb-log-skill">${entry.skill}</span>
      <span class="cb-log-detail">${detail}</span>
      <span class="cb-log-time">${entry.timestamp}</span>
    </div>`
  }).join('')

  // Scroll to latest
  logEl.scrollTop = logEl.scrollHeight
}

function updateMemoryDisplay() {
  if (!panels.memory) return

  const countEl    = document.getElementById('cb-mem-count')
  const exploredEl = document.getElementById('cb-mem-explored')
  const affordEl   = document.getElementById('cb-mem-afford')
  const lessonsEl  = document.getElementById('cb-mem-lessons')

  if (countEl)    countEl.textContent    = episodicMemory.count
  if (exploredEl) exploredEl.textContent = `${(worldModel.explorationRatio * 100).toFixed(0)}%`
  if (affordEl)   affordEl.textContent   = worldModel.affordanceCount

  if (lessonsEl) {
    const lessons = episodicMemory.getRecentLessons(3)
    if (lessons.length === 0) {
      lessonsEl.innerHTML = '<div class="cb-mem-lesson-empty">No lessons yet</div>'
    } else {
      lessonsEl.innerHTML = lessons.map(l =>
        `<div class="cb-mem-lesson">${l.slice(0, 60)}${l.length > 60 ? '…' : ''}</div>`
      ).join('')
    }
  }
}

function updateSkillsDisplay() {
  if (!skillRegistry._initialized) return

  const countEl = document.getElementById('cb-skills-count')
  const listEl  = document.getElementById('cb-skills-list')
  if (!countEl || !listEl) return

  const names = skillRegistry.getAllNames()
  countEl.textContent = names.length

  // Rebuild whenever count changes OR any skill has been executed since last render
  const execTotal = names.reduce((sum, n) => {
    const s = skillRegistry.get(n)
    return sum + (s?.metadata?.executionCount || 0)
  }, 0)
  const cacheKey = `${names.length}-${execTotal}`
  if (listEl.dataset.cacheKey === cacheKey) return
  listEl.dataset.cacheKey = cacheKey

  listEl.innerHTML = names.slice(0, 12).map(name => {
    const skill = skillRegistry.get(name)
    const badge = skill?.source === 'synthesized'
      ? '<span class="cb-skill-badge cb-badge-synth">LLM</span>'
      : skill?.source === 'learned'
        ? '<span class="cb-skill-badge cb-badge-rl">RL</span>'
        : ''

    // Show execution stats if there are any runs
    const meta = skill?.metadata
    let statsHtml = ''
    if (meta && meta.executionCount > 0) {
      const rate = (meta.successCount / meta.executionCount * 100).toFixed(0)
      const cls  = Number(rate) >= 70 ? 'cb-stat-good' : Number(rate) >= 40 ? 'cb-stat-warn' : 'cb-stat-bad'
      statsHtml = `<span class="cb-skill-stat ${cls}">${meta.executionCount}x ${rate}%</span>`
    }

    return `<div class="cb-list-item">${badge}<span class="cb-obj-type">${name}</span>${statsHtml}</div>`
  }).join('')

  if (names.length > 12) {
    listEl.innerHTML += `<div class="cb-list-more">+${names.length - 12} more</div>`
  }
}

// ─── Helpers ───────────────────────────────────────────────────────────────────

function createPanel(title, content) {
  const panel = document.createElement('div')
  panel.className = 'cb-panel'

  const header = document.createElement('div')
  header.className = 'cb-panel-header'
  header.textContent = title
  header.onclick = () => panel.classList.toggle('cb-panel-closed')

  panel.appendChild(header)
  panel.appendChild(content)

  return { el: panel, content }
}

// ─── Styles ────────────────────────────────────────────────────────────────────

function injectStyles() {
  if (document.getElementById('cb-dashboard-styles')) return

  const style = document.createElement('style')
  style.id = 'cb-dashboard-styles'
  style.textContent = `
    .cb-dashboard {
      position: fixed;
      z-index: 10000;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
      font-size: 11px;
      color: #e0e0e0;
      pointer-events: auto;
      transition: transform 0.3s ease;
      display: flex;
      flex-direction: column;
      gap: 6px;
      max-height: 90vh;
      overflow-y: auto;
    }
    .cb-pos-top-right { top: 10px; right: 10px; }
    .cb-pos-top-left { top: 10px; left: 10px; }
    .cb-pos-bottom-right { bottom: 10px; right: 10px; }
    .cb-dashboard.cb-collapsed { transform: translateX(calc(100% - 24px)); }
    .cb-dashboard.cb-collapsed .cb-panel { display: none; }

    .cb-toggle {
      align-self: flex-start;
      background: rgba(20, 20, 30, 0.85);
      border: 1px solid rgba(100, 200, 255, 0.3);
      color: #7af;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 10px;
      pointer-events: auto;
    }
    .cb-toggle:hover { border-color: #7af; }

    .cb-panel {
      background: rgba(10, 10, 20, 0.88);
      border: 1px solid rgba(100, 200, 255, 0.2);
      border-radius: 6px;
      min-width: 180px;
      overflow: hidden;
      backdrop-filter: blur(8px);
    }
    .cb-panel-header {
      padding: 5px 8px;
      background: rgba(50, 120, 200, 0.15);
      border-bottom: 1px solid rgba(100, 200, 255, 0.15);
      font-weight: 600;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #7af;
      cursor: pointer;
      user-select: none;
    }
    .cb-panel-closed > *:not(.cb-panel-header) { display: none; }

    .cb-row {
      display: flex;
      justify-content: space-between;
      padding: 3px 8px;
      border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .cb-label { color: #888; }
    .cb-value { color: #ddd; font-weight: 500; }
    .cb-truncate { max-width: 110px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

    .cb-state-idle { color: #888; }
    .cb-state-thinking { color: #ffa500; }
    .cb-state-planning { color: #ffa500; }
    .cb-state-executing { color: #4fc3f7; }
    .cb-state-success { color: #66bb6a; }
    .cb-state-error { color: #ef5350; }
    .cb-state-failed { color: #ef5350; }
    .cb-state-manual { color: #ab47bc; }
    /* UI-5: added planning/failed/manual classes to match lowercase state.js values */

    .cb-list { padding: 3px 8px; max-height: 100px; overflow-y: auto; }
    .cb-list-item { display: flex; justify-content: space-between; padding: 1px 0; }
    .cb-obj-type { color: #aaa; }
    .cb-obj-conf { color: #7af; font-size: 10px; }
    .cb-list-more { color: #666; font-style: italic; font-size: 10px; padding-top: 2px; }

    .cb-notification {
      padding: 6px 10px;
      border-radius: 4px;
      opacity: 0;
      transform: translateY(-8px);
      transition: opacity 0.3s, transform 0.3s;
      pointer-events: none;
    }
    .cb-note-visible { opacity: 1; transform: translateY(0); }
    .cb-note-info { background: rgba(33, 150, 243, 0.85); }
    .cb-note-success { background: rgba(76, 175, 80, 0.85); }
    .cb-note-warning { background: rgba(255, 152, 0, 0.85); }
    .cb-note-error { background: rgba(244, 67, 54, 0.85); }

    .cb-skill-badge {
      font-size: 9px;
      padding: 0 4px;
      border-radius: 3px;
      margin-right: 4px;
      font-weight: 700;
      letter-spacing: 0.02em;
    }
    .cb-badge-synth { background: rgba(156, 39, 176, 0.5); color: #ce93d8; }
    .cb-badge-rl    { background: rgba(255, 152, 0, 0.4);  color: #ffcc80; }

    .cb-skill-stat { font-size: 9px; margin-left: auto; padding-left: 4px; }
    .cb-stat-good  { color: #66bb6a; }
    .cb-stat-warn  { color: #ffa726; }
    .cb-stat-bad   { color: #ef5350; }

    /* Brain panel */
    .cb-brain-thinking {
      padding: 4px 8px;
      font-size: 10px;
      color: #b0c4de;
      min-height: 36px;
      max-height: 72px;
      overflow-y: auto;
      white-space: pre-wrap;
      word-break: break-word;
      border-bottom: 1px solid rgba(100,200,255,0.1);
      line-height: 1.4;
    }
    .cb-brain-log-header {
      padding: 3px 8px 1px;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #555;
    }
    .cb-brain-log {
      padding: 2px 6px 4px;
      max-height: 80px;
      overflow-y: auto;
    }
    .cb-log-entry {
      display: flex;
      align-items: center;
      gap: 4px;
      padding: 1px 0;
      font-size: 9.5px;
      border-bottom: 1px solid rgba(255,255,255,0.03);
    }
    .cb-log-icon   { font-size: 9px; flex-shrink: 0; }
    .cb-log-skill  { font-weight: 600; flex-shrink: 0; }
    .cb-log-detail { color: #666; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
    .cb-log-time   { color: #444; flex-shrink: 0; font-size: 8.5px; margin-left: auto; }
    .cb-log-success .cb-log-icon  { color: #66bb6a; }
    .cb-log-success .cb-log-skill { color: #a5d6a7; }
    .cb-log-failure .cb-log-icon  { color: #ef5350; }
    .cb-log-failure .cb-log-skill { color: #ef9a9a; }
    .cb-log-skip    .cb-log-icon  { color: #78909c; }
    .cb-log-skip    .cb-log-skill { color: #90a4ae; }

    /* Memory panel */
    .cb-mem-lessons-header {
      padding: 3px 8px 1px;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #555;
    }
    .cb-mem-lessons {
      padding: 2px 8px 4px;
      max-height: 60px;
      overflow-y: auto;
    }
    .cb-mem-lesson {
      font-size: 9.5px;
      color: #9db4cc;
      padding: 1px 0;
      border-bottom: 1px solid rgba(255,255,255,0.03);
      line-height: 1.4;
    }
    .cb-mem-lesson-empty {
      font-size: 9px;
      color: #444;
      font-style: italic;
    }
  `
  document.head.appendChild(style)
}
