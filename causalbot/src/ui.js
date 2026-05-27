import { state } from './state.js'
import { handleInstruction } from './executor.js'

// ─── Mode label shown in the status bar ───────────────────────────────────────
const MODE_LABELS = {
  ai:    '🧠 AI',
  debug: '🕹 Debug',
  rl:    '🤖 RL',
}

export function initUI() {
  const input = document.getElementById('instruction')

  input.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return
    const text = input.value.trim()
    if (!text) return

    input.value = ''

    if (state.controlMode === 'rl') {
      // In RL mode forward the prompt to Python via WebSocket
      input.disabled = true
      const { sendPromptRL } = await import('./rl.js')
      sendPromptRL(text)
      setStatus('Prompt sent to RL agent.')
      input.disabled = false
      input.focus()
      return
    }

    // AI mode: normal LLM instruction handling
    if (state.execution.running) return
    input.disabled = true
    await handleInstruction(text)
    input.disabled = false
    input.focus()
  })

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      state.execution.running = false
      state.robot.status      = 'idle'
      setStatus('Cancelled.')
    }
  })

  // Update the placeholder to reflect current mode
  _updateInputHint(input)
  // Poll — mode can change at runtime (Python connects / disconnects)
  setInterval(() => _updateInputHint(input), 1000)

  console.log('UI ready')
}

function _updateInputHint(input) {
  if (!input) return
  const mode = state.controlMode
  if (mode === 'rl') {
    input.placeholder = 'send goal to RL agent... (Python must be running)'
  } else if (mode === 'debug') {
    input.placeholder = 'WASD to drive · press 1 for AI mode'
  } else {
    input.placeholder = 'tell the robot what to do...'
  }
}

// ─── Thought sidebar ──────────────────────────────────────────────────────────

export function showThoughts(thoughtArray) {
  const sidebar = document.getElementById('thought-sidebar')
  const list    = document.getElementById('thought-list')
  if (!sidebar || !list) return

  clearThoughts()
  sidebar.classList.add('visible')

  thoughtArray.forEach((text, i) => {
    const el = document.createElement('div')
    el.className  = 'thought-step'
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

// ─── Status helpers ───────────────────────────────────────────────────────────

export function setStatus(text) {
  const el = document.getElementById('status-bar')
  if (el) el.textContent = text
}

export function setAgentStatus(text, type = 'thinking') {
  const el     = document.getElementById('agent-status')
  const textEl = document.getElementById('agent-status-text')
  if (!el || !textEl) return

  if (!text) {
    el.classList.remove('visible')
    return
  }

  el.classList.remove('thinking', 'navigating', 'scanning', 'error', 'success')
  el.classList.add(type)
  textEl.textContent = text
  el.classList.add('visible')
}