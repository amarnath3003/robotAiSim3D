import { state } from './state.js'
import { handleInstruction } from './executor.js'
import { isRLConnected, sendPromptRL } from './rl.js'

export function initUI() {
  const input = document.getElementById('instruction')

  input.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return
    const text = input.value.trim()
    if (!text) return
    input.value = ''

    const mode = state.controlMode

    if (mode === 'rl') {
      // ── RL mode: forward to Python ─────────────────────────────────────
      sendPromptRL(text)
      return
    }

    // ── AI mode: LLM instruction ───────────────────────────────────────────
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

  // Refresh placeholder every second to match current mode
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

  if (!text) {
    el.classList.remove('visible')
    return
  }

  el.classList.remove('thinking', 'navigating', 'scanning', 'error', 'success')
  el.classList.add(type)
  textEl.textContent = text
  el.classList.add('visible')
}