import { state } from './state.js'
import { handleInstruction } from './executor.js'

export function initUI() {
  const input = document.getElementById('instruction')

  input.addEventListener('keydown', async (e) => {
    if (e.key !== 'Enter') return
    const text = input.value.trim()
    if (!text) return
    if (state.execution.running) return

    input.value = ''
    input.disabled = true
    await handleInstruction(text)
    input.disabled = false
    input.focus()
  })

  // Keyboard shortcuts
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      state.execution.running = false
      state.robot.status = 'idle'
      document.getElementById('status-bar').textContent = 'Cancelled.'
    }
  })

  console.log('UI ready')
}

export function showThoughts(thoughtArray) {
  const sidebar = document.getElementById('thought-sidebar')
  const list = document.getElementById('thought-list')
  if (!sidebar || !list) return

  clearThoughts()
  sidebar.classList.add('visible')

  thoughtArray.forEach((text, i) => {
    const el = document.createElement('div')
    el.className = 'thought-step'
    el.textContent = text
    list.appendChild(el)

    // Staggered animation
    setTimeout(() => {
      el.classList.add('active')
    }, i * 600)
  })
}

export function clearThoughts() {
  const sidebar = document.getElementById('thought-sidebar')
  const list = document.getElementById('thought-list')
  if (sidebar) sidebar.classList.remove('visible')
  if (list) list.innerHTML = ''
}