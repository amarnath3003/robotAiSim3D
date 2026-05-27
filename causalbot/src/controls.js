import { state } from './state.js'

const keys = {
  w: false,
  a: false,
  s: false,
  d: false,
  space: false,
}

export function initControls() {
  window.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase()
    if (key === 'w') keys.w = true
    if (key === 'a') keys.a = true
    if (key === 's') keys.s = true
    if (key === 'd') keys.d = true
    if (key === ' ') { keys.space = true; e.preventDefault() }

    // Mode switching
    // 1 = AI (LLM instruction mode)
    // 2 = Debug (manual WASD control)
    // 3 = RL (Python/Gymnasium bridge)
    if (key === '1') {
      state.controlMode = 'ai'
      console.log('[Controls] Mode → AI')
    }
    if (key === '2') {
      state.controlMode = 'debug'
      console.log('[Controls] Mode → Debug')
    }
    if (key === '3') {
      state.controlMode = 'rl'
      console.log('[Controls] Mode → RL')
    }
  })

  window.addEventListener('keyup', (e) => {
    const key = e.key.toLowerCase()
    if (key === 'w') keys.w = false
    if (key === 'a') keys.a = false
    if (key === 's') keys.s = false
    if (key === 'd') keys.d = false
    if (key === ' ') keys.space = false
  })
}

export function getKeys() {
  return keys
}