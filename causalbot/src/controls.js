import { state } from './state.js'

const keys = {
  w: false,
  a: false,
  s: false,
  d: false,
  space: false
}

export function initControls() {
  window.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase()
    if (key === 'w') keys.w = true
    if (key === 'a') keys.a = true
    if (key === 's') keys.s = true
    if (key === 'd') keys.d = true
    if (key === ' ') {
      keys.space = true
      e.preventDefault() // prevent page scroll
    }

    // Toggle Mode
    if (key === '1') state.controlMode = 'ai'
    if (key === '2') state.controlMode = 'debug'
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
