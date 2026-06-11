/**
 * ui/controls.js — Input Controls for CausalBot
 * 
 * Creates and manages:
 * - Instruction input bar (text → LLM planner)
 * - Keyboard shortcuts (WASD for manual override, Escape to abort)
 * - Touch controls (mobile joystick for manual override)
 * - Mode indicators
 * 
 * Manual override sends motor commands directly, bypassing the brain.
 * This is useful for testing and for giving the human a "takeover" ability.
 */

import { subscribe, getState, setState } from '../core/state.js'
import { getActiveRobot } from '../core/engine.js'
import { handleInstruction, abortExecution, isExecuting } from '../brain/planner.js'
import { skillRegistry } from '../skills/registry.js'
import { showNotification } from './dashboard.js'

// ─── State ─────────────────────────────────────────────────────────────────────

let controlsEl = null
let inputEl = null
let keysDown = new Set()
let manualOverride = false
let manualInterval = null
// UI-1: store blur handler reference so destroyControls() can remove it
let _blurHandler = null
// UI-5: guard against re-entrant submissions (e.g. Enter key + button click same frame)
let _submitting = false

// Movement config (manual mode)
const MANUAL_SPEED = 1.5          // m/s
const MANUAL_TURN_SPEED = 2.5    // rad/s

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the controls overlay.
 * 
 * @param {Object} options
 * @param {boolean} [options.showInput=true] - Show instruction input bar
 * @param {boolean} [options.enableKeyboard=true] - Enable WASD manual override
 * @param {boolean} [options.enableTouch=false] - Enable mobile touch joystick
 */
export function initControls(options = {}) {
  const {
    showInput = true,
    enableKeyboard = true,
    enableTouch = false,
  } = options

  injectControlStyles()

  // Create control bar container
  controlsEl = document.createElement('div')
  controlsEl.id = 'cb-controls'
  controlsEl.className = 'cb-controls'

  if (showInput) {
    controlsEl.appendChild(buildInputBar())
  }

  document.body.appendChild(controlsEl)

  // Keyboard
  if (enableKeyboard) {
    initKeyboard()
  }

  // Touch (mobile)
  if (enableTouch) {
    initTouch()
  }

  console.log('[Controls] Initialized')
}

/**
 * Destroy controls and clean up event listeners.
 */
export function destroyControls() {
  if (controlsEl) controlsEl.remove()
  if (manualInterval) clearInterval(manualInterval)
  window.removeEventListener('keydown', onKeyDown)
  window.removeEventListener('keyup', onKeyUp)
  // UI-1: remove blur handler using stored reference
  if (_blurHandler) window.removeEventListener('blur', _blurHandler)
  controlsEl = null
  inputEl = null
  _blurHandler = null
}

/**
 * Focus the instruction input programmatically.
 */
export function focusInput() {
  if (inputEl) inputEl.focus()
}

/**
 * Check if manual override is active.
 */
export function isManualMode() {
  return manualOverride
}

// ─── Input Bar ─────────────────────────────────────────────────────────────────

function buildInputBar() {
  const bar = document.createElement('div')
  bar.className = 'cb-input-bar'

  // Status indicator
  const indicator = document.createElement('div')
  indicator.id = 'cb-input-indicator'
  indicator.className = 'cb-indicator cb-ind-idle'
  bar.appendChild(indicator)

  // Text input
  inputEl = document.createElement('input')
  inputEl.type = 'text'
  inputEl.id = 'instruction-input'
  inputEl.className = 'cb-input'
  inputEl.placeholder = 'Give an instruction... (Enter to submit, Esc to cancel)'
  inputEl.autocomplete = 'off'
  inputEl.spellcheck = false
  bar.appendChild(inputEl)

  // Submit button
  const submit = document.createElement('button')
  submit.className = 'cb-submit'
  submit.textContent = '▶'
  submit.onclick = () => submitInstruction()
  bar.appendChild(submit)

  // Event handlers
  inputEl.addEventListener('keydown', (e) => {
    e.stopPropagation()  // Don't trigger manual controls while typing

    if (e.key === 'Enter' && inputEl.value.trim()) {
      submitInstruction()
    } else if (e.key === 'Escape') {
      if (isExecuting()) {
        abortExecution()
        showNotification('Execution aborted', 'warning')
      }
      inputEl.blur()
    } else if (e.key === 'ArrowUp') {
      // Navigate backward through history
      e.preventDefault()
      if (_history.length > 0) {
        historyIndex = Math.min(historyIndex + 1, _history.length - 1)
        inputEl.value = _history[historyIndex]
        // Move cursor to end
        setTimeout(() => inputEl.setSelectionRange(inputEl.value.length, inputEl.value.length), 0)
      }
    } else if (e.key === 'ArrowDown') {
      // Navigate forward through history
      e.preventDefault()
      if (historyIndex > 0) {
        historyIndex--
        inputEl.value = _history[historyIndex]
      } else {
        historyIndex = -1
        inputEl.value = ''
      }
    }
  })

  // Subscribe to status changes for indicator
  subscribe('robot.status', (status) => {
    indicator.className = `cb-indicator cb-ind-${(status || 'idle').toLowerCase()}`
  })

  return bar
}

async function submitInstruction() {
  if (!inputEl || !inputEl.value.trim()) return
  // UI-5: prevent double-submission from simultaneous Enter key + button click
  if (_submitting) return
  _submitting = true

  const instruction = inputEl.value.trim()
  inputEl.value = ''
  inputEl.disabled = true
  // SM-4: use lowercase status values matching state.js schema
  setState('robot.status', 'planning')

  // Show what was submitted
  addToHistory(instruction)

  try {
    const result = await handleInstruction(instruction, skillRegistry)

    if (result.success) {
      setState('robot.status', 'idle')
    } else {
      setState('robot.status', 'failed')
      showNotification(result.reason || 'Instruction failed', 'error')
    }
  } catch (err) {
    setState('robot.status', 'failed')
    showNotification(`Error: ${err.message}`, 'error')
    console.error('[Controls] Instruction error:', err)
  } finally {
    // UI-4: always re-enable input even if handleInstruction throws
    _submitting = false
    if (inputEl) {
      inputEl.disabled = false
      inputEl.focus()
    }
  }
}

// ─── History ───────────────────────────────────────────────────────────────────
// UI-3: renamed from `history` → `_history` to avoid shadowing window.history

const _history = []
let historyIndex = -1

function addToHistory(instruction) {
  _history.unshift(instruction)
  if (_history.length > 50) _history.pop()
  historyIndex = -1
}

// ─── Keyboard Manual Override ──────────────────────────────────────────────────

function initKeyboard() {
  window.addEventListener('keydown', onKeyDown)
  window.addEventListener('keyup', onKeyUp)
  // UI-1: store reference so destroyControls() can remove it later
  _blurHandler = () => {
    keysDown.clear()
    if (manualOverride) {
      manualOverride = false
      const robot = getActiveRobot()
      if (robot) robot.stop()
      setState('robot.status', 'idle')  // SM-4: lowercase
    }
  }
  window.addEventListener('blur', _blurHandler)

  // Manual control tick
  manualInterval = setInterval(manualControlTick, 16)  // ~60Hz
}

function onKeyDown(e) {
  // Ignore if typing in input
  if (document.activeElement === inputEl) return

  // Shortcut: '/' focuses input
  if (e.key === '/' && !e.ctrlKey && !e.metaKey) {
    e.preventDefault()
    focusInput()
    return
  }

  // Shortcut: Escape aborts execution
  if (e.key === 'Escape' && isExecuting()) {
    abortExecution()
    showNotification('Execution aborted', 'warning')
    return
  }

  // WASD or arrow keys → manual override
  const moveKeys = new Set(['w', 'a', 's', 'd', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'])
  if (moveKeys.has(e.key.toLowerCase()) || moveKeys.has(e.key)) {
    keysDown.add(e.key.toLowerCase())

    if (!manualOverride) {
      manualOverride = true
      setState('robot.status', 'manual')  // SM-4: lowercase
    }
  }
}

function onKeyUp(e) {
  keysDown.delete(e.key.toLowerCase())

  if (keysDown.size === 0 && manualOverride) {
    manualOverride = false
    const robot = getActiveRobot()
    if (robot) robot.stop()
    setState('robot.status', 'idle')  // SM-4: lowercase
  }
}

function manualControlTick() {
  if (!manualOverride) return

  const robot = getActiveRobot()
  if (!robot) return

  let forward = 0
  let turn = 0

  if (keysDown.has('w') || keysDown.has('arrowup')) forward += MANUAL_SPEED
  if (keysDown.has('s') || keysDown.has('arrowdown')) forward -= MANUAL_SPEED
  if (keysDown.has('a') || keysDown.has('arrowleft')) turn += MANUAL_TURN_SPEED
  if (keysDown.has('d') || keysDown.has('arrowright')) turn -= MANUAL_TURN_SPEED

  robot.moveForward(forward)
  robot.rotate(turn)
}

// ─── Touch Controls (Stub) ─────────────────────────────────────────────────────

function initTouch() {
  // TODO: Virtual joystick for mobile testing
  // Would create a translucent joystick overlay in bottom-left
  console.log('[Controls] Touch controls not yet implemented')
}

// ─── Styles ────────────────────────────────────────────────────────────────────

function injectControlStyles() {
  if (document.getElementById('cb-control-styles')) return

  const style = document.createElement('style')
  style.id = 'cb-control-styles'
  style.textContent = `
    .cb-controls {
      position: fixed;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10001;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 8px;
      pointer-events: none;
    }
    .cb-controls > * { pointer-events: auto; }

    .cb-input-bar {
      display: flex;
      align-items: center;
      gap: 8px;
      background: rgba(10, 10, 20, 0.92);
      border: 1px solid rgba(100, 200, 255, 0.25);
      border-radius: 24px;
      padding: 8px 16px;
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
      min-width: 400px;
      max-width: 600px;
    }

    .cb-indicator {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      flex-shrink: 0;
      transition: background 0.3s;
    }
    .cb-ind-idle { background: #555; }
    .cb-ind-thinking { background: #ffa500; animation: cb-pulse 1s infinite; }
    .cb-ind-executing { background: #4fc3f7; animation: cb-pulse 0.6s infinite; }
    .cb-ind-manual { background: #ab47bc; }
    .cb-ind-success { background: #66bb6a; }
    .cb-ind-error { background: #ef5350; }

    @keyframes cb-pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    .cb-input {
      flex: 1;
      background: transparent;
      border: none;
      outline: none;
      color: #e0e0e0;
      font-family: 'JetBrains Mono', 'Fira Code', monospace;
      font-size: 13px;
      padding: 4px 0;
    }
    .cb-input::placeholder { color: #555; }
    .cb-input:disabled { opacity: 0.5; }

    .cb-submit {
      background: rgba(100, 200, 255, 0.15);
      border: 1px solid rgba(100, 200, 255, 0.3);
      border-radius: 50%;
      width: 28px;
      height: 28px;
      color: #7af;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 12px;
      transition: all 0.2s;
    }
    .cb-submit:hover {
      background: rgba(100, 200, 255, 0.3);
      border-color: #7af;
    }
  `
  document.head.appendChild(style)
}
