/**
 * brain/background_agent.js — Autonomous Idle Behavior Agent
 *
 * Autonomous background movement is currently disabled so the robot stays
 * completely still when the user is not giving commands.
 *
 * The infrastructure (initBackgroundAgent, resetIdleTimer, etc.) is kept
 * so it can be re-enabled later without changing call sites in main.js.
 */

// ─── Configuration ─────────────────────────────────────────────────────────────

const IDLE_THRESHOLD_MS  = 60_000   // Wait 60 s of user inactivity before acting (was 8 s — far too short)
const EXPLORE_INTERVAL_MS = 300_000  // Run an explore pass every 5 min of idleness (was 45 s)
const POLL_INTERVAL_MS   = 2_000   // How often to check idle state

// ─── State ─────────────────────────────────────────────────────────────────────

let _lastActiveTime   = Date.now()
let _lastExploreTime  = 0
let _running          = false   // True while a background skill is executing
let _enabled          = false
let _registry         = null
let _getContext       = null    // () => executionContext
let _intervalId       = null
let _onActivity       = null    // Optional callback: (goal: string) => void

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Start the background agent.
 *
 * @param {Object}   skillRegistry — The global skill registry
 * @param {function} getContextFn  — Returns a fresh skill execution context
 * @param {{onActivity?: function(string): void}} [options]
 */
export function initBackgroundAgent(skillRegistry, getContextFn, options = {}) {
  if (_intervalId !== null) return  // Already running

  _registry   = skillRegistry
  _getContext = getContextFn
  _onActivity = options.onActivity ?? null
  _enabled    = true
  _lastActiveTime  = Date.now()
  _lastExploreTime = Date.now() - EXPLORE_INTERVAL_MS * 0.7  // Allow first explore sooner

  _intervalId = setInterval(_tick, POLL_INTERVAL_MS)
  console.log('[BgAgent] Background agent started.')
}

/** Stop the background agent permanently. */
export function stopBackgroundAgent() {
  if (_intervalId !== null) {
    clearInterval(_intervalId)
    _intervalId = null
  }
  _enabled = false
  console.log('[BgAgent] Background agent stopped.')
}

/** Enable or disable without stopping the interval. */
export function setBackgroundAgentEnabled(enabled) {
  _enabled = enabled
}

/**
 * Reset the idle timer. Call this whenever the planner begins executing
 * a user instruction so the background agent backs off.
 */
export function resetIdleTimer() {
  _lastActiveTime = Date.now()
}

/** Is the agent currently running a background skill? */
export function isBackgroundAgentActive() {
  return _running
}

// ─── Internal Tick ─────────────────────────────────────────────────────────────

async function _tick() {
  // Background autonomous movement disabled — robot stays still when idle.
  // Both the stale-object rescan (spin) and random explore (wander) were
  // causing the robot to move without the user asking it to.
  // Re-enable specific goals here if autonomous behaviour is ever needed.
}

/**
 * Execute a single background goal skill by name.
 * @param {string} skillName
 * @param {string} description
 */
async function _runGoal(skillName, description) {
  const skill = _registry.get(skillName)
  if (!skill) return

  _running = true
  console.log(`[BgAgent] Running background goal: ${skillName} — ${description}`)
  _onActivity?.(description)

  try {
    const ctx = _getContext()
    await skill.execute(ctx)
    console.log(`[BgAgent] Goal complete: ${skillName}`)
  } catch (e) {
    console.warn(`[BgAgent] Goal "${skillName}" failed:`, e.message)
  } finally {
    _running = false
  }
}
