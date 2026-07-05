/**
 * brain/cot_trace.js — Chain-of-Thought Trace Logger + Ablation Mode Flag
 *
 * Records every reasoning event (thoughts, actions, observations, plans,
 * failure analyses, synthesis reasoning) per instruction episode so CoT
 * behaviour is measurable and exportable for experiments.
 *
 * Ablation modes (localStorage 'cot_mode' or VITE_COT_MODE, default 'adaptive'):
 *   off      — legacy single-call planning, no structured CoT
 *   prompt   — always single-call planning WITH structured CoT fields
 *   react    — every LLM-planned instruction runs the ReAct loop
 *   adaptive — router picks per-instruction; failures escalate to ReAct
 *
 * Console helpers: window.exportCoTTraces(), window.setCoTMode('react')
 */

const STORAGE_KEY = 'causalbot_cot_traces'
const MODE_KEY    = 'cot_mode'
const MAX_STORED_EPISODES = 40
const VALID_MODES = ['off', 'prompt', 'react', 'adaptive']

// ─── Mode Flag ─────────────────────────────────────────────────────────────────

export function getCoTMode() {
  try {
    const stored = localStorage.getItem(MODE_KEY)
    if (stored && VALID_MODES.includes(stored)) return stored
  } catch { /* SSR / privacy mode */ }
  const env = import.meta.env?.VITE_COT_MODE
  if (env && VALID_MODES.includes(env)) return env
  return 'adaptive'
}

export function setCoTMode(mode) {
  if (!VALID_MODES.includes(mode)) {
    console.warn(`[CoT] Invalid mode "${mode}". Valid: ${VALID_MODES.join(', ')}`)
    return getCoTMode()
  }
  try { localStorage.setItem(MODE_KEY, mode) } catch { /* ignore */ }
  console.info(`[CoT] Mode set to "${mode}"`)
  return mode
}

// ─── Trace Recorder ────────────────────────────────────────────────────────────

class CoTTrace {
  constructor() {
    this.episodes = this._load()
    this._current = null
  }

  /**
   * Begin a new trace episode for one user instruction.
   * @param {string} instruction
   * @param {string} mode - Effective CoT mode for this episode
   * @param {string} tier - 'direct' | 'plan' | 'react'
   */
  startEpisode(instruction, mode, tier) {
    this._current = {
      id: `ep_${Date.now()}_${Math.floor(Math.random() * 1e6)}`,
      instruction,
      mode,
      tier,
      startedAt: Date.now(),
      events: [],
      outcome: null,
    }
    return this._current.id
  }

  /**
   * Record a reasoning event in the current episode.
   * @param {string} type - 'thought'|'action'|'observation'|'plan'|'critique'|
   *                        'failure_analysis'|'synthesis_reasoning'|'escalation'|'error'
   * @param {Object} data - Arbitrary payload (kept small)
   */
  record(type, data = {}) {
    if (!this._current) return
    this._current.events.push({ t: Date.now() - this._current.startedAt, type, ...data })
  }

  /**
   * Close the current episode.
   * @param {{success: boolean, reason?: string, llmCalls?: number, tokens?: Object}} outcome
   */
  endEpisode(outcome) {
    if (!this._current) return
    this._current.outcome = outcome
    this._current.durationMs = Date.now() - this._current.startedAt
    this.episodes.push(this._current)
    if (this.episodes.length > MAX_STORED_EPISODES) {
      this.episodes.splice(0, this.episodes.length - MAX_STORED_EPISODES)
    }
    this._save()
    this._current = null
  }

  /** All recorded episodes (persisted across reloads). */
  getAll() { return [...this.episodes] }

  /** Serialize all traces to a JSON string (for experiment analysis). */
  export() { return JSON.stringify(this.episodes, null, 2) }

  /** Trigger a browser download of the trace log. */
  download() {
    const blob = new Blob([this.export()], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `cot_traces_${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  clear() {
    this.episodes = []
    this._save()
  }

  _load() {
    try {
      return JSON.parse(localStorage.getItem(STORAGE_KEY)) || []
    } catch { return [] }
  }

  _save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.episodes))
    } catch (e) {
      // Quota exceeded — drop oldest half and retry once
      this.episodes.splice(0, Math.ceil(this.episodes.length / 2))
      try { localStorage.setItem(STORAGE_KEY, JSON.stringify(this.episodes)) } catch { /* give up */ }
    }
  }
}

export const cotTrace = new CoTTrace()

// ─── Console Helpers ───────────────────────────────────────────────────────────

if (typeof window !== 'undefined') {
  window.exportCoTTraces = () => cotTrace.download()
  window.getCoTTraces    = () => cotTrace.getAll()
  window.clearCoTTraces  = () => cotTrace.clear()
  window.setCoTMode      = (m) => setCoTMode(m)
  window.getCoTMode      = () => getCoTMode()
}
