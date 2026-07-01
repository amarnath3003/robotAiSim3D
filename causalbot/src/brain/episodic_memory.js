/**
 * brain/episodic_memory.js — Persistent Episodic Memory
 *
 * Stores structured memories of past task executions, lessons, and spatial facts.
 * Persists to localStorage so the robot remembers across page reloads.
 *
 * Key design:
 * - Episodes capped at 500 (oldest dropped when exceeded)
 * - Keyword + recency scoring for recall
 * - Lessons: weight ×1.5, successes ×1.2, failures ×0.8
 * - getRelevantContext(instruction) returns top-K memories for LLM prompt injection
 */

const STORAGE_KEY  = 'causalbot_episodic_memory_v1'
const MAX_EPISODES = 500
const RECALL_TOP_K = 5

// ─── EpisodicMemory Class ─────────────────────────────────────────────────────

export class EpisodicMemory {
  constructor() {
    /** @type {Array<Object>} */
    this._episodes = []
    this._initialized = false
  }

  // ─── Persistence ───────────────────────────────────────────────────────────

  /** Load episodes from localStorage. Call once at boot. */
  load() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (raw) {
        const parsed = JSON.parse(raw)
        this._episodes = Array.isArray(parsed) ? parsed : []
        console.log(`[EpisodicMemory] Loaded ${this._episodes.length} episodes from storage`)
      }
    } catch (e) {
      console.warn('[EpisodicMemory] Failed to load from storage:', e)
      this._episodes = []
    }
    this._initialized = true
  }

  /** Serialize and persist current episodes. */
  _save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this._episodes))
    } catch (e) {
      console.warn('[EpisodicMemory] Failed to save:', e.message)
    }
  }

  // ─── Episode Recording ─────────────────────────────────────────────────────

  /**
   * Record a task episode (called by planner after every handled instruction).
   * @param {Object} params
   * @param {string}   params.instruction  - Original user instruction
   * @param {boolean}  params.success      - Did the task succeed?
   * @param {string}   params.summary      - Brief outcome summary
   * @param {string[]} [params.skillsUsed] - Skills that were executed
   * @param {Object}   [params.robotPos]   - Robot position {x,y,z} at task end
   */
  recordTask({ instruction, success, summary, skillsUsed = [], robotPos = null }) {
    this._push({
      type: 'task',
      instruction,
      success,
      summary,
      skillsUsed,
      robotPos,
      timestamp: Date.now(),
    })
  }

  /**
   * Record a lesson extracted by LLM reflection.
   * @param {string}   lesson    - The lesson text
   * @param {string[]} keywords  - Keywords for scoring
   */
  recordLesson(lesson, keywords = []) {
    this._push({
      type: 'lesson',
      lesson,
      keywords,
      timestamp: Date.now(),
    })
  }

  /**
   * Record a spatial fact (e.g. "charging dock is near the north wall").
   * @param {string}   fact     - The spatial fact
   * @param {string[]} keywords - Keywords for scoring
   */
  recordSpatialFact(fact, keywords = []) {
    this._push({
      type: 'spatial',
      fact,
      keywords,
      timestamp: Date.now(),
    })
  }

  _push(episode) {
    this._episodes.push(episode)
    // Cap at MAX_EPISODES — drop oldest entries
    if (this._episodes.length > MAX_EPISODES) {
      this._episodes.splice(0, this._episodes.length - MAX_EPISODES)
    }
    this._save()
  }

  // ─── Recall ────────────────────────────────────────────────────────────────

  /**
   * Recall the most relevant episodes for a given instruction.
   *
   * Scoring:
   *   score = (jaccardOverlap × typeWeight × recency) + pureRecencyBonus
   *
   * Type weights: lesson ×1.5, success ×1.2, failure ×0.8
   * Recency: exponential decay with ~3-day half-life
   * Pure-recency bonus: +0.10 for episodes within the last 24 h
   *
   * @param {string} instruction - The query instruction
   * @param {number} [k]         - Max results
   * @returns {Object[]} Top-k episodes sorted by score (score > 0 only)
   */
  recall(instruction, k = RECALL_TOP_K) {
    if (this._episodes.length === 0) return []

    const queryTokens = _tokenize(instruction)
    const now = Date.now()

    const scored = this._episodes.map(ep => {
      const epTokens = _episodeTokens(ep)
      const overlap  = _jaccard(queryTokens, epTokens)

      // Type weight
      let weight = 1.0
      if (ep.type === 'lesson') weight = 1.5
      else if (ep.type === 'task' && ep.success)  weight = 1.2
      else if (ep.type === 'task' && !ep.success) weight = 0.8

      // Recency (exponential decay; 3-day half-life)
      const ageDays = (now - ep.timestamp) / 86_400_000
      const recency = Math.exp(-ageDays / 3)

      const base = overlap * weight * recency

      // Pure-recency nudge for very recent episodes — but ONLY when the episode
      // is actually relevant (overlap > 0). Applying it unconditionally made the
      // score > 0 for EVERY episode from the last 24 h, so unrelated recent
      // memories leaked past the score>0 filter and polluted the planning prompt.
      const pureBonus = (overlap > 0 && ageDays < 1) ? 0.10 : 0

      return { ep, score: base + pureBonus }
    })

    scored.sort((a, b) => b.score - a.score)
    return scored
      .slice(0, k)
      .filter(s => s.score > 0)
      .map(s => s.ep)
  }

  // ─── LLM Context ───────────────────────────────────────────────────────────

  /**
   * Build a formatted context string for injection into the LLM planning prompt.
   * @param {string} instruction - Current user instruction
   * @returns {string} Empty string if nothing relevant found
   */
  getRelevantContext(instruction) {
    const memories = this.recall(instruction)
    if (memories.length === 0) return ''

    const lines = ['# Past Experience (relevant memories):']
    for (const ep of memories) {
      if (ep.type === 'task') {
        const tag = ep.success ? 'SUCCEEDED' : 'FAILED'
        lines.push(`- [${tag}] "${ep.instruction}" → ${ep.summary}`)
        if (!ep.success) {
          lines.push(`  (Do not repeat this failure pattern)`)
        }
      } else if (ep.type === 'lesson') {
        lines.push(`- [LESSON] ${ep.lesson}`)
      } else if (ep.type === 'spatial') {
        lines.push(`- [SPATIAL] ${ep.fact}`)
      }
    }
    return lines.join('\n')
  }

  // ─── Accessors ─────────────────────────────────────────────────────────────

  /** Total number of stored episodes */
  get count() {
    return this._episodes.length
  }

  /**
   * Get the most recent lesson texts (for the Memory panel display).
   * @param {number} n
   * @returns {string[]}
   */
  getRecentLessons(n = 3) {
    return this._episodes
      .filter(ep => ep.type === 'lesson')
      .slice(-n)
      .map(ep => ep.lesson)
  }

  /** Clear all episodes (debug / reset). */
  clear() {
    this._episodes = []
    this._save()
  }
}

// ─── Tokenization & Scoring Helpers ───────────────────────────────────────────

/**
 * Convert text into a set of lower-case tokens (≥ 3 chars, alphanumeric only).
 * @param {string} text
 * @returns {Set<string>}
 */
function _tokenize(text) {
  if (!text) return new Set()
  return new Set(
    text.toLowerCase()
      .replace(/[^a-z0-9\s]/g, ' ')
      .split(/\s+/)
      .filter(t => t.length >= 3)
  )
}

/** Build a combined token set from all text fields of an episode. */
function _episodeTokens(ep) {
  const parts = []
  if (ep.instruction) parts.push(ep.instruction)
  if (ep.summary)     parts.push(ep.summary)
  if (ep.lesson)      parts.push(ep.lesson)
  if (ep.fact)        parts.push(ep.fact)
  if (Array.isArray(ep.keywords))   parts.push(...ep.keywords)
  if (Array.isArray(ep.skillsUsed)) parts.push(...ep.skillsUsed)
  return _tokenize(parts.join(' '))
}

/**
 * Jaccard similarity between two token sets: |A∩B| / |A∪B|
 * @param {Set<string>} setA
 * @param {Set<string>} setB
 * @returns {number} 0–1
 */
function _jaccard(setA, setB) {
  if (setA.size === 0 || setB.size === 0) return 0
  let intersection = 0
  for (const tok of setA) {
    if (setB.has(tok)) intersection++
  }
  const union = setA.size + setB.size - intersection
  return union === 0 ? 0 : intersection / union
}

// ─── Singleton ────────────────────────────────────────────────────────────────

export const episodicMemory = new EpisodicMemory()
