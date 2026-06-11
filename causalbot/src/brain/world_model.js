/**
 * brain/world_model.js — World Belief State and Spatial Model
 *
 * Maintains the robot's internal model of the world:
 * - Navigation grid (20×20, 1 m/cell): traversability beliefs
 * - Skill affordances: which skills work on which object types
 * - General belief state: arbitrary key-value facts
 *
 * Grid design:
 *   GRID_SIZE  = 20 cells per axis
 *   CELL_SIZE  = 1.0 m per cell
 *   GRID_ORIGIN = -10.0 (world-space min x and min z)
 *   Cell index = (gz * 20) + gx  where gx = floor((worldX + 10) / 1)
 *
 * Affordance key format: "${objectType}::${skill}"
 */

const GRID_SIZE   = 20      // cells per axis (20 × 20 = 400 cells total)
const CELL_SIZE   = 1.0     // meters per cell
const GRID_ORIGIN = -10.0   // world-space coordinate of grid's min edge

// Cell state constants
const CELL_UNKNOWN  = 0     // Never observed
const CELL_FREE     = 1     // Observed clear
const CELL_OBSTACLE = 2     // Obstacle detected
const CELL_VISITED  = 3     // Robot navigated through

// ─── WorldModel Class ─────────────────────────────────────────────────────────

export class WorldModel {
  constructor() {
    // Flat Uint8Array: GRID_SIZE × GRID_SIZE cells, all initially UNKNOWN
    this._grid = new Uint8Array(GRID_SIZE * GRID_SIZE).fill(CELL_UNKNOWN)

    // Map<"objectType::skill", {success: number, failure: number}>
    this._affordances = new Map()

    // Map<string, any> — general named beliefs about the world
    this._beliefs = new Map()
  }

  // ─── Navigation Grid ──────────────────────────────────────────────────────

  /**
   * Convert world (x, z) coordinates to a flat grid index.
   * @returns {number} Index into this._grid, or -1 if out of bounds
   */
  _cellIndex(worldX, worldZ) {
    const gx = Math.floor((worldX - GRID_ORIGIN) / CELL_SIZE)
    const gz = Math.floor((worldZ - GRID_ORIGIN) / CELL_SIZE)
    if (gx < 0 || gx >= GRID_SIZE || gz < 0 || gz >= GRID_SIZE) return -1
    return gz * GRID_SIZE + gx
  }

  /**
   * Mark a world position as visited (robot traversed it).
   * Overrides any prior state (visited > free > unknown).
   */
  markVisited(worldX, worldZ) {
    const idx = this._cellIndex(worldX, worldZ)
    if (idx >= 0) this._grid[idx] = CELL_VISITED
  }

  /**
   * Mark a world position as free (observed, no obstacle).
   * Only upgrades UNKNOWN cells — does not demote VISITED.
   */
  markFree(worldX, worldZ) {
    const idx = this._cellIndex(worldX, worldZ)
    if (idx >= 0 && this._grid[idx] === CELL_UNKNOWN) {
      this._grid[idx] = CELL_FREE
    }
  }

  /**
   * Mark a world position as blocked (obstacle detected here).
   */
  markObstacle(worldX, worldZ) {
    const idx = this._cellIndex(worldX, worldZ)
    if (idx >= 0) this._grid[idx] = CELL_OBSTACLE
  }

  /** Fraction of cells that have been observed (non-UNKNOWN). 0–1. */
  get explorationRatio() {
    let known = 0
    for (let i = 0; i < this._grid.length; i++) {
      if (this._grid[i] !== CELL_UNKNOWN) known++
    }
    return known / this._grid.length
  }

  /** Number of cells the robot has physically moved through. */
  get visitedCount() {
    let count = 0
    for (let i = 0; i < this._grid.length; i++) {
      if (this._grid[i] === CELL_VISITED) count++
    }
    return count
  }

  /** Reset the navigation grid to all-UNKNOWN (affordances/beliefs preserved). */
  resetGrid() {
    this._grid.fill(CELL_UNKNOWN)
  }

  // ─── Affordances ──────────────────────────────────────────────────────────

  /**
   * Record the outcome of executing a skill on an object of a given type.
   * @param {string}  objectType - e.g. 'box', 'door', 'button'
   * @param {string}  skill      - e.g. 'pick_up', 'push', 'interact'
   * @param {boolean} success
   */
  recordAffordance(objectType, skill, success) {
    const key   = `${objectType}::${skill}`
    const entry = this._affordances.get(key) || { success: 0, failure: 0 }
    if (success) entry.success++
    else         entry.failure++
    this._affordances.set(key, entry)
  }

  /**
   * Get the learned success rate for a skill on an object type.
   * @returns {number|null} 0–1 rate, or null if no data yet
   */
  getAffordanceRate(objectType, skill) {
    const key   = `${objectType}::${skill}`
    const entry = this._affordances.get(key)
    if (!entry) return null
    const total = entry.success + entry.failure
    return total === 0 ? null : entry.success / total
  }

  /** Total number of distinct objectType::skill pairs learned. */
  get affordanceCount() {
    return this._affordances.size
  }

  // ─── Belief State ─────────────────────────────────────────────────────────

  /**
   * Store a named belief about the world (e.g. 'charging_dock_pos': {x, z}).
   * @param {string} key
   * @param {*}      value
   */
  setBelief(key, value) {
    this._beliefs.set(key, value)
  }

  /** Get a belief value by key. */
  getBelief(key) {
    return this._beliefs.get(key)
  }

  /** True if a belief has been recorded for this key. */
  hasBelief(key) {
    return this._beliefs.has(key)
  }

  // ─── LLM Context ──────────────────────────────────────────────────────────

  /**
   * Format world model state for injection into the LLM planning prompt.
   * @param {Object} [robotPos] - {x, y, z} current robot world position
   * @returns {string}
   */
  getWorldModelContext(robotPos) {
    const lines = []

    // Exploration progress
    const expPct = (this.explorationRatio * 100).toFixed(0)
    lines.push(`# World Model:`)
    lines.push(
      `Exploration: ${expPct}% of arena mapped ` +
      `(${this.visitedCount}/${GRID_SIZE * GRID_SIZE} cells visited)`
    )

    // Robot's current grid cell (helps the LLM reason about coverage)
    if (robotPos) {
      const gx = Math.floor((robotPos.x - GRID_ORIGIN) / CELL_SIZE)
      const gz = Math.floor((robotPos.z - GRID_ORIGIN) / CELL_SIZE)
      if (gx >= 0 && gx < GRID_SIZE && gz >= 0 && gz < GRID_SIZE) {
        lines.push(`Robot grid cell: (${gx}, ${gz}) of (${GRID_SIZE - 1}, ${GRID_SIZE - 1})`)
      }
    }

    // Learned affordances (only those with ≥2 samples)
    if (this._affordances.size > 0) {
      const entries = []
      for (const [key, val] of this._affordances.entries()) {
        const total = val.success + val.failure
        if (total >= 2) {
          const rate = (val.success / total * 100).toFixed(0)
          entries.push(`${key}=${rate}%`)
        }
      }
      if (entries.length > 0) {
        lines.push(`Learned affordances: ${entries.slice(0, 8).join(', ')}`)
      }
    }

    // Named beliefs
    if (this._beliefs.size > 0) {
      const beliefLines = []
      for (const [k, v] of this._beliefs.entries()) {
        beliefLines.push(`${k}: ${JSON.stringify(v)}`)
      }
      lines.push(`Beliefs: ${beliefLines.slice(0, 5).join('; ')}`)
    }

    return lines.join('\n')
  }
}

// ─── Singleton ────────────────────────────────────────────────────────────────

export const worldModel = new WorldModel()
