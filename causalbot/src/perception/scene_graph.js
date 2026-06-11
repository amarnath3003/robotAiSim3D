/**
 * perception/scene_graph.js — Semantic Scene Graph
 *
 * Maintains a rich object model on top of the flat perception memory.
 * Each observed object gets: type inference, exponential-weighted position
 * smoothing, velocity estimation, spatial relations, and motion detection.
 *
 * The scene graph is updated by state.js whenever updatePerceptionMemory() fires,
 * so it always mirrors the robot's perception memory in real time.
 *
 * getLLMContext(robotPos) generates a dense, human-readable multi-line
 * description used by the planning prompt — far richer than the raw flat list.
 */

// ─── Type Inference ────────────────────────────────────────────────────────────

const TYPE_PATTERNS = [
  [/box|crate|cube|package/,  'box'       ],
  [/ball|sphere|orb/,         'ball'      ],
  [/wall|barrier|fence/,      'wall'      ],
  [/goal|target|flag|waypoint/,'goal'     ],
  [/door|gate|portal/,        'door'      ],
  [/obstacle|block|pillar/,   'obstacle'  ],
  [/table|platform|shelf/,    'surface'   ],
  [/robot|agent|bot/,         'robot'     ],
  [/light|lamp|torch/,        'light'     ],
  [/button|switch|lever/,     'control'   ],
]

/**
 * Infer object type from its ID string.
 * @param {string} id
 * @returns {string}
 */
function inferType(id) {
  const lower = id.toLowerCase()
  for (const [pattern, type] of TYPE_PATTERNS) {
    if (pattern.test(lower)) return type
  }
  return 'object'
}

// ─── Compass Direction ─────────────────────────────────────────────────────────

const COMPASS_DIRS = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']

/**
 * Get compass direction label from a delta vector (dx, dz in world space).
 * Z is forward in Three.js / Rapier (negative Z = north by convention here).
 */
function compassDir(dx, dz) {
  const angle = Math.atan2(dx, -dz) * (180 / Math.PI)
  return COMPASS_DIRS[Math.round(((angle + 360) % 360) / 45) % 8]
}

// ─── SceneObject ───────────────────────────────────────────────────────────────

export class SceneObject {
  /**
   * @param {string} id
   * @param {{x: number, y: number, z: number}} position
   * @param {number} confidence — 0..1
   */
  constructor(id, position, confidence) {
    this.id         = id
    this.type       = inferType(id)
    this.position   = { ...position }
    this.confidence = Math.min(1, Math.max(0, confidence))

    // Velocity estimated from consecutive position updates
    this.velocity   = { x: 0, y: 0, z: 0 }

    // Tracking
    this._prevPos       = { ...position }
    this._prevPosTime   = Date.now()
    this.lastSeen       = Date.now()
    this.firstSeen      = Date.now()
    this._updateCount   = 0
  }

  /**
   * Incorporate a new observation using exponential weighted averaging.
   * Alpha is clamped to [0, 0.8] — high confidence observations pull
   * position strongly but never fully overwrite (avoids sensor spikes).
   */
  update(position, confidence) {
    const now   = Date.now()
    const alpha = Math.min(confidence, 0.8)

    // Velocity: finite difference from previous position
    const dtSec = Math.max((now - this._prevPosTime) / 1000, 0.016)
    this.velocity.x = (position.x - this.position.x) / dtSec
    this.velocity.y = (position.y - this.position.y) / dtSec
    this.velocity.z = (position.z - this.position.z) / dtSec

    // Save prev before overwriting
    this._prevPos     = { ...this.position }
    this._prevPosTime = now

    // Weighted position blend
    this.position.x = this.position.x * (1 - alpha) + position.x * alpha
    this.position.y = this.position.y * (1 - alpha) + position.y * alpha
    this.position.z = this.position.z * (1 - alpha) + position.z * alpha

    // Confidence: blend toward new value (slower decay when frequently observed)
    this.confidence   = Math.min(1.0, this.confidence * 0.4 + confidence * 0.6)
    this.lastSeen     = now
    this._updateCount++
  }

  /** True if the object is moving faster than the noise threshold. */
  isMoving() {
    return Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z) > 0.1
  }

  /** True if this entry hasn't been refreshed in > 30 seconds. */
  isStale() {
    return (Date.now() - this.lastSeen) > 30_000
  }

  /** Seconds since this object was last observed. */
  ageSeconds() {
    return (Date.now() - this.lastSeen) / 1000
  }

  /** Euclidean distance on the XZ plane from a given position. */
  distanceTo(pos) {
    return Math.hypot(this.position.x - pos.x, this.position.z - pos.z)
  }
}

// ─── SceneGraph ────────────────────────────────────────────────────────────────

export class SceneGraph {
  constructor() {
    /** @type {Map<string, SceneObject>} */
    this._objects = new Map()
  }

  // ── Mutation ──────────────────────────────────────────────────────────────

  /**
   * Add or update an object observation.
   * @param {string} id
   * @param {{x: number, y: number, z: number}} position
   * @param {number} confidence
   */
  update(id, position, confidence) {
    if (this._objects.has(id)) {
      this._objects.get(id).update(position, confidence)
    } else {
      this._objects.set(id, new SceneObject(id, position, confidence))
    }
  }

  /** Decay confidences over time. Call once per second (not every frame). */
  decay(decayRate = 0.04, dtSec = 1) {
    for (const [id, obj] of this._objects) {
      obj.confidence -= decayRate * dtSec
      if (obj.confidence <= 0) this._objects.delete(id)
    }
  }

  /** Remove all objects. */
  clear() { this._objects.clear() }

  // ── Queries ───────────────────────────────────────────────────────────────

  /** @returns {SceneObject|null} */
  get(id) { return this._objects.get(id) ?? null }

  /** @returns {SceneObject[]} */
  getAll() { return [...this._objects.values()] }

  /** @returns {SceneObject[]} */
  getByType(type) { return this.getAll().filter(o => o.type === type) }

  /**
   * Get objects within `radius` metres (XZ plane) of `position`.
   * @param {{x,z}} position
   * @param {number} radius
   * @returns {SceneObject[]}
   */
  getNearby(position, radius) {
    return this.getAll().filter(o => o.distanceTo(position) <= radius)
  }

  /**
   * Objects that are stale and low-confidence — candidates for a re-scan.
   * @returns {SceneObject[]}
   */
  getStaleCandidates() {
    return this.getAll().filter(o => o.confidence < 0.4 && o.isStale())
  }

  /**
   * Objects above a confidence threshold, sorted by confidence descending.
   * @param {number} [minConf=0.2]
   * @returns {SceneObject[]}
   */
  getConfident(minConf = 0.2) {
    return this.getAll()
      .filter(o => o.confidence >= minConf)
      .sort((a, b) => b.confidence - a.confidence)
  }

  // ── LLM Context ───────────────────────────────────────────────────────────

  /**
   * Generate a rich multi-line scene description for the LLM planning prompt.
   * Shows up to 12 most-confident objects with distance, direction, motion,
   * staleness, and type — far denser than the raw flat object list.
   *
   * @param {{x: number, y: number, z: number}} robotPos
   * @returns {string}
   */
  getLLMContext(robotPos) {
    const objs = this.getConfident(0.15).slice(0, 12)

    if (objs.length === 0) {
      return 'No objects observed yet. Robot must SCAN to discover the environment.'
    }

    const rp = robotPos ?? { x: 0, y: 0, z: 0 }
    const lines = [`${objs.length} known object(s):`]

    for (const o of objs) {
      const dx   = o.position.x - rp.x
      const dz   = o.position.z - rp.z
      const dist = Math.hypot(dx, dz).toFixed(1)
      const dir  = compassDir(dx, dz)

      const tags = []
      if (o.isMoving()) tags.push('moving')
      if (o.isStale())  tags.push('STALE')
      const tagStr = tags.length ? `, ${tags.join(', ')}` : ''

      lines.push(
        `  • ${o.id} [${o.type}] — ${dist}m ${dir}, conf=${o.confidence.toFixed(2)}` +
        `, pos=(${o.position.x.toFixed(1)},${o.position.z.toFixed(1)})${tagStr}`
      )
    }

    return lines.join('\n')
  }
}

// ─── Singleton ─────────────────────────────────────────────────────────────────

/** Global scene graph — updated automatically by state.js updatePerceptionMemory(). */
export const sceneGraph = new SceneGraph()
