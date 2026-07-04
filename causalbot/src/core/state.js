/**
 * core/state.js — Global State Management
 * 
 * Centralized reactive state for the CausalBot framework.
 * All systems read/write through this interface.
 * 
 * Key difference from the old state.js:
 * - Robot state is now driven by the manifest (not hardcoded)
 * - Perception is always active (no "omniscient" mode)
 * - No separate "modes" — the system is unified
 * - Observable: systems can subscribe to state changes
 */

// ─── State Schema ──────────────────────────────────────────────────────────────

const _state = {
  // Active robot (populated after manifest + model load)
  robot: {
    instance: null,       // RobotInstance from adapter.js
    manifest: null,       // Loaded manifest object
    status: 'unloaded',   // 'unloaded' | 'loading' | 'idle' | 'planning' | 'executing' | 'failed'
  },
  
  // Perception state (always active — robot is blind by default)
  perception: {
    observations: {},     // Latest sensor readings {sensorId: data}
    memory: new Map(),    // Spatial memory: objectId → {position, confidence, lastSeen}
    lastScanTime: 0,
    visibleObjects: [],   // Objects currently in sensor FOV
  },
  
  // Execution state
  execution: {
    currentPlan: null,    // Current action plan from LLM
    currentStep: 0,       // Current step in plan
    currentSkill: null,   // Currently executing skill
    queue: [],            // Queued plans
    history: [],          // Execution history {action, result, timestamp}
  },
  
  // World state (known objects — ONLY from perception, never omniscient)
  world: {
    knownObjects: {},     // Objects discovered through perception
    environment: {        // Static environment info
      bounds: null,
      floorY: 0,
    },
  },
  
  // RL state
  rl: {
    connected: false,
    training: false,
    episode: 0,
    totalSteps: 0,
    episodeReward: 0,
    stats: {
      deaths: 0,
      successes: 0,
      winRate: 0,
    },
  },
  
  // Scene references
  scene: {
    three: null,
    camera: null,
    renderer: null,
    controls: null,
    rapierWorld: null,
    rapier: null,        // RAPIER module reference
  },
  
  // UI state
  ui: {
    inputMode: 'text',   // 'text' | 'voice'
    debugOverlay: false,
    showSensorRays: false,
    showPathfinding: false,
    showColliders: false,
  },
}

// ─── Subscribers ───────────────────────────────────────────────────────────────

const _subscribers = new Map()  // path → Set<callback>

/**
 * Subscribe to state changes at a specific path.
 * @param {string} path - Dot-separated path (e.g., 'robot.status', 'rl.connected')
 * @param {function(any, any): void} callback - Called with (newValue, oldValue)
 * @returns {function(): void} Unsubscribe function
 */
export function subscribe(path, callback) {
  if (!_subscribers.has(path)) {
    _subscribers.set(path, new Set())
  }
  _subscribers.get(path).add(callback)
  
  return () => {
    const subs = _subscribers.get(path)
    if (subs) subs.delete(callback)
  }
}

function notifySubscribers(path, newValue, oldValue) {
  // Notify exact path
  const subs = _subscribers.get(path)
  if (subs) {
    for (const cb of subs) {
      try { cb(newValue, oldValue) } catch (e) { console.error(`[State] Subscriber error at "${path}":`, e) }
    }
  }
  
  // Notify parent paths (e.g., 'robot' for 'robot.status')
  const parts = path.split('.')
  for (let i = 1; i < parts.length; i++) {
    const parentPath = parts.slice(0, i).join('.')
    const parentSubs = _subscribers.get(parentPath)
    if (parentSubs) {
      for (const cb of parentSubs) {
        try { cb(getState(parentPath), null) } catch (e) { console.error(`[State] Subscriber error at "${parentPath}":`, e) }
      }
    }
  }
}

// ─── Change Detection ──────────────────────────────────────────────────────────

/**
 * Shallow equality check used by setState to suppress no-op updates.
 *
 * Rules:
 *  - Primitives (string, number, boolean, null, undefined): use ===
 *  - Plain objects ({…}) whose prototype is Object.prototype: shallow key=value compare
 *  - Everything else (class instances, arrays, Maps, …): always unequal so
 *    subscribers always fire — avoids accidentally suppressing updates on
 *    complex objects like RobotInstance or THREE.Scene.
 *
 * @param {any} a
 * @param {any} b
 * @returns {boolean}
 */
function _shallowEquals(a, b) {
  if (a === b) return true
  if (a == null || b == null) return false
  if (typeof a !== 'object' || typeof b !== 'object') return false
  // Only do shallow comparison for plain data objects
  if (Object.getPrototypeOf(a) !== Object.prototype) return false
  if (Object.getPrototypeOf(b) !== Object.prototype) return false
  const keysA = Object.keys(a)
  const keysB = Object.keys(b)
  if (keysA.length !== keysB.length) return false
  return keysA.every(k => a[k] === b[k])
}

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Get a value from state by dot-separated path.
 * @param {string} path - e.g., 'robot.status', 'rl.stats.winRate'
 * @returns {any}
 */
export function getState(path) {
  const parts = path.split('.')
  let current = _state
  for (const part of parts) {
    if (current == null) return undefined
    current = current[part]
  }
  return current
}

/**
 * Set a value in state by dot-separated path. Notifies subscribers.
 * @param {string} path
 * @param {any} value
 */
export function setState(path, value) {
  const parts = path.split('.')
  let current = _state
  
  for (let i = 0; i < parts.length - 1; i++) {
    if (current[parts[i]] == null) {
      current[parts[i]] = {}
    }
    current = current[parts[i]]
  }
  
  const lastKey = parts[parts.length - 1]
  const oldValue = current[lastKey]
  
  // SE-1: use shallowEquals so that plain {x,y,z} position objects written every
  //        frame don't flood subscribers, while class instances (RobotInstance,
  //        THREE.Scene) are always treated as changed to avoid missing updates.
  if (_shallowEquals(oldValue, value)) return
  
  current[lastKey] = value
  notifySubscribers(path, value, oldValue)
}

/**
 * Update multiple state values at once.
 * @param {Object} updates - {path: value, ...}
 */
export function batchUpdate(updates) {
  for (const [path, value] of Object.entries(updates)) {
    setState(path, value)
  }
}

/**
 * Get the full state tree (read-only reference for debugging).
 * DO NOT mutate — use setState() instead.
 * @returns {Object}
 */
export function getFullState() {
  return _state
}

// ─── Convenience Accessors ─────────────────────────────────────────────────────

/** Get the active robot instance */
export function getRobot() {
  return _state.robot.instance
}

/** Set the active robot instance */
export function setRobot(instance) {
  setState('robot.instance', instance)
  setState('robot.manifest', instance?.manifest ?? null)
  setState('robot.status', instance ? 'idle' : 'unloaded')
}

/** Get robot status */
export function getRobotStatus() {
  return _state.robot.status
}

/** Set robot status */
export function setRobotStatus(status) {
  setState('robot.status', status)
}

/** Is the RL bridge connected? */
export function isRLConnected() {
  return _state.rl.connected
}

/** Get the Three.js scene */
export function getScene() {
  return _state.scene.three
}

/** Set scene references */
export function setSceneRefs(refs) {
  for (const [key, value] of Object.entries(refs)) {
    _state.scene[key] = value
  }
}

// ─── Perception Memory Helpers ─────────────────────────────────────────────────

// Lazy import of the scene graph to avoid circular dependency at module load time.
// (scene_graph.js has no imports from state.js.)
let _sceneGraph = null
async function _getSceneGraph() {
  if (!_sceneGraph) {
    const mod = await import('../perception/scene_graph.js')
    _sceneGraph = mod.sceneGraph
  }
  return _sceneGraph
}

/**
 * Update perception memory with a newly observed object.
 * Also feeds the semantic scene graph for richer LLM context.
 * @param {string} objectId
 * @param {{x: number, y: number, z: number}} position
 * @param {number} confidence - 0 to 1
 * @param {Object} [meta] - Optional sensor metadata (colorName, label, radius…)
 */
export function updatePerceptionMemory(objectId, position, confidence = 1.0, meta = null) {
  const memory = _state.perception.memory
  const existing = memory.get(objectId)

  if (existing) {
    // Weighted blend toward new observation
    const weight = confidence
    existing.position.x = existing.position.x * (1 - weight) + position.x * weight
    existing.position.y = existing.position.y * (1 - weight) + position.y * weight
    existing.position.z = existing.position.z * (1 - weight) + position.z * weight
    existing.confidence = Math.min(1.0, existing.confidence * 0.5 + confidence * 0.5)
    existing.lastSeen = Date.now()
    if (meta) existing.meta = { ...(existing.meta || {}), ...meta }
  } else {
    memory.set(objectId, {
      position: { ...position },
      confidence,
      lastSeen: Date.now(),
      firstSeen: Date.now(),
      meta: meta ? { ...meta } : {},
    })
  }

  // Mirror into semantic scene graph (non-blocking — resolves on first call)
  _getSceneGraph().then(sg => sg.update(objectId, position, confidence)).catch(() => {})
}

/**
 * Decay all perception memory (call once per frame or per second).
 * Objects not re-observed will fade toward confidence=0.
 * @param {number} decayRate - Confidence decay per second (default 0.05)
 * @param {number} dt - Delta time in seconds
 */
export function decayPerceptionMemory(decayRate = 0.05, dt = 1/60) {
  const memory = _state.perception.memory
  const toRemove = []
  
  for (const [objectId, entry] of memory) {
    entry.confidence -= decayRate * dt
    if (entry.confidence <= 0) {
      toRemove.push(objectId)
    }
  }
  
  for (const id of toRemove) {
    memory.delete(id)
  }
}

/**
 * Get the remembered position of an object (or null if unknown/decayed).
 * @param {string} objectId
 * @returns {{position: {x,y,z}, confidence: number}|null}
 */
export function getRememberedObject(objectId) {
  const entry = _state.perception.memory.get(objectId)
  if (!entry || entry.confidence <= 0.1) return null
  return {
    position: entry.position,
    confidence: entry.confidence,
    lastSeen: entry.lastSeen,
    meta: entry.meta || {},
  }
}

/**
 * Get all remembered objects above a confidence threshold.
 * @param {number} minConfidence
 * @returns {Array<{id: string, position: {x,y,z}, confidence: number, lastSeen: number, meta: Object}>}
 */
export function getKnownObjects(minConfidence = 0.2) {
  const results = []
  for (const [id, entry] of _state.perception.memory) {
    if (entry.confidence >= minConfidence) {
      results.push({
        id,
        position: entry.position,
        confidence: entry.confidence,
        lastSeen: entry.lastSeen,
        meta: entry.meta || {},
      })
    }
  }
  return results
}

// ─── Execution History ─────────────────────────────────────────────────────────

/**
 * Log an execution result to history.
 * @param {string} action - What was attempted
 * @param {'success'|'failure'|'partial'} result
 * @param {string} details - Additional context
 */
export function logExecution(action, result, details = '') {
  _state.execution.history.push({
    action,
    result,
    details,
    timestamp: Date.now(),
  })
  
  // Keep history bounded (last 100 entries)
  if (_state.execution.history.length > 100) {
    _state.execution.history.shift()
  }
}

/**
 * Get recent execution history for LLM context.
 * @param {number} count - How many recent entries
 * @returns {Array}
 */
export function getRecentHistory(count = 10) {
  return _state.execution.history.slice(-count)
}
