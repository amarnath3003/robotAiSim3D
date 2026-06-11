/**
 * nav/pathfinder.js — A* Grid Pathfinding + Robot Navigation
 *
 * Provides:
 * - 2D occupancy grid (80×80 cells, 0.4 m/cell, covers ±16 m)
 * - AABB obstacle registration API (called by env/objects.js)
 * - A* search with 8-directional movement and corner-cutting prevention
 * - Line-of-sight path smoothing (reduces waypoint count)
 * - Promise-based navigateTo(robot, x, z, speed) driven by requestAnimationFrame
 *
 * No external dependencies beyond Three.js (for quaternion math).
 */

import * as THREE from 'three'

// ─── Grid Config ───────────────────────────────────────────────────────────────

const CELL_SIZE  = 0.4   // metres per cell
const GRID_HALF  = 16    // world extent in each direction (±16 m)
const GRID_SIZE  = Math.round((GRID_HALF * 2) / CELL_SIZE)  // 80

const ROBOT_RADIUS = 0.5  // metres — inflate obstacles by this much (was 0.3 — too small, caused collisions)

// ─── Occupancy Grid ────────────────────────────────────────────────────────────

// 1 = blocked, 0 = free  (static environment obstacles)
let _grid = new Uint8Array(GRID_SIZE * GRID_SIZE)

// Dynamic obstacle layer — rebuilt every perception tick from LiDAR hits.
// Kept separate so clearing it never erases static env data.
let _dynamicGrid = new Uint8Array(GRID_SIZE * GRID_SIZE)

// Registry: id → {minX, maxX, minZ, maxZ} (world-space, pre-inflation)
const _obstacles = new Map()

/**
 * Rebuild dynamic obstacle layer from the latest LiDAR hit distances.
 * Call this every perception tick (after castLidar) to keep moving obstacles current.
 *
 * @param {number}      robotX     World X of robot
 * @param {number}      robotZ     World Z of robot
 * @param {number}      yaw        Robot heading in radians (world Y rotation)
 * @param {Float32Array} distances  Per-ray distances returned by castLidar
 * @param {number}      fov        LiDAR field-of-view in radians (e.g. 2π for 360°)
 * @param {number}      range      Maximum sensor range in metres
 * @param {number}      rays       Number of rays
 */
export function updateDynamicObstacles(robotX, robotZ, yaw, distances, fov, range, rays) {
  _dynamicGrid.fill(0)

  const halfFov   = fov / 2
  const step      = rays > 1 ? fov / (rays - 1) : 0
  const inflCells = Math.ceil(ROBOT_RADIUS / CELL_SIZE)

  for (let i = 0; i < rays && i < distances.length; i++) {
    if (distances[i] >= range * 0.98) continue  // no real hit — ray reached max range

    const angle = yaw - halfFov + i * step
    const hitX  = robotX + Math.sin(angle) * distances[i]
    const hitZ  = robotZ + Math.cos(angle) * distances[i]

    const { col, row } = _worldToCell(hitX, hitZ)
    if (!_inBounds(col, row)) continue

    // Inflate by robot radius so A* keeps a safe margin
    for (let dr = -inflCells; dr <= inflCells; dr++) {
      for (let dc = -inflCells; dc <= inflCells; dc++) {
        const nc = col + dc
        const nr = row + dr
        if (_inBounds(nc, nr)) _dynamicGrid[nr * GRID_SIZE + nc] = 1
      }
    }
  }
}

/**
 * Clear the dynamic obstacle layer (e.g. when sensors are disabled).
 */
export function clearDynamicObstacles() {
  _dynamicGrid.fill(0)
}



/**
 * Register an obstacle AABB. Marks cells in the occupancy grid.
 * @param {string} id         Unique ID (for later removal)
 * @param {Object} aabb       {minX, maxX, minZ, maxZ} in world space
 */
export function registerObstacle(id, aabb) {
  _obstacles.set(id, aabb)
  _markAABB(aabb, true)
}

/**
 * Remove an obstacle from the occupancy grid.
 * Rebuilds the grid from scratch to avoid ghost cells.
 * @param {string} id
 */
export function unregisterObstacle(id) {
  _obstacles.delete(id)
  rebuildGrid()
}

/**
 * Clear all obstacles and reset the grid.
 */
export function clearObstacles() {
  _obstacles.clear()
  _grid.fill(0)
}

/**
 * Rebuild the full grid from the current obstacle registry.
 * Call after bulk changes.
 */
export function rebuildGrid() {
  _grid.fill(0)
  for (const aabb of _obstacles.values()) {
    _markAABB(aabb, true)
  }
}

/**
 * Check whether a world-space point is inside an obstacle.
 * @param {number} x
 * @param {number} z
 * @returns {boolean}
 */
export function isOccupied(x, z) {
  const { col, row } = _worldToCell(x, z)
  return _inBounds(col, row) && _grid[row * GRID_SIZE + col] === 1
}

// ─── Public: Path Finding ─────────────────────────────────────────────────────

/**
 * Find a world-space path from (startX, startZ) to (goalX, goalZ).
 * Returns an array of {x, z} waypoints, or null if no path exists.
 * The start position is NOT included in the returned array.
 *
 * @param {number} startX
 * @param {number} startZ
 * @param {number} goalX
 * @param {number} goalZ
 * @returns {Array<{x:number,z:number}>|null}
 */
export function findPath(startX, startZ, goalX, goalZ) {
  let startCell = _worldToCell(startX, startZ)
  let goalCell  = _worldToCell(goalX, goalZ)

  // Clamp to grid bounds
  startCell.col = Math.max(0, Math.min(GRID_SIZE - 1, startCell.col))
  startCell.row = Math.max(0, Math.min(GRID_SIZE - 1, startCell.row))

  // If start is blocked (robot inside dynamic obstacle zone), snap to nearest clear
  if (_blocked(startCell.col, startCell.row)) {
    const alt = _nearestClear(startCell.col, startCell.row)
    if (alt) startCell = alt
  }

  // If goal is blocked, find nearest clear cell
  if (_blocked(goalCell.col, goalCell.row)) {
    const alt = _nearestClear(goalCell.col, goalCell.row)
    if (!alt) return null
    goalCell = alt
  }

  const IDX = (c, r) => r * GRID_SIZE + c
  const startIdx = IDX(startCell.col, startCell.row)
  const goalIdx  = IDX(goalCell.col,  goalCell.row)

  if (startIdx === goalIdx) return []

  // A* data arrays
  const gCost  = new Float32Array(GRID_SIZE * GRID_SIZE).fill(Infinity)
  const parent = new Int32Array(GRID_SIZE * GRID_SIZE).fill(-1)
  const closed = new Uint8Array(GRID_SIZE * GRID_SIZE)

  gCost[startIdx] = 0

  const open = new _MinHeap()
  open.push(startIdx, _heuristic(startCell.col, startCell.row, goalCell.col, goalCell.row))

  // 8-directional movement: [dc, dr, cost]
  const DIRS = [
    [ 0,  1, 1],          [ 0, -1, 1],
    [ 1,  0, 1],          [-1,  0, 1],
    [ 1,  1, Math.SQRT2], [ 1, -1, Math.SQRT2],
    [-1,  1, Math.SQRT2], [-1, -1, Math.SQRT2],
  ]

  while (open.size > 0) {
    const ci = open.pop()

    if (ci === goalIdx) {
      return _smoothPath(_reconstructPath(parent, startIdx, goalIdx))
    }

    if (closed[ci]) continue
    closed[ci] = 1

    const c = ci % GRID_SIZE
    const r = Math.floor(ci / GRID_SIZE)

    for (const [dc, dr, cost] of DIRS) {
      const nc = c + dc
      const nr = r + dr
      if (!_inBounds(nc, nr)) continue
      const ni = IDX(nc, nr)
      if (closed[ni] || _blocked(nc, nr)) continue

      // Prevent diagonal corner-cutting through walls
      if (dc !== 0 && dr !== 0) {
        if (_blocked(nc, r) || _blocked(c, nr)) continue
      }

      const tentG = gCost[ci] + cost
      if (tentG < gCost[ni]) {
        gCost[ni]  = tentG
        parent[ni] = ci
        const f = tentG + _heuristic(nc, nr, goalCell.col, goalCell.row)
        open.push(ni, f)
      }
    }
  }

  return null  // No path found
}

// ─── Public: Robot Navigation ──────────────────────────────────────────────────

const ARRIVAL_THRESHOLD = 0.35   // metres — how close = "arrived at waypoint"
const TURN_THRESHOLD    = 0.12   // radians — face within this before driving
const NAV_STEP_TIMEOUT  = 15000  // ms per waypoint before giving up (was 8000 — too short for distant targets)

// EC-1: abort flag — set by abortNavigation(), checked in _followWaypoint tick
let _navAbortRequested = false

// RAF abort: track the latest frame handle and the active waypoint Promise's resolve
// so abortNavigation() can cancel the pending frame AND immediately settle the promise.
let _currentRafHandle = 0
let _currentResolve   = null

// Active path for debug visualisation
let _activePath = []

/**
 * Get the currently planned A* path (for debug visualisation).
 * Returns an array of {x, z} waypoints, or an empty array if not navigating.
 * @returns {Array<{x:number,z:number}>}
 */
export function getActivePath() {
  return _activePath
}

/**
 * Signal any in-progress navigateTo() to stop immediately.
 * Cancels the pending requestAnimationFrame AND resolves the active waypoint
 * promise right away — abort latency drops to one microtask instead of one frame.
 */
export function abortNavigation() {
  _navAbortRequested = true
  if (_currentRafHandle) {
    cancelAnimationFrame(_currentRafHandle)
    _currentRafHandle = 0
  }
  if (_currentResolve) {
    _currentResolve(false)
    _currentResolve = null
  }
}

/**
 * Drive the robot from its current position to (targetX, targetZ).
 * Uses A* + path smoothing, then follows waypoints with heading correction.
 * Resolves when the robot arrives at the goal (or times out on a segment).
 *
 * @param {import('../core/adapter.js').RobotInstance} robot
 * @param {number} targetX
 * @param {number} targetZ
 * @param {number} speed  m/s (defaults to robot's maxSpeed)
 */
export async function navigateTo(robot, targetX, targetZ, speed) {
  // EC-1: clear any stale abort flag from a previous navigation
  _navAbortRequested = false

  const maxSpeed = robot.manifest?.constraints?.maxSpeed ?? 2.0
  speed = Math.min(speed || maxSpeed, maxSpeed)

  const MAX_REPLAN = 3   // maximum mid-path replan attempts before giving up

  let remainingTarget = { x: targetX, z: targetZ }
  let replanCount = 0

  while (replanCount <= MAX_REPLAN) {
    const path = findPath(robot.position.x, robot.position.z, remainingTarget.x, remainingTarget.z)

    // Publish path for debug visualiser
    _activePath = path || []

    if (!path) {
      console.warn('[Nav] No path found — attempting direct move')
      await _followWaypoint(robot, remainingTarget.x, remainingTarget.z, speed * 0.6)
      break
    }

    if (path.length === 0) break   // already at goal

    let stuckOnPath = false
    for (const waypoint of path) {
      if (_navAbortRequested) break

      const arrived = await _followWaypoint(robot, waypoint.x, waypoint.z, speed)
      if (!arrived && !_navAbortRequested) {
        // Stuck or timed out — replan from current position
        stuckOnPath = true
        console.warn(`[Nav] Stuck at waypoint (${waypoint.x.toFixed(2)}, ${waypoint.z.toFixed(2)}) — replanning (#${replanCount + 1})`)
        break
      }
    }

    if (_navAbortRequested) break

    if (!stuckOnPath) {
      // Fine-positioning: skip if the exact goal is inside an obstacle (object centre).
      // The robot is already at the nearest clear cell — close enough for grab/interact.
      const goalCell = _worldToCell(remainingTarget.x, remainingTarget.z)
      if (!_blocked(goalCell.col, goalCell.row)) {
        const dx = remainingTarget.x - robot.position.x
        const dz = remainingTarget.z - robot.position.z
        if (Math.sqrt(dx * dx + dz * dz) > ARRIVAL_THRESHOLD) {
          await _followWaypoint(robot, remainingTarget.x, remainingTarget.z, speed * 0.5)
        }
      }
      break   // navigation complete
    }

    replanCount++
  }

  if (replanCount > MAX_REPLAN) {
    console.warn('[Nav] Max replan attempts reached — giving up')
  }

  robot.stop()
  _activePath = []
}

// ─── Debug: Grid Snapshot ──────────────────────────────────────────────────────

/**
 * Returns grid metadata for debugging/visualisation.
 */
export function getGridInfo() {
  return {
    size: GRID_SIZE,
    cellSize: CELL_SIZE,
    halfExtent: GRID_HALF,
    obstacleCount: _obstacles.size,
    blockedCells: _grid.reduce((s, v) => s + v, 0),
  }
}

// ─── Internal: Grid Helpers ────────────────────────────────────────────────────

function _worldToCell(x, z) {
  return {
    col: Math.floor((x + GRID_HALF) / CELL_SIZE),
    row: Math.floor((z + GRID_HALF) / CELL_SIZE),
  }
}

function _cellToWorld(col, row) {
  return {
    x: col * CELL_SIZE - GRID_HALF + CELL_SIZE * 0.5,
    z: row * CELL_SIZE - GRID_HALF + CELL_SIZE * 0.5,
  }
}

function _inBounds(col, row) {
  return col >= 0 && col < GRID_SIZE && row >= 0 && row < GRID_SIZE
}

function _blocked(col, row) {
  if (!_inBounds(col, row)) return true
  const idx = row * GRID_SIZE + col
  return _grid[idx] === 1 || _dynamicGrid[idx] === 1
}

function _markAABB(aabb, occupied) {
  const inf = ROBOT_RADIUS + CELL_SIZE * 0.5
  const c0 = Math.max(0, Math.floor((aabb.minX - inf + GRID_HALF) / CELL_SIZE))
  const c1 = Math.min(GRID_SIZE - 1, Math.floor((aabb.maxX + inf + GRID_HALF) / CELL_SIZE))
  const r0 = Math.max(0, Math.floor((aabb.minZ - inf + GRID_HALF) / CELL_SIZE))
  const r1 = Math.min(GRID_SIZE - 1, Math.floor((aabb.maxZ + inf + GRID_HALF) / CELL_SIZE))
  const val = occupied ? 1 : 0
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      _grid[r * GRID_SIZE + c] = val
    }
  }
}

function _heuristic(c0, r0, c1, r1) {
  // Octile distance (admissible for 8-direction movement)
  const dx = Math.abs(c1 - c0)
  const dz = Math.abs(r1 - r0)
  return CELL_SIZE * (Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz))
}

function _nearestClear(col, row, maxRadius = 10) {
  for (let r = 1; r <= maxRadius; r++) {
    for (let dc = -r; dc <= r; dc++) {
      for (let dr = -r; dr <= r; dr++) {
        if (Math.abs(dc) !== r && Math.abs(dr) !== r) continue
        const c = col + dc
        const rr = row + dr
        if (_inBounds(c, rr) && !_blocked(c, rr)) {
          return { col: c, row: rr }
        }
      }
    }
  }
  return null
}

// ─── Internal: Path Reconstruction ────────────────────────────────────────────

function _reconstructPath(parent, startIdx, goalIdx) {
  const raw = []
  let ci = goalIdx

  while (ci !== startIdx && ci !== -1) {
    const col = ci % GRID_SIZE
    const row = Math.floor(ci / GRID_SIZE)
    raw.unshift(_cellToWorld(col, row))
    ci = parent[ci]
  }

  return raw
}

// ─── Internal: Path Smoothing (Line-of-Sight) ──────────────────────────────────

function _smoothPath(path) {
  if (!path || path.length <= 2) return path

  const smoothed = [path[0]]
  let i = 0

  while (i < path.length - 1) {
    let j = path.length - 1
    // Walk backwards: skip to the furthest point we have safe LoS to
    while (j > i + 1) {
      if (_hasLOS(path[i], path[j])) break
      j--
    }
    smoothed.push(path[j])
    i = j
  }

  return smoothed
}

/**
 * Check single-cell-wide line-of-sight between two world points.
 * Used internally by _hasLOS to check multiple parallel corridors.
 */
function _hasLOSLine(ax, az, bx, bz) {
  let { col: c0, row: r0 } = _worldToCell(ax, az)
  const { col: c1, row: r1 } = _worldToCell(bx, bz)

  const dc  = Math.abs(c1 - c0)
  const dr  = Math.abs(r1 - r0)
  const sc  = c0 < c1 ? 1 : -1
  const sr  = r0 < r1 ? 1 : -1
  let err   = dc - dr

  while (c0 !== c1 || r0 !== r1) {
    if (_blocked(c0, r0)) return false
    const e2 = 2 * err
    if (e2 > -dr) { err -= dr; c0 += sc }
    if (e2 <  dc) { err += dc; r0 += sr }
  }

  return !_blocked(c1, r1)
}

/**
 * Fat line-of-sight check that accounts for robot width.
 * Tests the center path PLUS two paths offset ±ROBOT_RADIUS perpendicular
 * to the direction of travel. All three must be clear.
 * Prevents path smoothing from cutting corners that the robot body would clip.
 */
function _hasLOS(a, b) {
  // Center line
  if (!_hasLOSLine(a.x, a.z, b.x, b.z)) return false

  // Perpendicular offset (world space) = ROBOT_RADIUS in the direction ⊥ to travel
  const dx  = b.x - a.x
  const dz  = b.z - a.z
  const len = Math.sqrt(dx * dx + dz * dz)
  if (len < 0.01) return true   // zero-length segment — trivially clear

  const px = (-dz / len) * ROBOT_RADIUS   // perpendicular unit × ROBOT_RADIUS
  const pz = ( dx / len) * ROBOT_RADIUS

  // Left-offset path
  if (!_hasLOSLine(a.x + px, a.z + pz, b.x + px, b.z + pz)) return false
  // Right-offset path
  if (!_hasLOSLine(a.x - px, a.z - pz, b.x - px, b.z - pz)) return false

  return true
}

// ─── Internal: Waypoint Following ─────────────────────────────────────────────

function _getRobotHeading(robot) {
  // Extract yaw (rotation around world Y) from orientation quaternion.
  // Robot forward is +Z. After rotating by heading h around Y:
  // forward direction = (sin h, 0, cos h). Heading = 0 means facing +Z.
  const q = robot.orientation
  return Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z))
}

function _desiredHeading(dx, dz) {
  // Robot forward is +Z. After rotating by heading h around Y:
  // forward = (sin h, 0, cos h). To face toward (dx, 0, dz): h = atan2(dx, dz).
  // NOTE: old code had atan2(-dx, -dz) which was for forward = (0,0,-1) — that is WRONG now.
  return Math.atan2(dx, dz)
}

function _angularDiff(desired, current) {
  let diff = desired - current
  while (diff >  Math.PI) diff -= 2 * Math.PI
  while (diff < -Math.PI) diff += 2 * Math.PI
  return diff
}

function _followWaypoint(robot, wx, wz, speed) {
  return new Promise((resolve) => {
    // Register the active resolve so abortNavigation() can settle it instantly
    _currentResolve = resolve
    const start = Date.now()

    // Stuck detection: track position periodically
    let stuckCheckTime  = Date.now()
    let stuckLastX      = robot.position.x
    let stuckLastZ      = robot.position.z
    const STUCK_INTERVAL = 1500   // ms between stuck checks
    const STUCK_MIN_MOVE = 0.08   // metres — less than this = stuck

    function tick() {
      // EC-1: honour abort request — stop robot and resolve early
      if (_navAbortRequested) {
        robot.stop()
        _currentResolve = null
        resolve(false)
        return
      }

      // Timeout guard
      if (Date.now() - start > NAV_STEP_TIMEOUT) {
        robot.stop()
        _currentResolve = null
        resolve(false)
        return
      }

      // Stuck detection: if barely moved since last check, give up on this waypoint
      const now = Date.now()
      if (now - stuckCheckTime > STUCK_INTERVAL) {
        const movedX = robot.position.x - stuckLastX
        const movedZ = robot.position.z - stuckLastZ
        if (Math.sqrt(movedX * movedX + movedZ * movedZ) < STUCK_MIN_MOVE) {
          // Robot is stuck — stop and let navigateTo replan
          robot.stop()
          _currentResolve = null
          resolve(false)
          return
        }
        stuckLastX    = robot.position.x
        stuckLastZ    = robot.position.z
        stuckCheckTime = now
      }

      const dx   = wx - robot.position.x
      const dz   = wz - robot.position.z
      const dist = Math.sqrt(dx * dx + dz * dz)

      if (dist < ARRIVAL_THRESHOLD) {
        _currentResolve = null
        resolve(true)
        return
      }

      const desiredH  = _desiredHeading(dx, dz)
      const currentH  = _getRobotHeading(robot)
      const angDiff   = _angularDiff(desiredH, currentH)
      const turnSpeed = Math.sign(angDiff) * Math.min(Math.abs(angDiff) * 4.0, 3.0)

      if (Math.abs(angDiff) > 0.5) {
        // Large angle (>28°) — rotate in place; do NOT creep forward into obstacles
        robot.rotate(turnSpeed)
        robot.moveForward(0)
      } else if (Math.abs(angDiff) > TURN_THRESHOLD) {
        // Moderate angle — turn and very slight creep
        robot.rotate(turnSpeed)
        robot.moveForward(speed * 0.1)
      } else {
        // Aligned — drive forward with gentle heading correction
        robot.rotate(angDiff * 1.5)
        robot.moveForward(speed)
      }

      _currentRafHandle = requestAnimationFrame(tick)
    }

    _currentRafHandle = requestAnimationFrame(tick)
  })
}

// ─── Internal: Min-Heap ────────────────────────────────────────────────────────

class _MinHeap {
  constructor() {
    this._idx = []   // node indices
    this._pri = []   // f-scores
  }

  get size() { return this._idx.length }

  push(idx, priority) {
    this._idx.push(idx)
    this._pri.push(priority)
    this._bubbleUp(this._idx.length - 1)
  }

  pop() {
    const top = this._idx[0]
    const last = this._idx.pop()
    const lastP = this._pri.pop()
    if (this._idx.length > 0) {
      this._idx[0] = last
      this._pri[0] = lastP
      this._sinkDown(0)
    }
    return top
  }

  _bubbleUp(i) {
    while (i > 0) {
      const p = (i - 1) >> 1
      if (this._pri[p] <= this._pri[i]) break
      this._swap(p, i)
      i = p
    }
  }

  _sinkDown(i) {
    const n = this._idx.length
    while (true) {
      let s = i
      const l = 2 * i + 1
      const r = 2 * i + 2
      if (l < n && this._pri[l] < this._pri[s]) s = l
      if (r < n && this._pri[r] < this._pri[s]) s = r
      if (s === i) break
      this._swap(s, i)
      i = s
    }
  }

  _swap(a, b) {
    ;[this._idx[a], this._idx[b]] = [this._idx[b], this._idx[a]]
    ;[this._pri[a], this._pri[b]] = [this._pri[b], this._pri[a]]
  }
}
