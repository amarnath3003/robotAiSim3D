/**
 * nav/pathfinder.js — Clearance-Aware A* Pathfinding + Reactive Navigation
 *
 * The navigation core every movement primitive builds on. Guarantees:
 * - Obstacle-avoiding paths on a fine occupancy grid (0.25 m cells, ±16 m)
 * - Clearance-weighted A*: paths prefer corridor centres instead of hugging walls
 * - Two obstacle layers: static (registered AABBs) + dynamic (rebuilt from LiDAR)
 * - Fixed-timestep waypoint follower that runs as the engine's 'motor' system
 *   (60 Hz, same clock as physics — no more requestAnimationFrame drift)
 * - Reactive LiDAR layer while driving: slows near obstacles, steers through
 *   gaps, and triggers an immediate replan when the route is blocked mid-drive
 * - navigatePath(): follows arbitrary waypoint lists (shapes, patrol routes,
 *   any geometry) — blocked segments are detoured around with A*, then the
 *   original route resumes
 * - Honest failures: unreachable destinations reject with a clear Error so the
 *   planner/LLM can replan instead of silently pretending success
 *
 * No external dependencies.
 */

// ─── Grid Config ───────────────────────────────────────────────────────────────

const CELL_SIZE  = 0.25  // metres per cell (was 0.4 — finer grid, tighter paths)
const GRID_HALF  = 16    // world extent in each direction (±16 m)
const GRID_SIZE  = Math.round((GRID_HALF * 2) / CELL_SIZE)  // 128

const ROBOT_RADIUS = 0.5   // metres — inflate obstacles by this much

// Clearance costing: cells closer than this to any obstacle cost extra,
// so A* prefers routes with breathing room ("best path", not just shortest).
const CLEAR_SOFT_M  = 1.1
const CLEAR_PENALTY = 6.0

// ─── Occupancy Grids ───────────────────────────────────────────────────────────

// 1 = blocked, 0 = free  (static environment obstacles)
let _grid = new Uint8Array(GRID_SIZE * GRID_SIZE)

// Dynamic obstacle layer — rebuilt every perception tick from LiDAR hits.
// Kept separate so clearing it never erases static env data.
let _dynamicGrid = new Uint8Array(GRID_SIZE * GRID_SIZE)

// Clearance field (distance-to-nearest-obstacle, in cells). Recomputed at the
// start of each findPath() call from the combined static+dynamic grids.
let _clearance = new Float32Array(GRID_SIZE * GRID_SIZE)

// Registry: id → {minX, maxX, minZ, maxZ} (world-space, pre-inflation)
const _obstacles = new Map()

// Latest raw LiDAR scan — cached by updateDynamicObstacles() for the reactive
// steering layer in the motor tick.
let _lastScan = null   // {x, z, yaw, distances, fov, range, rays}

/**
 * Rebuild dynamic obstacle layer from the latest LiDAR hit distances,
 * and cache the raw scan for the reactive driving layer.
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
  _lastScan = { x: robotX, z: robotZ, yaw, distances, fov, range, rays }

  _dynamicGrid.fill(0)

  const halfFov   = fov / 2
  const step      = rays > 1 ? fov / (rays - 1) : 0
  const inflCells = Math.ceil(ROBOT_RADIUS / CELL_SIZE)

  for (let i = 0; i < rays && i < distances.length; i++) {
    const d = distances[i]
    if (d >= range * 0.98) continue  // no real hit — ray reached max range
    if (d < 0.35) continue           // touching the robot — don't block our own cell

    const angle = yaw - halfFov + i * step
    const hitX  = robotX + Math.sin(angle) * d
    const hitZ  = robotZ + Math.cos(angle) * d

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
  _lastScan = null
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
 * Check whether a world-space point is inside a static obstacle.
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
 * Clearance-weighted A*: prefers routes that keep distance from obstacles.
 * Returns an array of {x, z} waypoints, or null if no path exists.
 * The start position is NOT included in the returned array.
 */
export function findPath(startX, startZ, goalX, goalZ) {
  let startCell = _worldToCell(startX, startZ)
  let goalCell  = _worldToCell(goalX, goalZ)

  // Clamp to grid bounds
  startCell.col = Math.max(0, Math.min(GRID_SIZE - 1, startCell.col))
  startCell.row = Math.max(0, Math.min(GRID_SIZE - 1, startCell.row))
  goalCell.col  = Math.max(0, Math.min(GRID_SIZE - 1, goalCell.col))
  goalCell.row  = Math.max(0, Math.min(GRID_SIZE - 1, goalCell.row))

  // If start is blocked (robot inside an inflation zone), snap to nearest clear
  if (_blocked(startCell.col, startCell.row)) {
    const alt = _nearestClear(startCell.col, startCell.row)
    if (alt) startCell = alt
  }

  // If goal is blocked (e.g. the target object itself), plan to nearest clear cell
  if (_blocked(goalCell.col, goalCell.row)) {
    const alt = _nearestClear(goalCell.col, goalCell.row, 16)
    if (!alt) return null
    goalCell = alt
  }

  _computeClearance()

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

      // Clearance-weighted step cost: hugging obstacles is expensive
      const tentG = gCost[ci] + cost * _cellCostMul(ni)
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

// Follower tuning
const WP_ADVANCE      = 0.45   // metres — advance to next waypoint within this
const WP_ADVANCE_TIGHT = 0.28  // strict mode (shape tracing) — keep corners crisp
const STOP_DIST_DEF   = 0.45   // metres — reactive hard-stop clearance
const SLOW_DIST       = 1.6    // metres — start slowing below this clearance
const BLOCK_PATIENCE  = 0.9    // seconds fully blocked before requesting replan
const STUCK_WINDOW    = 1.4    // seconds between stuck checks
const STUCK_MIN_MOVE  = 0.07   // metres — less than this while driving = stuck
const PATH_CHECK_SEC  = 0.3    // seconds between mid-route path validity checks

let _navAbortRequested = false
let _nav        = null   // active leg state (owned by navMotorTick)
let _activePath = []     // remaining path for debug visualisation

/**
 * Get the currently planned path (for debug visualisation).
 * @returns {Array<{x:number,z:number}>}
 */
export function getActivePath() {
  return _activePath
}

/** Is a navigation currently in progress? */
export function isNavigating() {
  return _nav !== null
}

/**
 * Signal any in-progress navigation to stop immediately.
 */
export function abortNavigation() {
  _navAbortRequested = true
  if (_nav) _finishLeg('aborted')
}

/**
 * Fixed-timestep navigation follower. Registered as the engine's 'motor'
 * system in main.js — runs every physics substep (60 Hz) BEFORE movement is
 * committed to physics, so velocity commands are always physics-clocked.
 *
 * Combines waypoint following with a reactive LiDAR layer:
 *   - pure-pursuit style lookahead on smoothed A* paths
 *   - speed governor from live obstacle clearance
 *   - gap-seeking steering when blocked, escalating to a replan request
 *   - stuck watchdog with reverse-out recovery
 *   - periodic path re-validation against the live occupancy grids
 *
 * @param {number} dt     Fixed timestep (1/60 s)
 * @param {Object} robot  Active RobotInstance
 */
export function navMotorTick(dt, robot) {
  const nav = _nav
  if (!nav || nav.robot !== robot) return

  if (_navAbortRequested) return _finishLeg('aborted')

  nav.t += dt
  if (nav.t > nav.opts.legTimeout / 1000) return _finishLeg('timeout')

  // ── Recovery: reverse away from whatever we're wedged against ──────────────
  if (nav.recovery) {
    nav.recovery.t -= dt
    robot.rotate(0)
    robot.moveForward(-0.9)
    if (nav.recovery.t <= 0) return _finishLeg('stuck')
    return
  }

  const px = robot.position.x
  const pz = robot.position.z

  // ── Arrival on the TRUE goal (not just the last grid waypoint) ─────────────
  const goalDist = Math.hypot(nav.goal.x - px, nav.goal.z - pz)
  if (goalDist <= nav.opts.arrive) return _finishLeg('arrived')

  // ── Sensor-guided final creep ───────────────────────────────────────────────
  // The grid (robot-radius inflated twice over) can only take us so close to a
  // physical target. From the best grid-reachable point, inch straight toward
  // the goal on live LiDAR clearance, stopping at a hard safety floor.
  if (nav.creep) {
    nav.creep.t += dt
    const wantH = Math.atan2(nav.goal.x - px, nav.goal.z - pz)
    const curH  = _getRobotHeading(robot)
    const diff  = _angularDiff(wantH, curH)
    const clearAhead = _sectorClear(curH, 0.5)

    if (clearAhead < 0.42 || nav.creep.t > 3.0) return _finishLeg('arrived')

    if (Math.abs(diff) > 0.2) {
      robot.moveForward(0)
      robot.rotate(_clamp(diff * 3.0, -2.5, 2.5))
    } else {
      robot.rotate(_clamp(diff * 2.0, -1.5, 1.5))
      robot.moveForward(0.35)
    }
    return
  }

  // ── Waypoint bookkeeping ────────────────────────────────────────────────────
  const advanceDist = nav.opts.strict ? WP_ADVANCE_TIGHT : WP_ADVANCE
  let wp = nav.path[nav.wpIndex]
  while (
    nav.wpIndex < nav.path.length - 1 &&
    Math.hypot(wp.x - px, wp.z - pz) < advanceDist
  ) {
    nav.wpIndex++
    wp = nav.path[nav.wpIndex]
    _activePath = nav.path.slice(nav.wpIndex)
  }

  const wpDist = Math.hypot(wp.x - px, wp.z - pz)

  // On the final waypoint with the true goal still out of tolerance:
  // - goal cell blocked (object target / point inside a wall) → hand over to
  //   the sensor-guided creep for the last stretch
  // - free goal, reasonably close → accept
  // - free goal, far → the plan under-delivered, ask for a replan
  if (nav.wpIndex >= nav.path.length - 1) {
    const endReached =
      wpDist < 0.26 ||
      (goalDist <= Math.max(nav.opts.arrive, 1.35) && wpDist < advanceDist)
    if (endReached) {
      if (nav.goalBlocked && goalDist > nav.opts.arrive) {
        nav.creep = { t: 0 }
        return
      }
      if (goalDist <= Math.max(nav.opts.arrive, 1.35)) return _finishLeg('arrived')
      return _finishLeg('stuck')
    }
    // else fall through — keep driving onto the final waypoint
  }

  // ── Lookahead (skip ahead on the path when there's a clear straight line) ──
  // Centre-line LOS is the right test here: grid cells are already inflated
  // by the robot radius, so a clear centre line IS a driveable line.
  let target = wp
  if (!nav.opts.strict) {
    const maxSkip = Math.min(nav.wpIndex + 3, nav.path.length - 1)
    for (let j = maxSkip; j > nav.wpIndex; j--) {
      const cand = nav.path[j]
      if (
        Math.hypot(cand.x - px, cand.z - pz) <= 1.25 &&
        _hasLOSLine(px, pz, cand.x, cand.z)
      ) {
        nav.wpIndex = j
        target = cand
        _activePath = nav.path.slice(j)
        break
      }
    }
  }

  // ── Heading control ─────────────────────────────────────────────────────────
  const dx = target.x - px
  const dz = target.z - pz
  const desiredH = Math.atan2(dx, dz)
  const curH     = _getRobotHeading(robot)
  const aDiff    = _angularDiff(desiredH, curH)
  const turnCmd  = _clamp(aDiff * 3.5, -3.2, 3.2)

  let commandedSpeed = 0

  if (Math.abs(aDiff) > 0.55) {
    // Large error — rotate in place, never creep blindly into obstacles
    robot.rotate(turnCmd)
    robot.moveForward(0)
  } else {
    // ── Reactive clearance from the live LiDAR scan ───────────────────────────
    const clearAhead = _sectorClear(curH, 0.55)       // robot's actual forward cone
    const clearRoute = _sectorClear(curH, 0.50, aDiff) // toward the target direction
    const effClear   = Math.min(clearAhead, clearRoute)

    let speed = nav.opts.speed
    speed *= Math.max(0.25, Math.cos(aDiff))                                   // turning → slower
    speed *= _clamp(goalDist / 1.4, nav.opts.strict ? 0.45 : 0.35, 1)          // arrival ramp
    if (nav.opts.strict) speed *= _clamp(wpDist / 1.0, 0.4, 1)                 // crisp corners

    const stopDist  = nav.opts.stopDist
    const goalIsNear = goalDist < nav.opts.arrive + 0.9

    if (goalIsNear) {
      // Final approach: the "obstacle" ahead is usually the target itself.
      // Keep a hard floor so we still never physically strike anything.
      if (effClear < 0.30) speed = 0
      else speed *= 0.6
    } else {
      if (effClear < stopDist) speed = 0
      else if (effClear < SLOW_DIST) {
        speed *= Math.max(0.22, (effClear - stopDist) / (SLOW_DIST - stopDist))
      }
    }

    if (speed > 0.01) {
      robot.rotate(_clamp(aDiff * 2.5, -2.5, 2.5))
      robot.moveForward(speed)
      commandedSpeed = speed
      nav.blockT = 0
    } else {
      // Blocked dead ahead — steer toward the most promising gap while the
      // block timer runs; if no gap opens, request a replan.
      const esc = _bestGapDirection(curH, desiredH)
      if (esc !== null) {
        robot.rotate(_clamp(_angularDiff(esc, curH) * 3.0, -3.2, 3.2))
        robot.moveForward(0.12)
      } else {
        robot.stop()
      }
      nav.blockT += dt
      if (nav.blockT > BLOCK_PATIENCE) return _finishLeg('blocked')
    }
  }

  // ── Stuck watchdog ──────────────────────────────────────────────────────────
  nav.stuck.t += dt
  if (nav.stuck.t > STUCK_WINDOW) {
    const moved = Math.hypot(px - nav.stuck.x, pz - nav.stuck.z)
    if (moved < STUCK_MIN_MOVE && commandedSpeed > 0.05) {
      nav.recovery = { t: 0.5 }   // reverse ~0.45 m, then resolve 'stuck' → replan
      return
    }
    nav.stuck = { x: px, z: pz, t: 0 }
  }

  // ── Periodic mid-route path validation (walls/objects moved into the way) ──
  nav.checkT += dt
  if (nav.checkT > PATH_CHECK_SEC) {
    nav.checkT = 0
    if (!_pathStillClear(nav, px, pz)) return _finishLeg('blocked')
  }
}

/**
 * Drive the robot to (targetX, targetZ) with full obstacle avoidance.
 *
 * Options (or a plain number = speed, for backwards compatibility):
 *   speed      m/s (clamped to manifest maxSpeed)
 *   arrive     arrival tolerance in metres (default 0.3)
 *   approach   stop this far from the goal point (for objects — overrides
 *              arrive when larger; the goal cell may be inside the object)
 *   face       {x,z} point to rotate toward after arriving
 *   timeout    total ms budget (default 45000)
 *
 * Resolves {arrived, finalDist}. THROWS when the destination is genuinely
 * unreachable or navigation keeps failing — callers (skills/LLM) get an
 * honest signal instead of silent failure.
 */
export async function navigateTo(robot, targetX, targetZ, optsOrSpeed) {
  const opts = _normOpts(robot, optsOrSpeed)
  _navAbortRequested = false

  const goal = { x: _clampArena(targetX), z: _clampArena(targetZ) }
  const deadline = Date.now() + opts.timeout

  const MAX_ATTEMPTS = 8
  let attempts = 0
  let noPathRetries = 0

  while (true) {
    if (_navAbortRequested) {
      robot.stop()
      _activePath = []
      return { arrived: false, finalDist: _distTo(robot, goal), reason: 'aborted' }
    }
    if (Date.now() > deadline) {
      robot.stop()
      _activePath = []
      throw new Error(`Navigation to (${goal.x.toFixed(1)}, ${goal.z.toFixed(1)}) timed out`)
    }

    if (_distTo(robot, goal) <= opts.arrive) break   // already there

    const path = findPath(robot.position.x, robot.position.z, goal.x, goal.z)

    if (!path) {
      // Might be a transient dynamic blockage (something rolling past) — wait briefly
      if (noPathRetries < 2) {
        noPathRetries++
        await _sleep(300)
        continue
      }
      robot.stop()
      _activePath = []
      throw new Error(
        `No path to (${goal.x.toFixed(1)}, ${goal.z.toFixed(1)}) — destination is unreachable from here`
      )
    }

    if (path.length === 0) break   // same cell as goal

    _activePath = path
    const outcome = await _startLeg(robot, path, opts, goal)

    if (outcome === 'arrived') break
    if (outcome === 'aborted') {
      robot.stop()
      _activePath = []
      return { arrived: false, finalDist: _distTo(robot, goal), reason: 'aborted' }
    }

    // blocked / stuck / timeout → replan from wherever we are now
    attempts++
    if (attempts >= MAX_ATTEMPTS) {
      robot.stop()
      _activePath = []
      throw new Error(
        `Could not reach (${goal.x.toFixed(1)}, ${goal.z.toFixed(1)}) — route kept getting blocked (${attempts} replans)`
      )
    }
    await _sleep(120)   // let dynamic obstacles move / grids refresh
  }

  robot.stop()
  _activePath = []

  if (opts.face) await _faceTowards(robot, opts.face.x, opts.face.z)

  return { arrived: true, finalDist: _distTo(robot, goal) }
}

/**
 * Follow an ordered list of world-space waypoints (any geometry: shapes,
 * patrol routes, LLM-generated trajectories). Each leg is driven straight
 * when the line is clear; blocked legs are detoured around via A*, then the
 * route resumes at the next vertex. Unreachable vertices are skipped and
 * reported rather than failing the whole path.
 *
 * @param {Object} robot
 * @param {Array<{x:number,z:number}>} points
 * @param {Object} [options]  {speed, arrive (default 0.32), strict (default true)}
 * @returns {Promise<{reached:number, skipped:number, total:number, aborted:boolean}>}
 */
export async function navigatePath(robot, points, options = {}) {
  const opts = _normOpts(robot, {
    ...options,
    arrive: options.arrive ?? 0.32,
    strict: options.strict ?? true,
  })
  _navAbortRequested = false

  const result = { reached: 0, skipped: 0, total: points.length, aborted: false }

  for (const raw of points) {
    if (_navAbortRequested) { result.aborted = true; break }

    const pt = { x: _clampArena(raw.x), z: _clampArena(raw.z) }
    let done = false

    for (let attempt = 0; attempt < 3 && !done; attempt++) {
      if (_navAbortRequested) { result.aborted = true; break }

      if (_distTo(robot, pt) <= opts.arrive) { done = true; break }

      // Straight leg when the direct line is clear, A* detour otherwise
      let path
      if (_hasLOS({ x: robot.position.x, z: robot.position.z }, pt)) {
        path = [pt]
      } else {
        path = findPath(robot.position.x, robot.position.z, pt.x, pt.z)
        if (!path) break              // vertex unreachable → skip it
        if (path.length === 0) { done = true; break }
      }

      _activePath = path
      const outcome = await _startLeg(robot, path, opts, pt)

      if (outcome === 'arrived') done = true
      else if (outcome === 'aborted') { result.aborted = true; break }
      else await _sleep(100)          // blocked/stuck → retry with a detour
    }

    if (result.aborted) break

    if (done && _distTo(robot, pt) <= Math.max(opts.arrive + 0.5, 0.8)) {
      result.reached++
    } else {
      result.skipped++
    }
  }

  robot.stop()
  _activePath = []
  return result
}

// ─── Debug: Grid Snapshot ──────────────────────────────────────────────────────

export function getGridInfo() {
  return {
    size: GRID_SIZE,
    cellSize: CELL_SIZE,
    halfExtent: GRID_HALF,
    obstacleCount: _obstacles.size,
    blockedCells: _grid.reduce((s, v) => s + v, 0),
  }
}

// ─── Internal: Leg Lifecycle ───────────────────────────────────────────────────

function _startLeg(robot, path, opts, goal) {
  return new Promise((resolve) => {
    // Take over from any leg another caller left active
    if (_nav) {
      const prev = _nav.resolve
      _nav = null
      prev('aborted')
    }
    // Is the true goal cell blocked right now? Then the path's end point is
    // the best physically reachable spot (object targets, points in walls).
    const gc = _worldToCell(goal.x, goal.z)
    const goalBlocked = _blocked(
      Math.max(0, Math.min(GRID_SIZE - 1, gc.col)),
      Math.max(0, Math.min(GRID_SIZE - 1, gc.row))
    )
    _nav = {
      robot,
      path,
      wpIndex: 0,
      opts,
      goal,
      goalBlocked,
      resolve,
      t: 0,
      blockT: 0,
      checkT: 0,
      stuck: { x: robot.position.x, z: robot.position.z, t: 0 },
      recovery: null,
    }
  })
}

function _finishLeg(outcome) {
  const nav = _nav
  if (!nav) return
  _nav = null
  nav.robot.stop()
  if (globalThis.__NAV_DEBUG) {
    console.log(
      `[Nav] leg → ${outcome} @ (${nav.robot.position.x.toFixed(2)}, ${nav.robot.position.z.toFixed(2)})` +
      ` wp ${nav.wpIndex + 1}/${nav.path.length} blockT ${nav.blockT.toFixed(2)}`
    )
  }
  nav.resolve(outcome)
}

function _pathStillClear(nav, px, pz) {
  // Centre-line checks only — the grid is already robot-radius inflated.
  // (The fat-LOS variant would flag legitimate narrow corridors as blocked.)
  let prev = { x: px, z: pz }
  let span = 0
  for (let i = nav.wpIndex; i < nav.path.length && span < 6; i++) {
    const p = nav.path[i]
    // Final approach near the goal is owned by the reactive layer — the target
    // object itself shows up as an "obstacle" there, which is expected.
    if (Math.hypot(p.x - nav.goal.x, p.z - nav.goal.z) < 1.3) break
    if (!_hasLOSLine(prev.x, prev.z, p.x, p.z)) return false
    span += Math.hypot(p.x - prev.x, p.z - prev.z)
    prev = p
  }
  return true
}

// ─── Internal: Reactive LiDAR Layer ────────────────────────────────────────────

/**
 * Minimum LiDAR distance within a cone around a direction.
 * @param {number} yaw        Robot's current heading (world)
 * @param {number} halfWidth  Cone half-angle (radians)
 * @param {number} [relDir]   Cone centre relative to heading (default 0 = dead ahead)
 */
function _sectorClear(yaw, halfWidth, relDir = 0) {
  const s = _lastScan
  if (!s || !s.distances || s.rays < 2) return Infinity

  // Scan ray i sits at (scanYaw - fov/2 + i*step) in world space.
  // We want rays near (yaw + relDir) → relative to scan frame:
  const centre = _angNorm(relDir + (yaw - s.yaw))
  const step   = s.fov / (s.rays - 1)

  let min = s.range
  for (let i = 0; i < s.rays; i++) {
    const rel = -s.fov / 2 + i * step
    if (Math.abs(_angNorm(rel - centre)) <= halfWidth) {
      if (s.distances[i] < min) min = s.distances[i]
    }
  }
  return min
}

/**
 * Find the most promising open direction when blocked ahead: maximise
 * clearance while staying as close to the goal direction as possible.
 * Returns a world heading, or null when everything nearby is walled off.
 */
function _bestGapDirection(curH, desiredH) {
  const s = _lastScan
  if (!s) return null

  let best = null
  let bestScore = -Infinity

  for (let deg = -110; deg <= 110; deg += 10) {
    const rel = deg * (Math.PI / 180)
    const clear = _sectorClear(curH, 0.30, rel)
    if (clear < STOP_DIST_DEF + 0.15) continue

    const world = _angNorm(curH + rel)
    const score = Math.min(clear, 2.2) - 0.55 * Math.abs(_angularDiff(world, desiredH))
    if (score > bestScore) {
      bestScore = score
      best = world
    }
  }
  return best
}

// ─── Internal: Options / Small Helpers ─────────────────────────────────────────

function _normOpts(robot, o) {
  if (typeof o === 'number') o = { speed: o }
  o = o || {}
  const maxSpeed = robot.manifest?.constraints?.maxSpeed ?? 2.0
  return {
    speed:      Math.min(o.speed || maxSpeed, maxSpeed),
    arrive:     Math.max(o.arrive ?? 0.3, o.approach ?? 0),
    face:       o.face || null,
    strict:     !!o.strict,
    stopDist:   o.stopDist ?? STOP_DIST_DEF,
    timeout:    o.timeout ?? 45000,
    legTimeout: o.legTimeout ?? 20000,
  }
}

function _distTo(robot, p) {
  return Math.hypot(p.x - robot.position.x, p.z - robot.position.z)
}

function _clampArena(v) {
  const lim = GRID_HALF - 0.8
  return Math.max(-lim, Math.min(lim, v))
}

function _clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v))
}

function _sleep(ms) {
  return new Promise(r => setTimeout(r, ms))
}

async function _faceTowards(robot, tx, tz) {
  const deadline = Date.now() + 3000
  while (Date.now() < deadline && !_navAbortRequested) {
    const want = Math.atan2(tx - robot.position.x, tz - robot.position.z)
    const diff = _angularDiff(want, _getRobotHeading(robot))
    if (Math.abs(diff) < 0.09) break
    robot.moveForward(0)
    robot.rotate(_clamp(diff * 3.0, -2.5, 2.5))
    await _sleep(33)
  }
  robot.stop()
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
  return Math.max(dx, dz) + (Math.SQRT2 - 1) * Math.min(dx, dz)
}

function _nearestClear(col, row, maxRadius = 12) {
  // Scan expanding Chebyshev rings, but return the EUCLIDEAN-closest free
  // cell among the first two rings that contain any (first-hit-in-scan-order
  // could be a far corner cell, sending the robot to the wrong side).
  let best = null
  let bestD = Infinity
  let foundRing = -1

  for (let r = 1; r <= maxRadius; r++) {
    if (foundRing >= 0 && r > foundRing + 1) break
    for (let dc = -r; dc <= r; dc++) {
      for (let dr = -r; dr <= r; dr++) {
        if (Math.abs(dc) !== r && Math.abs(dr) !== r) continue
        const c = col + dc
        const rr = row + dr
        if (_inBounds(c, rr) && !_blocked(c, rr)) {
          if (foundRing < 0) foundRing = r
          const d = dc * dc + dr * dr
          if (d < bestD) { bestD = d; best = { col: c, row: rr } }
        }
      }
    }
  }
  return best
}

// ─── Internal: Clearance Field (distance transform) ────────────────────────────

/**
 * Two-pass chamfer distance transform over the combined static+dynamic grids.
 * _clearance[i] = distance (in cells) to the nearest blocked cell.
 */
function _computeClearance() {
  const N = GRID_SIZE
  const BIG = 1e6
  const cl = _clearance

  for (let i = 0; i < N * N; i++) {
    cl[i] = (_grid[i] === 1 || _dynamicGrid[i] === 1) ? 0 : BIG
  }

  // Forward pass
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const i = r * N + c
      let v = cl[i]
      if (v === 0) continue
      if (c > 0)          v = Math.min(v, cl[i - 1] + 1)
      if (r > 0)          v = Math.min(v, cl[i - N] + 1)
      if (c > 0 && r > 0) v = Math.min(v, cl[i - N - 1] + Math.SQRT2)
      if (c < N - 1 && r > 0) v = Math.min(v, cl[i - N + 1] + Math.SQRT2)
      cl[i] = v
    }
  }
  // Backward pass
  for (let r = N - 1; r >= 0; r--) {
    for (let c = N - 1; c >= 0; c--) {
      const i = r * N + c
      let v = cl[i]
      if (v === 0) continue
      if (c < N - 1)              v = Math.min(v, cl[i + 1] + 1)
      if (r < N - 1)              v = Math.min(v, cl[i + N] + 1)
      if (c < N - 1 && r < N - 1) v = Math.min(v, cl[i + N + 1] + Math.SQRT2)
      if (c > 0 && r < N - 1)     v = Math.min(v, cl[i + N - 1] + Math.SQRT2)
      cl[i] = v
    }
  }
}

function _cellCostMul(idx) {
  const clearM = _clearance[idx] * CELL_SIZE
  if (clearM >= CLEAR_SOFT_M) return 1
  const t = (CLEAR_SOFT_M - clearM) / CLEAR_SOFT_M
  return 1 + CLEAR_PENALTY * t * t
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
    // Walk backwards: skip to the furthest point with safe, wide line-of-sight.
    // minClear=2 cells keeps shortcuts out of the tight-clearance band the
    // A* cost field just paid to avoid.
    while (j > i + 1) {
      if (_hasLOS(path[i], path[j], 2)) break
      j--
    }
    smoothed.push(path[j])
    i = j
  }

  return smoothed
}

/**
 * Check single-cell-wide line-of-sight between two world points.
 * @param {number} minClear  Optional minimum clearance (cells) along the line —
 *                           only meaningful right after _computeClearance().
 */
function _hasLOSLine(ax, az, bx, bz, minClear = 0) {
  let { col: c0, row: r0 } = _worldToCell(ax, az)
  const { col: c1, row: r1 } = _worldToCell(bx, bz)

  const dc  = Math.abs(c1 - c0)
  const dr  = Math.abs(r1 - r0)
  const sc  = c0 < c1 ? 1 : -1
  const sr  = r0 < r1 ? 1 : -1
  let err   = dc - dr

  while (c0 !== c1 || r0 !== r1) {
    if (_blocked(c0, r0)) return false
    if (minClear > 0 && _inBounds(c0, r0) && _clearance[r0 * GRID_SIZE + c0] < minClear) return false
    const e2 = 2 * err
    if (e2 > -dr) { err -= dr; c0 += sc }
    if (e2 <  dc) { err += dc; r0 += sr }
  }

  return !_blocked(c1, r1)
}

/**
 * Fat line-of-sight check that accounts for robot width.
 * Tests the centre path PLUS two paths offset ±ROBOT_RADIUS perpendicular
 * to the direction of travel. All three must be clear.
 */
function _hasLOS(a, b, minClear = 0) {
  if (!_hasLOSLine(a.x, a.z, b.x, b.z, minClear)) return false

  const dx  = b.x - a.x
  const dz  = b.z - a.z
  const len = Math.sqrt(dx * dx + dz * dz)
  if (len < 0.01) return true

  const px = (-dz / len) * ROBOT_RADIUS
  const pz = ( dx / len) * ROBOT_RADIUS

  if (!_hasLOSLine(a.x + px, a.z + pz, b.x + px, b.z + pz)) return false
  if (!_hasLOSLine(a.x - px, a.z - pz, b.x - px, b.z - pz)) return false

  return true
}

// ─── Internal: Heading Math ────────────────────────────────────────────────────

function _getRobotHeading(robot) {
  // Extract yaw (rotation around world Y). Robot forward is +Z:
  // forward = (sin h, 0, cos h); heading 0 = facing +Z.
  const q = robot.orientation
  return Math.atan2(2 * (q.w * q.y + q.x * q.z), 1 - 2 * (q.y * q.y + q.z * q.z))
}

function _angularDiff(desired, current) {
  let diff = desired - current
  while (diff >  Math.PI) diff -= 2 * Math.PI
  while (diff < -Math.PI) diff += 2 * Math.PI
  return diff
}

function _angNorm(a) {
  while (a >  Math.PI) a -= 2 * Math.PI
  while (a < -Math.PI) a += 2 * Math.PI
  return a
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
