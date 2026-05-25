import * as THREE from 'three'

// ─── Maze Configuration ──────────────────────────────────────────────────────
const ROWS   = 15       // 15×15 cells
const COLS   = 15
const CELL   = 3.0      // meters per cell (Wider corridors!)
const WALL_H = 3.5      // much taller walls
const WALL_T = 0.30     // thicker walls

// ─── State ───────────────────────────────────────────────────────────────────
let mazeMeshGroup  = null
let goalMarkerGroup = null
let _goalPulse     = 0
let _passages      = []
let _goalPos       = { x: 0, z: 0 }
let _startPos      = { x: 0, z: 0 }
let _wallSpecs     = []   // {x, y, z, hw, hh, hd} for physics

// ─── Helpers ─────────────────────────────────────────────────────────────────
function cellPos(r, c) {
  return {
    x: (c + 0.5 - COLS / 2) * CELL,
    z: (r + 0.5 - ROWS / 2) * CELL,
  }
}

// ─── Recursive-backtracker maze generator ────────────────────────────────────
function generateMaze() {
  const p = Array.from({ length: ROWS }, () =>
    Array.from({ length: COLS }, () => ({ right: false, down: false }))
  )
  const visited = Array.from({ length: ROWS }, () => new Array(COLS).fill(false))

  function carve(r, c) {
    visited[r][c] = true
    const dirs = [[0,1],[0,-1],[1,0],[-1,0]]
    for (let i = dirs.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [dirs[i], dirs[j]] = [dirs[j], dirs[i]]
    }
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc
      if (nr >= 0 && nr < ROWS && nc >= 0 && nc < COLS && !visited[nr][nc]) {
        if (dc ===  1) p[r][c].right  = true
        if (dc === -1) p[r][nc].right = true
        if (dr ===  1) p[r][c].down   = true
        if (dr === -1) p[nr][c].down  = true
        carve(nr, nc)
      }
    }
  }

  const sr = Math.floor(ROWS / 2), sc = Math.floor(COLS / 2)
  carve(sr, sc)
  return p
}

// ─── BFS to find farthest reachable cell ─────────────────────────────────────
function bfsFarthest(p, sr, sc) {
  const dist = Array.from({ length: ROWS }, () => new Array(COLS).fill(-1))
  dist[sr][sc] = 0
  const queue = [[sr, sc]]
  let farthest = [sr, sc], maxD = 0

  while (queue.length) {
    const [r, c] = queue.shift()
    const nbrs = []
    if (p[r][c].right  && c + 1 < COLS)    nbrs.push([r, c + 1])
    if (p[r][c].down   && r + 1 < ROWS)    nbrs.push([r + 1, c])
    if (c > 0 && p[r][c - 1].right)        nbrs.push([r, c - 1])
    if (r > 0 && p[r - 1][c].down)         nbrs.push([r - 1, c])
    for (const [nr, nc] of nbrs) {
      if (dist[nr][nc] === -1) {
        dist[nr][nc] = dist[r][c] + 1
        queue.push([nr, nc])
        if (dist[nr][nc] > maxD) { maxD = dist[nr][nc]; farthest = [nr, nc] }
      }
    }
  }
  return farthest
}

// ─── Build maze (visuals only, walls collected for physics) ──────────────────
function _buildMaze(scene) {
  _passages  = generateMaze()
  _wallSpecs = []

  mazeMeshGroup = new THREE.Group()
  mazeMeshGroup.name = 'mazeGroup'

  const wallMat = new THREE.MeshStandardMaterial({
    color: 0x2e3d4f,
    roughness: 0.88,
    metalness: 0.08,
  })

  // Horizontal walls (run along X axis, separate cells in Z)
  for (let r = 0; r <= ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const draw = r === 0 || r === ROWS || !_passages[r - 1][c].down
      if (!draw) continue
      const wx = CELL + WALL_T, wz = WALL_T
      const x  = (c + 0.5 - COLS / 2) * CELL
      const z  = (r - ROWS / 2) * CELL
      _addWallMesh(mazeMeshGroup, wallMat, x, z, wx, wz)
      _wallSpecs.push({ x, y: WALL_H / 2, z, hw: wx / 2, hh: WALL_H / 2, hd: wz / 2 })
    }
  }

  // Vertical walls (run along Z axis, separate cells in X)
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c <= COLS; c++) {
      const draw = c === 0 || c === COLS || !_passages[r][c - 1].right
      if (!draw) continue
      const wx = WALL_T, wz = CELL + WALL_T
      const x  = (c - COLS / 2) * CELL
      const z  = (r + 0.5 - ROWS / 2) * CELL
      _addWallMesh(mazeMeshGroup, wallMat, x, z, wx, wz)
      _wallSpecs.push({ x, y: WALL_H / 2, z, hw: wx / 2, hh: WALL_H / 2, hd: wz / 2 })
    }
  }

  scene.add(mazeMeshGroup)

  // Compute start & goal
  const sr = Math.floor(ROWS / 2), sc = Math.floor(COLS / 2)
  const s  = cellPos(sr, sc)
  _startPos = { x: s.x, z: s.z }

  const [gr, gc] = bfsFarthest(_passages, sr, sc)
  const g = cellPos(gr, gc)
  _goalPos = { x: g.x, z: g.z }

  // Build goal marker
  _buildGoalMarker(scene, _goalPos.x, _goalPos.z)

  console.log(`[Maze] ${ROWS}×${COLS} | cells ${_wallSpecs.length} walls | Start(${_startPos.x.toFixed(2)}, ${_startPos.z.toFixed(2)}) → Goal(${_goalPos.x.toFixed(2)}, ${_goalPos.z.toFixed(2)})`)
}

function _addWallMesh(group, mat, x, z, wx, wz) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(wx, WALL_H, wz), mat)
  mesh.position.set(x, WALL_H / 2, z)
  mesh.castShadow = mesh.receiveShadow = true
  mesh.name = 'mazeWall'
  group.add(mesh)
}

function _buildGoalMarker(scene, x, z) {
  goalMarkerGroup = new THREE.Group()
  goalMarkerGroup.name = 'goalMarker'

  const ring = new THREE.Mesh(
    new THREE.RingGeometry(0.22, 0.42, 48),
    new THREE.MeshStandardMaterial({
      color: 0x00ffaa,
      emissive: 0x00ff66,
      emissiveIntensity: 3.5,
      side: THREE.DoubleSide,
    })
  )
  ring.rotation.x = -Math.PI / 2
  ring.position.y = 0.02
  goalMarkerGroup.add(ring)

  const light = new THREE.PointLight(0x00ffaa, 2.5, 4.0)
  light.position.y = 0.6
  goalMarkerGroup.add(light)

  goalMarkerGroup.position.set(x, 0, z)
  scene.add(goalMarkerGroup)
}

function _clearMaze(scene) {
  if (mazeMeshGroup)   { scene.remove(mazeMeshGroup);   mazeMeshGroup   = null }
  if (goalMarkerGroup) { scene.remove(goalMarkerGroup); goalMarkerGroup = null }
  _wallSpecs = []
}

// ─── Public API ──────────────────────────────────────────────────────────────

/**
 * Call AFTER initPhysics() and initScene().
 * Returns the wall spec list so main.js can pass it to addMazeWalls().
 */
export function initMaze(scene) {
  _buildMaze(scene)
  return _wallSpecs
}

/**
 * Regenerate the maze for a new episode.
 * Physics bodies are NOT removed (Rapier doesn't support runtime removal easily).
 * Instead we only regenerate when training restarts with a fresh env.
 */
export function resetMazeVisuals(scene) {
  _clearMaze(scene)
  _buildMaze(scene)
  return _wallSpecs
}

/** Animate the goal marker pulse — call from render loop */
export function updateMaze(delta) {
  if (!goalMarkerGroup) return
  _goalPulse += delta * 2.5
  const s = 1 + 0.12 * Math.sin(_goalPulse)
  goalMarkerGroup.scale.setScalar(s)
}

export function getGoalPosition()  { return { ..._goalPos } }
export function getStartPosition() { return { ..._startPos } }
export function getMazeWallSpecs() { return _wallSpecs }
