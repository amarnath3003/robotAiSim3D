/**
 * debug/visualizer.js — Real-Time Debug Overlays
 *
 * Renders live debug geometry directly into the Three.js scene:
 *   - LiDAR ray lines        (toggle: L)
 *   - CV Camera FOV Cone     (toggle: L)
 *   - A* navigation path     (toggle: P)
 *   - Occupancy grid overlay (toggle: G)
 *
 * Press V to toggle ALL layers at once.
 * Layers are intentionally OFF by default so they don't affect production runs.
 *
 * Usage:
 *   import { initVisualizer, updateLidarRays, updateFovCone, setNavPath, rebuildOccupancyGrid } from '../debug/visualizer.js'
 *   initVisualizer(scene)
 *   // each tick:
 *   updateLidarRays(robot.position, robot.heading, lidarDistances)
 *   updateFovCone(robot.position, robot.heading)
 */

import * as THREE from 'three'
import { getGridInfo, isOccupied } from '../nav/pathfinder.js'
import { getManifest } from '../core/manifest.js'
import { getState, setState, subscribe } from '../core/state.js'

// ─── Constants ─────────────────────────────────────────────────────────────────

const MAX_LIDAR_RAYS  = 128   // Pre-allocated vertex buffer size (72-ray LiDAR fits)
const MAX_PATH_POINTS = 128   // Max waypoints to visualise

// ─── Module State ──────────────────────────────────────────────────────────────

const LAYERS = {
  lidar: false,
  path:  true,
  grid:  false,
}

let _scene       = null
let _lidarLines  = null   // THREE.LineSegments — one segment per LiDAR ray
let _coneViz     = null   // THREE.LineSegments — camera FOV cone wires
let _pathLine    = null   // THREE.Line         — active A* path
let _gridPoints  = null   // THREE.Points       — occupied cells (rebuilt on demand)
let _keyHandler  = null
let _unsubscribers = []

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialize the visualizer. Must be called after initScene().
 * @param {THREE.Scene} scene
 */
export function initVisualizer(scene) {
  if (_scene) destroyVisualizer()   // guard against double-init

  _scene = scene
  _buildLidarGeometry()
  _buildFovConeGeometry()
  _buildPathGeometry()

  // Sync layers with state initially
  LAYERS.lidar = !!getState('ui.showSensorRays')
  _applyVisibility()

  // Subscribe to state changes
  _unsubscribers.push(subscribe('ui.showSensorRays', (val) => {
    LAYERS.lidar = !!val
    _applyVisibility()
  }))

  _keyHandler = (e) => {
    // Ignore key presses if typing in text inputs
    if (document.activeElement?.tagName === 'INPUT' || document.activeElement?.tagName === 'TEXTAREA') {
      return
    }

    switch (e.key.toUpperCase()) {
      case 'V': {
        const next = !getState('ui.showSensorRays')
        setState('ui.showSensorRays', next)
        LAYERS.path  = next
        LAYERS.grid  = next
        _applyVisibility()
        if (next) rebuildOccupancyGrid()
        console.log(`[Viz] All layers ${next ? 'ON' : 'OFF'}`)
        break
      }
      case 'L': {
        const next = !getState('ui.showSensorRays')
        setState('ui.showSensorRays', next)
        console.log(`[Viz] LiDAR/CV rays ${next ? 'ON' : 'OFF'}`)
        break
      }
      case 'P':
        LAYERS.path = !LAYERS.path
        if (_pathLine) _pathLine.visible = LAYERS.path
        console.log(`[Viz] Nav path ${LAYERS.path ? 'ON' : 'OFF'}`)
        break
      case 'G':
        LAYERS.grid = !LAYERS.grid
        if (LAYERS.grid) rebuildOccupancyGrid()
        else if (_gridPoints) _gridPoints.visible = false
        console.log(`[Viz] Occupancy grid ${LAYERS.grid ? 'ON' : 'OFF'}`)
        break
    }
  }

  window.addEventListener('keydown', _keyHandler)
  console.log('[Visualizer] Ready  V=all  L=lidar  P=path  G=grid')
}

/**
 * Destroy all visualizer objects and remove event listeners.
 */
export function destroyVisualizer() {
  if (_keyHandler) {
    window.removeEventListener('keydown', _keyHandler)
    _keyHandler = null
  }

  _unsubscribers.forEach(fn => fn?.())
  _unsubscribers = []

  for (const obj of [_lidarLines, _coneViz, _pathLine, _gridPoints]) {
    if (obj) {
      obj.geometry?.dispose()
      obj.material?.dispose()
      obj.parent?.remove(obj)
    }
  }

  _lidarLines = _coneViz = _pathLine = _gridPoints = null
  _scene = null
}

/**
 * Update LiDAR ray line positions from the latest sensor readings.
 * Color-codes lines dynamically based on proximity to obstacles.
 *
 * @param {{x:number,y:number,z:number}} origin   Robot world position
 * @param {number}       facingAngle               Robot Y-rotation (radians)
 * @param {Float32Array} distances                 LiDAR distance readings (one per ray)
 */
export function updateLidarRays(origin, facingAngle, distances) {
  if (!_lidarLines) return
  _lidarLines.visible = LAYERS.lidar
  if (!LAYERS.lidar || !distances || distances.length === 0) return

  const manifest = getManifest?.()
  const lidarSensor = manifest?.sensors?.find(s => s.type === 'lidar') || {}
  const fov = lidarSensor.config?.fov || 360
  const range = lidarSensor.config?.range || 5.0
  const sensorOffset = lidarSensor.position || { x: 0, y: 0.5, z: 0 }

  const sensorPos = new THREE.Vector3(
    origin.x + (sensorOffset.x || 0),
    origin.y + (sensorOffset.y || 0.5),
    origin.z + (sensorOffset.z || 0)
  )

  const rays       = Math.min(distances.length, MAX_LIDAR_RAYS)
  const fovRad     = fov * (Math.PI / 180)
  const angleStep  = fovRad / Math.max(rays - 1, 1)
  const startAngle = rays > 1 ? facingAngle - (fovRad / 2) : facingAngle

  const posAttr = _lidarLines.geometry.attributes.position
  const colAttr = _lidarLines.geometry.attributes.color

  const C_OK   = new THREE.Color(0x00ff88)
  const C_MID  = new THREE.Color(0xffaa00)
  const C_BAD  = new THREE.Color(0xff2244)
  const safeDist = range * 0.4

  for (let i = 0; i < MAX_LIDAR_RAYS; i++) {
    if (i < rays) {
      const angle = startAngle + angleStep * i
      const dist  = distances[i]
      const ex    = sensorPos.x + Math.sin(angle) * dist
      const ez    = sensorPos.z + Math.cos(angle) * dist

      posAttr.setXYZ(i * 2,     sensorPos.x, sensorPos.y, sensorPos.z)
      posAttr.setXYZ(i * 2 + 1, ex, sensorPos.y, ez)

      // Color mapping
      const t = 1.0 - Math.min(dist / safeDist, 1.0)
      const c = t < 0.5
        ? C_OK.clone().lerp(C_MID, t * 2)
        : C_MID.clone().lerp(C_BAD, (t - 0.5) * 2)

      colAttr.setXYZ(i * 2,     c.r, c.g, c.b)
      colAttr.setXYZ(i * 2 + 1, c.r, c.g, c.b)
    } else {
      posAttr.setXYZ(i * 2,     0, 0, 0)
      posAttr.setXYZ(i * 2 + 1, 0, 0, 0)
      colAttr.setXYZ(i * 2,     0, 0, 0)
      colAttr.setXYZ(i * 2 + 1, 0, 0, 0)
    }
  }

  posAttr.needsUpdate = true
  colAttr.needsUpdate = true
}

/**
 * Update CV Camera FOV Cone wireframe from active configuration.
 *
 * @param {{x:number,y:number,z:number}} origin      Robot world position
 * @param {number}                       facingAngle  Robot Y-rotation (radians)
 */
export function updateFovCone(origin, facingAngle) {
  if (!_coneViz) return
  _coneViz.visible = LAYERS.lidar
  if (!LAYERS.lidar) return

  const manifest = getManifest?.()
  const cameraSensor = manifest?.sensors?.find(s => s.type === 'camera') || {}
  const fov = cameraSensor.config?.fov || 150
  const range = cameraSensor.config?.range || 8.0
  const sensorOffset = cameraSensor.position || { x: 0, y: 0.5, z: 0.15 }

  const sensorPos = new THREE.Vector3(
    origin.x + (sensorOffset.x || 0),
    origin.y + (sensorOffset.y || 0.5),
    origin.z + (sensorOffset.z || 0)
  )

  const pts = []
  const segs = 20
  const fovRad = fov * (Math.PI / 180)
  const half = fovRad / 2

  // Center ray  (+sin/+cos = robot forward, matches castVision)
  pts.push(
    sensorPos.x, sensorPos.y, sensorPos.z,
    sensorPos.x + Math.sin(facingAngle) * range, sensorPos.y, sensorPos.z + Math.cos(facingAngle) * range
  )

  // Outer bounds & arc
  for (let i = 0; i <= segs; i++) {
    const a = (facingAngle - half) + (fovRad / segs) * i
    const ex = sensorPos.x + Math.sin(a) * range
    const ez = sensorPos.z + Math.cos(a) * range
    pts.push(sensorPos.x, sensorPos.y, sensorPos.z, ex, sensorPos.y, ez)
  }
  for (let i = 0; i < segs; i++) {
    const a1 = (facingAngle - half) + (fovRad / segs) * i
    const a2 = (facingAngle - half) + (fovRad / segs) * (i + 1)
    pts.push(
      sensorPos.x + Math.sin(a1) * range, sensorPos.y, sensorPos.z + Math.cos(a1) * range,
      sensorPos.x + Math.sin(a2) * range, sensorPos.y, sensorPos.z + Math.cos(a2) * range
    )
  }

  _coneViz.geometry.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  _coneViz.geometry.attributes.position.needsUpdate = true
}

/**
 * Update the navigation path line.
 * Call from planner/pathfinder whenever a new A* path is computed.
 *
 * @param {Array<{x:number,z:number}>} waypoints
 */
export function setNavPath(waypoints) {
  if (!_pathLine) return

  const pts   = waypoints || []
  const count = Math.min(pts.length, MAX_PATH_POINTS)

  _pathLine.visible = LAYERS.path && count > 0
  if (count === 0) {
    _pathLine.geometry.setDrawRange(0, 0)
    return
  }

  const posAttr = _pathLine.geometry.attributes.position

  for (let i = 0; i < count; i++) {
    posAttr.setXYZ(i, pts[i].x, 0.08, pts[i].z)
  }
  // Zero out tail so stale waypoints don't appear
  for (let i = count; i < MAX_PATH_POINTS; i++) {
    posAttr.setXYZ(i, 0, 0, 0)
  }

  _pathLine.geometry.setDrawRange(0, count)
  posAttr.needsUpdate = true
}

/**
 * Rebuild the occupancy grid points mesh from the current pathfinder state.
 * Relatively expensive (O(grid²)) — call when obstacles change, not every tick.
 */
export function rebuildOccupancyGrid() {
  if (!_scene) return

  // Remove old mesh
  if (_gridPoints) {
    _gridPoints.geometry.dispose()
    _gridPoints.material.dispose()
    _scene.remove(_gridPoints)
    _gridPoints = null
  }

  if (!LAYERS.grid) return

  const info    = getGridInfo()
  const blocked = []

  for (let r = 0; r < info.size; r++) {
    for (let c = 0; c < info.size; c++) {
      const wx = c * info.cellSize - info.halfExtent + info.cellSize * 0.5
      const wz = r * info.cellSize - info.halfExtent + info.cellSize * 0.5
      if (isOccupied(wx, wz)) {
        blocked.push(wx, 0.05, wz)
      }
    }
  }

  if (blocked.length === 0) return

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(blocked, 3))

  const mat = new THREE.PointsMaterial({
    color:       0xff2244,
    size:        info.cellSize * 0.75,
    transparent: true,
    opacity:     0.55,
    depthTest:   false,
  })

  _gridPoints = new THREE.Points(geo, mat)
  _gridPoints.name        = 'debug_occupancy'
  _gridPoints.visible     = LAYERS.grid
  _gridPoints.renderOrder = 997
  _scene.add(_gridPoints)
}

/**
 * Programmatically toggle a specific layer.
 * @param {'lidar'|'path'|'grid'} layer
 * @param {boolean} visible
 */
export function setLayerVisible(layer, visible) {
  if (!(layer in LAYERS)) return
  LAYERS[layer] = visible
  _applyVisibility()
  if (layer === 'grid' && visible) rebuildOccupancyGrid()
}

/**
 * Get the current visibility state of each layer (useful for HUD display).
 * @returns {{lidar:boolean, path:boolean, grid:boolean}}
 */
export function getLayerState() {
  return { ...LAYERS }
}

// ─── Internal ──────────────────────────────────────────────────────────────────

function _buildLidarGeometry() {
  const positions = new Float32Array(MAX_LIDAR_RAYS * 2 * 3)
  const colors = new Float32Array(MAX_LIDAR_RAYS * 2 * 3)
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
  geo.setDrawRange(0, MAX_LIDAR_RAYS * 2)

  const mat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity:     0.75,
    depthTest:   false,
    linewidth:   2,
  })

  _lidarLines = new THREE.LineSegments(geo, mat)
  _lidarLines.name        = 'debug_lidar'
  _lidarLines.visible     = LAYERS.lidar
  _lidarLines.renderOrder = 999
  _scene.add(_lidarLines)
}

function _buildFovConeGeometry() {
  const geo = new THREE.BufferGeometry()
  const mat = new THREE.LineBasicMaterial({
    color:       0x2266ff,
    transparent: true,
    opacity:     0.35,
    depthTest:   false,
    linewidth:   2,
  })

  _coneViz = new THREE.LineSegments(geo, mat)
  _coneViz.name        = 'debug_fov_cone'
  _coneViz.visible     = LAYERS.lidar
  _coneViz.renderOrder = 999
  _scene.add(_coneViz)
}

function _buildPathGeometry() {
  const positions = new Float32Array(MAX_PATH_POINTS * 3)
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geo.setDrawRange(0, 0)

  const mat = new THREE.LineBasicMaterial({
    color:       0xff8800,
    transparent: true,
    opacity:     0.85,
    depthTest:   false,
    linewidth:   2,
  })

  _pathLine = new THREE.Line(geo, mat)
  _pathLine.name        = 'debug_path'
  _pathLine.visible     = LAYERS.path
  _pathLine.renderOrder = 998
  _scene.add(_pathLine)
}

function _applyVisibility() {
  if (_lidarLines) _lidarLines.visible = LAYERS.lidar
  if (_coneViz)     _coneViz.visible     = LAYERS.lidar
  if (_pathLine)   _pathLine.visible   = LAYERS.path
  if (_gridPoints) _gridPoints.visible = LAYERS.grid
}
