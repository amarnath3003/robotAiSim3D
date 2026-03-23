export const state = {
  robot: {
    // Real world position — navigation writes here, robot.js reads it
    position: new Float32Array([0, 1.2, 2]),
    rotation: 0,
    status: 'idle', // idle | thinking | executing | failed
    armAngle: 0,    // radians — arm rotation from shoulder
    armExtend: 0,   // 0-1 — how extended arm is
    heldObject: null,
    eyeColor: 0x4488ff,
  },

  // Ground truth world objects — single source of truth
  world: {
    objects: {
      object_glass: { id: 'object_glass', name: 'glass', mass: 0.3, fragility: 0.7, snapable: true,  status: 'intact', position: [0, 0.9, 3], size: [0.08, 0.12, 0.08] },
      object_box:   { id: 'object_box',   name: 'box',   mass: 2.0, fragility: 0.1, snapable: true,  status: 'intact', position: [-1.9, 0.3, 1.1],  size: [0.35, 0.35, 0.35] },
      object_ball:  { id: 'object_ball',  name: 'ball',  mass: 0.5, fragility: 0.2, snapable: true,  status: 'intact', position: [2.2, 0.2, -2.0],  size: [0.2,  0.2,  0.2]  },
      table:        { id: 'table',        name: 'table', mass: 20,  fragility: 0.0, snapable: false, status: 'intact', position: [0,   0.4,  0],    size: [1.2, 0.8, 0.7]   },
    },
    roomBounds: { minX: -3, maxX: 3, minZ: -3, maxZ: 3 },
    floorY: 0,
  },

  execution: {
    running: false,
    currentSkill: null,
    queue: [],
  },

  memory: [],

  scene: {
    three: null,
    camera: null,
    renderer: null,
    controls: null,
  }
}

// Helper — get object by name or id
export function getObject(nameOrId) {
  const objs = state.world.objects
  // Try direct id match first
  if (objs[nameOrId]) return objs[nameOrId]
  // Try name match (case insensitive)
  return Object.values(objs).find(o => 
    o.name.toLowerCase() === nameOrId.toLowerCase() ||
    o.id.toLowerCase() === nameOrId.toLowerCase() ||
    o.id.toLowerCase().includes(nameOrId.toLowerCase()) ||
    nameOrId.toLowerCase().includes(o.name.toLowerCase())
  ) || null
}

// Helper — get robot position as plain object
export function getRobotPos() {
  return {
    x: state.robot.position[0],
    y: state.robot.position[1],
    z: state.robot.position[2],
  }
}

// Helper — set robot position
export function setRobotPos(x, y, z) {
  state.robot.position[0] = x
  state.robot.position[1] = y
  state.robot.position[2] = z
}