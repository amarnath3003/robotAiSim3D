export const state = {
  robot: {
    position: new Float32Array([0, 0.35, 2]),
    rotation: 0,
    status: 'idle', // idle | thinking | executing | failed
    armAngle: 0,
    armExtend: 0,
    heldObject: null,
    eyeColor: 0x4488ff,
  },

  debugRobot: {
    position: new Float32Array([1.5, 0.35, 1.5]),
    rotation: 0,
    eyeColor: 0xffaa00,
    status: 'idle',
  },

  // ── Default to 'ai' so the LLM input works on first load.
  // Switch to 'rl'  by pressing key '3' or opening ?mode=maze.
  // Switch to 'debug' with key '2'.
  controlMode: 'ai', // 'ai' | 'debug' | 'rl'

  perceptionMode: 'omniscient', // 'omniscient' | 'vision'

  world: {
    objects: {
      object_glass: { id: 'object_glass', name: 'glass', mass: 0.3, fragility: 0.7, snapable: true,  status: 'intact', position: [0,    0.87, 3],    size: [0.08, 0.12, 0.08] },
      object_box:   { id: 'object_box',   name: 'box',   mass: 2.0, fragility: 0.1, snapable: true,  status: 'intact', position: [-1.9, 0.22, 1.1],  size: [0.35, 0.35, 0.35] },
      object_ball:  { id: 'object_ball',  name: 'ball',  mass: 0.5, fragility: 0.2, snapable: true,  status: 'intact', position: [2.2,  0.18, -2.0], size: [0.2,  0.2,  0.2]  },
    },
    roomBounds: { minX: -3, maxX: 3, minZ: -3, maxZ: 3 },
    floorY: 0,
  },

  execution: {
    running: false,
    currentSkill: null,
    queue: [],
    pendingApproval: null,
  },

  memory: [],

  scene: {
    three: null,
    camera: null,
    renderer: null,
    controls: null,
    rapierWorld: null,
  },
}

export function getObject(nameOrId) {
  const objs = state.world.objects
  if (objs[nameOrId]) return objs[nameOrId]
  return Object.values(objs).find(o =>
    o.name.toLowerCase() === nameOrId.toLowerCase() ||
    o.id.toLowerCase()   === nameOrId.toLowerCase() ||
    o.id.toLowerCase().includes(nameOrId.toLowerCase()) ||
    nameOrId.toLowerCase().includes(o.name.toLowerCase())
  ) || null
}

export function getRobotPos() {
  return {
    x: state.robot.position[0],
    y: state.robot.position[1],
    z: state.robot.position[2],
  }
}

export function setRobotPos(x, y, z) {
  state.robot.position[0] = x
  state.robot.position[1] = y
  state.robot.position[2] = z
}