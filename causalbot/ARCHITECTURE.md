# CausalBot — System Architecture

This document defines the technical architecture, data flows, and design decisions for the CausalBot universal robot framework.

---

## Design Principles

1. **Manifest-First**: Every robot-specific behavior derives from the manifest. No hardcoded assumptions about morphology.
2. **Perception is Reality**: The robot never has omniscient access to world state. All knowledge comes through sensors.
3. **Skills are Composable**: Complex behaviors compose from simple, verified atomic skills.
4. **Sim = Real**: Every interface (observations, actions, rewards) uses the same dimensionality and semantics that physical hardware would.
5. **LLM Reasons, Physics Decides**: The LLM proposes; the physics simulator disposes. Failed skills are rejected, not faked.

---

## System Layers

```
┌─────────────────────────────────────────────────────────────────────┐
│ LAYER 5: USER INTERFACE                                              │
│ Natural language input • Dashboard • Debug visualization             │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 4: LLM BRAIN                                                   │
│ Task decomposition • Feasibility checking • Skill synthesis          │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 3: SKILL ORCHESTRATOR                                          │
│ Skill registry • Skill selection • Execution sequencing              │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 2: MOTOR CONTROL                                               │
│ Joint commands • IK • Locomotion • Constraint enforcement            │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 1: PHYSICS + PERCEPTION                                        │
│ Rapier3D simulation • Sensor raycasting • Proprioception             │
├─────────────────────────────────────────────────────────────────────┤
│ LAYER 0: ROBOT MANIFEST + 3D MODEL                                   │
│ Static configuration • Joint definitions • Capability declarations   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Instruction → Execution

```
User: "Pick up the red ball and bring it to the table"
  │
  ▼
┌─── LLM Brain ──────────────────────────────────────────────────────┐
│ 1. Parse instruction                                                │
│ 2. Load manifest constraints:                                       │
│    - Can grasp? YES (manipulation:grasp, maxPayload=2kg)           │
│    - Ball mass? 0.5kg < 2kg → feasible                             │
│ 3. Check perception: Do I know where the red ball is?               │
│    - Memory confidence for "ball": 0.3 (decayed) → need to scan    │
│ 4. Decompose into sub-tasks:                                        │
│    [scan_for("ball"), navigate_to("ball"), grasp("ball"),           │
│     scan_for("table"), navigate_to("table"), release("ball")]      │
│ 5. For each sub-task: find matching skill or synthesize             │
└────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─── Skill Orchestrator ─────────────────────────────────────────────┐
│ Execute sequence:                                                    │
│   skill: "scan_for" → rotates head, updates perception memory       │
│   skill: "navigate_to" → A* path → motor commands                   │
│   skill: "grasp" → approach + arm extend + grip                     │
│   ... (re-plan if any step fails)                                   │
└────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─── Motor Controller ───────────────────────────────────────────────┐
│ Translate skill output → joint velocities                           │
│ Enforce manifest limits (max speed, joint bounds)                    │
│ Apply to physics bodies                                             │
└────────────────────────────────────────────────────────────────────┘
  │
  ▼
┌─── Physics + Perception ───────────────────────────────────────────┐
│ Rapier3D steps simulation                                           │
│ Sensors fire (lidar rays, vision cone)                              │
│ Proprioception updates (joint positions, contacts, balance)         │
│ Feed observations back to brain for next decision                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### `/src/core/` — Framework Core

| File | Responsibility |
|------|---------------|
| `manifest.js` | Load, validate, and expose the robot manifest. Provides `getManifest()`, `getCapabilities()`, `getConstraints()`, `getJoints()`. |
| `adapter.js` | Universal Robot Adapter. Loads any GLB/GLTF model, discovers bones, maps them to manifest joints, creates physics bodies from manifest config. |
| `engine.js` | Main loop orchestrator. Fixed timestep. Calls physics → perception → brain → motor in correct order. |
| `state.js` | Reactive global state. Robot state, world state, execution state. Observable for UI updates. |

### `/src/brain/` — LLM Integration

| File | Responsibility |
|------|---------------|
| `planner.js` | Takes instruction + manifest + perception state → produces action plan (sub-task list). |
| `feasibility.js` | Given a proposed action and the manifest, determines if the robot CAN physically do it. Returns yes/no + reasoning. |
| `synthesizer.js` | When no existing skill matches, generates new skill code. Outputs JavaScript function body constrained by manifest. |
| `llm.js` | API communication layer. Handles prompts, rate limiting, model selection. Provider-agnostic (NVIDIA, OpenAI, local). |

### `/src/skills/` — Skill System

| File | Responsibility |
|------|---------------|
| `registry.js` | Stores all skills (built-in, learned, invented). Lookup by capability match. Persists to localStorage / file. |
| `primitives.js` | Atomic built-in skills derived from manifest capabilities: `move_forward`, `rotate`, `extend_arm`, `grasp`, `release`, `jump`. |
| `learned.js` | Loads RL-trained policy weights (ONNX) and exposes them as callable skills. |
| `verifier.js` | Runs a candidate skill in a sandboxed physics "dry run". Checks for constraint violations, collisions, success. |

### `/src/perception/` — Sensing

| File | Responsibility |
|------|---------------|
| `vision.js` | Raycasting-based vision sensor. Configurable FOV, range, ray count from manifest. |
| `proprioception.js` | Internal state sensing: joint angles, velocity, contact forces, balance (center of mass vs. support polygon). |
| `memory.js` | Spatial memory with confidence decay. Objects remembered with position + confidence + timestamp. |
| `observer.js` | Unified observation builder. Combines all sensor data into a single observation vector for RL or a structured object for LLM. |

### `/src/motor/` — Low-Level Control

| File | Responsibility |
|------|---------------|
| `controller.js` | Takes high-level motor commands → applies joint velocities/forces respecting manifest limits. |
| `locomotion.js` | Movement patterns (forward, backward, turn). Reads manifest constraints for speed limits. |

### `/src/physics/` — Simulation

| File | Responsibility |
|------|---------------|
| `world.js` | Rapier3D world initialization, fixed timestep stepping, gravity. |
| `environment.js` | Manages environment objects (walls, floors, interactable objects). Spawning, removal, collision callbacks. |

### `/src/rl/` — Reinforcement Learning

| File | Responsibility |
|------|---------------|
| `bridge.js` | WebSocket connection to Python training script. Serializes observations, deserializes actions. |
| `observation.js` | Builds the observation vector from perception + proprioception. Format defined by manifest's `observationSpace`. |
| `reward.js` | Computes reward signals. Modular: navigation reward, manipulation reward, energy penalty, constraint violation penalty. |

### `/src/render/` — Visualization

| File | Responsibility |
|------|---------------|
| `scene.js` | Three.js scene setup, lighting, post-processing (bloom, HDRI). |
| `debug.js` | Debug overlays: sensor rays, paths, joint limits, colliders, observation space visualization. |

### `/src/ui/` — User Interface

| File | Responsibility |
|------|---------------|
| `dashboard.js` | Real-time status panels, skill registry display, memory state. |
| `controls.js` | Input handling: text commands, keyboard shortcuts, mode switching. |

---

## The Robot Manifest in Detail

The manifest serves multiple consumers:

| Consumer | What it reads |
|----------|--------------|
| **Robot Adapter** | `model`, `joints`, `physics` → creates 3D mesh + physics body |
| **LLM Brain** | `capabilities`, `constraints`, `joints` → feasibility checking, plan generation |
| **Motor Controller** | `joints.limits`, `constraints.maxSpeed` → enforces physical limits |
| **Perception** | `sensors` → configures raycasting parameters |
| **RL Bridge** | `actionSpace`, `observationSpace` → defines gym spaces |
| **Skill Verifier** | `constraints` → validates invented skills don't violate limits |

### Manifest → URDF Mapping (Sim-to-Real)

```
Manifest Joint                    URDF Joint
─────────────────                 ──────────────────
name: "arm_left"            →     <joint name="arm_left">
type: "revolute"            →       <type>revolute</type>
axis: "x"                   →       <axis xyz="1 0 0"/>
limits.lower: -90 (deg)     →       <limit lower="-1.5708" (rad)
limits.upper: 180 (deg)     →              upper="3.1416"
limits.velocity: 3.0        →              velocity="3.0"
limits.effort: 10.0         →              effort="10.0"/>
parent: "torso"             →       <parent link="torso"/>
child: "hand_left"          →       <child link="hand_left"/>
```

A `manifest-to-urdf` export tool will be provided for Phase 4.

---

## RL Training Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Python Side (train_rl.py)                                    │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  Gymnasium Environment (causalbot_env.py)            │     │
│  │  • Reads manifest → defines action/obs spaces       │     │
│  │  • WebSocket ↔ Browser (physics simulation)         │     │
│  │  • Reward computation (modular, per-task)           │     │
│  └─────────────────────────────────────────────────────┘     │
│                         │                                     │
│                         ▼                                     │
│  ┌─────────────────────────────────────────────────────┐     │
│  │  PPO / SAC Agent (Stable-Baselines3)                │     │
│  │  • Trains policies for specific skills              │     │
│  │  • Exports to ONNX for browser inference            │     │
│  │  • Exports to ONNX for real-robot deployment        │     │
│  └─────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
                         │
              WebSocket (ws://localhost:8765)
                         │
┌──────────────────────────────────────────────────────────────┐
│  Browser Side (Three.js + Rapier3D)                           │
│                                                               │
│  • Receives actions → applies via Motor Controller            │
│  • Steps physics → sensors fire                               │
│  • Builds observation vector from manifest spec               │
│  • Sends observation back to Python                           │
│  • Renders visualization in real-time                         │
└──────────────────────────────────────────────────────────────┘
```

### Skill-Specific RL Training

Instead of one monolithic policy, CausalBot trains **per-skill policies**:

| Skill | Observation | Action | Reward |
|-------|------------|--------|--------|
| `navigate_to` | target_dist, target_angle, lidar | linear_speed, angular_speed | progress toward target, collision penalty |
| `grasp_object` | arm_angles, object_distance, gripper_state | arm_velocities, gripper_force | proximity to object, successful grasp |
| `balance` | IMU, center_of_mass, foot_contacts | joint torques | upright reward, fall penalty |
| `avoid_obstacles` | lidar, velocity | steering | min clearance, forward progress |

Each policy exports as ONNX and registers in the skill system. The LLM selects which policy to activate.

---

## Sim-to-Real Transfer Strategy

### Domain Randomization (in simulation)
- Randomize physics: friction, mass ±20%, joint damping
- Randomize sensors: noise injection, dropout, latency
- Randomize environment: object positions, lighting, obstacle layouts

### Observation Normalization
- All observations are normalized to [-1, 1] or [0, 1]
- Same normalization parameters used in sim and real
- VecNormalize statistics saved alongside model checkpoints

### Action Space Consistency
- Actions are in physical units (m/s, rad/s, N)
- Same action space definition in sim and on real hardware
- Motor controller handles unit conversion to actuator commands

### Transfer Pipeline
```
1. Train in CausalBot sim (Rapier3D physics)
2. Export policy to ONNX
3. Export manifest to URDF
4. Load ONNX policy on real robot (edge inference)
5. Map manifest sensors → real sensor drivers
6. Map manifest actions → real actuator drivers
7. Fine-tune with real-world data if needed
```

---

## State Management

Global state is structured and observable:

```javascript
{
  robot: {
    manifest: { /* loaded manifest object */ },
    joints: { /* current joint states: position, velocity, effort */ },
    position: [x, y, z],
    orientation: [qx, qy, qz, qw],
    velocity: [vx, vy, vz],
    status: 'idle' | 'executing' | 'planning' | 'failed',
    heldObjects: [],
  },
  perception: {
    observations: { /* latest sensor readings */ },
    memory: { /* spatial memory map with confidence scores */ },
    lastScanTime: timestamp,
  },
  execution: {
    currentPlan: [],
    currentSkill: null,
    skillQueue: [],
    history: [],
  },
  world: {
    objects: { /* known objects (from perception only) */ },
    environment: { /* static geometry info */ },
  },
  rl: {
    connected: boolean,
    episode: number,
    training: boolean,
  }
}
```

---

## File Naming Conventions

- **Modules**: `camelCase.js` (e.g., `skillRegistry.js`)
- **Manifests**: `kebab-case.json` (e.g., `default-bot.json`)
- **Models**: `kebab-case.glb` (e.g., `robot-arm-v2.glb`)
- **Python**: `snake_case.py` (e.g., `causalbot_env.py`)
- **Tests**: `*.test.js` / `test_*.py`

---

## Error Handling Philosophy

- **Physics failures** (collision, constraint violation): Reported to skill layer, skill fails gracefully, LLM re-plans.
- **LLM failures** (bad plan, infeasible action): Feasibility checker rejects before execution. Never reaches physics.
- **Sensor failures** (no data, stale): Perception layer reports confidence=0. LLM knows to scan before acting.
- **Network failures** (WebSocket drop): RL bridge reconnects automatically. Training resumes from last checkpoint.

---

## Future: Multi-Robot

The manifest system naturally supports multiple robots in the same scene:

```javascript
const robots = [
  await loadRobot('./manifests/default-bot.json'),
  await loadRobot('./manifests/arm-only.json'),
  await loadRobot('./manifests/quadruped.json'),
]

// Same instruction dispatched to each — different execution based on capabilities
await brain.execute("bring the box to the corner", robots[0])  // walks, grasps, carries
await brain.execute("bring the box to the corner", robots[1])  // can't move — refuses
await brain.execute("bring the box to the corner", robots[2])  // walks to it, pushes with head
```
