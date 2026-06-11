# CausalBot

**A universal embodied AI framework: any robot + a capability manifest + an LLM brain = autonomous execution of arbitrary instructions in physics simulation, transferable to real hardware.**

---

## Research Thesis

> Given (1) an arbitrary robot morphology described by a simple manifest, (2) a physics simulator grounding actions in reality, and (3) an LLM brain for reasoning — the system can autonomously discover feasible behaviors, synthesize motor skills, and execute open-ended natural language instructions **without robot-specific programming**.

The same policies, skills, and perception strategies trained in simulation transfer directly to physical robots via the manifest abstraction.

---

## The Problem

Every robot today requires bespoke programming. Change the morphology (add a leg, swap a gripper) and you rewrite the control stack. LLMs can reason about tasks but have no physical grounding. RL can learn control but has no semantic understanding.

**CausalBot unifies all three:**

```
Human Instruction
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              LLM BRAIN (Planner)                     │
│  Reads: Robot Manifest + Perception State           │
│  Outputs: Feasibility check → Sub-task plan         │
│           → Skill selection or synthesis            │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              SKILL LAYER                             │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐   │
│  │ Built-in │  │ RL-      │  │ LLM-Invented   │   │
│  │ Primitives│  │ Trained  │  │ (physics-      │   │
│  │          │  │ Policies │  │  verified code) │   │
│  └──────────┘  └──────────┘  └────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         MOTOR CONTROLLER                             │
│  Manifest-constrained joint commands                 │
│  IK solver • Locomotion • Force control             │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│         PHYSICS (Rapier3D) ←→ PERCEPTION            │
│  Executes • Detects • Observes • Feeds back         │
└─────────────────────────────────────────────────────┘
```

---

## Key Idea: The Robot Manifest

The manifest is the central abstraction that makes everything robot-agnostic:

```json
{
  "name": "DefaultBot",
  "model": "./models/robot1.glb",
  "morphology": "bipedal-wheeled",
  "joints": [
    { "name": "arm_left", "type": "revolute", "axis": "x", "limits": [-90, 180] },
    { "name": "arm_right", "type": "revolute", "axis": "x", "limits": [-90, 180] }
  ],
  "capabilities": [
    "locomotion:ground",
    "manipulation:single-gripper",
    "jump:low"
  ],
  "constraints": {
    "maxSpeed": 2.5,
    "maxReach": 0.6,
    "canFly": false,
    "canSwim": false
  },
  "sensors": ["lidar_360", "front_camera"],
  "mass": 8.0
}
```

The LLM reads this manifest. When you say "do a backflip", it checks constraints, plans the joint sequence, synthesizes the skill, and verifies it in physics. If the robot can't physically do it — it tells you why and suggests alternatives.

**Swap the manifest + model = entirely different robot, same brain.**

---

## Sim-to-Real Pipeline

This is not just a toy. The architecture is designed for real-world transfer:

| Layer | Sim (Three.js + Rapier3D) | Real (Target) |
|-------|---------------------------|---------------|
| Manifest | JSON config | Same JSON → URDF/ROS params |
| Perception | Raycasting + virtual camera | LiDAR + RGB camera |
| Motor commands | Joint velocities → Rapier | Joint velocities → ROS/actuators |
| RL policies | Train in sim | Deploy via sim2real transfer |
| Skills | Verified in physics sim | Same code, real actuators |
| Observations | Normalized sensor vector | Same dimensionality from real sensors |

The manifest schema maps directly to URDF joint descriptions. Observation spaces match real sensor outputs. Trained policies export to ONNX for edge deployment.

---

## What Makes This Novel

1. **Universal Robot Adapter** — Load any robot model. Describe its joints in plain JSON. The system adapts automatically.

2. **LLM + RL Cooperation** — The LLM doesn't guess motor commands. It orchestrates RL-trained skills and invents new ones when needed, always checking feasibility against the manifest.

3. **Embodied Perception by Default** — The robot is always "blind". It must physically observe its environment through sensors, maintaining a confidence-decaying spatial memory. No omniscience.

4. **Physics-Verified Skill Synthesis** — When the LLM invents a new skill (e.g., "cartwheel"), it runs in the physics sim first. Only skills that succeed without violating constraints get saved.

5. **Sim-to-Real Ready** — Every abstraction (manifest, observation space, action space, reward signals) is designed to map 1:1 to physical hardware.

---

## Current Status

- Three.js + Rapier3D physics simulation (working)
- LLM-driven planning with Chain-of-Thought (working)
- Dynamic skill invention and registry (working)
- Raycasting perception with decaying memory (working)
- PPO training via WebSocket bridge to Python (working)
- A* pathfinding with obstacle avoidance (working)
- Procedural maze environments (working)

**In Progress:** Unifying these components under the manifest-driven architecture.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Simulation | Three.js (rendering) + Rapier3D (physics, WASM) |
| AI Brain | LLM API (NVIDIA/OpenAI/local — configurable) |
| RL Training | Python: Stable-Baselines3, Gymnasium, PPO |
| Communication | WebSocket bridge (Python ↔ Browser) |
| Build | Vite 5 (ES modules) |
| Future Real | ROS2, ONNX Runtime, real LiDAR/cameras |

---

## Getting Started

### Prerequisites
- Node.js v18+
- Python 3.10+ (for RL training)
- An LLM API key (NVIDIA, OpenAI, or local model)

### Installation

```bash
git clone https://github.com/amarnath3003/robotAiSim3D.git
cd robotAiSim3D/causalbot
npm install
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your LLM API key
```

### Run Simulation

```bash
npm run dev
```

### Run RL Training (separate terminal)

```bash
cd python
pip install -r requirements.txt
python train_rl.py
```

---

## Project Structure

```
causalbot/
├── src/
│   ├── core/              # Framework core
│   │   ├── manifest.js    # Robot manifest loader + validator
│   │   ├── adapter.js     # Universal robot adapter (GLB + constraints)
│   │   ├── engine.js      # Main simulation loop orchestrator
│   │   └── state.js       # Global state management
│   ├── brain/             # LLM integration
│   │   ├── planner.js     # High-level task decomposition
│   │   ├── feasibility.js # Constraint checking against manifest
│   │   ├── synthesizer.js # Skill code generation
│   │   └── llm.js         # LLM API communication
│   ├── skills/            # Skill system
│   │   ├── registry.js    # Skill storage and lookup
│   │   ├── primitives.js  # Built-in atomic actions
│   │   ├── learned.js     # RL-trained skill loader
│   │   └── verifier.js    # Physics verification of invented skills
│   ├── perception/        # Sensing (blind by default)
│   │   ├── vision.js      # Raycasting vision sensor
│   │   ├── proprioception.js  # Joint states, balance, contacts
│   │   ├── memory.js      # Decaying spatial memory
│   │   └── observer.js    # Unified observation builder
│   ├── motor/             # Low-level control
│   │   ├── controller.js  # Joint velocity/force application
│   │   └── locomotion.js  # Walking/movement patterns
│   ├── physics/           # Physics simulation
│   │   ├── world.js       # Rapier world setup
│   │   └── environment.js # Scene objects, terrain, walls
│   ├── rl/                # Reinforcement learning
│   │   ├── bridge.js      # WebSocket to Python
│   │   ├── observation.js # Observation space builder
│   │   └── reward.js      # Reward signal definitions
│   ├── render/            # Three.js rendering
│   │   ├── scene.js       # Scene setup, lighting, post-processing
│   │   └── debug.js       # Debug visualization (paths, rays, colliders)
│   └── ui/                # Dashboard
│       ├── dashboard.js   # Status panels
│       └── controls.js    # Input handling
├── manifests/             # Robot manifest files
│   ├── schema.json        # JSON Schema for validation
│   └── default-bot.json   # Built-in robot manifest
├── python/                # RL training (Python side)
│   ├── causalbot_env.py   # Gymnasium environment
│   ├── train_rl.py        # PPO training script
│   └── models/            # Saved checkpoints
├── models/                # Robot 3D models (GLB/GLTF)
└── main.js                # Entry point
```

---

## Roadmap

### Phase 1: Foundation (Current)
- [x] Physics simulation with Rapier3D
- [x] LLM planning with CoT
- [x] Perception system (blind mode)
- [x] RL training pipeline
- [ ] **Robot Manifest schema and loader**
- [ ] **Universal Robot Adapter**
- [ ] **Unified control pipeline (replace mode switching)**

### Phase 2: Intelligence
- [ ] LLM feasibility checking against manifest
- [ ] Physics-verified skill synthesis
- [ ] RL trains specific skills invokable by LLM
- [ ] Proprioception (joint states, balance sensing)

### Phase 3: Generalization
- [ ] Load multiple different robot models
- [ ] Same instruction → different execution per morphology
- [ ] Skill transfer between similar morphologies
- [ ] Multi-step mission execution with re-planning

### Phase 4: Sim-to-Real
- [ ] Manifest → URDF export
- [ ] Policy export to ONNX
- [ ] ROS2 bridge for real actuators
- [ ] Real sensor ingestion (camera, LiDAR)
- [ ] Domain randomization for transfer

---

## License

MIT

## Author

Amarnath — [@amarnath3003](https://github.com/amarnath3003)
