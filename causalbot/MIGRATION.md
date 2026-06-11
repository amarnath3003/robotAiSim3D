# Migration Guide: Old → New Architecture

This document maps the old flat file structure to the new modular architecture,
explaining how each old file's responsibilities are distributed in the new system.

---

## File Mapping

| Old File | New Location(s) | Notes |
|----------|-----------------|-------|
| `state.js` | `src/core/state.js` | Completely rewritten. Now reactive with subscriptions. Robot state comes from manifest, not hardcoded. |
| `robot.js` | `src/core/adapter.js` | Robot loading is now manifest-driven. Any GLB model works. Joint mapping is automatic. |
| `physics.js` | `src/physics/world.js` + `src/core/adapter.js` | Physics body creation comes from manifest config. World setup is separate from robot physics. |
| `scene.js` | `src/render/scene.js` | Mostly unchanged — rendering is independent of robot logic. |
| `llm.js` | `src/brain/llm.js` | Same responsibility, but now receives manifest context in prompts. |
| `executor.js` | `src/brain/planner.js` + `src/skills/registry.js` | Planning split from execution. Feasibility checking added. |
| `skillRegistry.js` | `src/skills/registry.js` + `src/skills/primitives.js` | Built-in skills now auto-generated from manifest capabilities. |
| `rl.js` | `src/rl/bridge.js` + `src/rl/observation.js` | WebSocket bridge separated from observation building. Obs space from manifest. |
| `maze.js` | `src/physics/environment.js` | Maze is one type of environment. Others can be added. |
| `pathfinder.js` | `src/skills/primitives.js` (navigate_to skill) | Pathfinding is a built-in skill, not a standalone system. |
| `controls.js` | `src/ui/controls.js` | Same, but no more "mode switching" — unified pipeline. |
| `ui.js` | `src/ui/dashboard.js` | Same responsibility. |
| `memory.js` | `src/core/state.js` (execution history) | Action history is now part of state. |
| `perception/visionSensor.js` | `src/perception/vision.js` | Same logic, but configured from manifest sensors. |
| `perception/perceptualMemory.js` | `src/core/state.js` (perception memory) | Integrated into the central state with confidence decay. |
| `perception/perceptionMode.js` | **REMOVED** | No more "omniscient" mode. Robot is always blind. Perception is reality. |
| `perception/scanner.js` | `src/perception/vision.js` | Scanning is part of the vision system, driven by manifest sensor config. |
| `main.js` | `main.js` (uses `src/core/engine.js`) | Entry point simplified. Engine handles the loop. |

---

## Key Conceptual Changes

### 1. No More "Modes" (AI / Debug / RL)

**Old**: Three separate control modes switched with keyboard (1/2/3).

**New**: One unified pipeline. The robot is always controlled by the brain/skill system.
- RL training = the skill system is disabled, Python controls directly
- Normal use = LLM brain plans, skills execute
- Debug = debug overlay enabled, but robot still runs through the pipeline

### 2. No More "Omniscient" Mode

**Old**: Toggle between seeing everything vs. using vision sensors.

**New**: The robot is ALWAYS blind. It must perceive through sensors.
This is critical for sim-to-real: real robots don't have omniscience.

### 3. Manifest Drives Everything

**Old**: Hardcoded values everywhere (speed=2.5, arm range, capsule size...).

**New**: All robot-specific values come from the manifest JSON.
Change the manifest = change the robot's behavior. No code changes.

### 4. Skills are Typed by Capability

**Old**: Skills are arbitrary code blobs stored by name.

**New**: Skills declare which capabilities they require:
```javascript
{
  name: 'pick_up',
  requires: ['manipulation:grasp', 'locomotion:ground'],
  params: { targetObject: 'string' },
  execute: async (context, params) => { ... }
}
```
If the loaded robot doesn't have 'manipulation:grasp', this skill is hidden.

### 5. State is Observable

**Old**: Direct mutation of `state.robot.position[0] = x`.

**New**: `setState('robot.position', {x, y, z})` triggers subscribers.
UI auto-updates. RL bridge auto-notifies. No manual sync.

---

## Migration Steps (for existing code)

If you want to gradually migrate the old code:

1. **Keep old files working** — they still run as-is
2. **Add manifest loading** at startup (new `main.js` calls `loadManifest()`)
3. **Replace hardcoded values** with manifest reads one file at a time
4. **Move files** into new directories as you refactor each one
5. **Delete old files** once their new equivalents are tested

The new architecture is additive — you can run both old and new code during transition.

---

## Example: Old vs New Startup

### Old (main.js)
```javascript
await initScene(MAZE_MODE)
await initRobot()           // Hardcoded robot1.glb, fixed capsule
await initDebugRobot()      // Second hardcoded robot
await initPhysics()         // Fixed physics config
initSkillRegistry()         // Skills not tied to capabilities
initControls()              // Mode switching (1/2/3)
initRL()                    // Fixed 4D action space
```

### New (main.js)
```javascript
import { loadManifest } from './src/core/manifest.js'
import { loadRobot } from './src/core/adapter.js'
import { initEngine, registerSystem, startEngine } from './src/core/engine.js'

// 1. Load manifest (defines everything about the robot)
const manifest = await loadManifest('./manifests/default-bot.json')

// 2. Initialize scene + physics
const { scene, rapierWorld, RAPIER } = await initScene()

// 3. Load robot from manifest (any robot, any model)
const robot = await loadRobot(manifest, scene, rapierWorld, RAPIER)

// 4. Initialize engine
initEngine(robot)

// 5. Register systems (plug in what you need)
registerSystem('physics', stepPhysics)
registerSystem('perception', updatePerception)
registerSystem('brain', updateBrain)
registerSystem('render', renderScene)
registerSystem('ui', updateUI)

// Optional: register RL bridge
if (manifest.actionSpace) {
  registerSystem('rl', updateRL)
}

// 6. Start
startEngine()
```

The new version is more explicit, more flexible, and doesn't assume anything
about the robot. Swap the manifest path → completely different robot.
