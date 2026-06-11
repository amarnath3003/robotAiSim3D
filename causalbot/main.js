/**
 * main.js — CausalBot Entry Point
 * 
 * The unified bootstrap that:
 * 1. Loads the robot manifest
 * 2. Initializes rendering (Three.js)
 * 3. Initializes physics (Rapier3D)
 * 4. Loads the robot via the universal adapter
 * 5. Initializes perception, brain, and skills
 * 6. Optionally connects the RL bridge
 * 7. Starts the engine loop
 * 
 * This replaces the old main.js with its three disconnected modes.
 * There is ONE pipeline now — the robot is always manifest-driven and perception-based.
 */

import { loadManifest, getManifest, generateLLMDescription } from './src/core/manifest.js'
import { loadRobot } from './src/core/adapter.js'
import { initEngine, registerSystem, startEngine, getActiveRobot } from './src/core/engine.js'
import { setRobot, setSceneRefs, setState } from './src/core/state.js'
import { initScene, renderFrame, getScene, setFollowTarget } from './src/render/scene.js'
import { initPhysicsWorld, stepPhysicsWorld, createFloor } from './src/physics/world.js'
import { initVision } from './src/perception/vision.js'
import { initObserver, updatePerception } from './src/perception/observer.js'
import { skillRegistry } from './src/skills/registry.js'
import { handleInstruction, onPlannerEvent, buildExecutionContext } from './src/brain/planner.js'
import { initRLBridge, sendObservation, isRLConnected } from './src/rl/bridge.js'
import { initDashboard, showNotification } from './src/ui/dashboard.js'
import { initControls } from './src/ui/controls.js'
import { initEnvironment, loadDefaultLayout, loadRLLayout, updateInteractables, applyRobotPush } from './src/env/objects.js'
import { initBackgroundAgent, resetIdleTimer } from './src/brain/background_agent.js'
import { initSkillEditor } from './src/ui/skill_editor.js'
import { initVisualizer } from './src/debug/visualizer.js'
import { episodicMemory } from './src/brain/episodic_memory.js'

// ─── Configuration ─────────────────────────────────────────────────────────────

// Manifest path — change this to load a different robot
const MANIFEST_PATH = './manifests/default-bot.json'

// URL params for optional modes
const params = new URLSearchParams(window.location.search)
const ENABLE_RL = params.get('rl') === 'true' || params.get('mode') === 'rl'
const DEBUG_MODE = params.get('debug') === 'true'

// ─── Bootstrap ─────────────────────────────────────────────────────────────────

async function boot() {
  console.log('═══════════════════════════════════════════')
  console.log('  CausalBot — Universal Robot Framework')
  console.log('═══════════════════════════════════════════')
  console.log('')
  
  try {
    // ─── Step 1: Load Robot Manifest ─────────────────────────────────────
    console.log('[Boot] Loading manifest...')
    const manifest = await loadManifest(MANIFEST_PATH)
    console.log(`[Boot] Robot: "${manifest.name}" (${manifest.morphology})`)
    console.log('')

    // ─── Step 1b: Load Episodic Memory ───────────────────────────────────
    episodicMemory.load()
    console.log('')
    
    // ─── Step 2: Initialize Rendering ────────────────────────────────────
    console.log('[Boot] Initializing renderer...')
    const { scene, camera, renderer } = await initScene({
      hdriPath: '/sky3.hdr',
      enableBloom: true,
    })
    console.log('')
    
    // ─── Step 3: Initialize Physics ──────────────────────────────────────
    console.log('[Boot] Initializing physics...')
    const { world: rapierWorld, RAPIER } = await initPhysicsWorld()
    createFloor(0)
    
    // Store scene refs in global state
    setSceneRefs({
      three: scene,
      camera,
      renderer,
      rapierWorld,
      rapier: RAPIER,
    })
    console.log('')

    // ─── Step 3b: Initialize Environment ─────────────────────────────────
    console.log('[Boot] Building environment...')
    initEnvironment(scene, { arenaSize: 16, showBoundary: true })
    loadDefaultLayout()
    console.log('')
    
    // ─── Step 4: Load Robot via Universal Adapter ─────────────────────────
    console.log('[Boot] Loading robot...')
    const robot = await loadRobot(manifest, scene, rapierWorld, RAPIER)
    setRobot(robot)

    // Spawn height: capsule halfHeight(0.35) + radius(0.25) = 0.60
    // This places the bottom of the capsule exactly on the floor (Y = 0).
    const SPAWN_Y = 0.60
    robot.position.set(0, SPAWN_Y, 1.8)
    if (robot.physicsBody) {
      robot.physicsBody.setNextKinematicTranslation({ x: 0, y: SPAWN_Y, z: 1.8 })
    }

    // Camera follows the robot from boot
    setFollowTarget(robot)
    console.log('')
    
    // ─── Step 5: Initialize Perception ───────────────────────────────────
    console.log('[Boot] Initializing perception...')
    initVision(scene)
    initObserver(scene)
    console.log('')
    
    // ─── Step 6: Initialize Skills ───────────────────────────────────────
    console.log('[Boot] Initializing skills...')
    skillRegistry.init()
    console.log(`[Boot] Available skills: ${skillRegistry.getAllNames().join(', ')}`)
    console.log('')
    
    // ─── Step 7: Initialize Engine ───────────────────────────────────────
    console.log('[Boot] Setting up engine...')
    initEngine(robot)
    
    // Register system update functions
    registerSystem('physics', (dt) => stepPhysicsWorld(dt))
    registerSystem('objects', (dt) => {
      // 1. Sync all dynamic interactable mesh positions from Rapier physics
      updateInteractables()
      // 2. Proximity push: apply impulse to objects the moving robot sweeps through
      //    (supplements the character controller's built-in impulse for satisfying feel)
      applyRobotPush(
        { x: robot.position.x, y: robot.position.y, z: robot.position.z },
        robot.velocity,
        dt
      )
    })   // sync dynamic boxes/balls from physics
    registerSystem('perception', (dt, r) => updatePerception(dt, r))
    registerSystem('render', (dt) => renderFrame(dt))
    
    // ─── Step 8: RL Bridge (optional) ────────────────────────────────────
    if (ENABLE_RL) {
      console.log('[Boot] RL mode enabled — connecting to Python agent...')
      initRLBridge(scene, robot, {
        onConnect: () => {
          console.log('[Boot] RL agent connected!')
          setState('rl.connected', true)
        },
        onDisconnect: () => {
          setState('rl.connected', false)
        },
        onAction: (action) => {
          // Action applied by bridge — observation sent after physics step
        },
        onReset: (msg) => {
          // Episode reset — rebuild maze walls and reposition robot
          console.log(`[RL] Episode reset. Goal: (${msg.goal?.x}, ${msg.goal?.z})`)
          if (msg.walls) {
            loadRLLayout(msg.walls, msg.goal)
          } else if (msg.goal) {
            loadRLLayout([], msg.goal)
          }
          // Teleport robot to start at correct floor height
          const startX = msg.start?.x ?? 0
          const startZ = msg.start?.z ?? 0
          robot.position.set(startX, 0.60, startZ)
          if (robot.physicsBody) {
            robot.physicsBody.setNextKinematicTranslation({ x: startX, y: 0.60, z: startZ })
          }
        },
      })
      
      // Register RL system (sends observations after physics)
      registerSystem('rl', (dt, r) => {
        if (isRLConnected()) {
          sendObservation()
        }
      })
    }
    
    // ─── Step 9: Wire up UI ──────────────────────────────────────────────
    console.log('[Boot] Initializing UI...')
    initDashboard({ position: 'top-right', showPerception: true, showStats: true, showBrain: true })
    initControls({ showInput: true, enableKeyboard: true })
    
    // ─── Step 9b: Skill Editor (Shift+E) ─────────────────────────────────
    initSkillEditor(skillRegistry)
    
    // ─── Step 9c: Debug Visualizer (L/P/G keys) ──────────────────────────
    initVisualizer(scene)
    
    // Planner events → notifications
    // SM-5: use lowercase status values to match state.js schema and the
    //        cb-ind-* CSS classes in controls.js (cb-ind-thinking, cb-ind-executing…)
    onPlannerEvent('thinking', () => {
      setState('robot.status', 'thinking')
      resetIdleTimer()   // keep background agent dormant while user instruction is active
    })
    onPlannerEvent('executing', (info) => {
      setState('robot.status', 'executing')
      setState('robot.task', info.description || `Step ${info.step}/${info.total}`)
      if (info.step != null && info.total != null) {
        setState('robot.step', `${info.step}/${info.total}`)
      }
    })
    onPlannerEvent('complete', () => {
      setState('robot.status', 'idle')
      setState('robot.task', '—')
      setState('robot.step', '—')
      showNotification('Task complete', 'success')
    })
    onPlannerEvent('error', (msg) => {
      setState('robot.status', 'error')
      setState('robot.task', '—')
      setState('robot.step', '—')
      showNotification(msg, 'error', 5000)
    })
    onPlannerEvent('skillApproval', (info) => {
      // A new skill was synthesized by the LLM and auto-approved.
      // Surface it so the user knows a new capability was added.
      showNotification(`New skill "${info.name}" learned and saved`, 'success', 4000)
      setState('robot.status', 'Idle')
    })
    console.log('')
    
    // ─── Step 10: Start Engine ───────────────────────────────────────────
    console.log('')
    console.log('═══════════════════════════════════════════')
    console.log('  All systems ready. Starting simulation.')
    console.log('═══════════════════════════════════════════')
    console.log('')
    console.log(`LLM Description for this robot:`)
    console.log(generateLLMDescription())
    console.log('')
    
    startEngine()

    // ─── Step 11: Background Agent ───────────────────────────────────────
    // Activates after 8 s of user idleness — rescans stale objects, explores.
    // Bypasses LLM entirely (uses skills directly) to avoid token waste.
    initBackgroundAgent(
      skillRegistry,
      () => buildExecutionContext(robot, {}, skillRegistry),
      {
        onActivity: (goal) => {
          setState('robot.status', 'thinking')
          setState('robot.task', `[Auto] ${goal}`)
          showNotification(`Background: ${goal}`, 'info', 3000)
        },
      }
    )

    // Reset the background agent's idle timer whenever the user submits an instruction.
    // Already wired into the 'thinking' onPlannerEvent handler above.

    // Fade out the boot splash
    const overlay = document.getElementById('boot-overlay')
    if (overlay) {
      overlay.classList.add('hidden')
      setTimeout(() => overlay.remove(), 700)
    }
    
  } catch (err) {
    console.error('[Boot] FATAL:', err)
    document.body.innerHTML = `
      <div style="color: #ff4444; font-family: monospace; padding: 2rem;">
        <h2>CausalBot Boot Failed</h2>
        <pre>${err.message}\n${err.stack}</pre>
      </div>
    `
  }
}


<<<<<<< HEAD

// ─── Launch ────────────────────────────────────────────────────────────────────
=======
  // C3 fix: 1 substep per frame — physics.js has its own fixed-step accumulator,
  // so multiplying here caused 10× speed divergence and wall tunneling during RL.
  const substepDelta = delta

  updateRobot(substepDelta)
  updateDebugRobot(substepDelta)
  stepPhysics(substepDelta)
  stepDebugRobotPhysics(keys, substepDelta)
  updateRL(substepDelta)
  applyRobotCollisions()
>>>>>>> eacc7579af6f7deeef492076810d8800c06d3aff

boot()
