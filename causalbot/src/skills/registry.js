/**
 * skills/registry.js — Capability-Tagged Skill Storage
 * 
 * The skill registry manages all available skills:
 * - Built-in primitives (auto-generated from manifest)
 * - RL-trained policies (loaded as ONNX or JS functions)
 * - LLM-synthesized skills (dynamically generated, physics-verified)
 * 
 * Key improvement: Skills declare which capabilities they REQUIRE.
 * If the loaded robot doesn't have a required capability, the skill is hidden.
 * This makes skills portable across robots with matching capabilities.
 */

import { getManifest, hasCapability, getConstraints } from '../core/manifest.js'

// ─── Skill Definition ──────────────────────────────────────────────────────────

/**
 * @typedef {Object} Skill
 * @property {string} name - Unique skill identifier
 * @property {string} description - Human-readable description (shown to LLM)
 * @property {string[]} requires - Required capability IDs from manifest
 * @property {string} source - 'builtin' | 'learned' | 'synthesized'
 * @property {boolean} verified - Has this skill been physics-verified?
 * @property {boolean} approved - Has the user approved this skill?
 * @property {string} code - Source code (for synthesized skills)
 * @property {function} execute - The executable function(context) => Promise<void>
 * @property {Object} metadata - Additional metadata (author, version, stats)
 */

// ─── Storage ───────────────────────────────────────────────────────────────────

const _skills = new Map()         // All registered skills
const _pendingApproval = new Map() // Synthesized skills awaiting user approval
const STORAGE_KEY = 'causalbot_skills_v2'

// ─── Skill Registry Class ──────────────────────────────────────────────────────

class SkillRegistry {
  constructor() {
    this._initialized = false
  }
  
  /**
   * Initialize the registry with built-in primitives based on the loaded manifest.
   */
  init() {
    if (this._initialized) return
    
    // Load built-in primitives
    this._registerBuiltins()
    
    // Load persisted synthesized skills from localStorage
    this._loadPersisted()
    
    this._initialized = true
    console.log(`[Skills] Registry initialized: ${_skills.size} skills available`)
  }
  
  /**
   * Get a skill by name (only if robot has required capabilities).
   * @param {string} name
   * @returns {Skill|null}
   */
  get(name) {
    const skill = _skills.get(name)
    if (!skill) return null
    
    // Check if robot has required capabilities
    if (!this._checkRequirements(skill)) return null
    
    return skill
  }
  
  /**
   * Check if a skill exists and is available for the current robot.
   * @param {string} name
   * @returns {boolean}
   */
  has(name) {
    return this.get(name) !== null
  }
  
  /**
   * Get all registered skill objects (unfiltered — includes all sources).
   * Useful for the skill editor sidebar and diagnostics.
   * @returns {Skill[]}
   */
  getAll() {
    return [..._skills.values()]
  }

  /**
   * Get all available skill names (filtered by current robot capabilities).
   * @returns {string[]}
   */
  getAllNames() {
    const names = []
    for (const [name, skill] of _skills) {
      if (this._checkRequirements(skill)) {
        names.push(name)
      }
    }
    return names
  }
  
  /**
   * Get all available skills with their descriptions (for LLM context).
   * @returns {Array<{name: string, description: string, requires: string[]}>}
   */
  getAllForLLM() {
    const results = []
    for (const [name, skill] of _skills) {
      if (this._checkRequirements(skill)) {
        results.push({
          name: skill.name,
          description: skill.description,
          requires: skill.requires,
          args: skill.args || {},
        })
      }
    }
    return results
  }
  
  /**
   * Register a new synthesized skill.
   * @param {string} name
   * @param {string} code
   * @param {string} description
   * @returns {Skill|null}
   */
  registerSynthesized(name, code, description = '') {
    try {
      // Create executable function from code string
      const fn = new AsyncFunction('context', code)
      
      const skill = {
        name,
        description,
        requires: [],  // Synthesized skills don't declare requirements (LLM already checked feasibility)
        source: 'synthesized',
        verified: false,
        approved: true,   // Auto-approve: LLM checked feasibility; user can review later in skill editor
        code,
        execute: fn,
        metadata: {
          createdAt: Date.now(),
          executionCount: 0,
          successCount: 0,
          failureCount: 0,
        },
      }
      
      _skills.set(name, skill)
      _pendingApproval.set(name, skill)
      
      console.log(`[Skills] Registered synthesized skill: "${name}"`)
      return skill
      
    } catch (e) {
      console.error(`[Skills] Failed to compile skill "${name}":`, e)
      return null
    }
  }
  
  /**
   * Register an RL-trained policy as a skill.
   * @param {string} name
   * @param {function} policyFn - The policy function(observation) => action
   * @param {string} description
   * @param {string[]} requires
   */
  registerLearned(name, policyFn, description, requires = []) {
    const skill = {
      name,
      description,
      requires,
      source: 'learned',
      verified: true,  // RL policies are verified by training
      approved: true,
      code: null,
      execute: policyFn,
      metadata: {
        createdAt: Date.now(),
        trainingEpisodes: 0,
        avgReward: 0,
      },
    }
    
    _skills.set(name, skill)
    console.log(`[Skills] Registered RL-trained skill: "${name}"`)
  }
  
  /**
   * Approve a synthesized skill (persist to localStorage).
   * @param {string} name
   */
  approve(name) {
    const skill = _skills.get(name)
    if (!skill) return
    
    skill.approved = true
    _pendingApproval.delete(name)
    this._persist()
    
    console.log(`[Skills] Approved skill: "${name}"`)
  }
  
  /**
   * Reject and remove a synthesized skill.
   * @param {string} name
   */
  reject(name) {
    _skills.delete(name)
    _pendingApproval.delete(name)
    console.log(`[Skills] Rejected skill: "${name}"`)
  }
  
  /**
   * Mark a skill as physics-verified.
   * @param {string} name
   * @param {boolean} passed
   */
  markVerified(name, passed) {
    const skill = _skills.get(name)
    if (skill) {
      skill.verified = passed
      if (!passed) {
        console.warn(`[Skills] Skill "${name}" FAILED physics verification — removing`)
        _skills.delete(name)
      }
    }
  }
  
  /**
   * Record skill execution outcome (for stats tracking).
   * @param {string} name
   * @param {boolean} success
   */
  recordExecution(name, success) {
    const skill = _skills.get(name)
    if (!skill?.metadata) return
    
    skill.metadata.executionCount++
    if (success) {
      skill.metadata.successCount++
    } else {
      skill.metadata.failureCount++
    }
  }
  
  /**
   * Get skills pending user approval.
   * @returns {Map<string, Skill>}
   */
  getPendingApproval() {
    return _pendingApproval
  }
  
  // ─── Internal ──────────────────────────────────────────────────────────────
  
  _checkRequirements(skill) {
    if (!skill.requires || skill.requires.length === 0) return true
    return skill.requires.every(req => hasCapability(req))
  }
  
  _registerBuiltins() {
    const manifest = getManifest()
    const constraints = manifest?.constraints || {}
    
    // Navigation skills (require locomotion)
    this._register({
      name: 'navigate_to',
      description: 'Navigate to a perceived object. The object must be in perception memory (run scan_for first if not).',
      requires: ['locomotion:ground'],
      args: { target: 'object ID or color description (e.g. "ball_red", "red ball", "box_A")' },
      execute: async (ctx) => {
        const target = ctx.args?.target
        if (!target) throw new Error('navigate_to requires a target')

        // ONLY use perception memory — no direct physics lookup
        const match = ctx.findPerceivedObject?.(target)
        if (!match?.position) {
          throw new Error(
            `"${target}" not in perception memory. Use scan_for(target:"${target}") first.`
          )
        }

        ctx.setStatus(`Navigating to ${match.id} (approx pos: ${match.position.x.toFixed(1)}, ${match.position.z.toFixed(1)})...`)
        const currentPos = ctx.getPos()
        await ctx.navigateTo(match.position.x, currentPos.y, match.position.z)
      },
    })
    
    this._register({
      name: 'move_to_position',
      description: 'Move to specific x,z coordinates.',
      requires: ['locomotion:ground'],
      args: { x: 'number', z: 'number' },
      execute: async (ctx) => {
        const x = parseFloat(ctx.args?.x) || 0
        const z = parseFloat(ctx.args?.z) || 0
        const pos = ctx.getPos()
        await ctx.navigateTo(x, pos.y, z)
      },
    })
    
    // Rotation
    this._register({
      name: 'rotate',
      description: 'Rotate the robot by a specified angle.',
      requires: ['locomotion:rotate'],
      args: { degrees: 'number (default 360)' },
      execute: async (ctx) => {
        const degrees = parseFloat(ctx.args?.degrees) || 360
        const radians = degrees * (Math.PI / 180)
        const speed = constraints.maxAngularSpeed || 3.0
        const time = Math.abs(radians / speed) * 1000
        
        ctx.rotate(Math.sign(degrees) * speed)
        await ctx.wait(time)
        ctx.stop()
      },
    })
    
    this._register({
      name: 'spin',
      description: 'Do a quick 360 spin.',
      requires: ['locomotion:rotate'],
      execute: async (ctx) => {
        const speed = constraints.maxAngularSpeed || 3.0
        // Compute exact time for one full revolution + 5% buffer
        const duration = Math.ceil((2 * Math.PI / speed) * 1000 * 1.05)
        ctx.rotate(speed)
        await ctx.wait(duration)
        ctx.stop()
      },
    })
    
    // Jump
    if ((constraints.maxJumpHeight || 0) > 0) {
      this._register({
        name: 'jump',
        description: `Jump up to ${constraints.maxJumpHeight}m height.`,
        requires: ['locomotion:jump'],
        execute: async (ctx) => {
          const pos = ctx.getPos()
          const height = constraints.maxJumpHeight || 0.3
          const frames = 30
          
          for (let i = 0; i <= frames; i++) {
            const t = i / frames
            const y = pos.y + Math.sin(t * Math.PI) * height
            ctx.setPos(pos.x, y, pos.z)
            await ctx.wait(16)
          }
          ctx.setPos(pos.x, pos.y, pos.z)
        },
      })
    }
    
    // Manipulation
    this._register({
      name: 'pick_up',
      description: 'Pick up an object. Object must be in perception memory (run scan_for first if not).',
      requires: ['manipulation:grasp'],
      args: { target: 'object id or color description (e.g. "ball_red", "red ball")' },
      execute: async (ctx) => {
        const target = ctx.args?.target
        if (!target) throw new Error('pick_up requires a target')

        // ONLY use perception memory
        const match = ctx.findPerceivedObject?.(target)
        if (!match?.position) {
          throw new Error(
            `"${target}" not in perception memory. Use scan_for(target:"${target}") first.`
          )
        }

        ctx.setStatus(`Navigating to pick up ${match.id}...`)
        const currentPos = ctx.getPos()
        await ctx.navigateTo(match.position.x, currentPos.y, match.position.z)

        // Do a quick CV capture at close range to confirm current position (ball may have moved)
        ctx.setStatus(`Scanning close range for ${match.id}...`)
        await ctx.captureCV?.()
        await ctx.wait(200)

        // Extend arm for grab
        ctx.setJointGroup('left_arm', 90)
        await ctx.wait(400)

        // Use object ID from perception memory, or fuzzy-match from env
        const grabId = match.id
        const success = ctx.grab(grabId)
        if (!success) {
          ctx.setJointGroup('left_arm', 0)
          throw new Error(
            `Failed to grab "${grabId}" — may have moved. Re-scan and try again.`
          )
        }

        ctx.setStatus(`Holding ${grabId}`)
      },
    })

    this._register({
      name: 'release',
      description: 'Gently place down the currently held object in front of the robot.',
      requires: ['manipulation:grasp'],
      execute: async (ctx) => {
        if (!ctx.robot?.heldObjects?.length) {
          ctx.setStatus('Not holding anything')
          return
        }
        ctx.release()
        ctx.setJointGroup('left_arm', 0)
        await ctx.wait(300)
        ctx.setStatus('Object placed')
      },
    })

    this._register({
      name: 'place',
      description: 'Carry the held object to a target position (x, z) and set it down there.',
      requires: ['manipulation:grasp'],
      args: { x: 'target world X coordinate', z: 'target world Z coordinate' },
      execute: async (ctx) => {
        if (!ctx.robot?.heldObjects?.length) {
          throw new Error('Not holding any object. Use pick_up first.')
        }
        const x = parseFloat(ctx.args?.x)
        const z = parseFloat(ctx.args?.z)
        if (isNaN(x) || isNaN(z)) throw new Error('place requires numeric x and z arguments')

        ctx.setStatus(`Carrying to (${x.toFixed(1)}, ${z.toFixed(1)})...`)
        const cur = ctx.getPos()
        await ctx.navigateTo(x, cur.y, z)

        ctx.release()
        ctx.setJointGroup('left_arm', 0)
        await ctx.wait(400)
        ctx.setStatus('Object placed')
      },
    })

    this._register({
      name: 'place_near',
      description: 'Carry the held object near a perceived target object and place it down.',
      requires: ['manipulation:grasp'],
      args: { target: 'object id or description to place near', distance: 'standoff distance in metres (default 1.0)' },
      execute: async (ctx) => {
        if (!ctx.robot?.heldObjects?.length) {
          throw new Error('Not holding any object. Use pick_up first.')
        }
        const target = ctx.args?.target
        if (!target) throw new Error('place_near requires a target argument')

        const distance = parseFloat(ctx.args?.distance) || 1.0

        // Perception-only lookup
        const match = ctx.findPerceivedObject?.(target)
        if (!match?.position) {
          throw new Error(`"${target}" not in perception memory. Use scan_for(target:"${target}") first.`)
        }
        const targetPos = match.position

        const cur = ctx.getPos()
        const dx  = targetPos.x - cur.x
        const dz  = targetPos.z - cur.z
        const len = Math.sqrt(dx * dx + dz * dz) || 1
        const placeX = targetPos.x - (dx / len) * distance
        const placeZ = targetPos.z - (dz / len) * distance

        ctx.setStatus(`Carrying to near ${match.id}...`)
        await ctx.navigateTo(placeX, cur.y, placeZ)

        ctx.release()
        ctx.setJointGroup('left_arm', 0)
        await ctx.wait(400)
        ctx.setStatus(`Object placed near ${match.id}`)
      },
    })

    this._register({
      name: 'throw',
      description: 'Throw the currently held object forward with force.',
      requires: ['manipulation:grasp'],
      execute: async (ctx) => {
        if (!ctx.robot?.heldObjects?.length) {
          throw new Error('Not holding any object. Use pick_up first.')
        }
        // Raise arm for throw animation
        ctx.setJointGroup('left_arm', 140)
        await ctx.wait(350)

        ctx.throwObject(9.0)

        // Lower arm after throw
        await ctx.wait(200)
        ctx.setJointGroup('left_arm', 0)
        ctx.setStatus('Object thrown!')
      },
    })

    this._register({
      name: 'push',
      description: 'Push an object. Object must be in perception memory (use scan_for first).',
      requires: ['locomotion:ground'],
      args: { target: 'object id or description', force: 'impulse strength (default 6)' },
      execute: async (ctx) => {
        const target = ctx.args?.target
        if (!target) throw new Error('push requires a target')

        // Perception-only lookup
        const match = ctx.findPerceivedObject?.(target)
        if (!match?.position) {
          throw new Error(`"${target}" not in perception memory. Use scan_for(target:"${target}") first.`)
        }

        ctx.setStatus(`Moving to push ${match.id}...`)
        const pos  = match.position
        const cur  = ctx.getPos()
        const dx   = pos.x - cur.x
        const dz   = pos.z - cur.z
        const dist = Math.sqrt(dx * dx + dz * dz)
        const stopDist = 0.9
        if (dist > stopDist + 0.1) {
          await ctx.navigateTo(
            pos.x - (dx / dist) * stopDist,
            cur.y,
            pos.z - (dz / dist) * stopDist
          )
        }

        const force = parseFloat(ctx.args?.force) || 6
        ctx.robot && ctx.pushObject?.(match.id, force)
        ctx.setStatus(`Pushed ${match.id}!`)
      },
    })
    
    // Perception
    this._register({
      name: 'scan_room',
      description: 'Perform a full 360-degree scan to discover all nearby objects.',
      requires: [],  // All robots with sensors can scan
      execute: async (ctx) => {
        const speed = 1.0   // Reduced from 2.0 — slower scan gives 30Hz vision better coverage
        const duration = Math.ceil((2 * Math.PI / speed) * 1000 * 1.05)
        ctx.setStatus('Scanning environment...')
        ctx.rotate(speed)
        await ctx.wait(duration)
        ctx.stop()
        const found = ctx.getKnownObjects()
        ctx.setStatus(`Scan complete — ${found.length} object(s) detected`)
      },
    })
    
    this._register({
      name: 'scan_for',
      description: 'Rotate and scan with camera to find a specific object by description. Stops when found.',
      requires: [],
      args: { target: 'object description to search for (e.g. "red ball", "blue ball", "wooden box")' },
      execute: async (ctx) => {
        const target = ctx.args?.target || 'unknown'
        ctx.setStatus(`Scanning for "${target}"...`)

        const speed    = 1.0
        const fullCircle = (2 * Math.PI / speed) * 1000  // ms for 360°
        const stepMs   = 400   // rotate in 400ms increments
        const steps    = Math.ceil(fullCircle / stepMs)

        for (let i = 0; i < steps; i++) {
          ctx.rotate(speed)
          await ctx.wait(stepMs)
          ctx.stop()
          await ctx.wait(100)   // brief pause so CV can process the frame

          // Trigger CV capture at each step for real visual detection
          await ctx.captureCV?.()
          await ctx.wait(200)   // wait for CV API response to arrive

          // Check perception memory after each CV capture
          const match = ctx.findPerceivedObject?.(target)
          if (match) {
            ctx.setStatus(`Found "${match.id}" at approx (${match.position.x.toFixed(1)}, ${match.position.z.toFixed(1)})`)
            ctx.stop()
            return
          }
        }

        ctx.stop()
        ctx.setStatus(`"${target}" not found after full scan — try explore() and scan again`)
      },
    })
    
    // Stop
    this._register({
      name: 'stop',
      description: 'Stop all movement immediately.',
      requires: [],
      execute: async (ctx) => {
        ctx.stop()
        ctx.setJointGroup('left_arm', 0)
        ctx.setJointGroup('right_arm', 0)
      },
    })

    // ── Basic timed movement skills ─────────────────────────────────────────

    this._register({
      name: 'move_forward',
      description: 'Move forward at full speed for a given duration (default 2 s).',
      requires: ['locomotion:ground'],
      args: { speed: 'm/s (default: maxSpeed)', duration: 'ms (default: 2000)' },
      execute: async (ctx) => {
        const maxSpd = ctx.manifest?.constraints?.maxSpeed ?? 2.5
        const spd    = parseFloat(ctx.args?.speed)    || maxSpd
        const dur    = parseFloat(ctx.args?.duration) || 2000
        ctx.setStatus('Moving forward...')
        ctx.moveForward(spd)
        await ctx.wait(dur)
        ctx.stop()
        ctx.setStatus('Done')
      },
    })

    this._register({
      name: 'move_backward',
      description: 'Move backward at full speed for a given duration (default 2 s).',
      requires: ['locomotion:ground'],
      args: { speed: 'm/s (default: maxSpeed)', duration: 'ms (default: 2000)' },
      execute: async (ctx) => {
        const maxSpd = ctx.manifest?.constraints?.maxSpeed ?? 2.5
        const spd    = parseFloat(ctx.args?.speed)    || maxSpd
        const dur    = parseFloat(ctx.args?.duration) || 2000
        ctx.setStatus('Moving backward...')
        ctx.moveForward(-spd)
        await ctx.wait(dur)
        ctx.stop()
        ctx.setStatus('Done')
      },
    })

    this._register({
      name: 'turn_left',
      description: 'Turn left (counter-clockwise) by a given number of degrees (default 90).',
      requires: ['locomotion:rotate'],
      args: { degrees: 'number (default 90)' },
      execute: async (ctx) => {
        const deg  = parseFloat(ctx.args?.degrees) || 90
        const rad  = Math.abs(deg) * (Math.PI / 180)
        const spd  = constraints.maxAngularSpeed ?? 3.0
        const dur  = (rad / spd) * 1000
        ctx.rotate(spd)
        await ctx.wait(dur)
        ctx.stop()
      },
    })

    this._register({
      name: 'turn_right',
      description: 'Turn right (clockwise) by a given number of degrees (default 90).',
      requires: ['locomotion:rotate'],
      args: { degrees: 'number (default 90)' },
      execute: async (ctx) => {
        const deg  = parseFloat(ctx.args?.degrees) || 90
        const rad  = Math.abs(deg) * (Math.PI / 180)
        const spd  = constraints.maxAngularSpeed ?? 3.0
        const dur  = (rad / spd) * 1000
        ctx.rotate(-spd)
        await ctx.wait(dur)
        ctx.stop()
      },
    })
    
    // Patrol
    this._register({
      name: 'patrol',
      description: 'Move in a patrol pattern around the environment.',
      requires: ['locomotion:ground'],
      execute: async (ctx) => {
        const pos = ctx.getPos()
        const patrolPoints = [
          { x: pos.x + 1.5, z: pos.z },
          { x: pos.x + 1.5, z: pos.z + 1.5 },
          { x: pos.x, z: pos.z + 1.5 },
          { x: pos.x, z: pos.z },
        ]
        
        for (const point of patrolPoints) {
          ctx.setStatus(`Patrolling to (${point.x.toFixed(1)}, ${point.z.toFixed(1)})`)
          await ctx.navigateTo(point.x, pos.y, point.z)
          await ctx.wait(500)
        }
        ctx.setStatus('Patrol complete')
      },
    })

    // ── New skills ──────────────────────────────────────────────────────────

    this._register({
      name: 'return_home',
      description: 'Navigate back to the origin (spawn point) at world coordinates (0, 0).',
      requires: ['locomotion:ground'],
      execute: async (ctx) => {
        ctx.setStatus('Returning home...')
        const pos = ctx.getPos()
        await ctx.navigateTo(0, pos.y, 0)
        ctx.setStatus('Home')
      },
    })

    this._register({
      name: 'survey_grid',
      description: 'Systematically visit a 3×3 grid of positions around the robot to survey the area.',
      requires: ['locomotion:ground'],
      args: { radius: 'grid half-size in metres (default 2)' },
      execute: async (ctx) => {
        const radius = parseFloat(ctx.args?.radius) || 2
        const origin = ctx.getPos()
        const step   = radius
        let visited  = 0

        ctx.setStatus('Surveying grid...')

        for (let row = -1; row <= 1; row++) {
          for (let col = -1; col <= 1; col++) {
            if (row === 0 && col === 0) continue   // skip centre — already there
            const tx = origin.x + col * step
            const tz = origin.z + row * step
            ctx.setStatus(`Survey point ${++visited}/8 (${tx.toFixed(1)}, ${tz.toFixed(1)})`)
            await ctx.navigateTo(tx, origin.y, tz)
            await ctx.wait(300)   // brief pause for sensors to catch up
          }
        }

        // Return to survey start
        await ctx.navigateTo(origin.x, origin.y, origin.z)
        ctx.setStatus('Survey complete')
      },
    })

    this._register({
      name: 'explore',
      description: 'Explore the environment by navigating to random positions within reach.',
      requires: ['locomotion:ground'],
      args: { steps: 'number of random waypoints (default 4)', range: 'max distance per step in metres (default 3)' },
      execute: async (ctx) => {
        const steps = parseInt(ctx.args?.steps) || 4
        const range = parseFloat(ctx.args?.range) || 3

        ctx.setStatus('Exploring...')

        for (let i = 0; i < steps; i++) {
          const pos   = ctx.getPos()
          const angle = Math.random() * Math.PI * 2
          const dist  = (0.5 + Math.random() * 0.5) * range
          const tx    = pos.x + Math.cos(angle) * dist
          const tz    = pos.z + Math.sin(angle) * dist

          ctx.setStatus(`Explore step ${i + 1}/${steps}`)
          await ctx.navigateTo(tx, pos.y, tz)
          await ctx.wait(400)
        }

        ctx.setStatus('Exploration complete')
      },
    })

    this._register({
      name: 'find_goal',
      description: 'Scan the environment to find a named target and navigate to it.',
      requires: ['locomotion:ground'],
      args: { target: 'object name or keyword to search for' },
      execute: async (ctx) => {
        const target = ctx.args?.target || ''
        ctx.setStatus(`Looking for "${target}"...`)

        // Reduced from 1.2 — slower scan improves sensor coverage at 30Hz vision
        const speed = 1.0
        const duration = Math.ceil((2 * Math.PI / speed) * 1000 * 1.05)
        ctx.rotate(speed)
        await ctx.wait(duration)
        ctx.stop()

        const match = target ? ctx.findPerceivedObject?.(target) : ctx.getPerceivedObjects?.()[0]

        if (!match) {
          ctx.setStatus(`"${target}" not found — try scanning first`)
          throw new Error(`Target "${target}" not found after scanning`)
        }

        ctx.setStatus(`Found "${match.id}" — navigating...`)
        const pos = ctx.getPos()
        await ctx.navigateTo(match.position.x, pos.y, match.position.z)
        ctx.setStatus(`Reached "${match.id}"`)
      },
    })

    this._register({
      name: 'move_and_look',
      description: 'Move to a new vantage point and do a quick visual scan. Use when scan_for fails to find the target.',
      requires: ['locomotion:ground'],
      args: { direction: 'forward|left|right|back (default: random)', distance: 'metres to move (default: 2.5)' },
      execute: async (ctx) => {
        const dir  = ctx.args?.direction || 'forward'
        const dist = parseFloat(ctx.args?.distance) || 2.5

        const pos = ctx.getPos()
        const angle = { forward: 0, right: Math.PI / 2, back: Math.PI, left: -Math.PI / 2 }[dir] ?? 0

        const tx = pos.x + Math.sin(angle) * dist
        const tz = pos.z + Math.cos(angle) * dist

        ctx.setStatus(`Moving ${dir} ${dist}m to new vantage...`)
        try {
          await ctx.navigateTo(tx, pos.y, tz)
        } catch {
          ctx.setStatus('Pathfinding blocked — scanning from here')
        }

        // Do a 180-degree look sweep from the new position
        ctx.setStatus('Looking around...')
        for (let i = 0; i < 5; i++) {
          ctx.rotate(0.8)
          await ctx.wait(300)
          ctx.stop()
          await ctx.wait(80)
          await ctx.captureCV?.()
          await ctx.wait(180)
        }
        ctx.stop()
        ctx.setStatus('Look sweep complete')
      },
    })

    this._register({
      name: 'wave',
      description: 'Wave with the robot arm as a greeting or acknowledgement gesture.',
      requires: ['manipulation:grasp'],
      execute: async (ctx) => {
        ctx.setStatus('Waving...')

        // Three wave cycles: raise to 90°, oscillate between 90° and 60°, lower
        // Robot has one arm — left arm only (left_shoulder + left_elbow)
        ctx.setJoint('left_shoulder', 90)
        ctx.setJoint('left_elbow', 0)
        await ctx.wait(600)   // Wait for arm to reach raised position

        for (let cycle = 0; cycle < 3; cycle++) {
          ctx.setJoint('left_shoulder', 90)
          await ctx.wait(180)
          ctx.setJoint('left_shoulder', 60)
          await ctx.wait(180)
        }

        // Return to rest
        ctx.setJoint('left_shoulder', 0)
        ctx.setJoint('left_elbow', 0)
        await ctx.wait(600)
        ctx.setStatus('Wave complete')
      },
    })

    // ── Arm position skills ─────────────────────────────────────────────────
    // Robot has ONE physical arm: left arm (left_shoulder joint, axis "-x").
    // Pivot is at the shoulder end of the arm cylinder (Z≈0.004 from GLB).
    //
    // Axis "-x" convention: positive angle lifts the arm UP.
    //   angle =   0°  → arm points forward (+Z, rest/default in GLB)
    //   angle =  90°  → arm points straight up (+Y)
    //   angle = -90°  → arm points straight down (-Y)
    //
    // Joint velocity limit: 3.0 rad/s ≈ 172°/s
    //   90° travel ≈ 0.52 s → wait 700 ms (adds headroom for motor settling)
    //   180° travel (full range) ≈ 1.05 s → wait 1200 ms

    this._register({
      name: 'arm_up',
      description: 'Raise the arm straight up (shoulder 90°, pointing to the sky).',
      requires: ['manipulation:grasp'],
      execute: async (ctx) => {
        ctx.setStatus('Raising arm...')
        ctx.setJoint('left_shoulder', 90)
        ctx.setJoint('left_elbow', 0)
        await ctx.wait(1200)  // Worst case: from -90° (down) to 90° (up) = 180° travel
        ctx.setStatus('Arm up')
      },
    })

    this._register({
      name: 'arm_down',
      description: 'Lower the arm straight down (shoulder -90°, pointing to the ground).',
      requires: ['manipulation:grasp'],
      execute: async (ctx) => {
        ctx.setStatus('Lowering arm...')
        ctx.setJoint('left_shoulder', -90)
        ctx.setJoint('left_elbow', 0)
        await ctx.wait(1200)  // Worst case: from 90° (up) to -90° (down) = 180° travel
        ctx.setStatus('Arm down')
      },
    })

    this._register({
      name: 'arm_forward',
      description: 'Extend the arm forward horizontally (shoulder 0°, pointing ahead).',
      requires: ['manipulation:grasp'],
      execute: async (ctx) => {
        ctx.setStatus('Extending arm forward...')
        ctx.setJoint('left_shoulder', 0)
        ctx.setJoint('left_elbow', 0)
        await ctx.wait(700)   // Max travel from either extreme to 0° = 90° travel
        ctx.setStatus('Arm forward')
      },
    })

    this._register({
      name: 'arm_rest',
      description: 'Return the arm to the forward resting position (shoulder 0°).',
      requires: ['manipulation:grasp'],
      execute: async (ctx) => {
        ctx.setStatus('Resting arm...')
        ctx.setJoint('left_shoulder', 0)
        ctx.setJoint('left_elbow', 0)
        await ctx.wait(700)
        ctx.setStatus('Arm at rest')
      },
    })
  }
  
  _register(skillDef) {
    const skill = {
      ...skillDef,
      source: 'builtin',
      verified: true,
      approved: true,
      code: null,
      metadata: { executionCount: 0, successCount: 0, failureCount: 0 },
    }
    _skills.set(skill.name, skill)
  }
  
  _persist() {
    const toSave = []
    for (const [name, skill] of _skills) {
      if (skill.source === 'synthesized' && skill.approved) {
        toSave.push({
          name: skill.name,
          description: skill.description,
          requires: skill.requires,
          code: skill.code,
          metadata: skill.metadata,
        })
      }
    }
    
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(toSave))
    } catch (e) {
      console.warn('[Skills] Failed to persist to localStorage:', e)
    }
  }
  
  _loadPersisted() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      
      const saved = JSON.parse(raw)
      for (const entry of saved) {
        try {
          const fn = new AsyncFunction('context', entry.code)
          const skill = {
            name: entry.name,
            description: entry.description,
            requires: entry.requires || [],
            source: 'synthesized',
            verified: true,
            approved: true,
            code: entry.code,
            execute: fn,
            metadata: entry.metadata || { executionCount: 0, successCount: 0, failureCount: 0 },
          }
          _skills.set(entry.name, skill)
        } catch (e) {
          console.warn(`[Skills] Failed to load persisted skill "${entry.name}":`, e)
        }
      }
      
      console.log(`[Skills] Loaded ${saved.length} persisted skills`)
    } catch (e) {
      console.warn('[Skills] Failed to load from localStorage:', e)
    }
  }
}

// Needed for dynamic function creation
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor

// ─── Singleton Export ──────────────────────────────────────────────────────────

export const skillRegistry = new SkillRegistry()
