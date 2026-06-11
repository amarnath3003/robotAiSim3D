/**
 * brain/behavior_tree.js — Async Behavior Tree Runtime
 *
 * Composable BT engine for robot task execution.
 * Every node implements async tick(blackboard) → BT_SUCCESS | BT_FAILURE | BT_RUNNING.
 *
 * Architecture:
 *  - ActionNode awaits an async fn directly (no frame-by-frame polling needed)
 *  - Composite nodes chain awaits — Sequence/Selector/Parallel all async
 *  - BTRunner.start(tree, bb) returns a Promise resolved when tree reaches terminal status
 *  - planToBehaviorTree() converts a flat LLM plan into a Sequence with retry/optional support
 */

// ─── Status Constants ──────────────────────────────────────────────────────────

export const BT_SUCCESS = 'success'
export const BT_FAILURE = 'failure'
export const BT_RUNNING  = 'running'

// ─── Blackboard ────────────────────────────────────────────────────────────────

/**
 * Shared key-value store passed through the entire BT during execution.
 * Nodes read/write here for cross-node communication.
 */
export class Blackboard {
  constructor(init = {}) {
    this._data = { ...init }
  }

  get(key)        { return this._data[key] }
  set(key, value) { this._data[key] = value; return this }
  has(key)        { return key in this._data }
  delete(key)     { delete this._data[key]; return this }
  toObject()      { return { ...this._data } }
}

// ─── Base Node ─────────────────────────────────────────────────────────────────

export class BTNode {
  /** @returns {Promise<'success'|'failure'|'running'>} */
  async tick(blackboard) { return BT_FAILURE }
  reset() {}
}

// ─── Composite Nodes ───────────────────────────────────────────────────────────

/**
 * Sequence — runs children left-to-right.
 * Returns SUCCESS only if ALL children succeed.
 * Returns FAILURE (and stops) on first child failure.
 */
export class Sequence extends BTNode {
  constructor(...children) {
    super()
    this.children = children
  }

  async tick(bb) {
    for (const child of this.children) {
      const status = await child.tick(bb)
      if (status !== BT_SUCCESS) return status
    }
    return BT_SUCCESS
  }

  reset() { this.children.forEach(c => c.reset()) }
}

/**
 * Selector — runs children left-to-right.
 * Returns SUCCESS on first successful child.
 * Returns FAILURE if ALL children fail.
 */
export class Selector extends BTNode {
  constructor(...children) {
    super()
    this.children = children
  }

  async tick(bb) {
    for (const child of this.children) {
      const status = await child.tick(bb)
      if (status === BT_SUCCESS) return BT_SUCCESS
    }
    return BT_FAILURE
  }

  reset() { this.children.forEach(c => c.reset()) }
}

/**
 * Parallel — runs all children concurrently via Promise.all.
 * @param {BTNode[]} children
 * @param {'all'|'any'} policy — 'all': succeed when all succeed; 'any': succeed when any succeeds
 */
export class Parallel extends BTNode {
  constructor(children, policy = 'all') {
    super()
    this.children = children
    this.policy   = policy
  }

  async tick(bb) {
    const results = await Promise.all(this.children.map(c => c.tick(bb)))
    if (this.policy === 'any') {
      return results.some(r => r === BT_SUCCESS) ? BT_SUCCESS : BT_FAILURE
    }
    return results.every(r => r === BT_SUCCESS) ? BT_SUCCESS : BT_FAILURE
  }

  reset() { this.children.forEach(c => c.reset()) }
}

// ─── Decorator Nodes ───────────────────────────────────────────────────────────

/** Flips SUCCESS ↔ FAILURE. Passes through RUNNING unchanged. */
export class Inverter extends BTNode {
  constructor(child) { super(); this.child = child }

  async tick(bb) {
    const s = await this.child.tick(bb)
    if (s === BT_SUCCESS) return BT_FAILURE
    if (s === BT_FAILURE) return BT_SUCCESS
    return s
  }

  reset() { this.child.reset() }
}

/** Swallows child FAILURE — always returns SUCCESS. Use for optional steps. */
export class AlwaysSucceed extends BTNode {
  constructor(child) { super(); this.child = child }
  async tick(bb) { await this.child.tick(bb); return BT_SUCCESS }
  reset() { this.child.reset() }
}

/**
 * RepeatUntilSuccess — retries child on FAILURE with exponential backoff.
 * @param {BTNode} child
 * @param {number} maxRetries — additional attempts after the first failure
 */
export class RepeatUntilSuccess extends BTNode {
  constructor(child, maxRetries = 3) {
    super()
    this.child      = child
    this.maxRetries = maxRetries
  }

  async tick(bb) {
    for (let i = 0; i <= this.maxRetries; i++) {
      if (i > 0) this.child.reset()
      const status = await this.child.tick(bb)
      if (status === BT_SUCCESS) return BT_SUCCESS
      if (i < this.maxRetries) {
        await new Promise(r => setTimeout(r, 150 * (i + 1)))
      }
    }
    return BT_FAILURE
  }

  reset() { this.child.reset() }
}

// ─── Leaf Nodes ────────────────────────────────────────────────────────────────

/** Evaluates a synchronous or async predicate. SUCCESS if truthy, FAILURE otherwise. */
export class Condition extends BTNode {
  constructor(predicate) { super(); this.predicate = predicate }

  async tick(bb) {
    try {
      return (await this.predicate(bb)) ? BT_SUCCESS : BT_FAILURE
    } catch {
      return BT_FAILURE
    }
  }
}

/**
 * ActionNode — wraps an async function as a BT leaf.
 * Returns SUCCESS on resolution, FAILURE on rejection.
 * On failure, stores error info on blackboard under 'lastError'.
 */
export class ActionNode extends BTNode {
  constructor(name, fn) {
    super()
    this.name = name
    this.fn   = fn
  }

  async tick(bb) {
    try {
      await this.fn(bb)
      return BT_SUCCESS
    } catch (e) {
      if (bb) bb.set('lastError', { node: this.name, message: e.message })
      return BT_FAILURE
    }
  }
}

/** Pauses execution for `ms` milliseconds, then succeeds. */
export class Wait extends BTNode {
  constructor(ms) { super(); this.ms = ms }
  async tick(_bb) {
    await new Promise(r => setTimeout(r, this.ms))
    return BT_SUCCESS
  }
}

// ─── BTRunner ──────────────────────────────────────────────────────────────────

/**
 * Runs a behavior tree from root to completion.
 * start() returns a Promise<{success, status, blackboard}>.
 * The runner is reusable — call start() again for a new run.
 */
export class BTRunner {
  constructor() {
    this._running  = false
    this._aborted  = false
  }

  isRunning() { return this._running }

  /** Request abort on next async boundary. */
  abort() { this._aborted = true }

  /**
   * Execute the tree until it reaches SUCCESS or FAILURE.
   * @param {BTNode} tree
   * @param {Blackboard} [blackboard]
   * @returns {Promise<{success: boolean, status: string, blackboard: Blackboard}>}
   */
  async start(tree, blackboard = new Blackboard()) {
    if (this._running) throw new Error('[BTRunner] Already running a tree')
    this._running  = true
    this._aborted  = false

    // Expose runner on blackboard so nodes can request abort
    blackboard.set('_runner', this)

    try {
      const status = await tree.tick(blackboard)
      return { success: status === BT_SUCCESS, status, blackboard }
    } finally {
      this._running = false
    }
  }
}

// ─── Plan → Behavior Tree ──────────────────────────────────────────────────────

/**
 * Convert a flat planner step array into a Sequence behavior tree.
 *
 * Each step gets:
 *  - Abort flag check (reads blackboard 'abortFlag')
 *  - UI hooks: onStepStart(i, step, total), onStepDone(i, step), onStepFail(i, step, errMsg)
 *  - Skill registry lookup + execution
 *  - Per-step retry via step.retries (RepeatUntilSuccess wrapper)
 *  - Optional step via step.optional (AlwaysSucceed wrapper)
 *  - Failure metadata on blackboard: 'failedStep' = { index, step, error, remaining }
 *
 * @param {Array<{skill, args, description, retries?, optional?}>} plan
 * @param {Object}   skillRegistry
 * @param {function(args): Object} contextFn — builds skill execution context per step
 * @param {{onStepStart?, onStepDone?, onStepFail?}} [hooks]
 * @returns {Sequence}
 */
export function planToBehaviorTree(plan, skillRegistry, contextFn, hooks = {}) {
  const { onStepStart, onStepDone, onStepFail } = hooks
  const totalSteps = plan.length

  const children = plan.map((step, i) => {
    // ── Core action: single execution attempt ─────────────────────────────
    const coreNode = new ActionNode(step.skill, async (bb) => {
      // Honor abort request
      if (bb.get('abortFlag')?.()) throw new Error('Aborted by user')

      onStepStart?.(i, step, totalSteps)

      const skill = skillRegistry.get(step.skill)
      if (!skill) throw new Error(`Skill "${step.skill}" not found in registry`)

      await skill.execute(contextFn(step.args))

      onStepDone?.(i, step)
      bb.set('lastSuccessStep', i)
    })

    // ── Retry wrapper ─────────────────────────────────────────────────────
    const retries  = typeof step.retries === 'number' ? step.retries : 0
    const withRetry = retries > 0
      ? new RepeatUntilSuccess(coreNode, retries)
      : coreNode

    // ── Failure-tracking wrapper ──────────────────────────────────────────
    // Captures the error onto the blackboard AFTER all retries are exhausted,
    // then propagates FAILURE upward for the Sequence to stop.
    const tracker = new ActionNode(`track:${step.skill}`, async (bb) => {
      const status = await withRetry.tick(bb)

      if (status !== BT_SUCCESS) {
        const err    = bb.get('lastError')
        const errMsg = err?.message || 'step failed'

        onStepFail?.(i, step, errMsg)

        bb.set('failedStep', {
          index:     i,
          step,
          error:     errMsg,
          remaining: plan.slice(i + 1),
        })

        // Re-throw so this ActionNode's own tick() returns FAILURE
        throw new Error(errMsg)
      }
    })

    // ── Optional steps succeed even if the tracker returns FAILURE ────────
    return step.optional ? new AlwaysSucceed(tracker) : tracker
  })

  return new Sequence(...children)
}
