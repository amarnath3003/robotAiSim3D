/**
 * ui/skill_editor.js — In-Browser Skill Code Editor
 *
 * A floating panel toggled with Shift+E that lets users:
 *  1. Browse and select synthesized skills from a sidebar
 *  2. Write or edit async skill code directly in-browser
 *  3. Check syntax live using the AsyncFunction constructor
 *  4. Save → skillRegistry.registerSynthesized() with immediate activation
 *
 * No external dependencies — pure DOM + inline styles.
 */

// ─── State ─────────────────────────────────────────────────────────────────────

let _panel    = null
let _registry = null
let _visible  = false

// ─── Styles ────────────────────────────────────────────────────────────────────

const CSS = `
#skill-editor-panel {
  display: none;
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 820px;
  max-width: 96vw;
  height: 560px;
  max-height: 90vh;
  background: #13141a;
  border: 1px solid #2a2d3a;
  border-radius: 10px;
  box-shadow: 0 24px 64px rgba(0,0,0,0.7);
  font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
  font-size: 13px;
  color: #c9d1d9;
  z-index: 9000;
  flex-direction: column;
  overflow: hidden;
}

#skill-editor-panel.se-visible {
  display: flex;
}

.se-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 10px 16px;
  background: #0d0e13;
  border-bottom: 1px solid #2a2d3a;
  flex-shrink: 0;
}

.se-header-title {
  font-weight: 600;
  font-size: 13px;
  color: #79c0ff;
  letter-spacing: 0.5px;
}

.se-header-hint {
  font-size: 11px;
  color: #484f58;
  margin-left: 10px;
}

.se-close-btn {
  background: none;
  border: none;
  color: #6e7681;
  font-size: 16px;
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 4px;
  line-height: 1;
}
.se-close-btn:hover { color: #e6edf3; background: #21262d; }

.se-body {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ── */
.se-sidebar {
  width: 180px;
  border-right: 1px solid #21262d;
  overflow-y: auto;
  flex-shrink: 0;
  background: #0d0e13;
}

.se-sidebar-label {
  padding: 8px 12px 4px;
  font-size: 10px;
  text-transform: uppercase;
  letter-spacing: 0.8px;
  color: #484f58;
}

.se-skill-item {
  padding: 7px 12px;
  cursor: pointer;
  border-left: 2px solid transparent;
  font-size: 12px;
  color: #8b949e;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.se-skill-item:hover       { background: #161b22; color: #c9d1d9; }
.se-skill-item.se-selected { border-left-color: #58a6ff; color: #58a6ff; background: #161b22; }

.se-empty-sidebar {
  padding: 16px 12px;
  font-size: 11px;
  color: #484f58;
  line-height: 1.5;
}

/* ── Main editor pane ── */
.se-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  padding: 12px 14px 10px;
  gap: 8px;
}

.se-row {
  display: flex;
  gap: 8px;
}

.se-input {
  flex: 1;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 5px;
  color: #e6edf3;
  font-family: inherit;
  font-size: 12px;
  padding: 6px 9px;
  outline: none;
  transition: border-color 0.15s;
}
.se-input:focus { border-color: #388bfd; }
.se-input::placeholder { color: #484f58; }

.se-code-area {
  flex: 1;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 5px;
  color: #e6edf3;
  font-family: inherit;
  font-size: 12px;
  padding: 9px 10px;
  outline: none;
  resize: none;
  line-height: 1.55;
  tab-size: 2;
  transition: border-color 0.15s;
}
.se-code-area:focus { border-color: #388bfd; }
.se-code-area::placeholder { color: #484f58; }

.se-code-hint {
  font-size: 10px;
  color: #484f58;
  padding: 2px 0;
  flex-shrink: 0;
}

.se-actions {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-shrink: 0;
}

.se-btn {
  padding: 6px 14px;
  border: 1px solid #30363d;
  border-radius: 5px;
  background: #21262d;
  color: #c9d1d9;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}
.se-btn:hover { background: #30363d; }

.se-btn-primary {
  background: #1f6feb;
  border-color: #388bfd;
  color: #fff;
  font-weight: 600;
}
.se-btn-primary:hover { background: #388bfd; }

.se-btn-new {
  background: none;
  border: 1px dashed #30363d;
  color: #58a6ff;
  font-size: 11px;
  padding: 5px 10px;
}
.se-btn-new:hover { border-color: #388bfd; background: #0d1117; }

.se-status {
  font-size: 11px;
  min-height: 16px;
  flex-shrink: 0;
  padding: 2px 0;
}
.se-status.ok    { color: #3fb950; }
.se-status.error { color: #f85149; }
.se-status.info  { color: #58a6ff; }
`

// ─── Panel HTML ────────────────────────────────────────────────────────────────

const PANEL_HTML = `
<div id="skill-editor-panel">
  <div class="se-header">
    <span>
      <span class="se-header-title">Skill Editor</span>
      <span class="se-header-hint">Shift+E to toggle</span>
    </span>
    <button class="se-close-btn" id="se-close" title="Close">✕</button>
  </div>

  <div class="se-body">
    <div class="se-sidebar" id="se-sidebar">
      <div class="se-sidebar-label">Synthesized</div>
      <div id="se-skill-list"></div>
      <div style="padding:6px 12px;">
        <button class="se-btn se-btn-new" id="se-new">+ New skill</button>
      </div>
    </div>

    <div class="se-main">
      <div class="se-row">
        <input id="se-name" class="se-input" placeholder="skill_name (snake_case)" style="max-width:200px;" />
        <input id="se-desc" class="se-input" placeholder="Short description for the LLM..." />
      </div>

      <textarea id="se-code" class="se-code-area"
        placeholder="// Async skill body — context API available:
// context.getPos() · context.navigateTo(x,y,z) · context.wait(ms)
// context.setJoint(name,deg) · context.getKnownObjects() · context.setStatus(text)
// context.moveForward(speed) · context.rotate(speed) · context.stop()

const pos = context.getPos()
await context.navigateTo(pos.x + 1, pos.y, pos.z)"></textarea>

      <div class="se-code-hint">
        Tip: <kbd>Tab</kbd> = 2-space indent. Code runs as <code>async (context) =&gt; { … }</code>
      </div>

      <div class="se-actions">
        <button class="se-btn" id="se-check">Check syntax</button>
        <button class="se-btn se-btn-primary" id="se-save">Save &amp; Register</button>
        <div class="se-status" id="se-status"></div>
      </div>
    </div>
  </div>
</div>
`

// ─── Public API ────────────────────────────────────────────────────────────────

/**
 * Initialise the skill editor.
 * @param {Object} skillRegistry — the global SkillRegistry instance
 */
export function initSkillEditor(skillRegistry) {
  if (_panel) return

  _registry = skillRegistry

  // Inject CSS
  const style = document.createElement('style')
  style.textContent = CSS
  document.head.appendChild(style)

  // Inject HTML
  const host = document.createElement('div')
  host.innerHTML = PANEL_HTML
  document.body.appendChild(host)
  _panel = document.getElementById('skill-editor-panel')

  // Tab key → 2-space indent in code area
  document.getElementById('se-code')?.addEventListener('keydown', _handleTabKey)

  // Keyboard toggle (Shift+E)
  document.addEventListener('keydown', (e) => {
    if (e.shiftKey && (e.key === 'E' || e.key === 'e') && !_inputFocused()) {
      e.preventDefault()
      toggle()
    }
  })

  document.getElementById('se-close')?.addEventListener('click', hide)
  document.getElementById('se-new')?.addEventListener('click', _clearForm)
  document.getElementById('se-check')?.addEventListener('click', checkSyntax)
  document.getElementById('se-save')?.addEventListener('click', saveSkill)

  hide()
}

/** Programmatically show the editor. */
export function showSkillEditor() { show() }

/** Programmatically hide the editor. */
export function hideSkillEditor() { hide() }

/** Toggle editor visibility. */
export function toggleSkillEditor() { toggle() }

// ─── Internal ──────────────────────────────────────────────────────────────────

function show() {
  if (!_panel) return
  _panel.classList.add('se-visible')
  _visible = true
  _renderSkillList()
  // Focus the name input if it's empty
  const nameEl = document.getElementById('se-name')
  if (nameEl && !nameEl.value) nameEl.focus()
}

function hide() {
  if (!_panel) return
  _panel.classList.remove('se-visible')
  _visible = false
}

function toggle() { _visible ? hide() : show() }

function _clearForm() {
  document.getElementById('se-name').value  = ''
  document.getElementById('se-desc').value  = ''
  document.getElementById('se-code').value  = ''
  _setStatus('', '')
  // Deselect sidebar
  document.querySelectorAll('.se-skill-item').forEach(el => el.classList.remove('se-selected'))
  document.getElementById('se-name').focus()
}

function _renderSkillList() {
  const listEl = document.getElementById('se-skill-list')
  if (!listEl || !_registry) return

  // Get all synthesized skills
  const skills = _registry.getAll
    ? _registry.getAll().filter(s => s.source === 'synthesized')
    : []

  if (skills.length === 0) {
    listEl.innerHTML = '<div class="se-empty-sidebar">No synthesized skills yet.<br>Write one and save it.</div>'
    return
  }

  listEl.innerHTML = skills.map(s =>
    `<div class="se-skill-item" data-name="${_esc(s.name)}" title="${_esc(s.description || s.name)}">${_esc(s.name)}</div>`
  ).join('')

  listEl.querySelectorAll('.se-skill-item').forEach(el => {
    el.addEventListener('click', () => _loadSkill(el.dataset.name))
  })
}

function _loadSkill(name) {
  const skill = _registry.get ? _registry.get(name) : null
  // Also try internal map (get() filters by capability; use raw map if available)
  const raw = _registry._skills?.get(name)
  const s   = skill ?? raw
  if (!s) return

  document.getElementById('se-name').value = s.name        ?? ''
  document.getElementById('se-desc').value = s.description ?? ''
  document.getElementById('se-code').value = s.code        ?? ''
  _setStatus(`Loaded: ${s.name}`, 'info')

  // Highlight in sidebar
  document.querySelectorAll('.se-skill-item').forEach(el => {
    el.classList.toggle('se-selected', el.dataset.name === name)
  })
}

function checkSyntax() {
  const code = document.getElementById('se-code')?.value ?? ''
  if (!code.trim()) { _setStatus('Code is empty', 'error'); return false }

  try {
    // Wrap in an AsyncFunction to check syntax without executing
    // eslint-disable-next-line no-new-func
    new Function('context', `return (async (context) => { ${code} })`)
    _setStatus('Syntax OK', 'ok')
    return true
  } catch (e) {
    _setStatus(`Syntax error: ${e.message}`, 'error')
    return false
  }
}

function saveSkill() {
  const name = document.getElementById('se-name')?.value?.trim()
  const desc = document.getElementById('se-desc')?.value?.trim()
  const code = document.getElementById('se-code')?.value?.trim()

  if (!name) { _setStatus('Skill name is required', 'error'); return }
  if (!code)  { _setStatus('Code is required', 'error'); return }

  // Validate name format
  if (!/^[a-z][a-z0-9_]*$/.test(name)) {
    _setStatus('Name must be snake_case (e.g. my_skill)', 'error')
    return
  }

  // Syntax check before saving
  try {
    // eslint-disable-next-line no-new-func
    new Function('context', `return (async (context) => { ${code} })`)
  } catch (e) {
    _setStatus(`Cannot save — syntax error: ${e.message}`, 'error')
    return
  }

  if (!_registry) {
    _setStatus('Registry not available', 'error')
    return
  }

  const skill = _registry.registerSynthesized(name, code, desc || name)

  if (skill) {
    _setStatus(`"${name}" saved and registered`, 'ok')
    _renderSkillList()
    // Re-select in sidebar
    setTimeout(() => {
      document.querySelectorAll('.se-skill-item').forEach(el => {
        el.classList.toggle('se-selected', el.dataset.name === name)
      })
    }, 50)
  } else {
    _setStatus(`Failed to register "${name}" — check console`, 'error')
  }
}

function _setStatus(msg, type) {
  const el = document.getElementById('se-status')
  if (!el) return
  el.textContent = msg
  el.className   = `se-status ${type}`
}

function _handleTabKey(e) {
  if (e.key !== 'Tab') return
  e.preventDefault()
  const ta    = e.target
  const start = ta.selectionStart
  const end   = ta.selectionEnd
  ta.value = ta.value.slice(0, start) + '  ' + ta.value.slice(end)
  ta.selectionStart = ta.selectionEnd = start + 2
}

function _inputFocused() {
  const tag = document.activeElement?.tagName
  return tag === 'INPUT' || tag === 'TEXTAREA' || document.activeElement?.isContentEditable
}

function _esc(str) {
  return String(str ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[c]))
}
