"""
causalbot_env.py — Gymnasium environment bridging the Three.js sim via WebSocket.

Architecture:
  - asyncio WebSocket server runs in a background daemon thread
  - Main thread (gym step/reset) communicates with it via threading.Event + shared state
  - All shared state access is protected by a single threading.Lock

Observation (13-dim):
  [target_dist, target_rel_angle, lidar_0..lidar_10]

Action (4-dim, continuous):
  [linear_speed (0..2.5 m/s),     — move forward
   angular_speed (-2..2 rad/s),   — rotate left/right
   arm_rotation (-π..π rad),      — rotate robot arms/hands
   jump (0..1 — hop if > 0.5)]    — small upward jump

Episode logic — Simulation of Death:
  - 3-7 random obstacle walls generated every episode
  - Walls sent to JS: Three.js renders them, Rapier adds colliders
  - Hit any wall (lidar_min < DEATH_DIST) → instant death, episode resets
  - Reach goal (dist < GOAL_RADIUS) → success, episode resets
  - max_steps exceeded → truncation, episode resets

Train mode:
  - When train_mode=True the env auto-resets immediately on every termination.
  - Episodes chain non-stop without waiting for user prompts.
"""

import asyncio
import json
import math
import random
import threading
import time
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WS_PORT      = 8765
MAX_LIDAR    = 5.0
OBS_DIM      = 13      # target_dist + target_angle + 11 lidar
DEATH_DIST   = 0.28    # lidar min → instant death
GOAL_RADIUS  = 0.45    # distance to goal → success
START_X      = 0.0
START_Z      = 1.8
ROOM_HALF    = 2.3     # room bounds ±


# ── AABB overlap check (outside any loop — B3 fix) ──────────────────────────
def _aabb_overlaps(ax, az, aw, ad, bx, bz, margin=0.0):
    return (abs(ax - bx) < (aw / 2 + margin) and
            abs(az - bz) < (ad / 2 + margin))


class CausalBotEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low  = np.array([0.0, -2.0, -math.pi, 0.0], dtype=np.float32),
            high = np.array([2.5,  2.0,  math.pi, 1.0], dtype=np.float32),
        )

        low_obs  = np.array([0.0, -math.pi] + [0.0] * 11, dtype=np.float32)
        high_obs = np.array([10.0, math.pi] + [MAX_LIDAR] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # ── Shared state (all protected by _lock) ─────────────────────────────
        self._lock          = threading.Lock()
        self._obs           = np.zeros(OBS_DIM, dtype=np.float32)
        self._raw_obs       = None       # raw 14-value array from JS
        self.target_goal    = np.array([0.0, -1.5], dtype=np.float32)
        self.latest_prompt  = None
        self.current_step   = 0
        self.max_steps      = 500
        self._prev_dist     = 0.0
        self.force_reset    = False

        # ── Train mode — auto-reset without waiting for prompts ───────────────
        self.train_mode     = True   # set False if you want prompt-gated episodes

        # ── Episode statistics ────────────────────────────────────────────────
        self.episode_num    = 0
        self.death_count    = 0
        self.success_count  = 0
        self.episode_reward = 0.0

        # ── History ring-buffer (last 20 outcomes for win-rate display) ───────
        self._history       = deque(maxlen=20)  # PERF2: O(1) append/evict vs list.pop(0)
        self._HISTORY_LEN   = 20

        # ── Sync events ───────────────────────────────────────────────────────
        self._state_event   = threading.Event()
        self._reset_event   = threading.Event()

        # ── WebSocket ─────────────────────────────────────────────────────────
        self._websocket     = None
        self._loop          = asyncio.new_event_loop()
        t = threading.Thread(target=self._start_server, daemon=True)
        t.start()
        print(f'[Env] WebSocket server starting on ws://localhost:{WS_PORT}')

    # ── WebSocket server ──────────────────────────────────────────────────────

    def _start_server(self):
        import websockets
        asyncio.set_event_loop(self._loop)

        async def serve():
            async with websockets.serve(self._handler, 'localhost', WS_PORT):
                await asyncio.Future()

        self._loop.run_until_complete(serve())

    async def _handler(self, ws):
        # Evict any existing connection
        with self._lock:
            old = self._websocket
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass

        with self._lock:
            self._websocket = ws

        print('[Env] Browser connected.')
        try:
            async for raw in ws:
                msg = json.loads(raw)
                t   = msg.get('type')

                if t == 'state':
                    self._ingest_obs(msg.get('observation', []))
                    self._state_event.set()

                elif t == 'reset_done':
                    self._ingest_obs(msg.get('observation', []))
                    self._reset_event.set()

                elif t == 'prompt':
                    with self._lock:
                        self.latest_prompt = msg.get('text', '')
                    print(f'[Env] Prompt: "{self.latest_prompt}"')

                elif t == 'goal_override':
                    x = float(msg.get('x', 0.0))
                    z = float(msg.get('z', 0.0))
                    with self._lock:
                        self.target_goal = np.array([x, z], dtype=np.float32)
                    print(f'[Env] Goal override → ({x:.2f}, {z:.2f})')

                elif t == 'params':
                    if 'max_steps' in msg:
                        with self._lock:
                            self.max_steps = max(50, int(msg['max_steps']))
                        print(f'[Env] max_steps → {self.max_steps}')
                    if 'train_mode' in msg:
                        with self._lock:
                            self.train_mode = bool(msg['train_mode'])
                        print(f'[Env] train_mode → {self.train_mode}')
                    if 'death_dist' in msg:
                        global DEATH_DIST
                        DEATH_DIST = float(msg['death_dist'])
                        print(f'[Env] death_dist → {DEATH_DIST:.2f}')
                    if msg.get('reset'):
                        with self._lock:
                            self.force_reset = True
                        print('[Env] Manual reset requested')

        except Exception as e:
            print(f'[Env] Connection error: {e}')
        finally:
            with self._lock:
                if self._websocket is ws:
                    self._websocket = None
            # Unblock any waiting step/reset so the agent loop can recover
            self._state_event.set()
            self._reset_event.set()
            print('[Env] Browser disconnected.')

    # ── Observation processing ────────────────────────────────────────────────

    def _ingest_obs(self, raw):
        """
        JS sends 14 values: [rx, rz, heading, lidar×11]
        Derives 13-dim gym obs: [dist_to_goal, angle_to_goal, lidar×11]
        """
        if len(raw) < 14:
            return

        rx      = float(raw[0])
        rz      = float(raw[1])
        heading = float(raw[2])
        lidar   = [min(float(v), MAX_LIDAR) for v in raw[3:14]]

        with self._lock:
            gx, gz = float(self.target_goal[0]), float(self.target_goal[1])

        dx    = gx - rx
        dz    = gz - rz
        dist  = math.sqrt(dx * dx + dz * dz)
        angle = math.atan2(dx, dz) - heading

        # Normalise to [-π, π]
        angle = (angle + math.pi) % (2 * math.pi) - math.pi

        with self._lock:
            self._obs     = np.array([dist, angle] + lidar, dtype=np.float32)
            self._raw_obs = raw

    # ── Wall generator ────────────────────────────────────────────────────────

    def _generate_walls(self, goal_x, goal_z):
        """
        Generate 3-7 random axis-aligned walls inside the room.
        Guaranteed clear zones: START (radius 0.75) and GOAL (radius 0.75).
        """
        EXCLUSION = 0.80
        walls     = []
        n_target  = random.randint(3, 7)
        attempts  = 0

        while len(walls) < n_target and attempts < 300:
            attempts += 1

            cx = random.uniform(-ROOM_HALF + 0.25, ROOM_HALF - 0.25)
            cz = random.uniform(-ROOM_HALF + 0.25, ROOM_HALF - 0.25)

            horiz  = random.random() > 0.5
            length = random.uniform(0.5, 1.9)
            w      = length if horiz else 0.15
            d      = 0.15   if horiz else length
            h      = random.uniform(0.55, 1.05)

            if _aabb_overlaps(cx, cz, w, d, START_X, START_Z, EXCLUSION):
                continue
            if _aabb_overlaps(cx, cz, w, d, goal_x, goal_z, EXCLUSION):
                continue

            # Don't overlap with already-placed walls (give 0.2 m gap)
            ok = True
            for ww in walls:
                if _aabb_overlaps(cx, cz, w + 0.2, d + 0.2, ww['x'], ww['z'], 0):
                    ok = False
                    break
            if not ok:
                continue

            walls.append({
                'x': round(cx, 3), 'z': round(cz, 3),
                'w': round(w, 3),  'd': round(d, 3),
                'h': round(h, 3),
            })

        return walls

    # ── WebSocket helpers ─────────────────────────────────────────────────────

    def _wait_for_ws(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._websocket is not None:
                    return True
            time.sleep(0.05)
        return False

    def _send(self, obj):
        """Thread-safe fire-and-forget send."""
        with self._lock:
            ws = self._websocket
        if ws is None:
            return False
        future = asyncio.run_coroutine_threadsafe(ws.send(json.dumps(obj)), self._loop)
        try:
            future.result(timeout=5)
            return True
        except Exception as e:
            print(f'[Env] Send error: {e}')
            return False

    def _send_telemetry(self, reward=None, mode=None, outcome=None):
        """Push live stats to the JS dashboard (non-blocking)."""
        with self._lock:
            gx, gz      = float(self.target_goal[0]), float(self.target_goal[1])
            dist_val    = float(self._obs[0]) if len(self._obs) > 0 else 0.0
            deaths      = self.death_count
            successes   = self.success_count
            episode     = self.episode_num
            ep_reward   = round(self.episode_reward, 3)
            history     = list(self._history)

        # Recent win-rate (last 20 episodes)
        total   = len(history)
        win_rate = round(history.count('success') / total, 3) if total > 0 else 0.0

        payload = {
            'type':      'telemetry',
            'goal':      {'x': gx, 'z': gz},
            'mode':      mode or 'EXECUTING',
            'deaths':    deaths,
            'successes': successes,
            'episode':   episode,
            'ep_reward': ep_reward,
            'win_rate':  win_rate,
            'dist':      round(dist_val, 3),
        }
        if outcome:
            payload['outcome'] = outcome
        if reward is not None:
            payload['reward'] = round(float(reward), 4)

        self._send(payload)

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._wait_for_ws():
            raise RuntimeError('[Env] Browser not connected after 60 s')

        self._reset_event.clear()
        self._state_event.clear()
        self.current_step   = 0
        self.episode_reward = 0.0
        self.episode_num   += 1

        # Random goal — must be >1.2 m from start
        rng = np.random.default_rng(seed)
        for _ in range(1000):
            gx = float(rng.uniform(-ROOM_HALF + 0.3, ROOM_HALF - 0.3))
            gz = float(rng.uniform(-ROOM_HALF + 0.3, ROOM_HALF - 0.3))
            if math.sqrt((gx - START_X) ** 2 + (gz - START_Z) ** 2) > 1.2:
                break

        with self._lock:
            self.target_goal = np.array([gx, gz], dtype=np.float32)

        walls = self._generate_walls(gx, gz)

        print(f'[Env] Ep {self.episode_num} | Goal ({gx:.2f}, {gz:.2f}) | '
              f'{len(walls)} walls | Deaths {self.death_count} | '
              f'Wins {self.success_count}')

        self._send({
            'type':  'reset',
            'goal':  {'x': round(gx, 3), 'z': round(gz, 3)},
            'walls': walls,
        })

        if not self._reset_event.wait(timeout=10):
            print('[Env] reset_done timeout — using stale observation')

        with self._lock:
            obs = self._obs.copy()

        self._prev_dist = float(obs[0])
        self._send_telemetry(mode='IDLE')
        return obs, {}

    def step(self, action):
        # Normalise + unpack 4-dim action (backwards compat: pad if shorter)
        action = np.asarray(action, dtype=np.float32)
        if len(action) < 4:
            action = np.concatenate([action, np.zeros(4 - len(action), dtype=np.float32)])

        linear   = float(np.clip(action[0],  0.0, 2.5))
        angular  = float(np.clip(action[1], -2.0, 2.0))
        arm_rot  = float(np.clip(action[2], -math.pi, math.pi))
        jump     = float(np.clip(action[3],  0.0, 1.0))

        self._state_event.clear()

        # Reconnect wait if browser dropped
        with self._lock:
            ws = self._websocket
        if ws is None:
            if not self._wait_for_ws(timeout=30):
                with self._lock:
                    obs = self._obs.copy()
                return obs, -1.0, True, False, {'disconnected': True}

        ok = self._send({'type': 'action', 'action': [linear, angular, arm_rot, jump]})
        if not ok:
            with self._lock:
                obs = self._obs.copy()
            return obs, -1.0, True, False, {'send_failed': True}

        # Wait for JS to apply movement and send back state
        if not self._state_event.wait(timeout=5):
            with self._lock:
                obs = self._obs.copy()
            return obs, 0.0, False, False, {'timeout': True}

        with self._lock:
            obs = self._obs.copy()

        target_dist = float(obs[0])
        lidar_min   = float(np.min(obs[2:]))
        self.current_step += 1

        # ── Reward: Simulation of Death ───────────────────────────────────────
        terminated = False
        truncated  = False
        outcome    = None

        # Dense progress reward
        progress = (self._prev_dist - target_dist) * 5.0
        reward   = progress - 0.005
        self._prev_dist = target_dist

        if lidar_min < DEATH_DIST:
            reward     = -20.0
            terminated = True
            outcome    = 'death'
            with self._lock:
                self.death_count += 1
                self._history.append('death')
            print(f'[Env] ☠  DEATH #{self.death_count} | '
                  f'ep {self.episode_num} step {self.current_step} | '
                  f'dist {target_dist:.2f}m | lidar_min {lidar_min:.2f}m')

        elif target_dist < GOAL_RADIUS:
            reward     = 20.0
            terminated = True
            outcome    = 'success'
            with self._lock:
                self.success_count += 1
                self._history.append('success')
            print(f'[Env] ✓  SUCCESS #{self.success_count} | '
                  f'ep {self.episode_num} step {self.current_step}')

        # L2 fix: truncated is only True if the episode was NOT already terminated
        # (SB3 treats simultaneous terminated+truncated as undefined behaviour)
        if not terminated and self.current_step >= self.max_steps:
            truncated = True
            if outcome is None:
                outcome = 'timeout'
                with self._lock:
                    self._history.append('timeout')

        self.episode_reward += reward

        mode = 'IDLE' if (terminated or truncated) else 'EXECUTING'
        self._send_telemetry(reward=reward, mode=mode, outcome=outcome)

        return obs, reward, terminated, truncated, {}

    # ── Utility ───────────────────────────────────────────────────────────────

    def get_latest_prompt(self):
        """Thread-safe prompt retrieval — returns None if no new prompt."""
        with self._lock:
            p = self.latest_prompt
            self.latest_prompt = None
        return p

    def set_goal(self, x, z):
        with self._lock:
            self.target_goal = np.array([x, z], dtype=np.float32)

    def render(self):
        pass

    def close(self):
        pass