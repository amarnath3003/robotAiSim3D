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

Any natural language command is accepted and parsed to a world coordinate via LLM:
  'go to the ball', 'move to the box', 'reach the glass', 'go to the corner', etc.

Episode logic — Simulation of Death:
  - 3-7 random obstacle walls generated every episode
  - Walls sent to JS: Three.js renders them, Rapier adds colliders
  - Hit any wall (lidar_min < DEATH_DIST) → instant death, episode resets
  - Reach goal (dist < GOAL_RADIUS) → success, episode resets
  - max_steps exceeded → truncation, episode resets
"""

import asyncio
import json
import math
import random
import threading
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WS_PORT      = 8765
MAX_LIDAR    = 5.0
OBS_DIM      = 13     # target_dist + target_angle + 11 lidar
DEATH_DIST   = 0.28   # lidar threshold for death
GOAL_RADIUS  = 0.45   # goal reached threshold
START_X      = 0.0
START_Z      = 1.8
ROOM_HALF    = 2.3    # room bounds ±


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
        self._lock         = threading.Lock()
        self._obs          = np.zeros(OBS_DIM, dtype=np.float32)
        self._raw_obs      = None        # [rx, rz, heading, lidar×11] from JS
        self.target_goal   = np.array([0.0, -1.5], dtype=np.float32)
        self.latest_prompt = None
        self.current_step  = 0
        self.max_steps     = 500
        self._prev_dist    = 0.0         # for progress reward

        # ── Episode statistics ─────────────────────────────────────────────────
        self.episode_num   = 0
        self.death_count   = 0
        self.success_count = 0
        self.episode_reward = 0.0

        # ── Sync events ────────────────────────────────────────────────────────
        self._state_event  = threading.Event()
        self._reset_event  = threading.Event()

        # ── WebSocket ──────────────────────────────────────────────────────────
        self._websocket    = None
        self._loop         = asyncio.new_event_loop()
        t = threading.Thread(target=self._start_server, daemon=True)
        t.start()
        print(f'[Env] WebSocket server starting on ws://localhost:{WS_PORT}')

    # ── WebSocket server ──────────────────────────────────────────────────────

    def _start_server(self):
        import websockets
        asyncio.set_event_loop(self._loop)

        async def serve():
            async with websockets.serve(self._handler, 'localhost', WS_PORT):
                await asyncio.Future()  # run forever

        self._loop.run_until_complete(serve())

    async def _handler(self, ws):
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
                    print(f'[Env] Prompt received: "{self.latest_prompt}"')

                elif t == 'goal_override':
                    x = float(msg.get('x', 0.0))
                    z = float(msg.get('z', 0.0))
                    with self._lock:
                        self.target_goal = np.array([x, z], dtype=np.float32)
                    print(f'[Env] Dashboard goal override: ({x:.2f}, {z:.2f})')

                elif t == 'params':
                    if 'max_steps' in msg:
                        with self._lock:
                            self.max_steps = int(msg['max_steps'])
                        print(f'[Env] max_steps → {self.max_steps}')

        except Exception as e:
            print(f'[Env] Connection closed: {e}')
        finally:
            with self._lock:
                if self._websocket is ws:
                    self._websocket = None
            print('[Env] Browser disconnected.')

    # ── Observation processing ────────────────────────────────────────────────

    def _ingest_obs(self, raw):
        """
        JS sends 14 values: [rx, rz, heading, lidar×11]
        We compute the 13-dim gym observation from them.
        """
        if len(raw) < 14:
            return

        rx, rz, heading = float(raw[0]), float(raw[1]), float(raw[2])
        lidar = [min(float(v), MAX_LIDAR) for v in raw[3:14]]

        with self._lock:
            gx, gz = float(self.target_goal[0]), float(self.target_goal[1])

        dx = gx - rx
        dz = gz - rz
        dist  = math.sqrt(dx * dx + dz * dz)
        angle = math.atan2(dx, dz) - heading

        # Wrap to [-π, π]
        while angle >  math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi

        with self._lock:
            self._obs     = np.array([dist, angle] + lidar, dtype=np.float32)
            self._raw_obs = raw

    # ── Wall generator ────────────────────────────────────────────────────────

    def _generate_walls(self, goal_x, goal_z):
        """
        Generate 3-7 random axis-aligned walls.
        Each wall: {x, z, w, d, h}  (centre position, full-width, full-depth, height)
        Guaranteed not to block the robot start or goal within exclusion_r.
        """
        exclusion_r = 0.75
        walls = []
        n_walls = random.randint(3, 7)
        attempts = 0

        while len(walls) < n_walls and attempts < 200:
            attempts += 1

            cx = random.uniform(-ROOM_HALF + 0.2, ROOM_HALF - 0.2)
            cz = random.uniform(-ROOM_HALF + 0.2, ROOM_HALF - 0.2)
            horiz  = random.random() > 0.5
            length = random.uniform(0.6, 1.8)
            w = length if horiz else 0.15
            d = 0.15 if horiz else length
            h = random.uniform(0.6, 1.0)

            def overlaps(ax, az, aw, ad, bx, bz, margin=0.0):
                hw = aw / 2 + margin
                hd = ad / 2 + margin
                return abs(ax - bx) < hw and abs(az - bz) < hd

            if overlaps(cx, cz, w, d, START_X, START_Z, exclusion_r):
                continue
            if overlaps(cx, cz, w, d, goal_x, goal_z, exclusion_r):
                continue

            walls.append({'x': round(cx, 3), 'z': round(cz, 3),
                          'w': round(w, 3),  'd': round(d, 3),
                          'h': round(h, 3)})

        return walls

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def _wait_for_ws(self, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._websocket is not None:
                    return True
            time.sleep(0.1)
        return False

    def _send(self, obj):
        """Thread-safe send to current WebSocket."""
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
        """Push live stats back to the JS dashboard."""
        with self._lock:
            gx, gz     = float(self.target_goal[0]), float(self.target_goal[1])
            dist       = float(self._obs[0]) if len(self._obs) > 0 else None
            deaths     = self.death_count
            successes  = self.success_count
            episode    = self.episode_num
            ep_reward  = round(self.episode_reward, 3)

        payload = {
            'type':       'telemetry',
            'goal':       {'x': gx, 'z': gz},
            'mode':       mode or 'EXECUTING',
            'deaths':     deaths,
            'successes':  successes,
            'episode':    episode,
            'ep_reward':  ep_reward,
        }
        if outcome:
            payload['outcome'] = outcome     # 'death' | 'success' | 'timeout'
        if reward is not None:
            payload['reward'] = round(float(reward), 4)
        if dist is not None:
            payload['dist']   = round(dist, 3)

        self._send(payload)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._wait_for_ws():
            raise RuntimeError('[Env] Browser did not connect within 60 s')

        self._reset_event.clear()
        self.current_step   = 0
        self.episode_reward = 0.0
        self.episode_num   += 1

        # ── Pick a random goal that isn't too close to start ──────────────────
        while True:
            gx = random.uniform(-ROOM_HALF + 0.3, ROOM_HALF - 0.3)
            gz = random.uniform(-ROOM_HALF + 0.3, ROOM_HALF - 0.3)
            dist_to_start = math.sqrt((gx - START_X)**2 + (gz - START_Z)**2)
            if dist_to_start > 1.2:
                break

        with self._lock:
            self.target_goal = np.array([gx, gz], dtype=np.float32)

        # ── Generate random walls ─────────────────────────────────────────────
        walls = self._generate_walls(gx, gz)

        print(f'[Env] Episode {self.episode_num} | Goal ({gx:.2f}, {gz:.2f}) | {len(walls)} walls')

        # ── Tell JS to reset: spawn robot, place goal ball, build walls ────────
        self._send({
            'type':  'reset',
            'goal':  {'x': round(gx, 3), 'z': round(gz, 3)},
            'walls': walls,
        })

        if not self._reset_event.wait(timeout=15):
            print('[Env] reset_done timeout — using last observation')

        with self._lock:
            obs = self._obs.copy()

        self._prev_dist = float(obs[0])
        self._send_telemetry(mode='IDLE')
        return obs, {}

    def step(self, action):
        # Unpack 4-dim action
        action = np.asarray(action, dtype=np.float32)
        if len(action) < 4:
            # Backwards compat: pad with zeros
            action = np.concatenate([action, np.zeros(4 - len(action), dtype=np.float32)])

        linear      = float(np.clip(action[0],  0.0, 2.5))
        angular     = float(np.clip(action[1], -2.0, 2.0))
        arm_rot     = float(np.clip(action[2], -math.pi, math.pi))
        jump        = float(np.clip(action[3],  0.0, 1.0))

        self._state_event.clear()

        # Check WS alive
        with self._lock:
            ws = self._websocket
        if ws is None:
            if not self._wait_for_ws(timeout=30):
                with self._lock:
                    obs = self._obs.copy()
                return obs, -1.0, True, False, {}

        ok = self._send({'type': 'action', 'action': [linear, angular, arm_rot, jump]})
        if not ok:
            with self._lock:
                obs = self._obs.copy()
            return obs, -1.0, True, False, {}

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

        # Progress reward: positive if moving toward goal
        progress = (self._prev_dist - target_dist) * 5.0
        reward   = progress - 0.005   # time penalty

        self._prev_dist = target_dist

        if lidar_min < DEATH_DIST:
            # DEATH — hit a wall
            reward     = -20.0
            terminated = True
            outcome    = 'death'
            with self._lock:
                self.death_count += 1
            print(f'[Env] ☠  DEATH #{self.death_count} | step {self.current_step} | dist {target_dist:.2f}')

        elif target_dist < GOAL_RADIUS:
            # SUCCESS — reached the ball
            reward     = 20.0
            terminated = True
            outcome    = 'success'
            with self._lock:
                self.success_count += 1
            print(f'[Env] ✓  SUCCESS #{self.success_count} | step {self.current_step}')

        if self.current_step >= self.max_steps:
            truncated = True
            outcome   = 'timeout'

        self.episode_reward += reward

        # Push live stats to JS dashboard
        mode = 'IDLE' if (terminated or truncated) else 'EXECUTING'
        self._send_telemetry(reward=reward, mode=mode, outcome=outcome)

        return obs, reward, terminated, truncated, {}

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