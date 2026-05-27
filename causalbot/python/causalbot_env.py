"""
causalbot_env.py — Gymnasium environment bridging the Three.js sim via WebSocket.

Architecture:
  - asyncio WebSocket server runs in a background daemon thread
  - Main thread (gym step/reset) communicates with it via threading.Event + shared state
  - All shared state access is protected by a single threading.Lock

Observation (13-dim):
  [target_dist, target_rel_angle, lidar_0..lidar_10]

Action (2-dim, continuous):
  [linear_speed (0..2.5 m/s), angular_speed (-2..2 rad/s)]
"""

import asyncio
import json
import math
import threading
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

WS_PORT     = 8765
MAX_LIDAR   = 5.0
OBS_DIM     = 13   # target_dist + target_angle + 11 lidar


class CausalBotEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super().__init__()

        self.action_space = spaces.Box(
            low  = np.array([0.0, -2.0], dtype=np.float32),
            high = np.array([2.5,  2.0], dtype=np.float32),
        )

        low_obs  = np.array([0.0, -math.pi] + [0.0] * 11, dtype=np.float32)
        high_obs = np.array([10.0, math.pi] + [MAX_LIDAR] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(low=low_obs, high=high_obs, dtype=np.float32)

        # Shared state (protected by _lock)
        self._lock          = threading.Lock()
        self._obs           = np.zeros(OBS_DIM, dtype=np.float32)
        self._raw_obs       = None   # [rx, rz, heading, lidar×11] from JS
        self.target_goal    = np.array([0.0, 0.0], dtype=np.float32)
        self.latest_prompt  = None
        self.current_step   = 0
        self.max_steps      = 1000

        # Sync events — set by async WS thread, waited on by main thread
        self._state_event   = threading.Event()
        self._reset_event   = threading.Event()

        # WebSocket
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
                await asyncio.Future()  # run forever

        self._loop.run_until_complete(serve())

    async def _handler(self, ws):
        # Accept new connection; close old one if any
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
            self._obs      = np.array([dist, angle] + lidar, dtype=np.float32)
            self._raw_obs  = raw

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

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if not self._wait_for_ws():
            raise RuntimeError('[Env] Browser did not connect within 60 s')

        self._reset_event.clear()
        self.current_step = 0

        # Random goal within room
        with self._lock:
            self.target_goal = np.array([
                np.random.uniform(-2.4, 2.4),
                np.random.uniform(-2.4, 2.4),
            ], dtype=np.float32)

        self._send({'type': 'reset'})

        if not self._reset_event.wait(timeout=15):
            print('[Env] reset_done timeout — using last observation')

        with self._lock:
            obs = self._obs.copy()
        return obs, {}

    def step(self, action):
        linear  = float(np.clip(action[0],  0.0, 2.5))
        angular = float(np.clip(action[1], -2.0, 2.0))

        self._state_event.clear()

        ok = self._send({'type': 'action', 'action': [linear, angular]})
        if not ok:
            with self._lock:
                obs = self._obs.copy()
            return obs, -1.0, True, False, {}

        if not self._state_event.wait(timeout=5):
            # Browser paused or tab in background — return neutral step
            with self._lock:
                obs = self._obs.copy()
            return obs, 0.0, False, False, {'timeout': True}

        with self._lock:
            obs = self._obs.copy()

        target_dist = float(obs[0])
        lidar_min   = float(np.min(obs[2:]))
        self.current_step += 1

        # ── Reward ────────────────────────────────────────────────────────────
        reward     = -0.01     # small time cost
        terminated = False
        truncated  = False

        if lidar_min < 0.25:
            reward     = -10.0
            terminated = True
        elif target_dist < 0.4:
            reward     = 10.0
            terminated = True
        else:
            reward += 0.1 / (target_dist + 0.1)

        if self.current_step >= self.max_steps:
            truncated = True

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