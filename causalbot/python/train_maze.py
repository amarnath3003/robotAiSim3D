"""
train_maze.py — Incremental curriculum training on the maze environment.

This script CONTINUES from the latest checkpoint saved by train_rl.py.
The robot carries its wall-avoidance reflexes learned in the open room
and fine-tunes them on the harder maze navigation task.

Usage:
  1. Open browser at http://localhost:5173/?mode=maze
  2. Run: python python/train_maze.py
"""
import os
import glob
import json
import asyncio
import threading
import time
import math

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import websockets
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

# ─── Maze Environment ──────────────────────────────────────────────────────────
class MazeEnv(gym.Env):
    """
    Same 13-dim observation as CausalBotEnv:
      [target_dist, target_rel_angle, 11 lidar rays]
    But now connects to the maze scene (?mode=maze).
    The maze JS side sends { type: 'maze_goal', x, z } once connected,
    so Python knows where the goal is.
    """
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        # 13-dim obs: [dist, angle, 11 lidar]
        low  = np.array([0.0, -np.pi] + [0.0] * 11, dtype=np.float32)
        high = np.array([70.0, np.pi] + [5.0] * 11, dtype=np.float32)  # dist up to 70m for massive maze
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action space: Allow stopping [0.0], but we will penalize it to prevent endless spinning
        self.action_space = spaces.Box(
            low=np.array([0.0, -2.0]),
            high=np.array([2.5,  2.0]),
            dtype=np.float32
        )

        self.state        = np.zeros(13, dtype=np.float32)
        self.target_goal  = np.array([0.0, 0.0])
        self.prev_dist    = 0.0   # track progress each step
        self.start_dist   = 0.0   # track total progress to detect spinning
        self.max_steps    = 3000  # shorter episodes = faster learning signal
        self.current_step = 0
        self.stagnation_counter = 0
        self.websocket    = None
        self.latest_prompt = None

        self._state_event = threading.Event()
        self._reset_event = threading.Event()

        # Start WS server in background thread
        self.loop = asyncio.new_event_loop()
        t = threading.Thread(target=self._start_server, daemon=True)
        t.start()
        time.sleep(0.5)

    def _start_server(self):
        asyncio.set_event_loop(self.loop)
        async def run_server():
            async with websockets.serve(self._ws_handler, 'localhost', 8765):
                await asyncio.Future()
        self.loop.run_until_complete(run_server())

    async def _ws_handler(self, websocket):
        if self.websocket is not None:
            try: await self.websocket.close()
            except: pass
        self.websocket = websocket
        print('Maze JS client connected!')
        try:
            async for raw in websocket:
                data = json.loads(raw)
                t = data.get('type')

                if t == 'state':
                    obs = data.get('observation', [])
                    if len(obs) == 14:
                        self._process_obs(obs)
                        self._state_event.set()

                elif t == 'reset_done':
                    obs = data.get('observation', [])
                    if len(obs) == 14:
                        self._process_obs(obs)
                        self._reset_event.set()

                elif t == 'maze_goal':
                    # JS tells Python where the goal is after maze generation
                    gx = float(data.get('x', 0.0))
                    gz = float(data.get('z', 0.0))
                    self.target_goal = np.array([gx, gz])
                    print(f'[MazeEnv] Goal received: ({gx:.2f}, {gz:.2f})')

                elif t == 'prompt':
                    self.latest_prompt = data.get('text', '')

        except websockets.exceptions.ConnectionClosed:
            print('Maze JS client disconnected')
            self.websocket = None

    def _process_obs(self, obs_list):
        rx, rz, heading = obs_list[0], obs_list[1], obs_list[2]
        lidar = obs_list[3:]
        dx = self.target_goal[0] - rx
        dz = self.target_goal[1] - rz
        target_dist  = float(np.sqrt(dx**2 + dz**2))
        target_angle = float(np.arctan2(dx, dz)) - heading
        while target_angle >  np.pi: target_angle -= 2 * np.pi
        while target_angle < -np.pi: target_angle += 2 * np.pi
        self.state = np.array([target_dist, target_angle] + lidar, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.websocket is None:
            print('Waiting for maze JS client...')
            while self.websocket is None:
                time.sleep(0.1)
        self._reset_event.clear()
        self.current_step = 0

        # Tell JS to reset the robot to maze start
        reset_msg = json.dumps({'type': 'reset'})
        asyncio.run_coroutine_threadsafe(
            self.websocket.send(reset_msg), self.loop
        ).result(timeout=5)

        self._reset_event.wait(timeout=10)
        self.prev_dist    = float(self.state[0])   # record start distance
        self.start_dist   = float(self.state[0])
        self.current_step = 0
        self.stagnation_counter = 0
        return self.state, {}

    def step(self, action):
        if self.websocket is None:
            return self.state, -1.0, True, False, {}

        # Allow stopping to turn safely
        linear  = float(np.clip(action[0], 0.0, 2.5))
        angular = float(np.clip(action[1], -2.0, 2.0))

        self._state_event.clear()
        msg = json.dumps({'type': 'action', 'action': [linear, angular]})
        
        try:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(msg), self.loop
            ).result(timeout=5)
            self._state_event.wait(timeout=5)
        except Exception as e:
            print(f"WebSocket communication error: {e}")
            return self.state, -1.0, True, False, {}

        target_dist = float(self.state[0])
        lidar_min   = float(np.min(self.state[2:]))   # closest obstacle
        lidar_front = float(self.state[7])             # center-front ray
        self.current_step += 1

        terminated = False
        truncated  = False
        reward = 0.0

        if abs(self.prev_dist - target_dist) < 0.02:
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0

        # ── Goal reached (PRIORITY 1) ──
        if target_dist < 0.6:
            reward = 100.0  # Massive reward for winning
            terminated = True

        # ── Collision (PRIORITY 2) ──
        elif lidar_min < 0.28:
            reward = -50.0  # Harsher penalty to avoid wall tunneling
            terminated = True

        # ── Anti-Spin / Anti-Stuck ──
        elif self.stagnation_counter > 50:
            reward = -20.0  # Penalty for getting stuck or spinning in place
            terminated = True

        else:
            # PRIMARY: reward actual progress toward goal this step
            progress = self.prev_dist - target_dist    # positive = getting closer
            reward   = progress * 10.0   # heavily incentivize moving forward

            # BONUS: continuous reward for facing the goal
            angle_to_goal = float(self.state[1])
            reward += 0.1 * math.cos(angle_to_goal)

            # PENALTY: if front lidar is very close, discourage rushing into walls
            if lidar_front < 0.5:
                reward -= 0.5

            # PENALTY: heavy time cost if spinning in place (linear speed ~0)
            if linear < 0.1 and abs(angular) > 0.5:
                reward -= 0.5
            elif linear < 0.1:
                reward -= 0.2
            else:
                reward -= 0.01 # normal time cost

        self.prev_dist = target_dist

        if self.current_step >= self.max_steps:
            truncated = True

        return self.state, float(reward), terminated, truncated, {}

    def close(self):
        pass


# ─── Find latest checkpoint ────────────────────────────────────────────────────
def find_latest_checkpoint(model_dir='models'):
    checkpoints = glob.glob(os.path.join(model_dir, 'causalbot_ppo_*_steps.zip'))
    final = os.path.join(model_dir, 'causalbot_ppo_final.zip')
    if os.path.exists(final):
        return final
    if not checkpoints:
        return None
    # Sort by step count
    def step_count(p):
        try: return int(os.path.basename(p).split('_steps')[0].split('_')[-1])
        except: return 0
    return max(checkpoints, key=step_count)


# ─── Main training entry ───────────────────────────────────────────────────────
def train():
    os.makedirs('models', exist_ok=True)

    env = MazeEnv()

    checkpoint = find_latest_checkpoint()

    if checkpoint:
        print(f'\n✅ Found checkpoint: {checkpoint}')
        print('Loading and fine-tuning on maze environment...\n')
        custom_objects = {
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }
        model = PPO.load(checkpoint, env=env, custom_objects=custom_objects)
        # Lower LR for fine-tuning — don't forget what was learned
        model.learning_rate = 1e-4
    else:
        print('\n⚠️  No checkpoint found — training from scratch on maze.\n')
        model = PPO(
            'MlpPolicy', env, verbose=1,
            tensorboard_log='./tensorboard_logs/',
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            ent_coef=0.05,     # high entropy → more random exploration early on
            clip_range=0.2,
            vf_coef=0.5,
            normalize_advantage=True,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='causalbot_maze'
    )

    print('Starting maze training! Open browser at http://localhost:5173/?mode=maze')
    print('The robot will learn to navigate the maze to the glowing green goal.\n')

    # reset_num_timesteps=False keeps cumulative step count — fully incremental
    model.learn(
        total_timesteps=300000,
        callback=checkpoint_callback,
        reset_num_timesteps=False
    )

    print('\nTraining finished. Saving maze model...')
    model.save('models/causalbot_maze_final')
    env.close()


if __name__ == '__main__':
    train()
