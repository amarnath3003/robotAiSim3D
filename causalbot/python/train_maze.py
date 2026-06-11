"""
train_maze.py — Incremental curriculum training on the maze environment.

Architecture decisions (why each choice was made):
  - VecNormalize wraps the env to normalize obs + returns → critical because
    obs dims range from 0-70 (dist) vs 0-5 (lidar) vs -π..π (angle).
    Without this, the critic gradient is dominated by the large-magnitude dims.
  - n_steps=4096 (> max_steps=2000) so PPO collects COMPLETE episodes before
    updating. With n_steps=1024 and max_steps=3000 PPO used to train on
    partial trajectories, giving it a fragmented credit-assignment view.
  - gamma=0.98 (not 0.995!) — with 2000 max steps, the goal reward R=100
    discounted 2000 steps away is 100 * 0.98^2000 ≈ 4e-17 with 0.995!
    With gamma=0.98 it's still 2e-18 for the worst case but typical maze
    solve time is 500-800 steps → 100 * 0.98^600 ≈ 5.6. Survivable.
  - ent_coef=0.03 → mild entropy encourages exploration without randomness
    preventing learning. Previous 0.05 was too high and caused persistent
    spinning because random actions were too rewarded.
  - Stagnation detection: 200 consecutive steps of <0.02m progress (about 3s
    at 60fps) → terminate. 50 was triggering falsely on intentional slow turns.
  - Proximity wall penalty is distance-squared shaped for a steeper gradient.
  - Collision threshold raised to 0.35m (robot radius 0.25m + 0.10m margin)
    to compensate for the discrete physics timestep where the robot can move
    fast enough to skip collision detection (wall tunneling) in one substep.

Usage:
  1. Open browser at http://localhost:5173/?mode=maze
  2. Run: python python/train_maze.py
"""

import os
import sys
import glob
import json
import math
import asyncio
import threading
import time
import signal

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import websockets
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# ─── Constants ────────────────────────────────────────────────────────────────
WS_PORT        = 8765
MAX_DIST       = 75.0    # max observable distance (meters)
MAX_LIDAR      = 5.0     # max lidar range (meters)
GOAL_RADIUS    = 0.6     # metres — goal reached
COLLISION_DIST = 0.35    # lidar threshold for collision (robot r=0.25 + margin)
MAX_STEPS      = 2000    # per episode — shorter = faster learning signal
STAG_WINDOW    = 200     # steps with <0.02m progress → stuck
STAG_MIN_MOVE  = 0.02    # metres; below this counts as stagnation


# ─── Maze Environment ─────────────────────────────────────────────────────────
class MazeEnv(gym.Env):
    """
    13-dim observation: [target_dist, target_rel_angle, 11 lidar rays]
    Connects to the ?mode=maze Three.js scene via WebSocket.
    """
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()

        # Observation space — NOTE: VecNormalize will normalise these at runtime
        low  = np.array([0.0,    -np.pi] + [0.0] * 11, dtype=np.float32)
        high = np.array([MAX_DIST, np.pi] + [MAX_LIDAR] * 11, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # Action: [linear_speed (m/s), angular_speed (rad/s)]
        # Allow linear=0 so robot can turn in tight corridors
        self.action_space = spaces.Box(
            low =np.array([0.0, -2.0], dtype=np.float32),
            high=np.array([2.5,  2.0], dtype=np.float32),
        )

        self._reset_internal_state()

        self._state_event = threading.Event()
        self._reset_event = threading.Event()
        self._lock        = threading.Lock()

        # C6 fix: real asyncio.Lock for protecting self.websocket inside the async handler.
        # threading.Lock() is NOT awaitable; the old _lock_async stub did nothing.
        self._ws_lock = None   # created lazily inside the event loop

        # WebSocket server in daemon thread
        self.loop      = asyncio.new_event_loop()
        self.websocket = None
        t = threading.Thread(target=self._start_server, daemon=True)
        t.start()
        time.sleep(0.5)

    def _reset_internal_state(self):
        self.state            = np.zeros(13, dtype=np.float32)
        self.target_goal      = np.array([0.0, 0.0], dtype=np.float32)
        self.prev_dist        = MAX_DIST
        self.current_step     = 0
        self.stag_counter     = 0   # consecutive steps without meaningful progress

    # ── WebSocket server ──────────────────────────────────────────────────────
    def _start_server(self):
        asyncio.set_event_loop(self.loop)
        # C6 fix: create the asyncio.Lock inside the event loop where it will be used
        self._ws_lock = asyncio.Lock()
        async def run():
            async with websockets.serve(self._ws_handler, 'localhost', WS_PORT):
                await asyncio.Future()
        self.loop.run_until_complete(run())

    async def _ws_handler(self, websocket):
        # C6 fix: use a real asyncio.Lock to atomically swap self.websocket
        async with self._ws_lock:
            old = self.websocket
            self.websocket = websocket
        if old is not None:
            try:
                await old.close()
            except Exception:
                pass
        print('\n[OK] Maze JS client connected!\n')
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
                    gx = float(data.get('x', 0.0))
                    gz = float(data.get('z', 0.0))
                    self.target_goal = np.array([gx, gz], dtype=np.float32)
                    print(f'[MazeEnv] Goal received: ({gx:.2f}, {gz:.2f})')
        except websockets.exceptions.ConnectionClosed:
            print('[MazeEnv] JS client disconnected.')
            self.websocket = None

    # ── Observation processing ─────────────────────────────────────────────────
    def _process_obs(self, obs_list):
        rx, rz, heading = float(obs_list[0]), float(obs_list[1]), float(obs_list[2])
        lidar = [float(v) for v in obs_list[3:]]

        dx = float(self.target_goal[0]) - rx
        dz = float(self.target_goal[1]) - rz
        target_dist  = math.sqrt(dx * dx + dz * dz)
        target_angle = math.atan2(dx, dz) - heading

        # Wrap angle to [-π, π]
        while target_angle >  math.pi: target_angle -= 2 * math.pi
        while target_angle < -math.pi: target_angle += 2 * math.pi

        # Clamp lidar to max range (defensive against JS reporting > 5.0)
        lidar = [min(v, MAX_LIDAR) for v in lidar]

        self.state = np.array([target_dist, target_angle] + lidar, dtype=np.float32)

    # ── Gymnasium API ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Wait for the browser to connect
        if self.websocket is None:
            print('[MazeEnv] Waiting for maze JS client...')
            while self.websocket is None:
                time.sleep(0.1)

        self._reset_event.clear()
        self._reset_internal_state()

        try:
            asyncio.run_coroutine_threadsafe(
                self.websocket.send(json.dumps({'type': 'reset'})),
                self.loop
            ).result(timeout=10)
        except Exception as e:
            print(f'[MazeEnv] reset send error: {e}')

        got_reply = self._reset_event.wait(timeout=15)
        if not got_reply:
            print('[MazeEnv] ⚠️  reset_done timeout — using last state')

        self.prev_dist = float(self.state[0])
        return self.state.copy(), {}

    def step(self, action):
        # S2 fix: read websocket under the threading lock to avoid a TOCTOU race
        with self._lock:
            ws = self.websocket
        if ws is None:
            return self.state.copy(), -1.0, True, False, {}

        linear  = float(np.clip(action[0], 0.0, 2.5))
        angular = float(np.clip(action[1], -2.0, 2.0))

        self._state_event.clear()
        msg = json.dumps({'type': 'action', 'action': [linear, angular]})
        try:
            asyncio.run_coroutine_threadsafe(
                ws.send(msg), self.loop
            ).result(timeout=5)
            got_state = self._state_event.wait(timeout=5)
            if not got_state:
                # Browser paused/tab backgrounded — return neutral step
                return self.state.copy(), 0.0, False, False, {}
        except Exception as e:
            print(f'[MazeEnv] step WS error: {e}')
            return self.state.copy(), -1.0, True, False, {}

        self.current_step += 1

        target_dist = float(self.state[0])
        lidar       = self.state[2:]
        lidar_min   = float(np.min(lidar))
        lidar_front = float(self.state[7])      # index 7 = center-front of 11 rays

        # ── Stagnation tracking ──────────────────────────────────────────────
        progress = self.prev_dist - target_dist  # +ve = moving toward goal
        if abs(progress) < STAG_MIN_MOVE:
            self.stag_counter += 1
        else:
            self.stag_counter = 0

        terminated = False
        truncated  = False
        reward     = 0.0

        # ── PRIORITY 1: Goal reached ────────────────────────────────────────
        if target_dist < GOAL_RADIUS:
            reward     = 200.0
            terminated = True

        # ── PRIORITY 2: Collision ───────────────────────────────────────────
        elif lidar_min < COLLISION_DIST:
            # Scaled penalty: the closer the harder the punishment
            # Wall tunneling is explicitly penalised because any lidar < threshold
            # means the robot mesh has penetrated geometry.
            severity   = 1.0 - (lidar_min / COLLISION_DIST)  # 0..1
            reward     = -60.0 * (1.0 + severity)             # -60 to -120
            terminated = True

        # ── PRIORITY 3: Stagnation (spin/stuck) ────────────────────────────
        elif self.stag_counter >= STAG_WINDOW:
            reward     = -30.0
            terminated = True

        # ── Step reward ─────────────────────────────────────────────────────
        else:
            # 1. Progress toward goal (dominant signal)
            reward += progress * 10.0

            # 2. Continuous directional alignment reward
            angle_to_goal = float(self.state[1])
            reward += 0.15 * math.cos(angle_to_goal)

            # 3. Proximity wall avoidance (quadratic → stronger near walls)
            if lidar_min < 1.0:
                wall_penalty = ((1.0 - lidar_min) ** 2) * 2.0
                reward -= wall_penalty

            # 4. Punish spinning hard: high angular + low linear = spinning
            is_spinning = (linear < 0.15 and abs(angular) > 0.8)
            if is_spinning:
                reward -= 1.0
            elif linear < 0.15:
                # Slow intentional turn — mild penalty only
                reward -= 0.1
            else:
                # Normal forward motion — tiny time cost
                reward -= 0.005

            # 5. Bonus for moving fast when the path is clear ahead
            if lidar_front > 2.0 and linear > 1.5:
                reward += 0.05

        self.prev_dist = target_dist

        if self.current_step >= MAX_STEPS:
            truncated = True

        return self.state.copy(), float(reward), terminated, truncated, {}

    def close(self):
        pass


# ─── Training progress callback ───────────────────────────────────────────────
class MazeProgressCallback(BaseCallback):
    """Logs episode stats to console every N episodes."""
    def __init__(self, log_freq=10, verbose=0):
        super().__init__(verbose)
        self.log_freq       = log_freq
        self.ep_count       = 0
        self.ep_rewards     = []
        self.ep_lengths     = []
        self._ep_rew        = 0.0
        self._ep_len        = 0

    def _on_step(self) -> bool:
        self._ep_rew += self.locals['rewards'][0]
        self._ep_len += 1
        if self.locals['dones'][0]:
            self.ep_count   += 1
            self.ep_rewards.append(self._ep_rew)
            self.ep_lengths.append(self._ep_len)
            self._ep_rew = 0.0
            self._ep_len = 0
            if self.ep_count % self.log_freq == 0:
                avg_r = sum(self.ep_rewards[-self.log_freq:]) / self.log_freq
                avg_l = sum(self.ep_lengths[-self.log_freq:]) / self.log_freq
                ts    = self.num_timesteps
                print(f'  [Ep {self.ep_count:4d} | {ts:7d} steps]  '
                      f'avg_reward={avg_r:7.2f}  avg_len={avg_l:.0f}')
        return True


# ─── Checkpoint discovery ─────────────────────────────────────────────────────
def find_latest_checkpoint(model_dir='models'):
    """Return (path, type) tuple for the highest-step checkpoint.
    Q1 fix: always return a 2-tuple so callers can safely unpack result[0]/result[1].
    """
    # Prefer previously-saved maze checkpoints
    maze_cps = glob.glob(os.path.join(model_dir, 'causalbot_maze_*_steps.zip'))
    if maze_cps:
        def step_key(p):
            try: return int(os.path.basename(p).split('_steps')[0].split('_')[-1])
            except: return 0
        return max(maze_cps, key=step_key), 'maze'   # Q1: was missing ', maze' tuple tag

    # Fall back to room-training checkpoints (curriculum transfer)
    final = os.path.join(model_dir, 'causalbot_ppo_final.zip')
    if os.path.exists(final):
        return final, 'room'

    room_cps = glob.glob(os.path.join(model_dir, 'causalbot_ppo_*_steps.zip'))
    if room_cps:
        def step_key(p):
            try: return int(os.path.basename(p).split('_steps')[0].split('_')[-1])
            except: return 0
        return max(room_cps, key=step_key), 'room'

    return None, None


# ─── Main training entry ──────────────────────────────────────────────────────
def train():
    os.makedirs('models', exist_ok=True)

    print('\n' + '=' * 60)
    print('  CausalBot Maze Training')
    print('  Browser -> http://localhost:5173/?mode=maze')
    print('=' * 60 + '\n')

    # ── Build VecNormalize wrapped env ────────────────────────────────────────
    raw_env  = MazeEnv()
    vec_env  = DummyVecEnv([lambda: raw_env])
    env      = VecNormalize(
        vec_env,
        norm_obs     = True,   # normalise observations (critical for varied scales)
        norm_reward  = True,   # normalise returns (stabilises critic training)
        clip_obs     = 10.0,   # clip normalised obs to ±10σ
        clip_reward  = 10.0,
        gamma        = 0.98,
    )

    # ── Find checkpoint ───────────────────────────────────────────────────────
    result = find_latest_checkpoint()
    checkpoint = result[0] if isinstance(result, tuple) else result
    cp_type    = result[1] if isinstance(result, tuple) else 'maze'

    if checkpoint:
        print(f'[*] Found checkpoint: {checkpoint}')
        if cp_type == 'room':
            print('   (Curriculum transfer from room training -> maze fine-tune)\n')
        else:
            print('   (Resuming maze training)\n')

        custom_objects = {
            'observation_space': raw_env.observation_space,
            'action_space':      raw_env.action_space,
            # Override key PPO hyperparams for maze scale
            'n_steps':           4096,
            'batch_size':        256,
            'n_epochs':          10,
            'gamma':             0.98,
            'gae_lambda':        0.95,
            'ent_coef':          0.03,
            'clip_range':        0.2,
            'vf_coef':           0.5,
        }
        model = PPO.load(checkpoint, env=env, custom_objects=custom_objects)
        # Fine-tune at a lower learning rate — preserve room reflexes
        model.learning_rate = 5e-5
        print(f'  learning_rate = 5e-5  (fine-tune mode)')

    else:
        print('[!] No checkpoint found - training maze from scratch.\n')
        model = PPO(
            'MlpPolicy',
            env,
            verbose          = 0,          # we use our own callback for cleaner output
            tensorboard_log  = './tensorboard_logs/',
            learning_rate    = 2e-4,
            n_steps          = 4096,       # > MAX_STEPS → full episodes per update
            batch_size       = 256,
            n_epochs         = 10,
            gamma            = 0.98,       # see header for discount factor rationale
            gae_lambda       = 0.95,
            ent_coef         = 0.03,
            clip_range       = 0.2,
            vf_coef          = 0.5,
            normalize_advantage = True,
            policy_kwargs    = dict(
                net_arch = [dict(pi=[256, 256], vf=[256, 256])]
            ),
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 10_000,
        save_path   = './models/',
        name_prefix = 'causalbot_maze',
    )
    progress_cb = MazeProgressCallback(log_freq=10)

    # ── Train ─────────────────────────────────────────────────────────────────
    print('Starting maze training...')
    print('Watch the robot in your browser. Press Ctrl+C to stop and save.\n')

    try:
        model.learn(
            total_timesteps    = 500_000,
            callback           = [checkpoint_cb, progress_cb],
            reset_num_timesteps= False,   # keep cumulative step count (curriculum)
        )
    except KeyboardInterrupt:
        print('\n\nTraining interrupted by user.')

    print('\nSaving final model...')
    model.save('models/causalbot_maze_final')
    env.save('models/causalbot_maze_vecnorm.pkl')  # save normalisation stats
    print('[OK] Saved: models/causalbot_maze_final.zip')
    print('[OK] Saved: models/causalbot_maze_vecnorm.pkl\n')
    raw_env.close()


if __name__ == '__main__':
    train()
