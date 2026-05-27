"""
run_agent.py — Full RL agent loop.

Flow:
  1. Python starts WebSocket server (via CausalBotEnv)
  2. Browser connects → rl.js switches to RL mode
  3. Agent immediately starts in train_mode (episodes chain automatically)
  4. User can type a prompt to override the goal mid-episode
  5. LLM parses prompt → navigation goal (x, z) in background thread
  6. Trained PPO policy (or random) drives the robot step-by-step
  7. Death/success → episode ends → auto-reset (train_mode) or wait for prompt

Run:
  cd causalbot/python
  python run_agent.py
"""

import os
import sys
import json
import time
import threading
import requests
import numpy as np
from dotenv import load_dotenv

# Load .env from the causalbot directory (one level up from python/)
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
API_KEY = os.getenv('VITE_NVIDIA_API_KEY')
MODEL   = os.getenv('VITE_NVIDIA_MODEL', 'google/gemma-4-31b-it')
API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'


# ─── World objects + named locations ──────────────────────────────────────────
WORLD_OBJECTS = {
    'ball':         {'x':  2.2,  'z': -2.0, 'aliases': ['ball', 'red ball', 'sphere', 'orb']},
    'box':          {'x': -1.9,  'z':  1.1, 'aliases': ['box', 'crate', 'yellow box', 'cube', 'block']},
    'glass':        {'x':  0.0,  'z':  3.0, 'aliases': ['glass', 'cup', 'cylinder', 'bottle']},
    'center':       {'x':  0.0,  'z':  0.0, 'aliases': ['center', 'middle', 'origin']},
    'front_center': {'x':  0.0,  'z':  2.0, 'aliases': ['front', 'forward']},
    'back_center':  {'x':  0.0,  'z': -2.0, 'aliases': ['back', 'behind', 'rear']},
    'left_wall':    {'x': -2.2,  'z':  0.0, 'aliases': ['left', 'left wall']},
    'right_wall':   {'x':  2.2,  'z':  0.0, 'aliases': ['right', 'right wall']},
    'front_left':   {'x': -2.0,  'z':  2.0, 'aliases': ['front left', 'left corner']},
    'front_right':  {'x':  2.0,  'z':  2.0, 'aliases': ['front right', 'right corner']},
    'back_left':    {'x': -2.0,  'z': -2.0, 'aliases': ['back left', 'bottom left']},
    'back_right':   {'x':  2.0,  'z': -2.0, 'aliases': ['back right', 'bottom right']},
}
ROOM_BOUNDS = {'minX': -2.5, 'maxX': 2.5, 'minZ': -2.5, 'maxZ': 2.5}


# ─── LLM goal parser ──────────────────────────────────────────────────────────

def parse_goal(prompt_text: str) -> tuple[float, float]:
    """
    Map any natural language navigation command to (x, z) world coordinates.
    Runs in a background thread — never blocks the step loop.
    Returns (0, 0) on any failure.
    """
    if not API_KEY:
        print('[LLM] No API key — returning origin.')
        return 0.0, 0.0

    obj_lines = '\n'.join(
        f"  {name}: x={info['x']}, z={info['z']}  ({', '.join(info['aliases'])})"
        for name, info in WORLD_OBJECTS.items()
    )

    prompt = f"""You are a spatial navigator for a 3D robot.
Room: x=[{ROOM_BOUNDS['minX']}, {ROOM_BOUNDS['maxX']}], z=[{ROOM_BOUNDS['minZ']}, {ROOM_BOUNDS['maxZ']}].

Objects/locations:
{obj_lines}

Command: "{prompt_text}"

Return the (x, z) the robot should navigate to. Use object coords if named. Estimate otherwise.
Respond ONLY with JSON: {{"x": <number>, "z": <number>}}"""

    try:
        r = requests.post(
            API_URL,
            headers={'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'},
            json={'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}],
                  'temperature': 0.1, 'max_tokens': 48},
            timeout=15,
        )
        r.raise_for_status()
        content = r.json()['choices'][0]['message']['content']
        clean   = content.replace('```json', '').replace('```', '').strip()
        s, e    = clean.find('{'), clean.rfind('}')
        if s != -1 and e != -1:
            dec = json.loads(clean[s:e + 1])
            x   = float(np.clip(dec.get('x', 0.0), ROOM_BOUNDS['minX'], ROOM_BOUNDS['maxX']))
            z   = float(np.clip(dec.get('z', 0.0), ROOM_BOUNDS['minZ'], ROOM_BOUNDS['maxZ']))
            print(f'[LLM] "{prompt_text}" → ({x:.2f}, {z:.2f})')
            return x, z
    except Exception as ex:
        print(f'[LLM] Error: {ex}')

    return 0.0, 0.0


# ─── Agent ────────────────────────────────────────────────────────────────────

class Agent:
    """
    Wraps the PPO model (or random policy fallback) and drives the step loop.

    Improvements:
    - B2 fix: does NOT reset when a new goal arrives mid-episode; only updates target.
    - I4 fix: _parsing_lock prevents stacking multiple LLM calls.
    - train_mode: auto-resets on every termination for continuous RL training.
    """

    def __init__(self, env):
        self.env            = env
        self.model          = self._load_model()
        self._executing     = False
        self._pending_goal: tuple[float, float] | None = None
        self._goal_lock     = threading.Lock()
        self._parsing       = False   # I4: LLM dedup guard
        self._parsing_lock  = threading.Lock()

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'causalbot_ppo_final.zip')
        try:
            from stable_baselines3 import PPO
            if os.path.exists(model_path):
                print(f'[Agent] Loading PPO model: {model_path}')
                m = PPO.load(model_path)
                print('[Agent] PPO model loaded ✓')
                return m
            else:
                print(f'[Agent] No model at {model_path}')
                print('        → Run python/train_rl.py for a trained policy.')
                print('        → Using RANDOM policy for now.\n')
        except ImportError:
            print('[Agent] stable-baselines3 not installed → random policy.')
        return None

    def _act(self, obs):
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        return self.env.action_space.sample()

    # Called from background thread when LLM finishes
    def _on_goal_parsed(self, x: float, z: float):
        with self._goal_lock:
            self._pending_goal = (x, z)
        with self._parsing_lock:
            self._parsing = False

    def _parse_and_apply(self, text: str):
        """Background LLM call — queues result via _on_goal_parsed."""
        x, z = parse_goal(text)
        self._on_goal_parsed(x, z)

    def run(self):
        print('[Agent] Waiting for browser...')
        obs, _ = self.env.reset()
        self._executing = True   # train_mode: start executing immediately
        print(f'[Agent] Ready — train_mode={self.env.train_mode}\n')

        while True:
            # ── 1. Check for new user prompt ─────────────────────────────────
            prompt = self.env.get_latest_prompt()
            if prompt is not None:
                print(f'[Agent] Prompt: "{prompt}"')
                with self._parsing_lock:
                    if not self._parsing:
                        self._parsing = True
                        t = threading.Thread(
                            target=self._parse_and_apply,
                            args=(prompt,),
                            daemon=True,
                        )
                        t.start()
                    else:
                        print('[Agent] LLM already parsing — ignoring duplicate prompt')

            # ── 2. Apply pending goal (does NOT reset — B2 fix) ──────────────
            with self._goal_lock:
                goal = self._pending_goal
                self._pending_goal = None

            if goal is not None:
                gx, gz = goal
                self.env.set_goal(gx, gz)
                print(f'[Agent] Goal updated mid-episode → ({gx:.2f}, {gz:.2f})')
                # Only flip to executing (don't reset — let current episode continue)
                self._executing = True

            # ── 3. Step ───────────────────────────────────────────────────────
            with self.env._lock:
                do_reset = self.env.force_reset
                self.env.force_reset = False

            if do_reset:
                print('[Agent] Manual reset triggered from dashboard')
                obs, _ = self.env.reset()
                self._executing = self.env.train_mode
                continue

            if self._executing:
                action = self._act(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)

                if info.get('timeout') or info.get('disconnected'):
                    # Browser dropped or tab backgrounded — spin without resetting
                    time.sleep(0.05)
                    continue

                if terminated or truncated:
                    if reward >= 19.0:
                        print('[Agent] ✓  Goal reached!')
                    elif terminated:
                        print('[Agent] ☠  Collision — resetting.')
                    else:
                        print('[Agent] ⏱  Time limit — resetting.')

                    if self.env.train_mode:
                        # Continuous training: auto-reset immediately
                        obs, _ = self.env.reset()
                    else:
                        # Prompt-gated: wait idle until user sends next command
                        self._executing = False

            else:
                # IDLE: heartbeat step to keep WS alive + observe scene
                obs, _, _, _, _ = self.env.step([0.0, 0.0, 0.0, 0.0])
                time.sleep(0.05)   # ~20 Hz idle


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, os.path.dirname(__file__))
    from causalbot_env import CausalBotEnv

    env   = CausalBotEnv()
    agent = Agent(env)
    agent.run()


if __name__ == '__main__':
    main()