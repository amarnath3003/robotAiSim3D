"""
run_agent.py — Full RL agent loop.

Flow:
  1. Python starts WebSocket server (via CausalBotEnv)
  2. Browser connects → rl.js switches to RL mode
  3. User types a prompt in the browser → Python receives it
  4. LLM parses prompt → navigation goal (x, z)
  5. Trained PPO policy drives the robot step-by-step
  6. On goal-reach or collision → reset and wait for next prompt

Run:
  cd causalbot/python
  python run_agent.py

Browser:
  Open http://localhost:5173
  Python connects automatically → browser switches to RL mode
  Type e.g. "go to the ball" and press Enter
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

# ─── World objects — keep in sync with src/state.js ─────────────────────────
WORLD_OBJECTS = {
    'ball':  {'x':  2.2, 'z': -2.0, 'aliases': ['ball', 'red ball', 'sphere']},
    'box':   {'x': -1.9, 'z':  1.1, 'aliases': ['box', 'crate', 'yellow box', 'cube']},
    'glass': {'x':  0.0, 'z':  3.0, 'aliases': ['glass', 'cup', 'cylinder', 'bottle']},
}
ROOM_BOUNDS = {'minX': -2.5, 'maxX': 2.5, 'minZ': -2.5, 'maxZ': 2.5}


# ─── LLM goal parser ─────────────────────────────────────────────────────────

def parse_goal(prompt_text: str) -> tuple[float, float]:
    """
    Use the LLM to map a natural language prompt to (x, z) room coordinates.
    Runs in its own thread so it never blocks the step loop.
    Falls back to origin on any failure.
    """
    if not API_KEY:
        print('[LLM] VITE_NVIDIA_API_KEY not set — defaulting to origin.')
        return 0.0, 0.0

    obj_lines = '\n'.join(
        f"  - {name} (also: {', '.join(info['aliases'])}): "
        f"position x={info['x']}, z={info['z']}"
        for name, info in WORLD_OBJECTS.items()
    )

    prompt = f"""You are a spatial planner for a robot arm simulation.
Room is bounded x=[{ROOM_BOUNDS['minX']}, {ROOM_BOUNDS['maxX']}], z=[{ROOM_BOUNDS['minZ']}, {ROOM_BOUNDS['maxZ']}].

Objects in the room:
{obj_lines}

User instruction: "{prompt_text}"

Output the (x, z) world coordinates the robot should navigate TO.
If the instruction refers to a known object, use that object's coordinates.
If it refers to a room area (corner, centre, etc.), estimate coordinates.

Respond with ONLY a JSON object, no explanation:
{{"x": <number>, "z": <number>}}"""

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type':  'application/json',
    }
    body = {
        'model':       MODEL,
        'messages':    [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens':  48,
    }

    print(f'[LLM] Parsing: "{prompt_text}"')
    try:
        r = requests.post(API_URL, headers=headers, json=body, timeout=15)
        r.raise_for_status()
        content = r.json()['choices'][0]['message']['content']
        # Strip markdown fences if present
        clean = content.replace('```json', '').replace('```', '').strip()
        s, e  = clean.find('{'), clean.rfind('}')
        if s != -1 and e != -1:
            dec = json.loads(clean[s:e+1])
            x = float(np.clip(dec.get('x', 0.0), ROOM_BOUNDS['minX'], ROOM_BOUNDS['maxX']))
            z = float(np.clip(dec.get('z', 0.0), ROOM_BOUNDS['minZ'], ROOM_BOUNDS['maxZ']))
            print(f'[LLM] Goal → x={x:.2f}, z={z:.2f}')
            return x, z
    except Exception as ex:
        print(f'[LLM] Error: {ex}')

    return 0.0, 0.0


# ─── Agent ───────────────────────────────────────────────────────────────────

class Agent:
    """
    Wraps the PPO model (or random policy) and drives the step loop.
    LLM parsing runs in a background thread so the step loop never blocks.
    """

    def __init__(self, env):
        self.env       = env
        self.model     = self._load_model()
        self._mode     = 'IDLE'   # 'IDLE' | 'EXECUTING' | 'RESETTING'
        self._pending_goal: tuple[float, float] | None = None
        self._goal_lock = threading.Lock()

    def _load_model(self):
        model_path = os.path.join(os.path.dirname(__file__), 'models', 'causalbot_ppo_final.zip')
        try:
            from stable_baselines3 import PPO
            if os.path.exists(model_path):
                print(f'[Agent] Loading model: {model_path}')
                m = PPO.load(model_path)
                print('[Agent] Model loaded ✓')
                return m
            else:
                print(f'[Agent] No model at {model_path}')
                print('        → Run python/train_rl.py first for a trained policy.')
                print('        → Using RANDOM policy until then.\n')
        except ImportError:
            print('[Agent] stable-baselines3 not installed → random policy.')
        return None

    def _act(self, obs):
        if self.model is not None:
            action, _ = self.model.predict(obs, deterministic=True)
            return action
        return self.env.action_space.sample()

    # Called from background thread when LLM finishes
    def _apply_goal(self, x: float, z: float):
        with self._goal_lock:
            self._pending_goal = (x, z)

    def _parse_and_apply(self, text: str):
        """Background thread: call LLM, then queue the result."""
        x, z = parse_goal(text)
        self._apply_goal(x, z)

    def run(self):
        print('[Agent] Waiting for browser connection...')
        obs, _ = self.env.reset()
        print('[Agent] Ready. Type a command in the browser (RL mode).\n')

        while True:
            # ── Check for new prompt (non-blocking) ───────────────────────
            prompt = self.env.get_latest_prompt()
            if prompt is not None:
                print(f'[Agent] Prompt: "{prompt}"')
                # Parse asynchronously so step loop keeps running
                t = threading.Thread(target=self._parse_and_apply, args=(prompt,), daemon=True)
                t.start()

            # ── Apply pending goal if LLM finished ─────────────────────────
            with self._goal_lock:
                goal = self._pending_goal
                self._pending_goal = None

            if goal is not None:
                gx, gz = goal
                self.env.set_goal(gx, gz)
                print(f'[Agent] New goal set: ({gx:.2f}, {gz:.2f}) — executing')
                self._mode = 'EXECUTING'
                # Reset so robot starts from a clean position for the new goal
                obs, _ = self.env.reset()

            # ── Step ───────────────────────────────────────────────────────
            if self._mode == 'EXECUTING':
                action = self._act(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)

                if info.get('timeout'):
                    # Browser tab backgrounded — keep trying, don't reset
                    continue

                if terminated:
                    if reward >= 9.0:
                        print('[Agent] ✓ Goal reached!')
                    else:
                        print('[Agent] ✗ Collision detected — resetting.')
                    obs, _ = self.env.reset()
                    self._mode = 'IDLE'

                elif truncated:
                    print('[Agent] ✗ Time limit — resetting.')
                    obs, _ = self.env.reset()
                    self._mode = 'IDLE'

            else:
                # IDLE: send zero velocity to keep the WebSocket alive
                # This also keeps _state_event cycling so new prompts are noticed
                obs, _, _, _, _ = self.env.step([0.0, 0.0])
                time.sleep(0.016)   # ~60 fps idle


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    # Add python/ dir to path so causalbot_env imports cleanly
    sys.path.insert(0, os.path.dirname(__file__))
    from causalbot_env import CausalBotEnv

    env   = CausalBotEnv()
    agent = Agent(env)
    agent.run()


if __name__ == '__main__':
    main()