"""
run_agent.py — LLM parses a natural language prompt into a navigation goal,
then the trained PPO policy drives the robot to it.

Fixes over previous version:
  1. Object positions read from a shared config dict (single source of truth)
     instead of being hardcoded twice in the prompt string.
  2. Graceful fallback when no trained model exists — tells user clearly.
  3. IDLE loop sends zero-velocity steps so Python keeps the WS alive.
  4. Goal is printed and shown in status when received.
  5. WebSocket reconnect logic when browser refreshes.
"""

import os
import json
import time
import asyncio
import threading
import requests
import numpy as np
from dotenv import load_dotenv

load_dotenv('../.env')
API_KEY = os.getenv('VITE_NVIDIA_API_KEY')
MODEL   = os.getenv('VITE_NVIDIA_MODEL', 'google/gemma-4-31b-it')
API_URL = 'https://integrate.api.nvidia.com/v1/chat/completions'

# ─── World object registry ────────────────────────────────────────────────────
# Keep in sync with causalbot/src/state.js world.objects
WORLD_OBJECTS = {
    'ball':  {'x':  2.2, 'z': -2.0, 'aliases': ['red ball', 'ball']},
    'box':   {'x': -1.9, 'z':  1.1, 'aliases': ['yellow box', 'box', 'crate']},
    'glass': {'x':  0.0, 'z':  3.0, 'aliases': ['glass', 'cylinder', 'cup']},
}

ROOM_BOUNDS = {'minX': -2.5, 'maxX': 2.5, 'minZ': -2.5, 'maxZ': 2.5}

# ─── LLM goal parser ──────────────────────────────────────────────────────────

def get_goal_from_llm(prompt_text: str) -> list[float]:
    """Ask the LLM to map a natural language prompt to (x, z) coordinates."""
    if not API_KEY:
        print('[LLM] ERROR: VITE_NVIDIA_API_KEY not set. Defaulting to origin.')
        return [0.0, 0.0]

    # Build the object list dynamically from WORLD_OBJECTS
    obj_lines = '\n'.join(
        f"  - {name} ({', '.join(info['aliases'])}): x={info['x']}, z={info['z']}"
        for name, info in WORLD_OBJECTS.items()
    )

    prompt = f"""You are a spatial planner for a robot in a 3D room.
Room bounds: x [{ROOM_BOUNDS['minX']} to {ROOM_BOUNDS['maxX']}], z [{ROOM_BOUNDS['minZ']} to {ROOM_BOUNDS['maxZ']}].

Known objects:
{obj_lines}

User instruction: "{prompt_text}"

Output the (x, z) coordinates the robot should navigate to.
Respond ONLY with valid JSON containing 'x' and 'z'. No extra text.

Example: {{"x": 2.2, "z": -2.0}}"""

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type':  'application/json',
    }
    payload = {
        'model':       MODEL,
        'messages':    [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens':  64,
    }

    print(f"[LLM] Parsing: '{prompt_text}'...")
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        clean   = content.replace('```json', '').replace('```', '').strip()
        start, end = clean.find('{'), clean.rfind('}')
        if start != -1 and end != -1:
            dec = json.loads(clean[start:end+1])
            x = float(np.clip(dec.get('x', 0.0), ROOM_BOUNDS['minX'], ROOM_BOUNDS['maxX']))
            z = float(np.clip(dec.get('z', 0.0), ROOM_BOUNDS['minZ'], ROOM_BOUNDS['maxZ']))
            print(f'[LLM] Goal decoded: x={x:.2f}, z={z:.2f}')
            return [x, z]
    except Exception as e:
        print(f'[LLM] Error: {e}')

    return [0.0, 0.0]

# ─── Main agent loop ──────────────────────────────────────────────────────────

def run():
    from causalbot_env import CausalBotEnv

    model_path = 'models/causalbot_ppo_final.zip'
    model = None

    try:
        from stable_baselines3 import PPO
        if os.path.exists(model_path):
            print(f'[Agent] Loading model from {model_path}...')
            model = PPO.load(model_path)
            print('[Agent] Model loaded.')
        else:
            print(f'[Agent] WARNING: No model at {model_path}.')
            print('         Run python/train_rl.py first to train.')
            print('         Running with random policy for now.\n')
    except ImportError:
        print('[Agent] stable-baselines3 not installed — using random policy.')

    env = CausalBotEnv()
    print('[Agent] Waiting for browser to connect on ws://localhost:8765 ...')

    obs, _ = env.reset()
    print('[Agent] Connected. Send a prompt from the browser input field.')
    print('        (Make sure the browser is in RL mode — press key 3)\n')

    mode = 'IDLE'

    while True:
        # ── Check for new user prompt ──────────────────────────────────────
        if env.latest_prompt is not None:
            prompt_text       = env.latest_prompt
            env.latest_prompt = None

            goal              = get_goal_from_llm(prompt_text)
            env.target_goal   = np.array(goal, dtype=np.float32)
            mode              = 'EXECUTING'
            print(f'[Agent] Driving to goal {goal} for prompt: "{prompt_text}"')

        # ── Execute or idle ────────────────────────────────────────────────
        if mode == 'EXECUTING':
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)

            if terminated:
                if reward > 0:
                    print('[Agent] Goal reached!')
                else:
                    print('[Agent] Collision — resetting.')
                obs, _ = env.reset()
                mode   = 'IDLE'
            elif truncated:
                print('[Agent] Time limit — resetting.')
                obs, _ = env.reset()
                mode   = 'IDLE'
        else:
            # Keep WS alive; send zero velocity
            obs, _, _, _, _ = env.step([0.0, 0.0])
            time.sleep(0.05)


if __name__ == '__main__':
    run()