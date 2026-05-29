"""
llm_agent.py — Drive the robot using an LLM as the decision-maker.

O2/Q3 fix: The observation is 13-dim [target_dist, target_angle, lidar×11].
Previously the full obs was passed as "lidar distances", giving the LLM 13
values when it expected 11 and including distance/angle as fake lidar rays.
Now:
  - obs[0] → target distance  (given as goal info in the prompt)
  - obs[1] → target angle     (given as goal info in the prompt)
  - obs[2:13] → 11 lidar rays (passed to LLM for obstacle reasoning)

The returned action is also padded to 4-dim [lin, ang, 0, 0] to match the
4-dim action space expected by CausalBotEnv.step().
"""

import json
import os
import time

import requests
from dotenv import load_dotenv

from causalbot_env import CausalBotEnv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
API_KEY = os.getenv('VITE_NVIDIA_API_KEY')


def call_llm(obs) -> list:
    """
    Accepts the full 13-dim observation and returns a 4-dim action
    [linear, angular, 0.0, 0.0].

    obs[0] = distance to goal (m)
    obs[1] = relative angle to goal (rad, -π..π)
    obs[2:13] = 11 lidar rays, 165° FOV, 0 = left edge, 10 = right edge
    """
    if not API_KEY:
        print('ERROR: VITE_NVIDIA_API_KEY not set in .env')
        return [0.0, 0.0, 0.0, 0.0]

    target_dist  = round(float(obs[0]), 2)
    target_angle = round(float(obs[1]), 3)   # radians
    lidar_rays   = [round(float(v), 2) for v in obs[2:13]]

    prompt = f"""You are a robot navigating a 3D room to reach a goal.

Sensor readings:
- Goal distance: {target_dist} m  (0 = at goal, positive = far away)
- Goal angle: {target_angle} rad  (0 = straight ahead, negative = right, positive = left)
- Lidar (11 rays, 165° FOV, left→right): {lidar_rays}
  Index 0 = far left, 5 = straight ahead, 10 = far right.
  Max range = 5.0 m. Value < 0.3 m = DANGER (imminent collision).

Strategy:
1. If goal is close and mostly ahead, drive forward.
2. Turn toward the goal (match sign of goal_angle with angular sign).
3. Slow down or stop if an obstacle is within 0.5 m ahead (indices 3-7).

Respond ONLY with valid JSON:
{{"linear": <0.0–2.5>, "angular": <-2.0–2.0>}}"""

    headers = {
        'Authorization': f'Bearer {API_KEY}',
        'Content-Type': 'application/json',
    }
    payload = {
        'model': 'google/gemma-4-31b-it',
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.1,
        'max_tokens': 64,
    }

    try:
        url = 'https://integrate.api.nvidia.com/v1/chat/completions'
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        content = response.json()['choices'][0]['message']['content']
        clean   = content.replace('```json', '').replace('```', '').strip()
        s, e    = clean.find('{'), clean.rfind('}')
        if s != -1 and e != -1:
            dec     = json.loads(clean[s:e + 1])
            linear  = float(max(0.0, min(2.5,  dec.get('linear',  0.0))))
            angular = float(max(-2.0, min(2.0, dec.get('angular', 0.0))))
            return [linear, angular, 0.0, 0.0]   # pad to 4-dim action space

    except Exception as e:
        print(f'LLM Error: {e}')

    return [0.0, 0.0, 0.0, 0.0]


def main():
    env = CausalBotEnv()

    print('Environment created. Waiting for JS client...')
    obs, info = env.reset()
    print(f'Initial obs: dist={obs[0]:.2f}m  angle={obs[1]:.3f}rad  lidar={[round(x,1) for x in obs[2:]]}')

    step_count = 0
    while True:
        action = call_llm(obs)

        print(f'Step {step_count:4d} | dist={obs[0]:.2f}m angle={obs[1]:.2f}rad '
              f'| lin={action[0]:.2f} ang={action[1]:.2f}')

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f'Episode end after {step_count} steps. Resetting...')
            obs, info = env.reset()
            step_count = 0
        else:
            step_count += 1


if __name__ == '__main__':
    main()
