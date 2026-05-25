import gymnasium as gym
from causalbot_env import CausalBotEnv
import time
import os
import requests
from dotenv import load_dotenv

# Load env variables (VITE_NVIDIA_API_KEY)
load_dotenv('.env')
API_KEY = os.getenv("VITE_NVIDIA_API_KEY")

def call_llm(lidar_distances):
    if not API_KEY:
        print("ERROR: VITE_NVIDIA_API_KEY is not set in .env")
        return [0.0, 0.0]

    # Format the prompt
    formatted_distances = [round(float(d), 2) for d in lidar_distances]
    
    prompt = f"""You are a robot navigating a 3D environment. You have a 165-degree front-facing lidar with 11 rays (from far left to far right).
Your goal is to explore the environment without colliding into anything.
Max range is 4.5. A distance < 0.5 means you are dangerously close to an obstacle.

Current Lidar Distances: {formatted_distances}
(Index 0 is far left, Index 5 is straight ahead, Index 10 is far right)

Respond ONLY with a valid JSON containing your chosen linear velocity (forward speed) and angular velocity (turning speed).
Limits:
linear: [0.0 to 2.5] (0 is stop, 2.5 is fast forward)
angular: [-2.0 to 2.0] (negative is turn right, positive is turn left in ThreeJS/ROS standard)

JSON Format:
{{"linear": 1.0, "angular": 0.0}}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemma-4-31b-it",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 128
    }

    try:
        url = "https://integrate.api.nvidia.com/v1/chat/completions" # Adjust if different in JS
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        # Parse JSON
        import json
        clean_content = content.replace('```json', '').replace('```', '').strip()
        start = clean_content.find('{')
        end = clean_content.rfind('}')
        if start != -1 and end != -1:
            decision = json.loads(clean_content[start:end+1])
            linear = float(decision.get("linear", 0.0))
            angular = float(decision.get("angular", 0.0))
            
            # Clamp limits
            linear = max(0.0, min(2.5, linear))
            angular = max(-2.0, min(2.0, angular))
            
            return [linear, angular]
            
    except Exception as e:
        print("LLM Error:", e)
        
    return [0.0, 0.0]

def main():
    env = CausalBotEnv()
    
    print("Environment created. Waiting for JS client...")
    obs, info = env.reset()
    print("Initial observation:", obs)
    
    step_count = 0
    while True:
        # Ask LLM for action
        action = call_llm(obs)
        
        print(f"Step {step_count} | Lidar: {[round(x, 1) for x in obs]} | Action: Linear={action[0]}, Angular={action[1]}")
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Collision/End after {step_count} steps. Resetting...")
            obs, info = env.reset()
            step_count = 0
        else:
            step_count += 1

if __name__ == "__main__":
    main()
