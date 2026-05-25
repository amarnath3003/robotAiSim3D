import gymnasium as gym
from stable_baselines3 import PPO
from causalbot_env import CausalBotEnv
import time
import os
import requests
import json
from dotenv import load_dotenv
import numpy as np

# Load env variables (VITE_NVIDIA_API_KEY)
load_dotenv('.env')
API_KEY = os.getenv("VITE_NVIDIA_API_KEY")

def get_goal_from_llm(prompt_text):
    if not API_KEY:
        print("ERROR: VITE_NVIDIA_API_KEY is not set in .env")
        return [0.0, 0.0]

    prompt = f"""You are a high-level spatial planner for a robot in a 3D room. 
The room bounds are x: [-2.5 to 2.5] and z: [-2.5 to 2.5].
There are known objects:
- Red Ball: [2.2, -2.0]
- Yellow Box: [-1.0, -1.0]
- Glass Cylinder: [0.0, 2.0]

The user says: "{prompt_text}"

Based on the prompt, output the exact (x, z) coordinates the robot should navigate to.
Respond ONLY with a valid JSON containing 'x' and 'z'.

JSON Format:
{{"x": 2.2, "z": -2.0}}
"""

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemma-4-31b-it",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 128
    }

    print(f"[LLM] Parsing prompt: '{prompt_text}'...")
    try:
        url = "https://integrate.api.nvidia.com/v1/chat/completions"
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        
        clean_content = content.replace('```json', '').replace('```', '').strip()
        start = clean_content.find('{')
        end = clean_content.rfind('}')
        if start != -1 and end != -1:
            decision = json.loads(clean_content[start:end+1])
            x = float(decision.get("x", 0.0))
            z = float(decision.get("z", 0.0))
            print(f"[LLM] Goal Decoded: x={x}, z={z}")
            return [x, z]
            
    except Exception as e:
        print("[LLM Error]:", e)
        
    return [0.0, 0.0]

def run():
    env = CausalBotEnv()
    
    model_path = "models/causalbot_ppo_final.zip"
    
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model = PPO.load(model_path)
    else:
        print(f"WARNING: No trained model found at {model_path}.")
        print("Run `python train_rl.py` first to train the network.")
        print("Running with an untrained random policy for now...")
        model = None

    print("Environment created. Waiting for JS client...")
    obs, info = env.reset()
    print("Agent is ready. Waiting for user prompts in the browser...")
    
    current_mode = "IDLE" # IDLE or EXECUTING
    
    while True:
        # Check if the user submitted a new prompt
        if env.latest_prompt is not None:
            prompt_text = env.latest_prompt
            env.latest_prompt = None # Clear it
            
            # Use LLM to decode prompt into goal coordinates
            goal = get_goal_from_llm(prompt_text)
            env.target_goal = np.array(goal)
            
            current_mode = "EXECUTING"
            print(f"--- Execution Started: Driving to {goal} ---")
            
        if current_mode == "EXECUTING":
            # 1. Ask local trained RL model for the next action based on Lidar
            if model is not None:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # Fallback if not trained
                
            # 2. Step the environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Check if goal reached or collision
            if terminated:
                if reward > 0:
                    print("Goal reached successfully!")
                else:
                    print("Collision detected!")
                current_mode = "IDLE"
        else:
            # IDLE: send zero velocity to keep the simulation ticking
            obs, _, _, _, _ = env.step([0.0, 0.0])

if __name__ == "__main__":
    run()
