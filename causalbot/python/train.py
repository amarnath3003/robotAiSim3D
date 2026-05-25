import gymnasium as gym
from causalbot_env import CausalBotEnv
import time

def main():
    env = CausalBotEnv()
    
    print("Environment created. Waiting for reset...")
    obs, info = env.reset()
    print("Initial observation:", obs)
    
    for i in range(1000):
        # Random action
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            print(f"Step {i} | Reward: {reward:.3f} | Dist to ball: {((obs[0]-obs[10])**2 + (obs[2]-obs[12])**2)**0.5:.2f}")
            
        if terminated or truncated:
            print("Episode finished. Resetting...")
            obs, info = env.reset()
            
        time.sleep(0.05) # Throttle a bit so we can watch in the browser
        
    env.close()

if __name__ == "__main__":
    main()
