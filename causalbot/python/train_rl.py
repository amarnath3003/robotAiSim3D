import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from causalbot_env import CausalBotEnv
import glob

# Create directory to save models
os.makedirs("models", exist_ok=True)

def train():
    env = CausalBotEnv()
    
    # Save a checkpoint every 10000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/",
        name_prefix="causalbot_ppo"
    )
    
    # Find latest checkpoint
    checkpoints = glob.glob(os.path.join("models", "causalbot_ppo_*_steps.zip"))
    latest_checkpoint = None
    if checkpoints:
        def step_count(p):
            try: return int(os.path.basename(p).split('_steps')[0].split('_')[-1])
            except: return 0
        latest_checkpoint = max(checkpoints, key=step_count)

    if latest_checkpoint:
        print(f"\n✅ Found existing checkpoint: {latest_checkpoint}")
        print("Resuming training from where we left off...\n")
        model = PPO.load(latest_checkpoint, env=env)
    else:
        print("\n⚠️ No checkpoint found. Starting training from scratch...\n")
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log="./tensorboard_logs/",
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99
        )
    
    print("Starting Training! Open the browser and watch the robot train.")
    
    # Train for a longer time for better obstacle avoidance
    # (500,000 steps usually takes ~1-2 hours depending on speed)
    model.learn(total_timesteps=500000, callback=checkpoint_callback, reset_num_timesteps=False)
    
    print("Training finished. Saving final model...")
    model.save("models/causalbot_ppo_final")
    
    env.close()

if __name__ == "__main__":
    train()
