"""
train_rl.py — Room-mode PPO training.

Fixes applied:
  C4: VecNormalize wrapper added (obs dims span very different scales:
      dist 0-10, angle -π..π, lidar 0-5; without normalisation the critic
      is dominated by the largest-magnitude dimensions).
  C5: gamma=0.97 (was 0.99).  With max_steps=500 the goal reward R=20
      discounted back 500 steps is 20 * 0.97^500 ≈ 0.  0.97 gives
      20 * 0.97^500 ≈ 0.0015 — still tiny, but far less catastrophic than
      0.99^500 ≈ 0.007 * 20 = 0.14 effective reward.  Correct fix is a
      short max_steps AND high gamma; 0.97 with 500 steps is a workable
      middle ground while we tune curriculum.
  Q2: Model/log paths are now absolute (relative to this script's location)
      so training works regardless of the CWD.

Usage:
  cd causalbot/python
  python train_rl.py
"""

import os
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from causalbot_env import CausalBotEnv

# Q2: always resolve paths relative to this file so `python path/to/train_rl.py` works
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(SCRIPT_DIR, 'models')
TB_LOG_DIR  = os.path.join(SCRIPT_DIR, 'tensorboard_logs')
VECNORM_PATH = os.path.join(MODELS_DIR, 'causalbot_room_vecnorm.pkl')


def _step_count(path: str) -> int:
    try:
        return int(os.path.basename(path).split('_steps')[0].split('_')[-1])
    except Exception:
        return 0


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # C4: wrap in DummyVecEnv + VecNormalize for stable multi-scale obs training
    raw_env = CausalBotEnv()
    vec_env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(
        vec_env,
        norm_obs    = True,
        norm_reward = True,
        clip_obs    = 10.0,
        clip_reward = 10.0,
        gamma       = 0.97,   # C5: matches PPO gamma below
    )

    checkpoint_callback = CheckpointCallback(
        save_freq  = 10_000,
        save_path  = MODELS_DIR,
        name_prefix= 'causalbot_ppo',
    )

    # Find latest checkpoint
    checkpoints = glob.glob(os.path.join(MODELS_DIR, 'causalbot_ppo_*_steps.zip'))
    latest_checkpoint = max(checkpoints, key=_step_count) if checkpoints else None

    if latest_checkpoint:
        print(f'\n✅  Resuming from: {latest_checkpoint}')
        # Restore VecNormalize running stats BEFORE loading the model so the
        # model trains against the restored normalization, not a fresh one
        if os.path.exists(VECNORM_PATH):
            env = VecNormalize.load(VECNORM_PATH, vec_env)
            env.training = True
            env.norm_reward = True
            print(f'    VecNormalize stats loaded from {VECNORM_PATH}')
        model = PPO.load(latest_checkpoint, env=env)
    else:
        print('\n⚠️  No checkpoint found. Starting from scratch...\n')
        model = PPO(
            'MlpPolicy',
            env,
            verbose          = 1,
            tensorboard_log  = TB_LOG_DIR,
            learning_rate    = 3e-4,
            n_steps          = 2048,
            batch_size       = 64,
            n_epochs         = 10,
            gamma            = 0.97,   # C5: was 0.99 — goal reward barely visible at step 500
            gae_lambda       = 0.95,
        )

    print('Starting room training. Open the browser and watch the robot train.')
    model.learn(
        total_timesteps    = 500_000,
        callback           = checkpoint_callback,
        reset_num_timesteps= False,
    )

    print('\nTraining finished. Saving final model...')
    model.save(os.path.join(MODELS_DIR, 'causalbot_ppo_final'))
    env.save(VECNORM_PATH)
    print(f'[OK] Saved: {MODELS_DIR}/causalbot_ppo_final.zip')
    print(f'[OK] Saved: {VECNORM_PATH}')

    raw_env.close()


if __name__ == '__main__':
    train()
