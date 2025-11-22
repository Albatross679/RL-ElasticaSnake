"""
Main training script for RL Snake
Run this script to train the PPO agent on the snake environment.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from snake_env import FixedWavelengthXZOnlyContinuumSnakeEnv
from callbacks import RewardCallback, OverwriteCheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import config
import time


def create_environment():
    """Create and configure the environment"""
    env = FixedWavelengthXZOnlyContinuumSnakeEnv(
        fixed_wavelength=config.ENV_CONFIG["fixed_wavelength"],
        obs_keys=config.ENV_CONFIG["obs_keys"],
    )
    
    # Configure environment parameters
    env.period = config.ENV_CONFIG["period"]
    env.ratio_time = config.ENV_CONFIG["ratio_time"]
    env.rut_ratio = config.ENV_CONFIG["rut_ratio"]
    env.reward_weights = config.REWARD_WEIGHTS
    
    return env


def main():
    """Main training function"""
    start_time = time.time()
    print("=" * 70)
    print("Starting RL Training for Continuum Snake")
    print("=" * 70)
    
    # Create directories
    os.makedirs(config.PATHS["log_dir"], exist_ok=True)
    os.makedirs(config.PATHS["model_dir"], exist_ok=True)
    
    # Create environment
    print("\nCreating environment...")
    env = create_environment()
    env.reset()
    
    print(f"Action space shape: {env.action_space.shape}")
    print(f"Fixed wavelength: {env.fixed_wavelength}")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Observation keys: {env.obs_keys}")
    
    # Optional: Check environment (can be slow, comment out for production)
    # print("\nChecking environment...")
    # check_env(env, warn=True)
    
    # Create PPO model
    print("\nCreating PPO model...")
    model = PPO(
        config.MODEL_CONFIG["policy"],
        env,
        gamma=0.99,      # <--- The Discount Factor
        gae_lambda=0.95, # <--- The GAE Parameter
        n_steps=2048,    # <--- The Rollout Buffer Size
        verbose=config.MODEL_CONFIG["verbose"],
        # tensorboard_log=config.PATHS["log_dir"]  # Uncomment for tensorboard logging
    )
    
    # Create callbacks
    reward_callback = RewardCallback(
        print_freq=config.TRAIN_CONFIG["print_freq"],
        step_info_keys=config.TRAIN_CONFIG["step_info_keys"],
        save_dir=config.PATHS["log_dir"],  # Save training data to log directory
        save_freq=config.TRAIN_CONFIG.get("save_freq", 100),  # Save every N episodes
        save_steps=config.TRAIN_CONFIG.get("save_steps", True),  # Whether to save step-level data
    )
    
    # Add checkpoint callback to save model periodically (every 10k timesteps)
    checkpoint_callback = OverwriteCheckpointCallback(
        checkpoint_freq=config.TRAIN_CONFIG.get("checkpoint_freq", 10_000),
        save_path=config.PATHS["model_dir"],
        filename=config.PATHS.get("checkpoint_name", "checkpoint"),
        verbose=1,
        checkpoint_hooks=[reward_callback.checkpoint_hook],
    )
    
    # Combine callbacks
    callback = CallbackList([reward_callback, checkpoint_callback])
    
    # Train model
    print(f"\nStarting training for {config.TRAIN_CONFIG['total_timesteps']} timesteps...")
    print("=" * 70)
    
    try:
        model.learn(
            total_timesteps=config.TRAIN_CONFIG["total_timesteps"],
            callback=callback
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
    
    # Save model
    model_path = os.path.join(config.PATHS["model_dir"], config.PATHS["model_name"])
    model.save(model_path)
    print(f"\nTraining finished! Model saved to {model_path}")
    
    # Close environment
    env.close()
    
    print("\nTraining complete!")
    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()

