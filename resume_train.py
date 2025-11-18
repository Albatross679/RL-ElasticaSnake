"""
Script to resume training from an existing PPO model
Run this script to continue training a previously saved PPO agent.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from snake_env import FixedWavelengthContinuumSnakeEnv
from callbacks import RewardCallback
import config
import time


def create_environment():
    """Create and configure the environment"""
    env = FixedWavelengthContinuumSnakeEnv(
        fixed_wavelength=config.ENV_CONFIG["fixed_wavelength"],
        obs_keys=config.ENV_CONFIG["obs_keys"],
    )
    
    # Configure environment parameters
    env.period = config.ENV_CONFIG["period"]
    env.ratio_time = config.ENV_CONFIG["ratio_time"]
    env.rut_ratio = config.ENV_CONFIG["rut_ratio"]
    
    return env


def main():
    start = time.time()
    """Main training function"""
    print("=" * 70)
    print("Resuming RL Training for Continuum Snake")
    print("=" * 70)
    
    # Create directories
    os.makedirs(config.PATHS["log_dir"], exist_ok=True)
    os.makedirs(config.PATHS["model_dir"], exist_ok=True)
    
    # Determine model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Use default model path from config
        model_path = os.path.join(config.PATHS["model_dir"], config.PATHS["model_name"])
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"\nError: Model file not found at {model_path}")
        print("Please provide a valid model path as an argument, or ensure the default model exists.")
        print(f"Usage: python resume_train.py [model_path]")
        sys.exit(1)
    
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
    
    # Load existing PPO model
    print(f"\nLoading PPO model from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    # Create callback
    callback = RewardCallback(
        print_freq=config.TRAIN_CONFIG["print_freq"],
        step_print_interval=config.TRAIN_CONFIG["step_print_interval"]
    )
    
    # Train model
    print(f"\nResuming training for {config.TRAIN_CONFIG['total_timesteps']} timesteps...")
    print("=" * 70)
    
    try:
        model.learn(
            total_timesteps=config.TRAIN_CONFIG["total_timesteps"],
            callback=callback,
            reset_num_timesteps=False  # Don't reset timestep counter when resuming
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving model...")
    
    # Save model
    save_path = os.path.join(config.PATHS["model_dir"], config.PATHS["model_name"])
    model.save(save_path)
    print(f"\nTraining finished! Model saved to {save_path}")
    
    # Close environment
    env.close()
    
    print("\nTraining complete!")
    end = time.time()
    elapsed = end - start
    print(f"Total training time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()

