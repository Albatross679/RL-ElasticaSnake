"""
Script to resume training from an existing PPO model
Run this script to continue training a previously saved PPO agent.
"""

import os
import sys
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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
    env.max_episode_length = config.ENV_CONFIG["max_episode_length"]
    env.reward_weights = config.REWARD_WEIGHTS
    
    return env


def main():
    """Main training function"""
    start_time = time.time()
    print("=" * 70)
    print("Resuming RL Training for Continuum Snake")
    print("=" * 70)
    
    # Create directories
    os.makedirs(config.PATHS["log_dir"], exist_ok=True)
    os.makedirs(config.PATHS["model_dir"], exist_ok=True)
    
    # Determine model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        # If user provided path doesn't have .zip, check with .zip extension
        if not os.path.exists(model_path) and not model_path.endswith('.zip'):
            model_path_zip = model_path + '.zip'
            if os.path.exists(model_path_zip):
                model_path = model_path_zip
    else:
        # Use default model path from config
        model_path = os.path.join(config.PATHS["model_dir"], config.PATHS["model_name"])
        # Check with .zip extension (Stable Baselines3 saves with .zip)
        if not os.path.exists(model_path):
            model_path_zip = model_path + '.zip'
            if os.path.exists(model_path_zip):
                model_path = model_path_zip
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"\nError: Model file not found at {model_path}")
        print("Please provide a valid model path as an argument, or ensure the default model exists.")
        print(f"Usage: python resume_train.py [model_path]")
        # List available models
        model_dir = config.PATHS["model_dir"]
        if os.path.exists(model_dir):
            available_models = [f for f in os.listdir(model_dir) if f.endswith('.zip')]
            if available_models:
                print(f"\nAvailable models in {model_dir}:")
                for model_file in available_models:
                    print(f"  - {os.path.join(model_dir, model_file)}")
        sys.exit(1)
    
    # Create environment
    print("\nCreating environment...")
    base_env = create_environment()
    base_env.reset()
    
    print(f"Action space shape: {base_env.action_space.shape}")
    print(f"Fixed wavelength: {base_env.fixed_wavelength}")
    print(f"Observation space shape: {base_env.observation_space.shape}")
    print(f"Observation keys: {base_env.obs_keys}")
    print(f"\nReward weights (from config):")
    for key, value in base_env.reward_weights.items():
        print(f"  {key}: {value}")
    
    # Check if VecNormalize statistics file exists (for observation normalization)
    vec_normalize_path = model_path.replace('.zip', '_vec_normalize.pkl')
    if not vec_normalize_path.endswith('.pkl'):
        vec_normalize_path = model_path + '_vec_normalize.pkl'
    
    normalize_obs = config.MODEL_CONFIG.get("normalize_observations", False)
    has_vec_normalize_file = os.path.exists(vec_normalize_path)
    
    # Apply observation normalization if enabled or if stats file exists
    if normalize_obs or has_vec_normalize_file:
        print("\n  ✓ Observation normalization enabled (VecNormalize)")
        if has_vec_normalize_file:
            print(f"    Loading VecNormalize statistics from {vec_normalize_path}")
        else:
            print("    Creating new VecNormalize statistics")
        
        # Wrap in DummyVecEnv (required for VecNormalize)
        env = DummyVecEnv([create_environment])
        # Apply VecNormalize wrapper
        env = VecNormalize(
            env,
            training=config.MODEL_CONFIG.get("normalize_observations_training", True),
            norm_obs=True,
            norm_reward=False,
            clip_obs=config.MODEL_CONFIG.get("clip_obs", 10.0),
        )
        
        # Load existing statistics if available
        if has_vec_normalize_file:
            env = VecNormalize.load(vec_normalize_path, env)
            print("    ✓ VecNormalize statistics loaded successfully")
    else:
        # Wrap in DummyVecEnv for consistency
        env = DummyVecEnv([create_environment])
    
    # Optional: Check environment (can be slow, comment out for production)
    # print("\nChecking environment...")
    # check_env(env, warn=True)
    
    # Load existing PPO model
    print(f"\nLoading PPO model from {model_path}...")
    model = PPO.load(model_path, env=env)
    
    # Create callbacks
    reward_callback = RewardCallback(
        print_freq=config.TRAIN_CONFIG["print_freq"],
        step_info_keys=config.TRAIN_CONFIG["step_info_keys"],
        print_exclude_keys=config.TRAIN_CONFIG.get("print_exclude_keys", []),
        save_dir=config.PATHS["log_dir"],  # Save training data to log directory
        save_freq=config.TRAIN_CONFIG.get("save_freq", 100),  # Save every N episodes
        save_steps=config.TRAIN_CONFIG.get("save_steps", True),  # Whether to save step-level data
        total_timesteps=config.TRAIN_CONFIG["total_timesteps"],
    )
    
    # Add checkpoint callback to save model periodically (every 10 iterations = ~27k timesteps)
    checkpoint_callback = OverwriteCheckpointCallback(
        checkpoint_freq=config.TRAIN_CONFIG.get("checkpoint_freq", 10_000),
        save_path=config.PATHS["model_dir"],
        filename=config.PATHS.get("checkpoint_name", "checkpoint"),
        verbose=1,
        checkpoint_hooks=[reward_callback.checkpoint_hook],
        total_timesteps=config.TRAIN_CONFIG["total_timesteps"],
    )
    
    # Combine callbacks
    callback = CallbackList([reward_callback, checkpoint_callback])
    
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
    
    # Save model (SB3 automatically adds .zip extension)
    save_model_path = os.path.join(config.PATHS["model_dir"], config.PATHS["model_name"])
    model.save(save_model_path)
    print(f"\nTraining finished! Model saved to {save_model_path}.zip")
    
    # Save VecNormalize statistics if observation normalization is enabled
    if (normalize_obs or has_vec_normalize_file) and isinstance(env, VecNormalize):
        vec_normalize_save_path = os.path.join(config.PATHS["model_dir"], f"{config.PATHS['model_name']}_vec_normalize.pkl")
        env.save(vec_normalize_save_path)
        print(f"VecNormalize statistics saved to {vec_normalize_save_path}")
    
    # Close environment
    env.close()
    
    print("\nTraining complete!")
    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()

