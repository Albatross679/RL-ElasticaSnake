"""
Script to resume training from an existing PPO model
Run this script to continue training a previously saved PPO agent.
"""

import os
import sys
import numpy as np
import json
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from snake_env import FixedWavelengthXZOnlyContinuumSnakeEnv
from callbacks import RewardCallback, OverwriteCheckpointCallback
from stable_baselines3.common.callbacks import CallbackList
import config
import time


def create_environment(env_config=None):
    """Create and configure the environment"""
    # Use provided env_config or fall back to config.ENV_CONFIG
    if env_config is None:
        env_config = config.ENV_CONFIG
    
    env = FixedWavelengthXZOnlyContinuumSnakeEnv(
        fixed_wavelength=env_config["fixed_wavelength"],
        obs_keys=env_config["obs_keys"],
    )
    
    # Configure environment parameters
    env.period = env_config["period"]
    env.ratio_time = env_config["ratio_time"]
    env.rut_ratio = env_config["rut_ratio"]
    env.max_episode_length = env_config["max_episode_length"]
    env.reward_weights = config.REWARD_WEIGHTS
    
    return env


def main():
    """Main training function"""
    start_time = time.time()
    print("=" * 70)
    print("Resuming RL Training for Continuum Snake")
    print("=" * 70)
    
    # Determine device (GPU or CPU) - check early
    import torch
    # Check if GPU should be used (from config, environment variable, or auto-detect)
    use_gpu = config.MODEL_CONFIG.get("use_gpu", None)  # Can be True, False, or None (auto-detect)
    
    if use_gpu is None:
        # Auto-detect: use GPU if available and CUDA_VISIBLE_DEVICES is not set to -1
        if torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1":
            device = "cuda"
        else:
            device = "cpu"
    elif use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and use_gpu:
            print("\n  ⚠ Warning: GPU requested but not available, using CPU")
    else:
        device = "cpu"
    
    # Print device information prominently at the start
    print(f"\nDevice Configuration:")
    print(f"  Using: {device.upper()}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  PyTorch Version: {torch.__version__}")
    else:
        print(f"  CPU mode (GPU not available or disabled)")
    
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
    
    # Try to load saved configuration file
    # First, try to find config file based on model name
    model_basename = os.path.splitext(os.path.basename(model_path))[0]
    if model_basename.endswith('.zip'):
        model_basename = os.path.splitext(model_basename)[0]
    
    # Try checkpoint config first, then model config
    config_paths = [
        os.path.join(config.PATHS["log_dir"], f"{model_basename}_config.json"),
        os.path.join(config.PATHS["log_dir"], f"{config.PATHS['model_name']}_config.json"),
    ]
    
    saved_env_config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            print(f"\nLoading saved environment configuration from {config_path}...")
            with open(config_path, 'r') as f:
                saved_env_config = json.load(f)
            print("  ✓ Configuration loaded successfully")
            print(f"  Observation keys: {saved_env_config.get('obs_keys', 'N/A')}")
            break
    
    if saved_env_config is None:
        print("\n  ⚠ Warning: No saved configuration file found. Using config.ENV_CONFIG.")
        print("  This may cause issues if the observation space has changed.")
        saved_env_config = config.ENV_CONFIG
    
    # Create environment with saved config
    print("\nCreating environment...")
    base_env = create_environment(env_config=saved_env_config)
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
        # Use lambda to capture saved_env_config
        env = DummyVecEnv([lambda: create_environment(env_config=saved_env_config)])
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
            # Safety check: Validate observation space shape before loading
            expected_obs_shape = base_env.observation_space.shape
            print(f"    Expected observation shape: {expected_obs_shape}")
            
            # Try to load and validate the VecNormalize stats
            try:
                # Load the VecNormalize object to check its observation space
                import pickle
                with open(vec_normalize_path, 'rb') as f:
                    vec_normalize_data = pickle.load(f)
                
                # Check if the saved observation space matches
                if hasattr(vec_normalize_data, 'observation_space'):
                    saved_obs_shape = vec_normalize_data.observation_space.shape
                    print(f"    Saved observation shape: {saved_obs_shape}")
                    
                    if saved_obs_shape != expected_obs_shape:
                        print(f"    ⚠ WARNING: Observation shape mismatch!")
                        print(f"      Expected: {expected_obs_shape}")
                        print(f"      Saved:    {saved_obs_shape}")
                        print(f"      This may cause errors when loading VecNormalize statistics.")
                        print(f"      The saved config has obs_keys: {saved_env_config.get('obs_keys', 'N/A')}")
                        response = input("      Continue anyway? (y/n): ")
                        if response.lower() != 'y':
                            print("      Aborting. Please check your configuration.")
                            sys.exit(1)
                
                # Now load it properly
                env = VecNormalize.load(vec_normalize_path, env)
                print("    ✓ VecNormalize statistics loaded successfully")
            except Exception as e:
                print(f"    ⚠ ERROR: Failed to load VecNormalize statistics: {e}")
                print("    Continuing with new VecNormalize statistics...")
    else:
        # Wrap in DummyVecEnv for consistency
        env = DummyVecEnv([lambda: create_environment(env_config=saved_env_config)])
    
    # Optional: Check environment (can be slow, comment out for production)
    # print("\nChecking environment...")
    # check_env(env, warn=True)
    
    # Load existing PPO model
    print(f"\nLoading PPO model from {model_path}...")
    model = PPO.load(model_path, env=env, device=device)
    
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
        env_config=saved_env_config,
        config_save_dir=config.PATHS["log_dir"],
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

