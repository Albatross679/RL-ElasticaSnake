"""
Main training script for RL Snake
Run this script to train the PPO agent on the snake environment.
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
    print("Starting RL Training for Continuum Snake")
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
    
    # Apply observation normalization (VecNormalize) if enabled
    # This is HIGHLY RECOMMENDED for unbounded observation spaces!
    # Note: VecNormalize uses STANDARDIZATION (mean=0, std=1), NOT min-max scaling to [0,1]
    # After standardization, values can still be large (e.g., 5, 10, or more standard deviations)
    # Clipping prevents extreme outliers from destabilizing training
    normalize_obs = config.MODEL_CONFIG.get("normalize_observations", False)
    if normalize_obs:
        print("\n  ✓ Observation normalization enabled (VecNormalize)")
        print("    Standardizing observations to mean=0, std=1 using running statistics")
        print("    Note: Standardized values are NOT bounded to [0,1] - they can be large!")
        print("    Clipping to [-10, 10] to handle outliers (most values will be within [-3, 3])")
        # Wrap in DummyVecEnv (required for VecNormalize)
        env = DummyVecEnv([create_environment])
        # Apply VecNormalize wrapper
        clip_value = config.MODEL_CONFIG.get("clip_obs", 10.0)
        env = VecNormalize(
            env,
            training=config.MODEL_CONFIG.get("normalize_observations_training", True),
            norm_obs=True,  # Standardize observations (mean=0, std=1)
            norm_reward=False,  # Don't normalize rewards (you have custom reward structure)
            clip_obs=clip_value,  # Clip standardized observations to [-clip_value, clip_value]
        )
    else:
        # Wrap in DummyVecEnv for consistency (PPO works with vectorized environments)
        env = DummyVecEnv([create_environment])
    
    # Optional: Check environment (can be slow, comment out for production)
    # print("\nChecking environment...")
    # check_env(env, warn=True)
    
    # Create PPO model
    print("\nCreating PPO model...")
    
    # Configure policy kwargs (for layer normalization, custom network architecture, etc.)
    policy_kwargs = {}
    
    # Orthogonal initialization (often better for RL than default)
    if config.MODEL_CONFIG.get("use_orthogonal_init", False):
        policy_kwargs["ortho_init"] = True
        print("  ✓ Orthogonal weight initialization enabled")
    
    # Custom network architecture if specified
    if config.MODEL_CONFIG.get("net_arch") is not None:
        policy_kwargs["net_arch"] = config.MODEL_CONFIG["net_arch"]
        print(f"  Using custom network architecture: {config.MODEL_CONFIG['net_arch']}")
    
    # Layer normalization: Create custom feature extractor with layer normalization
    if config.MODEL_CONFIG.get("use_layer_norm", False):
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
        import torch.nn as nn
        import torch
        
        # Custom feature extractor with layer normalization
        class LayerNormFeatureExtractor(BaseFeaturesExtractor):
            """
            Feature extractor with layer normalization.
            Normalizes activations across features for each sample independently,
            which helps with training stability when observations have different scales.
            """
            def __init__(self, observation_space, features_dim: int = 64):
                super().__init__(observation_space, features_dim)
                n_input = observation_space.shape[0]
                self.net = nn.Sequential(
                    nn.Linear(n_input, features_dim),
                    nn.LayerNorm(features_dim),
                    nn.Tanh(),
                    nn.Linear(features_dim, features_dim),
                    nn.LayerNorm(features_dim),
                    nn.Tanh(),
                )
            
            def forward(self, observations: torch.Tensor) -> torch.Tensor:
                return self.net(observations)
        
        policy_kwargs["features_extractor_class"] = LayerNormFeatureExtractor
        policy_kwargs["features_extractor_kwargs"] = {"features_dim": 64}
        print("  ✓ Layer normalization enabled in policy network")
    
    # Get gradient clipping value
    # Note: Stable-Baselines3 PPO always applies gradient clipping, so we need a valid float
    # If you want to disable clipping, use a very large value (e.g., 1e6)
    max_grad_norm = config.MODEL_CONFIG.get("max_grad_norm", 0.5)
    if max_grad_norm is None:
        # If None is explicitly set, use a very large value to effectively disable clipping
        max_grad_norm = 1e6
        print("  Gradient clipping: disabled (using large max_grad_norm=1e6)")
    else:
        print(f"  ✓ Gradient clipping enabled: max_grad_norm={max_grad_norm}")
    
    model = PPO(
        config.MODEL_CONFIG["policy"],
        env,
        gamma=config.MODEL_CONFIG["gamma"],      # <--- The Discount Factor
        gae_lambda=config.MODEL_CONFIG["gae_lambda"], # <--- The GAE Parameter
        n_steps=config.MODEL_CONFIG["n_steps"],    # <--- The Rollout Buffer Size
        max_grad_norm=max_grad_norm,  # Gradient clipping to prevent exploding gradients
        verbose=config.MODEL_CONFIG["verbose"],
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        device=device,  # Explicitly set device (cuda or cpu)
        # tensorboard_log=config.PATHS["log_dir"]  # Uncomment for tensorboard logging
    )
    
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
    
    # Add checkpoint callback to save model periodically (every 10k timesteps)
    checkpoint_callback = OverwriteCheckpointCallback(
        checkpoint_freq=config.TRAIN_CONFIG.get("checkpoint_freq", 10_000),
        save_path=config.PATHS["model_dir"],
        filename=config.PATHS.get("checkpoint_name", "checkpoint"),
        verbose=1,
        checkpoint_hooks=[reward_callback.checkpoint_hook],
        total_timesteps=config.TRAIN_CONFIG["total_timesteps"],
        env_config=config.ENV_CONFIG,
        config_save_dir=config.PATHS["log_dir"],
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
    
    # Save model (SB3 automatically adds .zip extension)
    model_path = os.path.join(config.PATHS["model_dir"], config.PATHS["model_name"])
    model.save(model_path)
    print(f"\nTraining finished! Model saved to {model_path}.zip")
    
    # Save ENV_CONFIG to JSON file alongside the model
    config_path = os.path.join(config.PATHS["log_dir"], f"{config.PATHS['model_name']}_config.json")
    with open(config_path, 'w') as f:
        json.dump(config.ENV_CONFIG, f, indent=2)
    print(f"Environment configuration saved to {config_path}")
    
    # Save VecNormalize statistics if observation normalization is enabled
    if normalize_obs and isinstance(env, VecNormalize):
        vec_normalize_path = os.path.join(config.PATHS["model_dir"], f"{config.PATHS['model_name']}_vec_normalize.pkl")
        env.save(vec_normalize_path)
        print(f"VecNormalize statistics saved to {vec_normalize_path}")
        print("  (Load this when testing/evaluating the model to use the same normalization)")
    
    # Close environment
    env.close()
    
    print("\nTraining complete!")
    elapsed = time.time() - start_time
    print(f"Total training time: {elapsed:.1f} seconds ({elapsed/60:.2f} minutes)")


if __name__ == "__main__":
    main()

