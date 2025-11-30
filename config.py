"""
Configuration file for RL training
"""

# Environment configuration
# Step time calculation: step_time = (ratio_time + 0.001) * period
# With period=1.0, ratio_time=0.01: step_time = 0.011 seconds per RL step
# Max steps per episode = max_episode_length / step_time ≈ 20 / 0.011 ≈ 1818 steps
ENV_CONFIG = {
    "fixed_wavelength": 1.0,
    # Available observation keys: "time", "avg_position", "avg_velocity", "curvature",
    # "tangents", "position", "velocity", "director", "relative_position"
    "obs_keys": ["avg_velocity", "curvature", "velocity", "tangents", "relative_position"],
    # Example with relative_position: ["velocity", "curvature"]
    "period": 1.0,
    "ratio_time": 0.01,
    "rut_ratio": 0.001,
    "_n_elem": 10,
    "max_episode_length": 20,  # Maximum simulation time (seconds) before episode termination
}

# Reward weights configuration
# Note: Negative values for penalties, positive values for rewards/bonuses
REWARD_WEIGHTS = {
    "forward_progress": 0.0, # 5.0 * 1000,
    "lateral_penalty": 0.0,  # Negative for penalty (perpendicular to current heading)
    "lateral_speed_penalty": -1.0 * 10,  # Negative for penalty (perpendicular to target direction)
    "curvature_range_penalty": -0.1,  # Negative for penalty
    "curvature_oscillation_reward": 0.0, #0.01 / ENV_CONFIG["_n_elem"],  # Positive for reward
    "energy_penalty": 0.0, #-2.0e4,  # Negative for penalty (set to 0.0 to disable)
    "smoothness_penalty": 0.0, # -5.0e3,  # Negative for penalty
    "alignment_bonus": 0.0, #0.5 * 0.1,
    "streak_bonus":0.0, #100.0,
    "projected_speed": 5.0 * 10,
}

# Training configuration
TRAIN_CONFIG = {
    "total_timesteps": 2_000_000,  # Change to 50_000 or more for full training
    "print_freq": 100,  # Controls both step-level and episode-level printing frequency
    "step_info_keys": ["forward_speed",
                       "lateral_speed", 
                       "velocity_projection",
                       "action",  # Record per-step action vectors in snapshots
                       "curvatures",  # Saved but not printed (see print_exclude_keys)
                       "reward_terms",  # Saved but not printed (see print_exclude_keys)
                       "forward_progress",
                    #    "speed", 
                    #    "alignment",
                    #    "alignment_streak",
                    #    "alignment_goal_met"
                       ],
    "print_exclude_keys": ["action", "curvatures", "reward_terms"],  # Keys to save but exclude from printing
    "save_freq": 100,  # Save every N timesteps for training snapshots
    "save_steps": True,  # Whether to save step-level data
    "checkpoint_freq": 10_000,  # Timesteps between checkpoint saves
}

# Model configuration
MODEL_CONFIG = {
    "n_steps": 2048,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "policy": "MlpPolicy",
    "verbose": 1,
    # GPU configuration
    # Set use_gpu to True to force GPU usage, False to force CPU, or None to auto-detect
    # Auto-detect (None): Uses GPU if available and CUDA_VISIBLE_DEVICES is not set to -1
    "use_gpu": False,  # None = auto-detect, True = force GPU, False = force CPU
    # Policy network architecture options
    # Set use_layer_norm=True to enable layer normalization in the policy network
    # Layer normalization can help with training stability when observations have
    # different scales (e.g., velocities vs curvatures vs positions)
    "use_layer_norm": True,  # Set to True to enable layer normalization
    "net_arch": [dict(pi=[64, 64], vf=[64, 64])], # None uses default [64, 64]. Can specify custom: [dict(pi=[128, 128], vf=[128, 128])]
    
    # Weight initialization
    # Orthogonal initialization often works better than default for RL
    # Initializes weights as orthogonal matrices, which helps with gradient flow
    # and training stability, especially in deep networks
    "use_orthogonal_init": True,  # Use orthogonal weight initialization (recommended for RL)
    
    # Observation normalization (VecNormalize)
    # HIGHLY RECOMMENDED for unbounded observation spaces!
    # VecNormalize uses STANDARDIZATION (mean=0, std=1), NOT min-max scaling to [0,1]
    # After standardization, values can still be large (e.g., 5-10 standard deviations)
    # Clipping prevents extreme outliers from destabilizing training
    "normalize_observations": True,  # Enable observation standardization
    "normalize_observations_training": True,  # Update normalization stats during training
    "clip_obs": 10.0,  # Clip standardized observations to [-clip_obs, clip_obs] (default: 10.0)
    
    # Gradient clipping
    # Clips gradients if their norm exceeds max_grad_norm to prevent exploding gradients
    # Helps stabilize training, especially with deep networks or unstable environments
    # Common values: 0.5 (conservative), 1.0 (moderate), None (no clipping)
    "max_grad_norm": 0.5,  # Clip gradients if norm exceeds this value (default: 0.5)
}

# Paths
PATHS = {
    "log_dir": "Training/Logs",
    "model_dir": "Training/Saved_Models",
    "model_name": "PPO_Snake_Model",
    "checkpoint_name": "PPO_Snake_Checkpoint",
    "policy_gradient_viz_dir": "policy_gradient_visualizations",  # Output directory for policy gradient visualizations
}

