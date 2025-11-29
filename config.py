"""
Configuration file for RL training
"""

# Environment configuration
ENV_CONFIG = {
    "fixed_wavelength": 1.0,
    # Available observation keys: "time", "avg_position", "avg_velocity", "curvature",
    # "tangents", "position", "velocity", "director", "relative_position"
    "obs_keys": ["avg_velocity", "curvature", "velocity", "tangents"],
    # Example with relative_position: ["relative_position", "velocity", "curvature"]
    "period": 1.0,
    "ratio_time": 0.01,
    "rut_ratio": 0.001,
    "_n_elem": 10,
    "max_episode_length": 30,  # Maximum simulation time (seconds) before episode termination
}

# Reward weights configuration
# Note: Negative values for penalties, positive values for rewards/bonuses
REWARD_WEIGHTS = {
    "forward_progress": 5.0 * 1000,
    "lateral_penalty": -1.0,  # Negative for penalty
    "curvature_range_penalty": -0.1,  # Negative for penalty
    "curvature_oscillation_reward": 0.08 / ENV_CONFIG["_n_elem"],  # Positive for reward
    "energy_penalty": 0.0, #-2.0e4,  # Negative for penalty (set to 0.0 to disable)
    "smoothness_penalty": 0.0, # -5.0e3,  # Negative for penalty
    "alignment_bonus": 0.5,
    "streak_bonus": 100.0,
    "projected_speed": 5.0 * 10,
}

# Training configuration
TRAIN_CONFIG = {
    "total_timesteps": 1_000_000,  # Change to 50_000 or more for full training
    "print_freq": 100,  # Controls both step-level and episode-level printing frequency
    "step_info_keys": ["forward_speed",
                       "lateral_speed", 
                       "velocity_projection",
                       "action",  # Record per-step action vectors in snapshots
                       "curvatures",  # Saved but not printed (see print_exclude_keys)
                       "reward_terms",  # Saved but not printed (see print_exclude_keys)
                    #    "forward_progress",
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
    # Policy network architecture options
    # Set use_layer_norm=True to enable layer normalization in the policy network
    # Layer normalization can help with training stability when observations have
    # different scales (e.g., velocities vs curvatures vs positions)
    "use_layer_norm": True,  # Set to True to enable layer normalization
    "net_arch": None,  # None uses default [64, 64]. Can specify custom: [dict(pi=[128, 128], vf=[128, 128])]
}

# Paths
PATHS = {
    "log_dir": "Training/Logs",
    "model_dir": "Training/Saved_Models",
    "model_name": "PPO_Snake_Model",
    "checkpoint_name": "PPO_Snake_Checkpoint",
    "policy_gradient_viz_dir": "policy_gradient_visualizations",  # Output directory for policy gradient visualizations
}

