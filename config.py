"""
Configuration file for RL training
"""

# Environment configuration
ENV_CONFIG = {
    "fixed_wavelength": 1.0,
    "obs_keys": ["avg_velocity", "curvature", "velocity", "tangents"],
    "period": 1.0,
    "ratio_time": 0.01,
    "rut_ratio": 0.001,
    "_n_elem": 50,
}

# Reward weights configuration
# Note: Negative values for penalties, positive values for rewards/bonuses
REWARD_WEIGHTS = {
    "forward_progress": 1.0,
    "lateral_penalty": -1.0,  # Negative for penalty
    "curvature_penalty": -0.05,  # Negative for penalty
    "energy_penalty": -2.0e4,  # Negative for penalty (set to 0.0 to disable)
    "smoothness_penalty": -5.0e3,  # Negative for penalty
    "alignment_bonus": 0.5,
    "streak_bonus": 100.0,
    "projected_speed": 2.0,
}

# Training configuration
TRAIN_CONFIG = {
    "total_timesteps": 5000,  # Change to 50_000 or more for full training
    "n_steps": 2048,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "policy": "MlpPolicy",
    "verbose": 1,
    "print_freq": 100,  # Controls both step-level and episode-level printing frequency
    "step_info_keys": ["forward_speed",
                       "lateral_speed", 
                       "velocity_projection", 
                    #    "forward_progress",
                    #    "speed", 
                    #    "alignment",
                    #    "alignment_streak",
                    #    "alignment_goal_met"
                       ],
    "save_freq": 100,  # Save every N timesteps for training snapshots
    "save_steps": True,  # Whether to save step-level data
    "checkpoint_freq": 10_000,  # Timesteps between checkpoint saves
}

# Model configuration
MODEL_CONFIG = {
    "policy": "MlpPolicy",
    "verbose": 1,
}

# Paths
PATHS = {
    "log_dir": "Training/Logs",
    "model_dir": "Training/Saved_Models",
    "model_name": "PPO_Snake_Model",
    "checkpoint_name": "PPO_Snake_Checkpoint",
}

