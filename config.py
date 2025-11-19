"""
Configuration file for RL training
"""

# Environment configuration
ENV_CONFIG = {
    "fixed_wavelength": 1.0,
    "obs_keys": ["position", "velocity", "curvature"],
    "period": 1.0,
    "ratio_time": 0.01,
    "rut_ratio": 0.001,
}

# Training configuration
TRAIN_CONFIG = {
    "total_timesteps": 5000,  # Change to 50_000 or more for full training
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
    "save_freq": 100,  # Save every N episodes
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

