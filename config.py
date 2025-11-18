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
    "total_timesteps": 2000,  # Change to 50_000 or more for full training
    "print_freq": 5,
    "step_print_interval": 20,
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
}

