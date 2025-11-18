"""
Example usage script showing how to use the environment and model
"""

import numpy as np
from snake_env import FixedWavelengthContinuumSnakeEnv
from stable_baselines3 import PPO
import config


def example_environment_usage():
    """Example of using the environment directly"""
    print("=" * 70)
    print("Example: Using the environment directly")
    print("=" * 70)
    
    # Create environment
    env = FixedWavelengthContinuumSnakeEnv(
        fixed_wavelength=1.0,
        obs_keys=["position", "velocity", "curvature"],
    )
    
    # Configure environment
    env.period = 1.0
    env.ratio_time = 0.01
    env.rut_ratio = 0.001
    
    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps with a constant action
    default_action = np.array([-5e-3, 5e-3, 5e-3, 5e-3, -5e-3, -5e-3], dtype=np.float32)
    action = env._map_torque_to_action(default_action)
    
    print(f"\nRunning 5 steps with constant action...")
    for i in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, forward={info['forward_speed']:.4f} m/s")
        
        if terminated or truncated:
            print("Episode ended!")
            obs, info = env.reset()
    
    env.close()
    print("\nEnvironment example complete!\n")


def example_training():
    """Example of training a model"""
    print("=" * 70)
    print("Example: Training a model")
    print("=" * 70)
    print("To train a model, run: python train.py")
    print("Or import and use:")
    print("""
    from train import create_environment
    from stable_baselines3 import PPO
    from callbacks import RewardCallback
    import config
    
    env = create_environment()
    model = PPO("MlpPolicy", env, verbose=1)
    callback = RewardCallback(print_freq=5, step_print_interval=20)
    model.learn(total_timesteps=1000, callback=callback)
    model.save("my_model")
    """)


def example_testing():
    """Example of testing a trained model"""
    print("=" * 70)
    print("Example: Testing a trained model")
    print("=" * 70)
    print("To test a model, run: python test.py --model_path <path> --num_steps 100")
    print("Or import and use:")
    print("""
    from test import test_model
    
    test_model(
        model_path="Training/Saved_Models/PPO_Snake_Model",
        num_steps=100,
        deterministic=True
    )
    """)


if __name__ == "__main__":
    example_environment_usage()
    example_training()
    example_testing()

