"""
Test script for evaluating trained models
Run this script to test a trained model on the snake environment.
"""

import os
import sys
import numpy as np
import io
import contextlib
from stable_baselines3 import PPO

from snake_env import FixedWavelengthContinuumSnakeEnv
import config


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
    env.reward_weights = config.REWARD_WEIGHTS
    
    return env


def test_model(model_path: str, num_steps: int = 100, deterministic: bool = True):
    """
    Test a trained model
    
    Args:
        model_path: Path to the saved model
        num_steps: Number of steps to run
        deterministic: Whether to use deterministic actions
    """
    print("=" * 70)
    print(f"Testing model: {model_path}")
    print("=" * 70)
    
    # Create environment
    print("\nCreating environment...")
    env = create_environment()
    obs, _ = env.reset()
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file not found at {model_path}.zip")
        return
    
    model = PPO.load(model_path, env=env)
    print("Model loaded successfully!")
    
    # Run evaluation
    print(f"\nRunning {num_steps} steps with trained model...")
    print("=" * 70)
    
    total_reward = 0.0
    episode_count = 0
    
    for i in range(num_steps):
        # Get action from model
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Step environment (suppress stdout from elastica)
        with contextlib.redirect_stdout(io.StringIO()):
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Accumulate reward
        total_reward += reward
        
        # Print step information
        action_str = np.array2string(action, precision=6, suppress_small=True)
        status_line = (
            f"Step {i+1}/{num_steps} | "
            f"Time: {info['current_time']:.2f}s | "
            f"Forward: {info['forward_speed']:.4f} m/s | "
            f"Lateral: {info['lateral_speed']:.4f} m/s | "
            f"Reward: {reward:.4f} | "
            f"Pos: [{info['position'][0]:.3f}, {info['position'][1]:.3f}, {info['position'][2]:.3f}] | "
            f"Action: {action_str}"
        )
        print(status_line)
        
        # Check if episode ended
        if terminated or truncated:
            episode_count += 1
            print(f"\nEpisode {episode_count} ended at step {i+1}")
            print(f"  Final position: {info['position']}")
            print(f"  Episode reward: {total_reward:.4f}")
            obs, _ = env.reset()
            print("Environment reset for next episode\n")
            total_reward = 0.0
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"Evaluation complete!")
    print(f"Total steps: {num_steps}")
    print(f"Episodes completed: {episode_count}")
    print(f"Average reward per step: {total_reward/num_steps:.4f}")
    print("=" * 70)
    
    env.close()


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test a trained RL model")
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(config.PATHS["model_dir"], config.PATHS["model_name"]),
        help="Path to the saved model (without .zip extension)"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to run"
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions instead of deterministic"
    )
    
    args = parser.parse_args()
    
    test_model(
        model_path=args.model_path,
        num_steps=args.num_steps,
        deterministic=not args.stochastic
    )


if __name__ == "__main__":
    main()

