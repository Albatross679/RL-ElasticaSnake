"""
Script to generate diverse actions, run them in the environment for 1000 steps,
and record the resulting heading directions.

This script explores the action space systematically to understand the relationship
between actions and snake movement directions.
"""

import numpy as np
import json
import os
from pathlib import Path
import sys
from typing import List, Dict, Tuple
from collections import defaultdict

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from snake_env import FixedWavelengthContinuumSnakeEnv
import config


def generate_diverse_actions(n_samples: int = 200, action_dim: int = 6) -> np.ndarray:
    """
    Generate diverse action arrays that cover the action space comprehensively.
    
    Uses multiple strategies:
    1. Latin Hypercube Sampling (LHS) for uniform coverage
    2. Random sampling
    3. Corner cases (extreme values)
    4. Grid sampling
    5. Smooth variations (sine waves)
    
    Args:
        n_samples: Number of action samples to generate
        action_dim: Dimension of action space (6 for FixedWavelengthContinuumSnakeEnv)
    
    Returns:
        Array of shape (n_samples, action_dim) with actions in range [-1, 1]
    """
    actions = []
    
    # Strategy 1: Latin Hypercube Sampling (LHS) - 40% of samples
    n_lhs = int(0.4 * n_samples)
    if n_lhs > 0:
        try:
            from scipy.stats import qmc
            sampler = qmc.LatinHypercube(d=action_dim, seed=42)
            lhs_samples = sampler.random(n=n_lhs)
            # Scale from [0, 1] to [-1, 1]
            lhs_actions = 2 * lhs_samples - 1
            actions.append(lhs_actions)
        except ImportError:
            # Fallback to random sampling if scipy.stats.qmc is not available
            print("Warning: scipy.stats.qmc not available, using random sampling instead")
            random_actions = np.random.uniform(-1, 1, size=(n_lhs, action_dim))
            actions.append(random_actions)
    
    # Strategy 2: Random uniform sampling - 30% of samples
    n_random = int(0.3 * n_samples)
    if n_random > 0:
        random_actions = np.random.uniform(-1, 1, size=(n_random, action_dim))
        actions.append(random_actions)
    
    # Strategy 3: Corner cases (extreme values) - 10% of samples
    n_corners = int(0.1 * n_samples)
    if n_corners > 0:
        corner_actions = []
        # Generate corners: all combinations of -1, 0, 1 for each dimension
        # Limit to avoid too many combinations (2^6 = 64 corners)
        corner_values = [-1.0, -0.5, 0.0, 0.5, 1.0]
        for _ in range(min(n_corners, len(corner_values) ** action_dim)):
            corner = np.random.choice(corner_values, size=action_dim)
            corner_actions.append(corner)
        actions.append(np.array(corner_actions[:n_corners]))
    
    # Strategy 4: Grid sampling (coarse grid) - 10% of samples
    n_grid = int(0.1 * n_samples)
    if n_grid > 0:
        grid_values = np.linspace(-1, 1, 5)  # 5 values per dimension
        grid_actions = []
        # Sample from grid points
        for _ in range(n_grid):
            grid_action = np.array([np.random.choice(grid_values) for _ in range(action_dim)])
            grid_actions.append(grid_action)
        actions.append(np.array(grid_actions))
    
    # Strategy 5: Smooth variations (sine waves) - 10% of samples
    n_smooth = n_samples - sum([n_lhs, n_random, n_corners, n_grid])
    if n_smooth > 0:
        smooth_actions = []
        for i in range(n_smooth):
            # Create sine wave patterns with different frequencies and phases
            t = np.linspace(0, 2 * np.pi, action_dim)
            freq = np.random.uniform(0.5, 3.0)
            phase = np.random.uniform(0, 2 * np.pi)
            amplitude = np.random.uniform(0.3, 1.0)
            smooth_action = amplitude * np.sin(freq * t + phase)
            smooth_actions.append(smooth_action)
        actions.append(np.array(smooth_actions))
    
    # Combine all strategies
    all_actions = np.vstack(actions)
    
    # Ensure we have exactly n_samples
    if len(all_actions) > n_samples:
        # Randomly select n_samples
        indices = np.random.choice(len(all_actions), n_samples, replace=False)
        all_actions = all_actions[indices]
    elif len(all_actions) < n_samples:
        # Fill with random samples
        n_missing = n_samples - len(all_actions)
        additional = np.random.uniform(-1, 1, size=(n_missing, action_dim))
        all_actions = np.vstack([all_actions, additional])
    
    # Shuffle to mix strategies
    np.random.shuffle(all_actions)
    
    return all_actions


def calculate_heading_direction(env, action: np.ndarray, n_steps: int = 1000) -> Dict:
    """
    Run an action in the environment for n_steps and calculate the final heading direction.
    
    Args:
        env: The snake environment
        action: Action array to apply
        n_steps: Number of steps to run
    
    Returns:
        Dictionary with heading direction, final position, and other metrics
    """
    # Reset environment
    obs, info = env.reset()
    
    # Store initial position
    initial_pos = info.get("position", np.zeros(3)).copy()
    
    # Run for n_steps
    steps_completed = 0
    terminated = False
    truncated = False
    for step in range(n_steps):
        try:
            obs, reward, terminated, truncated, info = env.step(action)
            steps_completed += 1
            
            if terminated or truncated:
                # If episode ends early, break
                break
        except Exception as e:
            print(f"    Error at step {step}: {e}")
            break
    
    # Get final state
    final_pos = info.get("position", np.zeros(3))
    heading_dir = info.get("heading_dir", np.array([0.0, 0.0, 1.0]))
    
    # Calculate displacement
    displacement = final_pos - initial_pos
    displacement_magnitude = np.linalg.norm(displacement)
    
    # Calculate direction of displacement (normalized)
    if displacement_magnitude > 1e-10:
        displacement_dir = displacement / displacement_magnitude
    else:
        displacement_dir = np.array([0.0, 0.0, 1.0])
    
    # Calculate angle between heading and displacement
    if displacement_magnitude > 1e-10:
        cos_angle = np.clip(np.dot(heading_dir, displacement_dir), -1.0, 1.0)
        angle = np.arccos(cos_angle)
    else:
        angle = 0.0
    
    result = {
        "action": action.tolist(),
        "heading_direction": heading_dir.tolist(),
        "displacement_direction": displacement_dir.tolist(),
        "displacement_magnitude": float(displacement_magnitude),
        "initial_position": initial_pos.tolist(),
        "final_position": final_pos.tolist(),
        "heading_displacement_angle": float(angle),
        "speed": float(info.get("speed", 0.0)),
        "n_steps_completed": steps_completed,
        "terminated": bool(terminated),
    }
    
    return result


def determine_optimal_sample_size(action_dim: int = 6) -> int:
    """
    Determine optimal number of samples based on action space dimensionality.
    
    Uses a heuristic: for a d-dimensional space, we want enough samples to
    cover the space reasonably well. Common approaches:
    - 10-50 samples per dimension for low dimensions
    - Statistical coverage: ~100-200 samples for 6D space
    
    Args:
        action_dim: Dimension of action space
    
    Returns:
        Recommended number of samples
    """
    # Base samples per dimension
    samples_per_dim = 20
    
    # For 6D space, 20 * 6 = 120, but we want more for better coverage
    # Use a power law: n_samples = base * (d^1.5)
    base_samples = 30
    n_samples = int(base_samples * (action_dim ** 1.2))
    
    # Clamp to reasonable range
    n_samples = max(50, min(n_samples, 500))
    
    return n_samples


def main():
    """Main function to run action-direction matching"""
    print("=" * 70)
    print("Action-Direction Matching Script")
    print("=" * 70)
    
    # Determine optimal sample size
    action_dim = 6  # For FixedWavelengthContinuumSnakeEnv
    n_samples = determine_optimal_sample_size(action_dim)
    print(f"\nRecommended number of samples: {n_samples}")
    print(f"Action space dimension: {action_dim}")
    print(f"Action range: [-1, 1] for each dimension")
    
    # Ask user for confirmation or custom sample size
    try:
        user_input = input(f"\nUse {n_samples} samples? (y/n, or enter custom number): ").strip().lower()
        if user_input == 'n':
            n_samples = int(input("Enter number of samples: "))
        elif user_input.isdigit():
            n_samples = int(user_input)
    except (ValueError, KeyboardInterrupt):
        print("Using default sample size.")
    
    print(f"\nGenerating {n_samples} diverse action samples...")
    
    # Generate diverse actions
    actions = generate_diverse_actions(n_samples=n_samples, action_dim=action_dim)
    print(f"Generated {len(actions)} action samples")
    
    # Create environment
    print("\nCreating environment...")
    env = FixedWavelengthContinuumSnakeEnv(
        fixed_wavelength=config.ENV_CONFIG["fixed_wavelength"],
        obs_keys=config.ENV_CONFIG["obs_keys"],
    )
    env.period = config.ENV_CONFIG["period"]
    env.ratio_time = config.ENV_CONFIG["ratio_time"]
    env.rut_ratio = config.ENV_CONFIG["rut_ratio"]
    env.max_episode_length = config.ENV_CONFIG["max_episode_length"]
    
    print(f"Action space: {env.action_space}")
    print(f"Running each action for 1000 steps...")
    
    # Run each action and collect results
    results = []
    n_steps = 1000
    
    print("\nRunning actions in environment...")
    for i, action in enumerate(actions):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Progress: {i + 1}/{len(actions)} ({100 * (i + 1) / len(actions):.1f}%)")
        
        try:
            result = calculate_heading_direction(env, action, n_steps=n_steps)
            result["sample_id"] = i
            results.append(result)
        except Exception as e:
            print(f"  Error with action {i}: {e}")
            continue
    
    print(f"\nCompleted {len(results)}/{len(actions)} action runs")
    
    # Analyze results
    print("\nAnalyzing results...")
    
    # Group by heading direction (using spherical coordinates for clustering)
    heading_directions = np.array([r["heading_direction"] for r in results])
    
    # Calculate statistics
    stats = {
        "n_samples": len(results),
        "n_steps_per_sample": n_steps,
        "heading_directions": {
            "mean": np.mean(heading_directions, axis=0).tolist(),
            "std": np.std(heading_directions, axis=0).tolist(),
        },
        "displacement_magnitudes": {
            "mean": float(np.mean([r["displacement_magnitude"] for r in results])),
            "std": float(np.std([r["displacement_magnitude"] for r in results])),
            "min": float(np.min([r["displacement_magnitude"] for r in results])),
            "max": float(np.max([r["displacement_magnitude"] for r in results])),
        },
        "speeds": {
            "mean": float(np.mean([r["speed"] for r in results])),
            "std": float(np.std([r["speed"] for r in results])),
            "min": float(np.min([r["speed"] for r in results])),
            "max": float(np.max([r["speed"] for r in results])),
        },
    }
    
    # Save results
    output_dir = Path(__file__).parent.parent / "Training" / "Logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "action_direction_mapping.json"
    
    output_data = {
        "metadata": {
            "n_samples": len(results),
            "n_steps_per_sample": n_steps,
            "action_dim": action_dim,
            "action_space": "FixedWavelengthContinuumSnakeEnv",
        },
        "statistics": stats,
        "results": results,
    }
    
    print(f"\nSaving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved successfully!")
    print(f"\nSummary:")
    print(f"  Total samples: {len(results)}")
    print(f"  Steps per sample: {n_steps}")
    print(f"  Mean displacement magnitude: {stats['displacement_magnitudes']['mean']:.4f}")
    print(f"  Mean speed: {stats['speeds']['mean']:.4f}")
    print(f"  Mean heading direction: {stats['heading_directions']['mean']}")
    
    # Print recommendation
    print(f"\n{'=' * 70}")
    print("RECOMMENDATION:")
    print(f"{'=' * 70}")
    print(f"For a {action_dim}D action space, we recommend using {n_samples} samples.")
    print(f"This provides good coverage while remaining computationally feasible.")
    print(f"The script used multiple sampling strategies:")
    print(f"  - Latin Hypercube Sampling (40%)")
    print(f"  - Random sampling (30%)")
    print(f"  - Corner cases (10%)")
    print(f"  - Grid sampling (10%)")
    print(f"  - Smooth variations (10%)")
    print(f"\nResults are saved in: {output_file}")


if __name__ == "__main__":
    main()

