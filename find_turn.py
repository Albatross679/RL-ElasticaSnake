import numpy as np
from snake_env import FixedWavelengthXZOnlyContinuumSnakeEnv

def evaluate_gait(env, action_array):
    obs, _ = env.reset()
    
    # Run simulation for 100 steps
    for _ in range(100):
        # We wrap the action in the format the env expects
        # (Note: env expects raw actions [-8e-3, 8e-3], not pre-scaled torques)
        # So we pass the raw action directly.
        obs, _, done, _, info = env.step(action_array)
        if done:
            break  # Stop if episode terminates early
    
    # Return final X position (Left/Right)
    final_x_position = info['position'][0]
    return final_x_position

def main():
    # Initialize Env
    env = FixedWavelengthXZOnlyContinuumSnakeEnv(fixed_wavelength=1.0)
    
    # Start with the known Forward Gait (in action space [-8e-3, 8e-3])
    # Assuming the forward gait corresponds to roughly max torque in specific patterns
    action_max = 8e-3
    best_action = np.array([4e-3, 4e-3, 4e-3, 4e-3, 4e-3, 4e-3]) 
    best_score = -9999
    
    total_iterations = 500
    print(f"Searching for Left Turn gait ({total_iterations} iterations)...")
    
    for i in range(total_iterations):
        # Mutate the best action slightly
        noise = np.random.normal(0, 2e-3, size=6)
        candidate_action = np.clip(best_action + noise, -action_max, action_max)
        
        # Evaluate (Score = X displacement)
        score = evaluate_gait(env, candidate_action)
        
        # If it moved further Left (positive X) than previous best
        if score > best_score:
            best_score = score
            best_action = candidate_action
            print(f"[{i+1}/{total_iterations}] New Best Left X: {score:.4f} | Action: {np.round(best_action, 3)}")
        
        # Print progress every 50 iterations
        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{total_iterations}] Progress: {100*(i+1)/total_iterations:.0f}% | Best X: {best_score:.4f}")

    print(f"\nFINAL LEFT ACTION: {list(best_action)}")
    
    # For RIGHT turn, try flipping signs or running search again optimizing for -X
    
if __name__ == "__main__":
    main()