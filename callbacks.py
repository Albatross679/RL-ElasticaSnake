"""
Training callbacks for stable-baselines3
"""

import numpy as np
import os
import json
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional, Callable, Iterable
from datetime import datetime


class RewardCallback(BaseCallback):
    """
    Callback for printing reward information during training.
    Prints episode rewards and, optionally, step-level metrics.
    Can save training data to files periodically.
    """
    def __init__(
        self,
        verbose: int = 0,
        print_freq: int = 5,
        save_keys: Optional[list[str]] = None,
        print_keys: Optional[list[str]] = None,
        save_dir: Optional[str] = None,
        save_freq: int = 100,  # Save every N episodes
        save_steps: bool = True,  # Whether to save step-level data
        total_timesteps: Optional[int] = None,
    ):
        super(RewardCallback, self).__init__(verbose)
        self.print_freq = print_freq  # Controls both step and episode printing frequency
        self.save_keys = save_keys or []  # Keys to save to JSON file
        self.print_keys = set(print_keys or [])  # Keys to print to console
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_steps = save_steps
        self.total_timesteps = total_timesteps
        self.start_timesteps = None  # Will be set in _init_callback
        self._first_seen_timestep = None  # Track first timestep we see for lazy init
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.current_episode = 0  # Track current episode number (incremented when episode completes)
        self.timesteps_at_episode = []  # Track timesteps when each episode ended
        self.last_sim_time = None  # Track last sim_time to detect episode resets
        
        # Unified step data storage: list of dicts snapshotting info every save_freq episodes
        self.step_data = []  # Each entry captured at save_freq timesteps
        self.latest_step_entry = None
        self.last_snapshot_timestep = 0
        
        # Store latest gradient norms (computed after training updates)
        self.latest_gradient_norms = {
            "policy": None,
            "value": None,
        }
        
        # Create save directory if provided
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            # Single unified training data file (overwritten at checkpoints)
            self.training_data_file = os.path.join(self.save_dir, "training_data.json")
    
    def _init_callback(self) -> None:
        """Initialize callback - capture starting timestep for progress calculation"""
        self.start_timesteps = self.num_timesteps
        # Model is available through self.model (from BaseCallback)

    def _on_step(self) -> bool:
        # Track the first timestep we see for lazy initialization
        if self._first_seen_timestep is None:
            self._first_seen_timestep = self.num_timesteps
        
        # Ensure start_timesteps is set correctly
        # If it's None, set it to the first timestep we saw (handles case where _init_callback wasn't called)
        # If it's 0 but we've seen a higher timestep, use the first non-zero timestep we saw
        if self.start_timesteps is None:
            self.start_timesteps = self._first_seen_timestep
        elif self.start_timesteps == 0 and self._first_seen_timestep > 0:
            # If _init_callback captured 0 but we're resuming from a higher timestep, use the first timestep we saw
            self.start_timesteps = self._first_seen_timestep
        
        rewards = self.locals.get('rewards', [0])
        infos = self.locals.get('infos', [{}])
        info_dict = infos[0] if infos and isinstance(infos[0], dict) else {}

        # Track if we detected an episode boundary this step (via sim_time reset)
        episode_boundary_detected = False
        
        # Handle step-level rewards and printing
        if rewards:
            step_reward = float(np.mean(rewards))
            sim_time = info_dict.get('current_time')
            
            # Detect episode boundaries by sim_time reset (fallback if episode info not available)
            if sim_time is not None and self.last_sim_time is not None:
                # If sim_time decreased significantly (reset), likely a new episode started
                if sim_time < self.last_sim_time - 0.5:  # Threshold to avoid false positives from small fluctuations
                    # Episode boundary detected via sim_time reset
                    episode_boundary_detected = True
                    # Increment current episode number (we're now in a new episode)
                    # Note: current_episode starts at 0 (first episode), so when we detect the first
                    # boundary, we're starting episode 1, hence we increment
                    self.current_episode += 1
            self.last_sim_time = sim_time
            
            # Track latest step data (used when we snapshot every save_freq episodes)
            step_entry = {
                "timestep": self.num_timesteps,
                "reward": step_reward,
                "sim_time": sim_time if sim_time is not None else None,
            }
            for key in self.save_keys:
                if key in info_dict:
                    step_entry[key] = info_dict[key]
            self.latest_step_entry = step_entry

            if self.save_steps and (self.num_timesteps - self.last_snapshot_timestep) >= self.save_freq:
                # Print progress at snapshot steps
                if self.total_timesteps is not None:
                    # Calculate target total: starting timesteps + total timesteps for this session
                    # Ensure we have a valid start_timesteps (should be set by now due to lazy init above)
                    target_total = self.start_timesteps + self.total_timesteps
                    # Progress percentage is based on this session's progress (starts at 0%)
                    session_progress = self.num_timesteps - self.start_timesteps
                    progress_pct = min((session_progress / self.total_timesteps) * 100, 100.0)
                    print(f"\n[Snapshot] Progress: {self.num_timesteps:,} / {target_total:,} steps ({progress_pct:.2f}%)")
                else:
                    print(f"\n[Snapshot] Step: {self.num_timesteps:,}")
                self.record_step_snapshot()
                self.last_snapshot_timestep = self.num_timesteps

            if self.num_timesteps % self.print_freq == 0:
                # Only print keys that are in both save_keys and print_keys
                keys_to_print = [k for k in self.print_keys if k in info_dict and k in self.save_keys]
                info_bits = [f"{k}={info_dict[k]}" for k in keys_to_print]
                info_str = f" | {'; '.join(info_bits)}" if info_bits else ""
                sim_time_str = f" | sim_time={sim_time:.6f}" if sim_time is not None else ""
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Step {self.num_timesteps}] reward={step_reward:.6f}{sim_time_str}{info_str}")

        # Handle episode completion (primary method via stable-baselines3 episode info)
        episode_info = info_dict.get('episode')
        if episode_info:
            episode_reward = episode_info.get('r', 0)
            episode_length = episode_info.get('l', 0)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.timesteps_at_episode.append(self.num_timesteps)
            self.episode_count += 1
            # Increment current episode number for next episode
            # Only increment if we haven't already detected this boundary via sim_time reset
            if not episode_boundary_detected:
                self.current_episode += 1

            print(f"\n[Episode {self.episode_count}] Reward = {episode_reward:.4f}, Length = {episode_length} steps")

            if self.episode_count % self.print_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-self.print_freq:])
                print(f"  -> Average reward (last {self.print_freq} episodes) = {avg_reward:.4f}\n")

        return True
    
    def _on_rollout_end(self) -> None:
        """
        Called after each rollout collection and training update.
        Compute gradient norms here (right after training) and store them for snapshot recording.
        """
        try:
            if hasattr(self, 'model') and self.model is not None:
                # Compute gradient norms right after training update
                policy_norm = self._compute_gradient_norm(self.model.policy, network_type="policy")
                value_norm = self._compute_gradient_norm(self.model.policy, network_type="value")
                
                # Store for later use in snapshots
                self.latest_gradient_norms["policy"] = policy_norm
                self.latest_gradient_norms["value"] = value_norm
        except Exception:
            # Silently handle errors (gradients might not be available)
            pass
    
    def save_training_data(self):
        """Save all training data to a unified dictionary format"""
        if not self.save_dir:
            return
        
        # Create unified training data dictionary
        training_data = {
            "metadata": {
                "saved_at": datetime.now().isoformat(),
                "total_timesteps": self.num_timesteps,
                "episode_count": self.episode_count,
                "save_keys": self.save_keys,
                "print_keys": list(self.print_keys),
            },
            "episodes": {
                "rewards": self.episode_rewards,
                "lengths": self.episode_lengths,
                "timesteps_at_episode": self.timesteps_at_episode,
            },
            "steps": self.step_data if self.save_steps else [],
        }
        
        # Save to single file (overwrites previous)
        with open(self.training_data_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Always print the file path (not just when verbose > 0)
        print(f"  -> Training data saved to {self.training_data_file}")
    
    def _compute_gradient_norm(self, policy, network_type: str = "policy") -> Optional[float]:
        """
        Compute gradient norm for the specified network.
        
        Args:
            policy: The policy network
            network_type: "policy", "value", or "total"
        
        Returns:
            Gradient norm (L2 norm) as a float, or None if gradients not available
        """
        try:
            import torch
            total_norm = 0.0
            has_gradients = False
            
            if network_type == "policy":
                # Compute norm for policy network parameters only
                # Policy network includes: mlp_extractor (shared/pi), action_net
                for name, param in policy.named_parameters():
                    if param.grad is not None:
                        has_gradients = True
                        # Include action network and policy-specific parts
                        if "action_net" in name:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        elif "mlp_extractor" in name and ("pi" in name or "shared_net" in name):
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
            elif network_type == "value":
                # Compute norm for value network parameters only
                for name, param in policy.named_parameters():
                    if param.grad is not None:
                        has_gradients = True
                        if "value_net" in name:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
                        elif "mlp_extractor" in name and "vf" in name:
                            param_norm = param.grad.data.norm(2)
                            total_norm += param_norm.item() ** 2
            else:  # "total"
                # Compute norm for all parameters
                for param in policy.parameters():
                    if param.grad is not None:
                        has_gradients = True
                        param_norm = param.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
            
            if has_gradients:
                total_norm = total_norm ** (1. / 2)
                return float(total_norm)
            else:
                return None
        except Exception:
            # Gradients might not be available (cleared after optimizer step)
            return None
    
    def record_step_snapshot(self):
        """Capture a snapshot of the latest step info at the configured save frequency."""
        if not self.save_steps or not self.latest_step_entry:
            return
        snapshot = dict(self.latest_step_entry)
        snapshot["episode"] = self.current_episode  # Use current episode number, not count of completed episodes
        snapshot["captured_at"] = datetime.now().isoformat()
        
        # Add gradient norms if available (computed after training updates)
        if self.latest_gradient_norms["policy"] is not None:
            snapshot["gradient_norm_policy"] = self.latest_gradient_norms["policy"]
        if self.latest_gradient_norms["value"] is not None:
            snapshot["gradient_norm_value"] = self.latest_gradient_norms["value"]
        
        self.step_data.append(snapshot)

    def _on_training_end(self):
        """Called when training ends - save final data"""
        if self.save_dir:
            self.save_training_data()
            print(f"\nFinal training data saved to {self.training_data_file}")

    def checkpoint_hook(self, num_timesteps: int) -> None:
        """Hook executed at checkpoint save time - saves all training data"""
        self.save_training_data()


class OverwriteCheckpointCallback(BaseCallback):
    """
    Save checkpoints at a fixed timestep frequency, overwriting the previous
    checkpoint file each time.
    """

    def __init__(
        self,
        checkpoint_freq: int = 10_000,
        save_path: str | None = None,
        filename: str = "checkpoint",
        verbose: int = 0,
        checkpoint_hooks: Optional[Iterable[Callable[[int], None]]] = None,
        total_timesteps: Optional[int] = None,
        env_config: Optional[dict] = None,
        config_save_dir: Optional[str] = None,
        full_config: Optional[dict] = None,
    ):
        super().__init__(verbose)
        if save_path is None:
            raise ValueError("save_path must be provided for OverwriteCheckpointCallback")
        self.checkpoint_freq = int(checkpoint_freq)
        self.save_path = save_path
        self.filename = filename
        self._last_checkpoint_step = 0
        # Stable Baselines3 automatically adds .zip extension when saving
        self._checkpoint_file = os.path.join(self.save_path, self.filename)
        self._checkpoint_file_with_ext = self._checkpoint_file + ".zip"
        self.checkpoint_hooks = list(checkpoint_hooks) if checkpoint_hooks else []
        self.total_timesteps = total_timesteps
        self.start_timesteps = None  # Will be set in _init_callback
        self._first_seen_timestep = None  # Track first timestep we see for lazy init
        self.env_config = env_config  # Keep for backward compatibility
        self.full_config = full_config  # Full configuration dictionary
        self.config_save_dir = config_save_dir

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        self.start_timesteps = self.num_timesteps

    def _save_checkpoint(self) -> None:
        self.model.save(self._checkpoint_file)
        
        # Save configuration to JSON file if provided
        if self.config_save_dir is not None:
            os.makedirs(self.config_save_dir, exist_ok=True)
            config_path = os.path.join(self.config_save_dir, f"{self.filename}_config.json")
            
            # Save full config if available, otherwise fall back to env_config for backward compatibility
            if self.full_config is not None:
                with open(config_path, 'w') as f:
                    json.dump(self.full_config, f, indent=2)
            elif self.env_config is not None:
                with open(config_path, 'w') as f:
                    json.dump(self.env_config, f, indent=2)
        
        # Ensure start_timesteps is set (lazy initialization)
        if self._first_seen_timestep is None:
            self._first_seen_timestep = self.num_timesteps
        
        if self.start_timesteps is None:
            self.start_timesteps = self._first_seen_timestep
        elif self.start_timesteps == 0 and self._first_seen_timestep > 0:
            self.start_timesteps = self._first_seen_timestep
        
        # Print progress information
        if self.total_timesteps is not None:
            # Calculate target total: starting timesteps + total timesteps for this session
            target_total = self.start_timesteps + self.total_timesteps
            # Progress percentage is based on this session's progress (starts at 0%)
            session_progress = self.num_timesteps - self.start_timesteps
            progress_pct = min((session_progress / self.total_timesteps) * 100, 100.0)
            print(f"\n[Snapshot] Progress: {self.num_timesteps:,} / {target_total:,} steps ({progress_pct:.2f}%)")
        else:
            print(f"\n[Snapshot] Step: {self.num_timesteps:,}")
        
        # Always print the checkpoint file path (SB3 adds .zip automatically)
        print(f"Checkpoint saved to {self._checkpoint_file_with_ext}")
        
        for hook in self.checkpoint_hooks:
            try:
                hook(self.num_timesteps)
            except Exception as hook_exc:
                if self.verbose > 0:
                    print(f"Checkpoint hook error: {hook_exc}")

    def _on_step(self) -> bool:
        # Track the first timestep we see for lazy initialization
        if self._first_seen_timestep is None:
            self._first_seen_timestep = self.num_timesteps
        
        if self.num_timesteps - self._last_checkpoint_step >= self.checkpoint_freq:
            self._save_checkpoint()
            self._last_checkpoint_step = self.num_timesteps
        return True

