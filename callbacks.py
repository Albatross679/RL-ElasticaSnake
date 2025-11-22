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
        step_info_keys: Optional[list[str]] = None,
        print_exclude_keys: Optional[list[str]] = None,
        save_dir: Optional[str] = None,
        save_freq: int = 100,  # Save every N episodes
        save_steps: bool = True,  # Whether to save step-level data
        total_timesteps: Optional[int] = None,
    ):
        super(RewardCallback, self).__init__(verbose)
        self.print_freq = print_freq  # Controls both step and episode printing frequency
        self.step_info_keys = step_info_keys or []
        self.print_exclude_keys = set(print_exclude_keys or [])  # Keys to exclude from printing
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_steps = save_steps
        self.total_timesteps = total_timesteps
        self.start_timesteps = None  # Will be set in _init_callback
        self._first_seen_timestep = None  # Track first timestep we see for lazy init
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_count = 0
        self.timesteps_at_episode = []  # Track timesteps when each episode ended
        
        # Unified step data storage: list of dicts snapshotting info every save_freq episodes
        self.step_data = []  # Each entry captured at save_freq timesteps
        self.latest_step_entry = None
        self.last_snapshot_timestep = 0
        
        # Create save directory if provided
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            # Single unified training data file (overwritten at checkpoints)
            self.training_data_file = os.path.join(self.save_dir, "training_data.json")
    
    def _init_callback(self) -> None:
        """Initialize callback - capture starting timestep for progress calculation"""
        self.start_timesteps = self.num_timesteps

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

        # Handle step-level rewards and printing
        if rewards:
            step_reward = float(np.mean(rewards))
            sim_time = info_dict.get('current_time')
            
            # Track latest step data (used when we snapshot every save_freq episodes)
            step_entry = {
                "timestep": self.num_timesteps,
                "reward": step_reward,
                "sim_time": sim_time if sim_time is not None else None,
            }
            for key in self.step_info_keys:
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
                # Filter out excluded keys from printing
                keys_to_print = [k for k in self.step_info_keys if k in info_dict and k not in self.print_exclude_keys]
                info_bits = [f"{k}={info_dict[k]}" for k in keys_to_print]
                info_str = f" | {'; '.join(info_bits)}" if info_bits else ""
                sim_time_str = f" | sim_time={sim_time:.6f}" if sim_time is not None else ""
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [Step {self.num_timesteps}] reward={step_reward:.6f}{sim_time_str}{info_str}")

        # Handle episode completion
        episode_info = info_dict.get('episode')
        if episode_info:
            episode_reward = episode_info.get('r', 0)
            episode_length = episode_info.get('l', 0)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.timesteps_at_episode.append(self.num_timesteps)
            self.episode_count += 1

            print(f"\n[Episode {self.episode_count}] Reward = {episode_reward:.4f}, Length = {episode_length} steps")

            if self.episode_count % self.print_freq == 0:
                avg_reward = np.mean(self.episode_rewards[-self.print_freq:])
                print(f"  -> Average reward (last {self.print_freq} episodes) = {avg_reward:.4f}\n")

        return True
    
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
                "step_info_keys": self.step_info_keys,
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
        
        if self.verbose > 0:
            print(f"  -> Training data saved to {self.training_data_file}")
    
    def record_step_snapshot(self):
        """Capture a snapshot of the latest step info at the configured save frequency."""
        if not self.save_steps or not self.latest_step_entry:
            return
        snapshot = dict(self.latest_step_entry)
        snapshot["episode"] = self.episode_count
        snapshot["captured_at"] = datetime.now().isoformat()
        self.step_data.append(snapshot)

    def _on_training_end(self):
        """Called when training ends - save final data"""
        if self.save_dir:
            self.save_training_data()
            print(f"\nFinal training data saved to {self.save_dir}")

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
    ):
        super().__init__(verbose)
        if save_path is None:
            raise ValueError("save_path must be provided for OverwriteCheckpointCallback")
        self.checkpoint_freq = int(checkpoint_freq)
        self.save_path = save_path
        self.filename = filename
        self._last_checkpoint_step = 0
        self._checkpoint_file = os.path.join(self.save_path, self.filename)
        self.checkpoint_hooks = list(checkpoint_hooks) if checkpoint_hooks else []
        self.total_timesteps = total_timesteps
        self.start_timesteps = None  # Will be set in _init_callback
        self._first_seen_timestep = None  # Track first timestep we see for lazy init

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        self.start_timesteps = self.num_timesteps

    def _save_checkpoint(self) -> None:
        self.model.save(self._checkpoint_file)
        
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
        
        if self.verbose > 0:
            print(f"Checkpoint saved to {self._checkpoint_file}")
        
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

