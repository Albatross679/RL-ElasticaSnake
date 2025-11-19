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
        save_dir: Optional[str] = None,
        save_freq: int = 100,  # Save every N episodes
        save_steps: bool = True,  # Whether to save step-level data
    ):
        super(RewardCallback, self).__init__(verbose)
        self.print_freq = print_freq  # Controls both step and episode printing frequency
        self.step_info_keys = step_info_keys or []
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.save_steps = save_steps
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_rewards = []
        self.step_info_data = []  # Store step info keys data
        self.step_sim_times = []  # Store simulation time for each step
        self.episode_count = 0
        self.timesteps_at_episode = []  # Track timesteps when each episode ended
        self.last_saved_step_count = 0  # Track how many step rewards have been saved
        self.custom_step_snapshots = []  # Store periodic snapshots of custom info
        self.latest_step_info = {}
        
        # Create save directory if provided
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            # Use a single file for step rewards (will be appended to)
            self.step_rewards_file = os.path.join(self.save_dir, "step_rewards.npy")
            self.step_info_file = os.path.join(self.save_dir, "step_info_data.json")
            self.step_sim_times_file = os.path.join(self.save_dir, "step_sim_times.npy")
            self.step_snapshots_file = os.path.join(self.save_dir, "step_info_snapshots.json")

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [0])
        infos = self.locals.get('infos', [{}])
        info_dict = infos[0] if infos and isinstance(infos[0], dict) else {}

        # Handle step-level rewards and printing
        if rewards:
            step_reward = float(np.mean(rewards))
            self.step_rewards.append(step_reward)
            
            # Store step info data (the keys displayed in _on_step)
            step_info_dict = {k: info_dict[k] for k in self.step_info_keys if k in info_dict}
            self.step_info_data.append(step_info_dict)
            self.latest_step_info = step_info_dict
            
            # Store simulation time
            sim_time = info_dict.get('current_time')
            self.step_sim_times.append(sim_time if sim_time is not None else None)

            if self.num_timesteps % self.print_freq == 0:
                info_bits = [f"{k}={info_dict[k]}" for k in self.step_info_keys if k in info_dict]
                info_str = f" | {'; '.join(info_bits)}" if info_bits else ""
                sim_time_str = f" | sim_time={info_dict.get('current_time'):.6f}" if info_dict.get('current_time') is not None else ""
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

            if self.save_dir and self.episode_count % self.save_freq == 0:
                self.save_training_data()
                self.add_info_snapshot()

        return True
    
    def save_training_data(self):
        """Save collected training data to files"""
        if not self.save_dir:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save episode data
        episode_data = {
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "timesteps_at_episode": self.timesteps_at_episode,
            "episode_count": self.episode_count,
            "total_timesteps": self.num_timesteps,
        }
        
        episode_file = os.path.join(self.save_dir, f"episode_data_{timestamp}.json")
        with open(episode_file, 'w') as f:
            json.dump(episode_data, f, indent=2)
        
        # Save step data if enabled (append only new rewards since last save)
        if self.save_steps and len(self.step_rewards) > self.last_saved_step_count:
            new_rewards = self.step_rewards[self.last_saved_step_count:]
            new_info_data = self.step_info_data[self.last_saved_step_count:]
            new_sim_times = self.step_sim_times[self.last_saved_step_count:]
            
            # Load existing rewards if file exists, otherwise start fresh
            if os.path.exists(self.step_rewards_file):
                existing_rewards = np.load(self.step_rewards_file)
                all_rewards = np.concatenate([existing_rewards, new_rewards])
            else:
                all_rewards = np.array(new_rewards)
            
            np.save(self.step_rewards_file, all_rewards)
            
            # Save step info data (append to existing JSON)
            if os.path.exists(self.step_info_file):
                with open(self.step_info_file, 'r') as f:
                    existing_info_data = json.load(f)
                all_info_data = existing_info_data + new_info_data
            else:
                all_info_data = new_info_data
            
            with open(self.step_info_file, 'w') as f:
                json.dump(all_info_data, f, indent=2)
            
            # Save simulation times
            if os.path.exists(self.step_sim_times_file):
                existing_sim_times = np.load(self.step_sim_times_file, allow_pickle=True)
                # Handle None values by converting to list first
                existing_list = existing_sim_times.tolist() if isinstance(existing_sim_times, np.ndarray) else existing_sim_times
                new_list = [t for t in new_sim_times]
                all_sim_times = existing_list + new_list
            else:
                all_sim_times = [t for t in new_sim_times]
            
            np.save(self.step_sim_times_file, np.array(all_sim_times, dtype=object), allow_pickle=True)
            
            self.last_saved_step_count = len(self.step_rewards)
        
        if self.verbose > 0:
            print(f"  -> Training data saved to {self.save_dir}")
    
    def _on_training_end(self):
        """Called when training ends - save final data"""
        if self.save_dir:
            self.save_training_data()
            print(f"\nFinal training data saved to {self.save_dir}")
        self.flush_step_snapshots(self.num_timesteps, final=True)

    def add_info_snapshot(self):
        """Collect a snapshot of the configured step info keys."""
        if not self.save_dir or not self.save_steps:
            return
        snapshot_payload = {
            "timestamp": datetime.now().isoformat(),
            "episode": self.episode_count,
            "num_timesteps": self.num_timesteps,
            "info": {k: self.latest_step_info.get(k) for k in self.step_info_keys} if self.latest_step_info else {},
        }
        self.custom_step_snapshots.append(snapshot_payload)

    def flush_step_snapshots(self, num_timesteps: int, final: bool = False) -> None:
        """Write accumulated custom step info snapshots to disk."""
        if not self.save_dir or not self.custom_step_snapshots:
            return

        payload = {
            "generated_at": datetime.now().isoformat(),
            "num_timesteps": num_timesteps,
            "snapshots": self.custom_step_snapshots,
        }

        with open(self.step_snapshots_file, 'w') as f:
            json.dump(payload, f, indent=2)

        if not final:
            self.custom_step_snapshots = []

    def checkpoint_hook(self, num_timesteps: int) -> None:
        """Hook executed at checkpoint save time."""
        self.flush_step_snapshots(num_timesteps)


class OverwriteCheckpointCallback(BaseCallback):
    """
    Save checkpoints at a fixed timestep frequency, overwriting the previous
    checkpoint file each time.
    """

    def __init__(
        self,
        save_freq: int = 10_000,
        save_path: str | None = None,
        filename: str = "checkpoint",
        verbose: int = 0,
        checkpoint_hooks: Optional[Iterable[Callable[[int], None]]] = None,
    ):
        super().__init__(verbose)
        if save_path is None:
            raise ValueError("save_path must be provided for OverwriteCheckpointCallback")
        self.save_freq = int(save_freq)
        self.save_path = save_path
        self.filename = filename
        self._last_checkpoint_step = 0
        self._checkpoint_file = os.path.join(self.save_path, self.filename)
        self.checkpoint_hooks = list(checkpoint_hooks) if checkpoint_hooks else []

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def _save_checkpoint(self) -> None:
        self.model.save(self._checkpoint_file)
        if self.verbose > 0:
            print(f"Checkpoint saved to {self._checkpoint_file}")
        for hook in self.checkpoint_hooks:
            try:
                hook(self.num_timesteps)
            except Exception as hook_exc:
                if self.verbose > 0:
                    print(f"Checkpoint hook error: {hook_exc}")

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_checkpoint_step >= self.save_freq:
            self._save_checkpoint()
            self._last_checkpoint_step = self.num_timesteps
        return True

