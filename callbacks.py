"""
Training callbacks for stable-baselines3
"""

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Optional
from datetime import datetime


class RewardCallback(BaseCallback):
    """
    Callback for printing reward information during training.
    Prints episode rewards and, optionally, step-level metrics.
    """
    def __init__(
        self,
        verbose: int = 0,
        print_freq: int = 5,
        step_print_interval: Optional[int] = None,
        step_info_keys: Optional[list[str]] = None,
    ):
        super(RewardCallback, self).__init__(verbose)
        self.print_freq = print_freq
        self.step_print_interval = step_print_interval
        self.step_info_keys = step_info_keys or []
        self.episode_rewards = []
        self.episode_lengths = []
        self.step_rewards = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [0])
        infos = self.locals.get('infos', [{}])

        if len(rewards) > 0:
            step_reward = float(np.mean(rewards))
            self.step_rewards.append(step_reward)

            if self.step_print_interval and self.num_timesteps % self.step_print_interval == 0:
                info_dict = infos[0] if len(infos) > 0 and isinstance(infos[0], dict) else {}
                info_bits = []
                for key in self.step_info_keys:
                    if key in info_dict:
                        info_bits.append(f"{key}={info_dict[key]}")
                info_suffix = f" | {'; '.join(info_bits)}" if info_bits else ""
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sim_time = info_dict.get("current_time", None)
                sim_time_str = f" | sim_time={sim_time:.6f}" if sim_time is not None else ""
                print(f"[{current_time}] [Step {self.num_timesteps}] reward={step_reward:.6f}{sim_time_str}{info_suffix}")

        if len(infos) > 0 and isinstance(infos[0], dict):
            if 'episode' in infos[0]:
                episode_info = infos[0]['episode']
                if episode_info is not None:
                    episode_reward = episode_info.get('r', 0)
                    episode_length = episode_info.get('l', 0)
                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)
                    self.episode_count += 1

                    print(
                        f"\n[Episode {self.episode_count}] "
                        f"Reward = {episode_reward:.4f}, "
                        f"Length = {episode_length} steps"
                    )

                    if self.episode_count % self.print_freq == 0:
                        avg_reward = np.mean(self.episode_rewards[-self.print_freq:])
                        print(f"  -> Average reward (last {self.print_freq} episodes) = {avg_reward:.4f}\n")

        return True

