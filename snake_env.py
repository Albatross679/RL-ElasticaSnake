"""
Snake Environment for Reinforcement Learning
Contains the Gymnasium environment classes for the continuum snake.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import elastica as ea
from collections import defaultdict
from typing import Optional
from numpy.typing import NDArray
from elastica.typing import RodType

# Constants
CURVATURE_HISTORY_WINDOW = 50
CURVATURE_MIN = 9.0
CURVATURE_MAX = 15.0
RENDERING_FPS = 60


class RewardCalculator:
    """
    Base class for reward calculation in the continuum snake environment.
    Subclasses can override methods to customize reward schemes.
    """
    
    # Default reward weights (negative values for penalties, positive for rewards/bonuses)
    DEFAULT_REWARD_WEIGHTS = {
        "forward_progress": 1.0,
        "lateral_penalty": -1.0,
        "curvature_range_penalty": -0.1,
        "curvature_oscillation_reward": 1.0,
        "energy_penalty": -2.0e4,
        "smoothness_penalty": -5.0e3,
        "alignment_bonus": 0.5,
        "streak_bonus": 100.0,
        "projected_speed": 2.0,
        "lateral_speed_penalty": -1.0,  # Penalty for speed perpendicular to target direction
    }
    
    def __init__(self):
        """Initialize the reward calculator with default reward weights."""
        self.reward_weights = self.DEFAULT_REWARD_WEIGHTS.copy()
    
    def calculate_reward(
        self,
        forward_progress: float,
        lateral: float,
        curvature_array: NDArray[np.float64],
        curvature_history: list[NDArray[np.float64]],
        prev_curvature_array: Optional[NDArray[np.float64]],
        torque_coeffs: NDArray[np.float64],
        prev_torque_coeffs: Optional[NDArray[np.float64]],
        speed: float,
        alignment: float,
        alignment_streak: int,
        required_alignment_steps: int,
        alignment_speed_tol: float,
        velocity_projection: float,
        lateral_speed_perpendicular: float,
    ) -> tuple[float, dict]:
        """
        Calculate the total reward and individual reward terms.
        
        Args:
            forward_progress: Progress in the target direction
            lateral: Lateral movement magnitude
            curvature_array: Current curvature array (3, n_elem-1)
            curvature_history: List of curvature arrays from last CURVATURE_HISTORY_WINDOW steps
            prev_curvature_array: Previous step curvature array (for oscillation)
            torque_coeffs: Current torque coefficients
            prev_torque_coeffs: Previous torque coefficients (for smoothness)
            speed: Current speed magnitude
            alignment: Alignment with target direction (dot product)
            alignment_streak: Current alignment streak count
            required_alignment_steps: Required steps for alignment bonus
            alignment_speed_tol: Minimum speed for alignment calculations
            velocity_projection: Projection of velocity onto target direction
            lateral_speed_perpendicular: Speed component perpendicular to target direction
            
        Returns:
            Tuple of (total_reward, reward_terms_dict)
        """
        reward_terms = {}
        
        # Forward progress reward
        reward_terms["forward_progress"] = self._calculate_forward_progress(forward_progress)
        
        # Lateral penalty
        reward_terms["lateral_penalty"] = self._calculate_lateral_penalty(lateral)
        
        # Curvature range penalty (based on instantaneous curvature)
        reward_terms["curvature_range_penalty"] = self._calculate_curvature_range_penalty(
            curvature_array
        )
        
        # Curvature oscillation reward
        reward_terms["curvature_oscillation_reward"] = self._calculate_curvature_oscillation_reward(
            curvature_array, prev_curvature_array
        )
        
        # Energy penalty (only included if weight is non-zero)
        energy_penalty = self._calculate_energy_penalty(torque_coeffs)
        if energy_penalty != 0.0:
            reward_terms["energy_penalty"] = energy_penalty
        
        # Smoothness penalty
        reward_terms["smoothness_penalty"] = self._calculate_smoothness_penalty(
            torque_coeffs, prev_torque_coeffs
        )
        
        # Alignment bonus
        reward_terms["alignment_bonus"] = self._calculate_alignment_bonus(
            speed, alignment, alignment_streak, required_alignment_steps, alignment_speed_tol
        )
        
        # Projected speed reward
        reward_terms["projected_speed"] = self._calculate_projected_speed(velocity_projection)
        
        # Lateral speed penalty (perpendicular to target direction)
        reward_terms["lateral_speed_penalty"] = self._calculate_lateral_speed_penalty(lateral_speed_perpendicular)
        
        total_reward = sum(reward_terms.values())
        return float(total_reward), reward_terms
    
    def _calculate_forward_progress(self, forward_progress: float) -> float:
        """Calculate forward progress reward component."""
        return self.reward_weights["forward_progress"] * forward_progress
    
    def _calculate_lateral_penalty(self, lateral: float) -> float:
        """Calculate lateral movement penalty."""
        return self.reward_weights["lateral_penalty"] * lateral
    
    def _calculate_curvature_range_penalty(
        self, curvature_array: NDArray[np.float64]
    ) -> float:
        """
        Calculate curvature range penalty based on instantaneous curvature.
        If curvature of each element stays within [CURVATURE_MIN, CURVATURE_MAX], do nothing.
        If curvature goes beyond [CURVATURE_MIN, CURVATURE_MAX], penalize proportional to the difference.
        """
        # Calculate magnitude for each element: shape (n_elem-1,)
        curvature_magnitudes = np.linalg.norm(curvature_array, axis=0)
        
        # Penalize if instantaneous curvature is outside [CURVATURE_MIN, CURVATURE_MAX]
        penalty = 0.0
        for curv_mag in curvature_magnitudes:
            if curv_mag < CURVATURE_MIN:
                penalty += (CURVATURE_MIN - curv_mag)
            elif curv_mag > CURVATURE_MAX:
                penalty += (curv_mag - CURVATURE_MAX)
        
        return self.reward_weights["curvature_range_penalty"] * penalty
    
    def _calculate_curvature_oscillation_reward(
        self,
        curvature_array: NDArray[np.float64],
        prev_curvature_array: Optional[NDArray[np.float64]],
    ) -> float:
        """
        Calculate curvature oscillation reward.
        Sum of absolute differences of curvatures of each element from previous step.
        """
        if prev_curvature_array is None:
            return 0.0
        
        # Calculate magnitude for each element: shape (n_elem-1,)
        current_magnitudes = np.linalg.norm(curvature_array, axis=0)
        prev_magnitudes = np.linalg.norm(prev_curvature_array, axis=0)
        
        # Calculate absolute differences and sum
        abs_differences = np.abs(current_magnitudes - prev_magnitudes)
        oscillation_sum = np.sum(abs_differences)
        
        return self.reward_weights["curvature_oscillation_reward"] * oscillation_sum
    
    def _calculate_energy_penalty(self, torque_coeffs: NDArray[np.float64]) -> float:
        """Calculate energy penalty based on torque magnitude."""
        weight = self.reward_weights["energy_penalty"]
        if weight == 0.0:
            return 0.0
        return weight * np.linalg.norm(torque_coeffs) ** 2
    
    def _calculate_smoothness_penalty(
        self,
        torque_coeffs: NDArray[np.float64],
        prev_torque_coeffs: Optional[NDArray[np.float64]],
    ) -> float:
        """Calculate smoothness penalty based on torque changes."""
        if prev_torque_coeffs is None:
            return 0.0
        delta_torque = torque_coeffs - prev_torque_coeffs
        return self.reward_weights["smoothness_penalty"] * np.linalg.norm(delta_torque) ** 2
    
    def _calculate_alignment_bonus(
        self,
        speed: float,
        alignment: float,
        alignment_streak: int,
        required_alignment_steps: int,
        alignment_speed_tol: float,
    ) -> float:
        """Calculate alignment bonus."""
        if speed <= alignment_speed_tol or alignment <= 0.0:
            return 0.0
        
        bonus = self.reward_weights["alignment_bonus"] * alignment
        
        if alignment_streak >= required_alignment_steps:
            bonus += self.reward_weights["streak_bonus"]
        
        return bonus
    
    def _calculate_projected_speed(self, velocity_projection: float) -> float:
        """Calculate projected speed reward."""
        return self.reward_weights["projected_speed"] * velocity_projection
    
    def _calculate_lateral_speed_penalty(self, lateral_speed_perpendicular: float) -> float:
        """Calculate penalty for lateral speed perpendicular to target direction."""
        return self.reward_weights["lateral_speed_penalty"] * lateral_speed_perpendicular


class BaseContinuumSnakeEnv(gym.Env):
    """
    Shared implementation for the PyElastica Continuum Snake Gymnasium environment.
    Subclasses configure how wavelength is handled in the action space.
    """
    metadata = {"render_modes": []}

    def __init__(self, obs_keys: Optional[list] = None):
        super().__init__()

        self._n_elem = 10
        self._torque_min = 1e-3
        self._torque_max = 8e-3
        self._torque_span = self._torque_max - self._torque_min

        action_space_low, action_space_high = self._action_space_bounds()
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)

        if obs_keys is None:
            obs_keys = ["position", "velocity", "director"]
        self.obs_keys = obs_keys

        valid_keys = {
            "time", "avg_position", "avg_velocity", "curvature",
            "tangents", "position", "velocity", "director",
            "relative_position",
        }
        invalid_keys = [key for key in self.obs_keys if key not in valid_keys]
        if invalid_keys:
            raise ValueError(f"Invalid observation keys: {invalid_keys}. Valid keys are: {sorted(valid_keys)}")

        obs_size = self._calculate_obs_size()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        self.period = 2.0
        self.ratio_time = 11.0
        self.time_step = 1e-4
        self.rut_ratio = 1
        self.slip_velocity_tol = 1e-8
        self.max_episode_length = 300  # Maximum simulation time (seconds) before episode termination

        self.start = np.zeros((3,))
        self.direction = np.array([0.0, 0.0, 1.0])
        self.normal = np.array([0.0, 1.0, 0.0])
        self.base_length = 0.35
        self.base_radius = self.base_length * 0.011
        self.density = 1000
        self.E = 1e6
        self.poisson_ratio = 0.5
        self.shear_modulus = self.E / (1.0 + self.poisson_ratio)

        self.timestepper = ea.PositionVerlet()

        self.current_time = 0.0
        self.state_dict = self._create_state_dict()

        self._prev_com = None
        self._dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.forward = 0.0
        self.lateral = 0.0

        target = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        self.target_direction = target / (np.linalg.norm(target) + 1e-12)
        self.required_alignment_steps = 50
        self.alignment_angle_tolerance = np.deg2rad(5.0)
        self._alignment_dot_threshold = np.cos(self.alignment_angle_tolerance)
        self.alignment_speed_tol = 1e-4
        self._alignment_streak = 0
        self.velocity_projection = 0.0
        self.lateral_speed_perpendicular = 0.0  # Speed perpendicular to target direction
        self.reward = 0.0  # Initialize reward attribute

        # Initialize reward calculator with default weights
        self.reward_calculator = RewardCalculator()
        self._prev_torque_coeffs = None
        self._curvature_history = []  # Store last CURVATURE_HISTORY_WINDOW steps of curvature
        self._prev_curvature_array = None  # Previous step curvature for oscillation

        self.callback_data = defaultdict(list)

    def _create_state_dict(self) -> dict:
        """Create a new state dictionary with empty lists for all keys."""
        return {
            "time": [],
            "avg_position": [],
            "avg_velocity": [],
            "curvature": [],
            "tangents": [],
            "position": [],
            "velocity": [],
            "director": [],
            "relative_position": [],
            "torque_coeffs": [],
            "wave_number": [],
            "forward": [],
            "lateral": [],
            "reward": [],
        }

    @property
    def reward_weights(self) -> dict:
        """Get the current reward weights."""
        return self.reward_calculator.reward_weights.copy()
    
    @reward_weights.setter
    def reward_weights(self, value: dict):
        """Set reward weights and update the reward calculator."""
        self.reward_calculator.reward_weights = self.reward_calculator.DEFAULT_REWARD_WEIGHTS.copy()
        if value is not None:
            self.reward_calculator.reward_weights.update(value)

    def _action_space_bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        raise NotImplementedError

    def _default_action(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def _map_action_to_torque(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        action_arr = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        # Linear mapping: 0 -> 0 torque, 1 -> max torque
        torques = action_arr * self._torque_max
        return torques.astype(np.float64)

    def _map_torque_to_action(self, torque: NDArray[np.float64]) -> NDArray[np.float64]:
        torque_arr = np.asarray(torque, dtype=np.float64)
        magnitudes = np.abs(torque_arr)
        clipped = np.clip(magnitudes, 0.0, self._torque_max)
        abs_action = (self._torque_max - clipped) / self._torque_span
        abs_action = np.clip(abs_action, 0.0, 1.0)
        return (abs_action * np.sign(torque_arr)).astype(np.float64)

    def _augment_action(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        raise NotImplementedError

    @staticmethod
    def _default_torque_coeffs() -> NDArray[np.float64]:
        return np.array([3.4e-3, 3.3e-3, 4.2e-3, 2.6e-3, 3.6e-3, 3.5e-3], dtype=np.float64)

    def _calculate_obs_size(self) -> int:
        n_elem = self._n_elem
        n_nodes = n_elem + 1
        key_sizes = {
            "time": 1,
            "avg_position": 3,
            "avg_velocity": 3,
            "curvature": (n_elem - 1) * 3,
            "tangents": n_elem * 3,
            "position": n_nodes * 3,
            "velocity": n_nodes * 3,
            "director": n_elem * 9,
            "relative_position": n_nodes * 3,
        }
        return sum(key_sizes[key] for key in self.obs_keys)

    def _transition_model(self, b_coeff: NDArray[np.float64]):
        class SnakeSimulator(
            ea.BaseSystemCollection,
            ea.Constraints,
            ea.Forcing,
            ea.Damping,
            ea.CallBacks,
            ea.Contact,
        ):
            pass

        snake_sim = SnakeSimulator()
        snake_sim.append(self.shearable_rod)

        gravitational_acc = -9.80665
        snake_sim.add_forcing_to(self.shearable_rod).using(
            ea.GravityForces, acc_gravity=np.array([0.0, gravitational_acc, 0.0])
        )

        wave_length = b_coeff[-1]
        # Prevent division by zero - ensure minimum wavelength
        if abs(wave_length) < 1e-12:
            wave_length = 1e-12 if wave_length >= 0 else -1e-12
        b_coeff_only = b_coeff[:-1]
        snake_sim.add_forcing_to(self.shearable_rod).using(
            ea.MuscleTorques,
            base_length=self.base_length,
            b_coeff=b_coeff_only,
            period=self.period,
            wave_number=2.0 * np.pi / wave_length,
            phase_shift=0.0,
            rest_lengths=self.shearable_rod.rest_lengths,
            ramp_up_time=self.rut_ratio * self.period,
            direction=self.normal,
            with_spline=True,
        )

        ground_plane = ea.Plane(
            plane_origin=np.array([0.0, -self.base_radius, 0.0]),
            plane_normal=self.normal,
        )
        snake_sim.append(ground_plane)
        froude = 0.1
        mu = self.base_length / (self.period * self.period * np.abs(gravitational_acc) * froude)
        kinetic_mu_array = np.array([mu, 1.5 * mu, 2.0 * mu])
        static_mu_array = np.zeros(kinetic_mu_array.shape)
        snake_sim.detect_contact_between(self.shearable_rod, ground_plane).using(
            ea.RodPlaneContactWithAnisotropicFriction,
            k=1.0,
            nu=1e-6,
            slip_velocity_tol=self.slip_velocity_tol,
            static_mu_array=static_mu_array,
            kinetic_mu_array=kinetic_mu_array,
        )

        damping_constant = 2e-3
        snake_sim.dampen(self.shearable_rod).using(
            ea.AnalyticalLinearDamper,
            damping_constant=damping_constant,
            time_step=self.time_step,
        )

        if not hasattr(self, "callback_data") or self.callback_data is None:
            self.callback_data = defaultdict(list)
        step_skip = int(1.0 / (RENDERING_FPS * self.time_step))
        slip_velocity_tol = self.slip_velocity_tol

        class ContinuumSnakeCallBack(ea.CallBackBaseClass):
            def __init__(self, step_skip: int, callback_params: dict) -> None:
                ea.CallBackBaseClass.__init__(self)
                self.every = step_skip
                self.callback_params = callback_params

            def make_callback(
                self, system: ea.CosseratRod, time: float, current_step: int
            ) -> None:
                if current_step == 0:
                    return

                if current_step % self.every == 0:
                    self.callback_params["time"].append(time)
                    self.callback_params["step"].append(current_step)
                    self.callback_params["position"].append(
                        system.position_collection.copy()
                    )
                    self.callback_params["velocity"].append(
                        system.velocity_collection.copy()
                    )
                    self.callback_params["avg_velocity"].append(
                        system.compute_velocity_center_of_mass()
                    )
                    self.callback_params["center_of_mass"].append(
                        system.compute_position_center_of_mass()
                    )
                    self.callback_params["curvature"].append(system.kappa.copy())
                    self.callback_params["tangents"].append(system.tangents.copy())
                    self.callback_params["friction"].append(
                        self.get_slip_velocity(system).copy()
                    )
                    return

            def get_slip_velocity(self, system: RodType) -> NDArray[np.float64]:
                from elastica.contact_utils import (
                    _find_slipping_elements,
                    _node_to_element_velocity,
                )
                from elastica._linalg import _batch_dot, _batch_product_k_ik_to_ik

                axial_direction = system.tangents
                element_velocity = _node_to_element_velocity(
                    mass=system.mass, node_velocity_collection=system.velocity_collection
                )
                velocity_mag_along_axial_direction = _batch_dot(
                    element_velocity, axial_direction
                )
                velocity_along_axial_direction = _batch_product_k_ik_to_ik(
                    velocity_mag_along_axial_direction, axial_direction
                )
                return _find_slipping_elements(
                    velocity_along_axial_direction, slip_velocity_tol
                )

        snake_sim.collect_diagnostics(self.shearable_rod).using(
            ContinuumSnakeCallBack,
            step_skip=step_skip,
            callback_params=self.callback_data,
        )

        snake_sim.finalize()
        return snake_sim

    def _get_obs_value_from_rod(self, key: str):
        """Get observation value from rod using dictionary lookup."""
        rod = self.shearable_rod
        obs_map = {
            "position": lambda: rod.position_collection.ravel(),
            "velocity": lambda: rod.velocity_collection.ravel(),
            "director": lambda: rod.director_collection.ravel(),
            "curvature": lambda: rod.kappa.ravel(),
            "tangents": lambda: rod.tangents.ravel(),
            "avg_position": lambda: rod.compute_position_center_of_mass(),
            "avg_velocity": lambda: rod.compute_velocity_center_of_mass(),
            "time": lambda: np.array([self.current_time]),
            "relative_position": lambda: (rod.position_collection - rod.compute_position_center_of_mass().reshape(3, 1)).ravel(),
        }
        if key not in obs_map:
            raise ValueError(f"Unknown observation key: {key}")
        return obs_map[key]()

    def _get_obs(self):
        obs_parts = []
        use_state_dict = (
            len(self.obs_keys) > 0
            and self.obs_keys[0] in self.state_dict
            and len(self.state_dict[self.obs_keys[0]]) > 0
        )

        for key in self.obs_keys:
            if use_state_dict and key in self.state_dict and len(self.state_dict[key]) > 0:
                latest_value = self.state_dict[key][-1]
                value = latest_value.ravel() if isinstance(latest_value, np.ndarray) else np.array([latest_value])
                obs_parts.append(value)
            else:
                obs_parts.append(self._get_obs_value_from_rod(key))

        return np.concatenate(obs_parts).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_time = 0.0
        self.state_dict = self._create_state_dict()

        # Reset state variables
        self._prev_com = None
        self._dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.forward = 0.0
        self.lateral = 0.0
        self.reward = 0.0
        self.velocity_projection = 0.0
        self.lateral_speed_perpendicular = 0.0
        self._alignment_streak = 0
        self._prev_torque_coeffs = None
        self._curvature_history = []
        self._prev_curvature_array = None
        self.callback_data = defaultdict(list)

        self.shearable_rod = ea.CosseratRod.straight_rod(
            self._n_elem,
            self.start,
            self.direction,
            self.normal,
            self.base_length,
            self.base_radius,
            self.density,
            youngs_modulus=self.E,
            shear_modulus=self.shear_modulus,
        )

        reset_action = self._default_action()
        if options is not None and "action" in options:
            reset_action = np.asarray(options["action"], dtype=np.float64)
        else:
            reset_action = np.asarray(reset_action, dtype=np.float64)
        b_coeff = self._augment_action(reset_action)

        self.sim = self._transition_model(b_coeff)

        observation = self._get_obs()
        info = {}
        return observation, info

    def _forward_lateral_from_last_step(
        self, sim_time: float
    ) -> tuple[float, float, NDArray[np.float64]]:
        com = self.shearable_rod.compute_position_center_of_mass()
        if self._prev_com is None:
            self._prev_com = com.copy()
            return 0.0, 0.0, np.zeros(3, dtype=np.float64)

        disp = com - self._prev_com
        velocity_vec = disp / (sim_time + 1e-12)
        self._dir = velocity_vec / (np.linalg.norm(velocity_vec) + 1e-12)

        forward = float(np.dot(velocity_vec, self._dir))
        lateral = float(
            np.linalg.norm(velocity_vec - np.dot(velocity_vec, self._dir) * self._dir)
        )

        self._prev_com = com.copy()
        return forward, lateral, velocity_vec

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        b_coeff = self._augment_action(action)

        sim_time = (self.ratio_time + 0.001) * self.period
        total_steps = int(sim_time / self.time_step)

        self.sim = self._transition_model(b_coeff)

        prev_com_before = (
            self.shearable_rod.compute_position_center_of_mass().copy()
            if self._prev_com is None
            else self._prev_com.copy()
        )

        self.current_time = ea.integrate(
            self.timestepper,
            self.sim,
            sim_time,
            total_steps,
            self.current_time,
            progress_bar=False,
        )

        self.state_dict["time"].append(self.current_time)
        self.state_dict["avg_position"].append(
            self.shearable_rod.compute_position_center_of_mass()
        )
        self.state_dict["avg_velocity"].append(
            self.shearable_rod.compute_velocity_center_of_mass()
        )
        self.state_dict["curvature"].append(self.shearable_rod.kappa.copy())
        self.state_dict["tangents"].append(self.shearable_rod.tangents.copy())
        self.state_dict["position"].append(
            self.shearable_rod.position_collection.copy()
        )
        self.state_dict["velocity"].append(
            self.shearable_rod.velocity_collection.copy()
        )
        self.state_dict["director"].append(
            self.shearable_rod.director_collection.copy()
        )
        com = self.shearable_rod.compute_position_center_of_mass()
        relative_positions = self.shearable_rod.position_collection - com.reshape(3, 1)
        self.state_dict["relative_position"].append(relative_positions.copy())

        self._prev_com = prev_com_before
        self.forward, self.lateral, velocity_vec = self._forward_lateral_from_last_step(sim_time)
        current_com = self.shearable_rod.compute_position_center_of_mass()
        self.velocity_projection = float(np.dot(velocity_vec, self.target_direction))
        
        # Calculate lateral speed perpendicular to target direction
        lateral_velocity = velocity_vec - self.velocity_projection * self.target_direction
        self.lateral_speed_perpendicular = float(np.linalg.norm(lateral_velocity))

        speed = float(np.linalg.norm(velocity_vec))
        alignment = 0.0
        if speed > self.alignment_speed_tol:
            alignment = float(np.dot(velocity_vec / (speed + 1e-12), self.target_direction))
            if alignment >= self._alignment_dot_threshold:
                self._alignment_streak += 1
            else:
                self._alignment_streak = 0
        else:
            self._alignment_streak = 0

        torque_coeffs = b_coeff[:-1].astype(np.float64).copy()
        forward_progress = float(np.dot(current_com - prev_com_before, self.target_direction))
        curvature_array = self.shearable_rod.kappa.copy()
        
        # Update curvature history (keep last CURVATURE_HISTORY_WINDOW steps)
        self._curvature_history.append(curvature_array)
        if len(self._curvature_history) > CURVATURE_HISTORY_WINDOW:
            self._curvature_history.pop(0)

        # Calculate reward using the reward calculator
        reward, reward_terms = self.reward_calculator.calculate_reward(
            forward_progress=forward_progress,
            lateral=self.lateral,
            curvature_array=curvature_array,
            curvature_history=self._curvature_history,
            prev_curvature_array=self._prev_curvature_array,
            torque_coeffs=torque_coeffs,
            prev_torque_coeffs=self._prev_torque_coeffs,
            speed=speed,
            alignment=alignment,
            alignment_streak=self._alignment_streak,
            required_alignment_steps=self.required_alignment_steps,
            alignment_speed_tol=self.alignment_speed_tol,
            velocity_projection=self.velocity_projection,
            lateral_speed_perpendicular=self.lateral_speed_perpendicular,
        )

        self.reward = reward
        self._prev_torque_coeffs = torque_coeffs.copy()
        self._prev_curvature_array = curvature_array.copy()

        self.state_dict["torque_coeffs"].append(torque_coeffs.copy())
        wave_length_for_state = b_coeff[-1]
        if abs(wave_length_for_state) < 1e-12:
            wave_length_for_state = 1e-12 if wave_length_for_state >= 0 else -1e-12
        self.state_dict["wave_number"].append(2.0 * np.pi / wave_length_for_state)
        self.state_dict["forward"].append(float(self.forward))
        self.state_dict["lateral"].append(float(self.lateral))
        self.state_dict["reward"].append(float(self.reward))

        terminated_due_to_time = self.current_time >= self.max_episode_length
        terminated_due_to_alignment = self._alignment_streak >= self.required_alignment_steps
        terminated = terminated_due_to_time or terminated_due_to_alignment
        truncated = False

        observation = self._get_obs()

        info = {
            "reward": float(reward),
            "reward_terms": reward_terms,
            "forward_speed": float(self.forward),
            "lateral_speed": float(self.lateral),
            "lateral_speed_perpendicular": float(self.lateral_speed_perpendicular),
            "velocity_projection": float(self.velocity_projection),
            "forward_progress": forward_progress,
            "speed": speed,
            "alignment": alignment,
            "alignment_streak": int(self._alignment_streak),
            "alignment_goal_met": bool(terminated_due_to_alignment),
            "position": current_com.astype(float),
            "heading_dir": self._dir.astype(float),
            "current_time": float(self.current_time),
            "curvatures": curvature_array.astype(float).tolist(),  # For JSON serialization
            "action": action.astype(float).tolist(),  # Raw action for logging
        }

        return observation, reward, terminated, truncated, info

    def close(self):
        pass


class FixedWavelengthContinuumSnakeEnv(BaseContinuumSnakeEnv):
    def __init__(self, fixed_wavelength: float, obs_keys: Optional[list] = None):
        self.fixed_wavelength = float(fixed_wavelength)
        super().__init__(obs_keys=obs_keys)

    def _action_space_bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        low = -np.ones(6, dtype=np.float32)
        high = np.ones(6, dtype=np.float32)
        return low, high

    def _default_action(self) -> NDArray[np.float64]:
        torques = self._default_torque_coeffs()
        return self._map_torque_to_action(torques)

    def _augment_action(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        if action.shape[-1] != 6:
            raise ValueError("Expected 6 torque coefficients for fixed-wavelength control.")
        torques = self._map_action_to_torque(action)
        return np.append(torques, self.fixed_wavelength)


class VariableWavelengthContinuumSnakeEnv(BaseContinuumSnakeEnv):
    def __init__(
        self,
        obs_keys: Optional[list] = None,
        wavelength_bounds: tuple[float, float] = (-1.5, 1.5),
        default_wavelength: float = 1.0,
    ):
        self.wavelength_bounds = wavelength_bounds
        self.default_wavelength = default_wavelength
        super().__init__(obs_keys=obs_keys)

    def _action_space_bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        low = -np.ones(7, dtype=np.float32)
        high = np.ones(7, dtype=np.float32)
        low[-1] = np.float32(self.wavelength_bounds[0])
        high[-1] = np.float32(self.wavelength_bounds[1])
        return low, high

    def _default_action(self) -> NDArray[np.float64]:
        torques = self._default_torque_coeffs()
        torque_actions = self._map_torque_to_action(torques)
        return np.append(torque_actions, self.default_wavelength)

    def _augment_action(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        if action.shape[-1] != 7:
            raise ValueError(
                "Expected 7 action dimensions (6 torque coefficients + wavelength)."
            )
        action = np.asarray(action, dtype=np.float64).copy()
        torques = self._map_action_to_torque(action[:-1])
        min_w, max_w = self.wavelength_bounds
        wavelength = float(np.clip(action[-1], min_w, max_w))
        return np.append(torques, wavelength)


class FixedWavelengthXZOnlyContinuumSnakeEnv(FixedWavelengthContinuumSnakeEnv):
    """
    Variant of FixedWavelengthContinuumSnakeEnv that filters out y-components
    (vertical axis) from most spatial observations. Only x and z components are
    included for position, velocity, avg_position, avg_velocity, curvature, 
    tangents, and relative_position. Director is kept in full 3D.
    """
    
    def _calculate_obs_size(self) -> int:
        """Calculate observation size with 2D (x,z) for most spatial keys, director kept in 3D."""
        n_elem = self._n_elem
        n_nodes = n_elem + 1
        # Most spatial keys are 2D (x,z only), director is kept in full 3D
        # Director is 3x3 matrices, so we keep 2 rows (x,z) -> 2*3 = 6 per element
        key_sizes = {
            "time": 1,
            "avg_position": 2,  # 2D (x,z)
            "avg_velocity": 2,  # 2D (x,z)
            "curvature": (n_elem - 1) * 2,  # 2D (x,z)
            "tangents": n_elem * 2,  # 2D (x,z)
            "position": n_nodes * 2,  # 2D (x,z)
            "velocity": n_nodes * 2,  # 2D (x,z)
            "director": n_elem * 9,  # Full 3x3 matrix = 9 per element (kept in 3D)
            "relative_position": n_nodes * 2,  # 2D (x,z)
        }
        return sum(key_sizes[key] for key in self.obs_keys)
    
    def _filter_2d_value(self, key: str, value: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Filter out y-component (index 1) from observation values for all spatial keys.
        Director is kept in 3D even in 2D mode.
        """
        # All spatial keys are filtered to 2D (x,z) except director
        if key == "time":
            return value  # Scalar, no filtering needed
        
        if key == "director":
            # Shape: (3, 3, n_elem) -> keep full 3D -> flatten
            return value.ravel()
        elif key in {"avg_position", "avg_velocity"}:
            # Shape: (3,) -> select indices 0,2 (x,z)
            return value[[0, 2]]
        else:
            # Shape: (3, n) -> select rows 0,2 (x,z) -> flatten
            # Applies to: position, velocity, curvature, tangents, relative_position
            return value[[0, 2], :].ravel()
    
    def _get_obs(self):
        """
        Override to filter state_dict values when using them for 2D observations.
        """
        obs_parts = []
        use_state_dict = (
            len(self.obs_keys) > 0
            and self.obs_keys[0] in self.state_dict
            and len(self.state_dict[self.obs_keys[0]]) > 0
        )

        for key in self.obs_keys:
            if use_state_dict and key in self.state_dict and len(self.state_dict[key]) > 0:
                latest_value = self.state_dict[key][-1]
                if isinstance(latest_value, np.ndarray):
                    value = self._filter_2d_value(key, latest_value).ravel()
                else:
                    value = np.array([latest_value])
                obs_parts.append(value)
            else:
                obs_parts.append(self._get_obs_value_from_rod(key))

        return np.concatenate(obs_parts).astype(np.float32)
    
    def _get_obs_value_from_rod(self, key: str):
        """
        Get observation value from rod, filtering out y-component (index 1)
        for all spatial keys to make them 2D (x,z only).
        """
        rod = self.shearable_rod
        
        # Handle scalar key
        if key == "time":
            return np.array([self.current_time])
        
        # Handle 1D spatial keys (avg_position, avg_velocity)
        if key == "avg_position":
            return rod.compute_position_center_of_mass()[[0, 2]]
        elif key == "avg_velocity":
            return rod.compute_velocity_center_of_mass()[[0, 2]]
        
        # Handle 2D spatial keys (position, velocity, curvature, tangents, relative_position)
        elif key in {"position", "velocity", "curvature", "tangents", "relative_position"}:
            if key == "relative_position":
                com = rod.compute_position_center_of_mass()
                relative_positions = rod.position_collection - com.reshape(3, 1)
                return relative_positions[[0, 2], :].ravel()
            else:
                attr_map = {
                    "position": rod.position_collection,
                    "velocity": rod.velocity_collection,
                    "curvature": rod.kappa,
                    "tangents": rod.tangents,
                }
                return attr_map[key][[0, 2], :].ravel()
        
        # Handle director (3D matrix per element)
        elif key == "director":
            # Shape: (3, 3, n_elem) -> select rows 0,2 (x,z) -> (2, 3, n_elem) -> flatten
            return rod.director_collection[[0, 2], :, :].ravel()
        
        raise ValueError(f"Unknown observation key: {key}")

