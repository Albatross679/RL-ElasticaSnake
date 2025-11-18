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


class BaseContinuumSnakeEnv(gym.Env):
    """
    Shared implementation for the PyElastica Continuum Snake Gymnasium environment.
    Subclasses configure how wavelength is handled in the action space.
    """
    metadata = {"render_modes": []}

    def __init__(self, obs_keys: Optional[list] = None):
        super().__init__()

        self._n_elem = 50
        self._torque_min = 1e-3
        self._torque_max = 8e-3
        self._torque_span = self._torque_max - self._torque_min

        action_space_low, action_space_high = self._action_space_bounds()
        self.action_space = spaces.Box(low=action_space_low, high=action_space_high, dtype=np.float32)

        if obs_keys is None:
            obs_keys = ["position", "velocity", "director"]
        self.obs_keys = obs_keys

        valid_keys = [
            "time",
            "avg_position",
            "avg_velocity",
            "curvature",
            "tangents",
            "position",
            "velocity",
            "director",
        ]
        for key in self.obs_keys:
            if key not in valid_keys:
                raise ValueError(f"Invalid observation key: {key}. Valid keys are: {valid_keys}")

        obs_size = self._calculate_obs_size()
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_size,), dtype=np.float32)

        self.period = 2.0
        self.ratio_time = 11.0
        self.time_step = 1e-4
        self.rut_ratio = 1
        self.slip_velocity_tol = 1e-8

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
        self.state_dict = {
            "time": [],
            "avg_position": [],
            "avg_velocity": [],
            "curvature": [],
            "tangents": [],
            "position": [],
            "velocity": [],
            "director": [],
            "torque_coeffs": [],
            "wave_number": [],
            "forward": [],
            "lateral": [],
            "reward": [],
        }

        self._prev_com = None
        self._dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.forward = 0.0
        self.lateral = 0.0

        target = np.array([1.0, 0.0, 1.0], dtype=np.float64)
        self.target_direction = target / (np.linalg.norm(target) + 1e-12)
        self.required_alignment_steps = 15
        self.alignment_angle_tolerance = np.deg2rad(5.0)
        self._alignment_dot_threshold = np.cos(self.alignment_angle_tolerance)
        self.alignment_speed_tol = 1e-4
        self._alignment_streak = 0
        self.velocity_projection = 0.0
        self._last_velocity_vec = np.zeros(3, dtype=np.float64)

        self.reward_weights = {
            "forward_progress": 1.0,
            "lateral_penalty": 0.2,
            "curvature_penalty": 0.05,
            "energy_penalty": 2.0e4,
            "smoothness_penalty": 5.0e3,
            "alignment_bonus": 0.5,
            "streak_bonus": 1.0,
        }
        self._prev_torque_coeffs = None

        self.callback_data = defaultdict(list)

    def _action_space_bounds(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        raise NotImplementedError

    def _default_action(self) -> NDArray[np.float64]:
        raise NotImplementedError

    def _map_action_to_torque(self, action: NDArray[np.float64]) -> NDArray[np.float64]:
        action_arr = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        magnitudes = self._torque_max - self._torque_span * np.abs(action_arr)
        torques = np.sign(action_arr) * magnitudes
        torques[np.isclose(action_arr, 0.0)] = 0.0
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
        rendering_fps = 60
        step_skip = int(1.0 / (rendering_fps * self.time_step))
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
        if key == "position":
            return self.shearable_rod.position_collection.ravel()
        if key == "velocity":
            return self.shearable_rod.velocity_collection.ravel()
        if key == "director":
            return self.shearable_rod.director_collection.ravel()
        if key == "curvature":
            return self.shearable_rod.kappa.ravel()
        if key == "tangents":
            return self.shearable_rod.tangents.ravel()
        if key == "avg_position":
            return self.shearable_rod.compute_position_center_of_mass()
        if key == "avg_velocity":
            return self.shearable_rod.compute_velocity_center_of_mass()
        if key == "time":
            return np.array([self.current_time])
        raise ValueError(f"Unknown observation key: {key}")

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
                if isinstance(latest_value, np.ndarray):
                    obs_parts.append(latest_value.ravel())
                else:
                    obs_parts.append(np.array([latest_value]))
            else:
                obs_parts.append(self._get_obs_value_from_rod(key))

        return np.concatenate(obs_parts).astype(np.float32)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.current_time = 0.0
        self.state_dict = {
            "time": [],
            "avg_position": [],
            "avg_velocity": [],
            "curvature": [],
            "tangents": [],
            "position": [],
            "velocity": [],
            "director": [],
            "torque_coeffs": [],
            "wave_number": [],
            "forward": [],
            "lateral": [],
            "reward": [],
        }

        self._prev_com = None
        self._dir = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.forward = 0.0
        self.lateral = 0.0
        self.reward = 0.0
        self.velocity_projection = 0.0
        self._alignment_streak = 0
        self._last_velocity_vec = np.zeros(3, dtype=np.float64)
        self._prev_torque_coeffs = None

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
        new_dir = velocity_vec / (np.linalg.norm(velocity_vec) + 1e-12)
        alpha = 1.0
        self._dir = (1 - alpha) * self._dir + alpha * new_dir
        self._dir /= np.linalg.norm(self._dir) + 1e-12

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

        self._prev_com = prev_com_before
        self.forward, self.lateral, velocity_vec = self._forward_lateral_from_last_step(sim_time)
        self._last_velocity_vec = velocity_vec
        current_com = self.shearable_rod.compute_position_center_of_mass()
        self.velocity_projection = float(np.dot(velocity_vec, self.target_direction))

        speed = float(np.linalg.norm(velocity_vec))
        alignment = 0.0
        if speed > self.alignment_speed_tol:
            alignment = float(
                np.dot(velocity_vec / (speed + 1e-12), self.target_direction)
            )
        if speed > self.alignment_speed_tol and alignment >= self._alignment_dot_threshold:
            self._alignment_streak += 1
        else:
            self._alignment_streak = 0

        torque_coeffs = b_coeff[:-1].astype(np.float64).copy()
        forward_progress = float(np.dot(current_com - prev_com_before, self.target_direction))
        forward_term = self.reward_weights["forward_progress"] * forward_progress
        lateral_penalty = -self.reward_weights["lateral_penalty"] * float(self.lateral)
        curvature_array = self.shearable_rod.kappa
        curvature_magnitude = (
            float(np.mean(np.linalg.norm(curvature_array, axis=0))) if curvature_array.size > 0 else 0.0
        )
        curvature_penalty = -self.reward_weights["curvature_penalty"] * curvature_magnitude
        energy_penalty = -self.reward_weights["energy_penalty"] * float(np.linalg.norm(torque_coeffs) ** 2)

        smoothness_penalty = 0.0
        if self._prev_torque_coeffs is not None:
            delta_torque = torque_coeffs - self._prev_torque_coeffs
            smoothness_penalty = -self.reward_weights["smoothness_penalty"] * float(
                np.linalg.norm(delta_torque) ** 2
            )

        alignment_bonus = 0.0
        if speed > self.alignment_speed_tol and alignment > 0.0:
            alignment_bonus += self.reward_weights["alignment_bonus"] * alignment
            if self._alignment_streak >= self.required_alignment_steps:
                alignment_bonus += self.reward_weights["streak_bonus"]

        reward_terms = {
            "forward_progress": float(forward_term),
            "lateral_penalty": float(lateral_penalty),
            "curvature_penalty": float(curvature_penalty),
            "energy_penalty": float(energy_penalty),
            "smoothness_penalty": float(smoothness_penalty),
            "alignment_bonus": float(alignment_bonus),
        }

        self.reward = float(sum(reward_terms.values()))
        reward = float(self.reward)
        self._prev_torque_coeffs = torque_coeffs.copy()

        self.state_dict["torque_coeffs"].append(torque_coeffs.copy())
        self.state_dict["wave_number"].append(2.0 * np.pi / b_coeff[-1])
        self.state_dict["forward"].append(float(self.forward))
        self.state_dict["lateral"].append(float(self.lateral))
        self.state_dict["reward"].append(float(self.reward))

        max_sim_time = 300
        terminated_due_to_time = self.current_time >= max_sim_time
        terminated_due_to_alignment = self._alignment_streak >= self.required_alignment_steps
        terminated = terminated_due_to_time or terminated_due_to_alignment
        truncated = False

        observation = self._get_obs()

        info = {}
        info["reward"] = float(reward)
        info["reward_terms"] = reward_terms
        info["forward_speed"] = float(self.forward)
        info["lateral_speed"] = float(self.lateral)
        info["velocity_projection"] = float(self.velocity_projection)
        info["forward_progress"] = forward_progress
        info["speed"] = speed
        info["alignment"] = alignment
        info["alignment_streak"] = int(self._alignment_streak)
        info["alignment_goal_met"] = bool(terminated_due_to_alignment)
        info["position"] = current_com.astype(float)
        info["heading_dir"] = self._dir.astype(float)
        info["current_time"] = float(self.current_time)

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

