"""Road-following randomized behavior policy, NOT a learned world model.

The policy uses privileged simulator road geometry. It is intended to generate
varied trajectories for offline model training, not to guarantee safety.
Steering convention is Gymnasium's: negative=left, positive=right.
"""

from __future__ import annotations

import math

import numpy as np

from env.road_geometry import RoadContext, TrackGeometry, wrap_angle

MODE_NORMAL = 0
MODE_RECOVERY = 1
MODE_SPIN_PREVENTION = 2
MODE_BABBLE = 3
MODE_NAMES = ("normal", "boundary_recovery", "spin_prevention", "babble")


class RandomRoadAgent:
    """Follow the road with seeded, correlated steering exploration."""

    def __init__(
        self,
        geometry: TrackGeometry,
        rng: np.random.Generator,
        exploration: float = 0.08,
        max_speed: float = 48.0,
        corner_speed: float = 16.0,
        safe_exploration: bool = True,
        boundary_recovery: bool = True,
        speed_scale: tuple[float, float] = (0.96, 1.0),
        babble_prob: float = 0.0,
        babble_steps: tuple[int, int] = (5, 20),
        babble_amp: float = 1.0,
    ):
        self.geometry = geometry
        self.rng = rng
        self.exploration = float(exploration)
        self.noise_state = 0.0
        self.speed_scale = float(rng.uniform(*speed_scale))
        self.max_speed = float(max_speed)
        self.corner_speed = float(corner_speed)
        self.safe_exploration = bool(safe_exploration)
        self.boundary_recovery = bool(boundary_recovery)
        self.babble_prob = float(babble_prob)
        self.babble_steps = babble_steps
        self.babble_amp = float(babble_amp)
        self.babble_left = 0
        self.babble_action = np.zeros(3, dtype=np.float32)

    def act(self, state: np.ndarray, road: RoadContext) -> tuple[np.ndarray, int, dict]:
        yaw_rate = state[5]
        curvature = state[10]
        speed = state[11]
        heading_to_target_error, distance = self.steering_target(state, road)
        desired_speed, anticipated_curvature = self.plan_speed(road.progress, speed)
        distance_to_edge = min(state[8], state[9])
        noise = self.steering_noise(distance_to_edge, speed)

        # Pure pursuit with yaw-rate damping to counter rotation.
        steering = math.atan2(2.0 * 4.8 * math.sin(heading_to_target_error), distance)
        steering += 0.10 * (yaw_rate - speed * curvature) + noise
        speed_error = desired_speed - speed
        gas = float(np.clip(0.18 * speed_error, 0.0, 1.0))
        brake = float(np.clip(-0.10 * speed_error, 0.0, 0.75))
        steering, gas, brake, mode = self.apply_safety_controls(
            state, heading_to_target_error, steering, gas, brake
        )

        action = np.array(
            [
                np.clip(steering, -1.0, 1.0),
                np.clip(gas, 0.0, 1.0),
                np.clip(brake, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
        action, mode = self.apply_babble(action, mode)
        diagnostics = {
            "target_speed": desired_speed,
            "target_heading_error": heading_to_target_error,
            "preview_curvature_abs": anticipated_curvature,
            "steering_noise": noise,
        }
        return action, mode, diagnostics

    def apply_babble(self, action: np.ndarray, mode: int) -> tuple[np.ndarray, int]:
        """Sometimes hold a random command for a few steps, whatever the road looks like."""
        if self.babble_left == 0 and self.babble_prob > 0 and self.rng.random() < self.babble_prob:
            low, high = self.babble_steps
            self.babble_left = int(self.rng.integers(low, high + 1))
            self.babble_action = self.sample_babble_action()
        if self.babble_left == 0:
            return action, mode
        self.babble_left -= 1
        return self.babble_action.copy(), MODE_BABBLE

    def sample_babble_action(self) -> np.ndarray:
        """Random steering, and either some gas or some brake."""
        steering = self.rng.uniform(-self.babble_amp, self.babble_amp)
        if self.rng.random() < 0.5:
            gas, brake = self.rng.uniform(0.0, 1.0), 0.0
        else:
            gas, brake = 0.0, self.rng.uniform(0.0, 0.6)
        return np.array([steering, gas, brake], dtype=np.float32)

    def apply_safety_controls(self, state, target_heading_error, steering, gas, brake):
        """Apply boundary recovery before the higher-priority spin controls."""
        yaw_rate = state[5]
        (
            lateral_error,
            heading_error,
            left_clearance,
            right_clearance,
            curvature,
            speed,
            sideslip,
        ) = state[6:]
        distance_to_edge = min(left_clearance, right_clearance)
        mode = MODE_NORMAL

        if self.boundary_recovery and (
            distance_to_edge < 1.6
            or abs(lateral_error) > self.geometry.half_width * 0.65
        ):
            mode = MODE_RECOVERY
            steering += 0.12 * target_heading_error
            gas = min(gas, 0.35)
            if speed > self.corner_speed:
                brake = max(brake, 0.25)

        # Limit braking so locked wheels do not dominate recovery.
        if (
            abs(yaw_rate - speed * curvature) > 2.0
            or abs(heading_error) > 1.1
            or (speed > 3.0 and abs(sideslip) > 0.7)
        ):
            mode = MODE_SPIN_PREVENTION
            steering = 1.25 * target_heading_error + 0.55 * yaw_rate
            gas = 0.0
            if speed > 9:
                brake = 0.40
            elif speed > 5:
                brake = 0.20
            else:
                brake = 0.0

        return steering, gas, brake, mode

    def steering_target(self, state, road):
        """Return heading error and distance to the look-ahead target."""
        x, y, body_angle = state[:3]
        speed = state[11]
        preview_distance = 6.0 + 0.42 * speed
        target = self.geometry.point_at(road.progress + preview_distance)
        desired_heading = math.atan2(float(target[1]) - y, float(target[0]) - x)
        actual_heading = body_angle + math.pi / 2.0
        # Left of target gives a positive command (right steering).
        heading_error = wrap_angle(actual_heading - desired_heading)
        distance = max(float(np.linalg.norm(target - np.array([x, y]))), 1.0)
        return heading_error, distance

    def plan_speed(self, progress, speed):
        """Plan braking for the upcoming bends along the road."""
        distances = np.linspace(0.0, max(35.0, speed * 2.5), 18)
        curvatures = [
            abs(self.geometry.curvature_at(progress + distance))
            for distance in distances
        ]
        speed_limits = [
            np.clip(
                math.sqrt(22.0 / max(curvature, 0.001)),
                self.corner_speed,
                self.max_speed,
            )
            for curvature in curvatures
        ]
        desired_speed = float(
            min(
                math.sqrt(speed_limit * speed_limit + 2.0 * 8.0 * distance)
                for speed_limit, distance in zip(speed_limits, distances)
            )
            * self.speed_scale
        )
        anticipated_curvature = max(curvatures)

        return desired_speed, anticipated_curvature

    def steering_noise(self, distance_to_edge, speed):
        # Ornstein-Uhlenbeck-like correlated noise. This encourages exploratory
        # behavior without abrupt independent random steering every frame.
        self.noise_state = (
            0.96 * self.noise_state + math.sqrt(1.0 - 0.96**2) * self.rng.normal()
        )
        safe_margin = 1.0
        if self.safe_exploration:
            safe_margin = float(np.clip((distance_to_edge - 0.8) / 3.0, 0.0, 1.0))
        return self.exploration * safe_margin * self.noise_state / (1.0 + speed / 8.0)
