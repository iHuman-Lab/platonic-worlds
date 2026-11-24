"""Track projection and the 13 CarRacing state variables (Box2D world frame).

x, y, vx, vy are at the hull's center of mass; theta is the Box2D body angle
(the car's forward direction is theta + pi/2). d_left/d_right are local
tangent-strip approximations of hull clearance, not exact distances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STATE_NAMES = ("x", "y", "theta", "vx", "vy", "omega", "e_y", "e_theta",
               "d_left", "d_right", "kappa", "speed", "beta")  # fmt: skip
STATE_UNITS = ("world", "world", "rad", "world/s", "world/s", "rad/s", "world",
               "rad", "world", "world", "1/world", "world/s", "rad")  # fmt: skip


def wrap_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class RoadContext:
    e_y: float
    e_theta: float
    d_left: float
    d_right: float
    curvature: float
    progress: float
    nearest_segment: int
    center_distance: float
    road_heading: float


class TrackGeometry:
    """Project positions and hull footprints onto a closed track centerline."""

    def __init__(self, track: np.ndarray, road_half_width: float):
        self.track = np.asarray(track, dtype=np.float64).copy()  # (alpha, beta, x, y)
        self.points = self.track[:, 2:4]
        self.n = len(self.points)
        self.deltas = np.roll(self.points, -1, axis=0) - self.points
        self.seg_lengths = np.linalg.norm(self.deltas, axis=1)
        self.tangents = self.deltas / self.seg_lengths[:, None]
        self.seg_headings = np.arctan2(self.tangents[:, 1], self.tangents[:, 0])
        self.cumulative = np.r_[0.0, np.cumsum(self.seg_lengths)]
        self.total_length = float(self.cumulative[-1])
        self.half_width = float(road_half_width)
        # Turn angle per distance, smoothed cyclically to reduce polygon noise.
        turns = np.arctan2(
            np.sin(np.roll(self.seg_headings, -1) - self.seg_headings),
            np.cos(np.roll(self.seg_headings, -1) - self.seg_headings),
        )
        raw = turns / (0.5 * (self.seg_lengths + np.roll(self.seg_lengths, -1)))
        self.curvatures = sum(
            weight * np.roll(raw, shift)
            for shift, weight in ((-2, 1), (-1, 2), (0, 3), (1, 2), (2, 1))
        ) / 9.0

    def closest_segment(self, p: np.ndarray, last_idx: int | None = None):
        """Nearest projection; prefers segments near last_idx for continuity."""
        point = np.asarray(p, dtype=np.float64)
        fractions = np.einsum("ij,ij->i", point - self.points, self.deltas)
        fractions = np.clip(fractions / self.seg_lengths**2, 0.0, 1.0)
        projections = self.points + fractions[:, None] * self.deltas
        offsets = point - projections
        squared = np.einsum("ij,ij->i", offsets, offsets)
        index = int(np.argmin(squared))
        if last_idx is not None:
            nearby = (int(last_idx) + np.arange(-18, 19)) % self.n
            local = int(nearby[np.argmin(squared[nearby])])
            if squared[local] <= squared[index] + 1.0:
                index = local
        return index, float(fractions[index]), projections[index], float(np.sqrt(squared[index]))

    def segment_at(self, progress: float):
        wrapped = float(progress) % self.total_length
        index = np.searchsorted(self.cumulative, wrapped, side="right") - 1
        return min(int(index), self.n - 1), wrapped

    def point_at(self, progress: float) -> np.ndarray:
        index, wrapped = self.segment_at(progress)
        fraction = (wrapped - self.cumulative[index]) / self.seg_lengths[index]
        return self.points[index] + fraction * self.deltas[index]

    def curvature_at(self, progress: float) -> float:
        return float(self.curvatures[self.segment_at(progress)[0]])

    def road_context(self, hull, last_idx: int | None = None) -> RoadContext:
        center = np.array([hull.worldCenter.x, hull.worldCenter.y], dtype=np.float64)
        index, fraction, projection, center_distance = self.closest_segment(center, last_idx)
        tangent = self.tangents[index]
        left_normal = np.array([-tangent[1], tangent[0]])
        road_heading = float(self.seg_headings[index])
        # Clearance: project every hull vertex onto the local road normal.
        offsets = [
            float(np.dot(np.array(hull.GetWorldPoint(vertex)) - projection, left_normal))
            for fixture in hull.fixtures
            for vertex in fixture.shape.vertices
        ]
        return RoadContext(
            e_y=float(np.dot(center - projection, left_normal)),  # positive is left
            e_theta=wrap_angle(float(hull.angle) + math.pi / 2 - road_heading),
            d_left=float(self.half_width - max(offsets)),
            d_right=float(self.half_width + min(offsets)),
            curvature=float(self.curvatures[index]),
            progress=float(self.cumulative[index] + fraction * self.seg_lengths[index]),
            nearest_segment=index,
            center_distance=center_distance,
            road_heading=road_heading,
        )


def extract_state(env, geometry: TrackGeometry, last_idx: int | None = None):
    """Return state vector (13,), road context, and diagnostic extras."""
    car = env.unwrapped.car
    hull = car.hull
    x, y = map(float, hull.worldCenter)
    vx, vy = map(float, hull.linearVelocity)
    theta = float(hull.angle)
    speed = math.hypot(vx, vy)
    forward = -vx * math.sin(theta) + vy * math.cos(theta)
    right = vx * math.cos(theta) + vy * math.sin(theta)
    beta_valid = speed >= 0.2  # sideslip is undefined near zero speed
    beta = math.atan2(-right, forward) if beta_valid else 0.0
    road = geometry.road_context(hull, last_idx)
    state = np.array(
        [x, y, theta, vx, vy, float(hull.angularVelocity), road.e_y, road.e_theta,
         road.d_left, road.d_right, road.curvature, speed, beta]
    )  # fmt: skip
    wheels = car.wheels
    extras = {
        "forward_velocity": forward,
        "lateral_velocity_right": right,
        "steering_angle_mean": float((wheels[0].joint.angle + wheels[1].joint.angle) / 2),
        "wheel_contact_count": sum(bool(wheel.tiles) for wheel in wheels),
        "progress": road.progress,
        "center_distance": road.center_distance,
        "nearest_segment": road.nearest_segment,
        "beta_valid": bool(beta_valid),
        "road_heading": road.road_heading,
    }
    return state, road, extras
