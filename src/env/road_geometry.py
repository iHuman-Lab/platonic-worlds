"""Track projection and the 13 CarRacing physical/road-state variables.

All positions use the Box2D world frame. x,y,vx,vy refer to the car hull's
CENTER OF MASS; theta is the Box2D body angle (local +X vs world +X).
The car's actual forward direction is local +Y, i.e. theta + pi/2.

Road d_left/d_right below are signed, *local tangent-strip approximations*
to clearance of the entire HULL FOOTPRINT, not exact shortest distances to
curved road-boundary polygons. This approximation must not be treated as a
certified safety guarantee.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

STATE_NAMES = (
    "x",
    "y",
    "theta",
    "vx",
    "vy",
    "omega",
    "e_y",
    "e_theta",
    "d_left",
    "d_right",
    "kappa",
    "speed",
    "beta",
)
STATE_UNITS = (
    "world",
    "world",
    "rad",
    "world/s",
    "world/s",
    "rad/s",
    "world",
    "rad",
    "world",
    "world",
    "1/world",
    "world/s",
    "rad",
)


def wrap_angle(angle: float) -> float:
    """Normalize an angle to the signed range from -pi to pi."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass
class RoadContext:
    """Local road measurements at the vehicle hull."""

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
        # Track entries in CarRacing-v3: (alpha, beta, center_x, center_y)
        self.track = np.asarray(track, dtype=np.float64).copy()
        if self.track.ndim != 2 or self.track.shape[1] != 4:
            raise ValueError(
                "Expected Gymnasium CarRacing track with (N,4) entries"
            )
        self.points = self.track[:, 2:4]
        self.n = len(self.points)
        if self.n < 4:
            raise ValueError("Too few track points")
        self.deltas = np.roll(self.points, -1, axis=0) - self.points
        self.seg_lengths = np.linalg.norm(self.deltas, axis=1)
        if np.any(self.seg_lengths < 1e-8):
            raise ValueError("Degenerate track segment")
        self.tangents = self.deltas / self.seg_lengths[:, None]
        self.seg_headings = np.arctan2(
            self.tangents[:, 1], self.tangents[:, 0]
        )
        self.cumulative = np.r_[0.0, np.cumsum(self.seg_lengths)]
        self.total_length = float(self.cumulative[-1])
        self.half_width = float(road_half_width)

        # Turn angle / distance, then cyclic smoothing to reduce polygon noise.
        next_heading = np.roll(self.seg_headings, -1)
        turns = np.arctan2(
            np.sin(next_heading - self.seg_headings),
            np.cos(next_heading - self.seg_headings),
        )
        segment_spacing = 0.5 * (
            self.seg_lengths + np.roll(self.seg_lengths, -1)
        )
        raw_curvatures = turns / segment_spacing
        self.curvatures = (
            np.roll(raw_curvatures, -2)
            + 2 * np.roll(raw_curvatures, -1)
            + 3 * raw_curvatures
            + 2 * np.roll(raw_curvatures, 1)
            + np.roll(raw_curvatures, 2)
        ) / 9.0

    def closest_segment(self, p: np.ndarray, last_idx: int | None = None):
        """Find the nearest projection, using continuity at crossings."""
        point = np.asarray(p, dtype=np.float64)
        fractions = np.einsum(
            "ij,ij->i", point[None, :] - self.points, self.deltas
        )
        fractions /= self.seg_lengths**2
        fractions = np.clip(fractions, 0.0, 1.0)
        projections = self.points + fractions[:, None] * self.deltas
        projection_offsets = point[None, :] - projections
        squared_distances = np.einsum(
            "ij,ij->i", projection_offsets, projection_offsets
        )
        global_index = int(np.argmin(squared_distances))
        nearest_index = global_index
        if last_idx is not None:
            nearby = (int(last_idx) + np.arange(-18, 19)) % self.n
            local_index = int(nearby[np.argmin(squared_distances[nearby])])
            # Prefer a similarly close local segment to preserve continuity.
            if (
                squared_distances[local_index]
                <= squared_distances[global_index] + 1.0
            ):
                nearest_index = local_index
        return (
            nearest_index,
            float(fractions[nearest_index]),
            projections[nearest_index],
            float(math.sqrt(squared_distances[nearest_index])),
        )

    def _segment_at(self, progress: float):
        wrapped_progress = float(progress) % self.total_length
        next_index = np.searchsorted(
            self.cumulative, wrapped_progress, side="right"
        )
        return min(int(next_index - 1), self.n - 1), wrapped_progress

    def point_at(self, progress: float) -> np.ndarray:
        index, wrapped_progress = self._segment_at(progress)
        segment_progress = wrapped_progress - self.cumulative[index]
        fraction = segment_progress / self.seg_lengths[index]
        return self.points[index] + fraction * self.deltas[index]

    def curvature_at(self, progress: float) -> float:
        index, _ = self._segment_at(progress)
        return float(self.curvatures[index])

    def road_context(self, hull, last_idx: int | None = None) -> RoadContext:
        center_of_mass = np.array(
            [hull.worldCenter.x, hull.worldCenter.y], dtype=np.float64
        )
        index, fraction, projection, center_distance = self.closest_segment(
            center_of_mass, last_idx
        )
        tangent = self.tangents[index]
        left_normal = np.array([-tangent[1], tangent[0]], dtype=np.float64)
        lateral_error = float(
            np.dot(center_of_mass - projection, left_normal)
        )  # Positive is left of the road.
        car_heading = float(hull.angle) + math.pi / 2.0
        road_heading = float(self.seg_headings[index])
        heading_error = wrap_angle(car_heading - road_heading)

        left_clearance, right_clearance = self._hull_clearances(
            hull, projection, left_normal
        )
        progress = float(
            self.cumulative[index] + fraction * self.seg_lengths[index]
        )
        return RoadContext(
            e_y=lateral_error,
            e_theta=heading_error,
            d_left=left_clearance,
            d_right=right_clearance,
            curvature=float(self.curvatures[index]),
            progress=progress,
            nearest_segment=index,
            center_distance=center_distance,
            road_heading=road_heading,
        )

    def _hull_clearances(self, hull, projection, left_normal):
        """Project all hull fixtures onto the local road normal."""
        offsets = []
        for fixture in hull.fixtures:
            for local_vertex in fixture.shape.vertices:
                world_vertex = hull.GetWorldPoint(local_vertex)
                vertex_offset = (
                    np.array([world_vertex.x, world_vertex.y]) - projection
                )
                offsets.append(float(np.dot(vertex_offset, left_normal)))
        return (
            float(self.half_width - max(offsets)),
            float(self.half_width + min(offsets)),
        )


def extract_state(env, geometry: TrackGeometry, last_idx: int | None = None):
    """Return state vector (13,), road info, and detailed diagnostic extras."""
    hull = env.unwrapped.car.hull
    x, y = map(float, hull.worldCenter)  # velocity also measured at COM
    vx, vy = map(float, hull.linearVelocity)
    theta = float(hull.angle)
    omega = float(hull.angularVelocity)
    speed = math.hypot(vx, vy)
    forward_velocity = -vx * math.sin(theta) + vy * math.cos(theta)
    lateral_velocity_right = vx * math.cos(theta) + vy * math.sin(theta)
    # Left-positive sideslip is undefined near zero speed.
    beta_valid = speed >= 0.2
    beta = (
        math.atan2(-lateral_velocity_right, forward_velocity)
        if beta_valid
        else 0.0
    )
    road = geometry.road_context(hull, last_idx)
    state = np.array(
        [
            x,
            y,
            theta,
            vx,
            vy,
            omega,
            road.e_y,
            road.e_theta,
            road.d_left,
            road.d_right,
            road.curvature,
            speed,
            beta,
        ],
        dtype=np.float64,
    )
    wheels = env.unwrapped.car.wheels
    extras = {
        "forward_velocity": forward_velocity,
        "lateral_velocity_right": lateral_velocity_right,
        "steering_angle_mean": float(
            (wheels[0].joint.angle + wheels[1].joint.angle) / 2
        ),
        "wheel_contact_count": sum(bool(wheel.tiles) for wheel in wheels),
        "progress": road.progress,
        "center_distance": road.center_distance,
        "nearest_segment": road.nearest_segment,
        "beta_valid": bool(beta_valid),
        "road_heading": road.road_heading,
    }
    return state, road, extras
