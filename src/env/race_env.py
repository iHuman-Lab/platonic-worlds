"""CarRacing with completion at a forward crossing of the actual start line."""

from __future__ import annotations

import numpy as np
from gymnasium.envs.box2d.car_racing import TRACK_WIDTH, CarRacing

from road_geometry import TrackGeometry


class FinishLineTracker:
    """Track net lap progress and forward crossings within the road width."""

    def __init__(self, geometry, position):
        self.geometry = geometry
        self.origin = geometry.points[0]
        self.forward = geometry.tangents[0]
        self.left = np.array([-self.forward[1], self.forward[0]])
        self.previous = np.asarray(position, dtype=float)
        index, fraction, _, _ = geometry.closest_segment(self.previous)
        self.segment = index
        self.progress = (
            geometry.cumulative[index] + fraction * geometry.seg_lengths[index]
        )
        self.distance = 0.0

    def update(self, position):
        position = np.asarray(position, dtype=float)
        geometry = self.geometry
        index, fraction, _, _ = geometry.closest_segment(
            position, self.segment
        )
        progress = (
            geometry.cumulative[index] + fraction * geometry.seg_lengths[index]
        )
        # Unwrap progress across the start line in either direction.
        self.distance += (
            progress - self.progress + geometry.total_length / 2
        ) % geometry.total_length - geometry.total_length / 2
        crossed = self._crossed_finish_line(position)
        self.previous = position
        self.progress = progress
        self.segment = index
        return crossed

    def _crossed_finish_line(self, position):
        previous_offset = float(
            np.dot(self.previous - self.origin, self.forward)
        )
        current_offset = float(np.dot(position - self.origin, self.forward))
        if not (
            previous_offset < 0 <= current_offset
            and self.distance >= 0.95 * self.geometry.total_length
        ):
            return False
        crossing = self.previous + (
            -previous_offset / (current_offset - previous_offset)
        ) * (position - self.previous)
        lateral_offset = float(np.dot(crossing - self.origin, self.left))
        return bool(abs(lateral_offset) <= self.geometry.half_width)


class FinishLineCarRacing(CarRacing):
    """Retain off-playfield termination, replacing tile-based lap termination.

    Direct instantiation bypasses Gym's default 1,000-step TimeLimit.
    The collector owns the explicit, recorded emergency step limit.
    """

    def reset(self, **kwargs):
        self.finish_tracker = None
        observation, info = super().reset(**kwargs)
        self.geometry = TrackGeometry(np.asarray(self.track), TRACK_WIDTH)
        self.finish_tracker = FinishLineTracker(
            self.geometry, self.car.hull.worldCenter
        )
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        if action is not None and self.finish_tracker is not None:
            finished = self.finish_tracker.update(self.car.hull.worldCenter)
            off_playfield = terminated and not info.get("lap_finished", False)
            terminated = bool(finished or off_playfield)
            info["lap_finished"] = finished
            info["lap_progress"] = (
                self.finish_tracker.distance / self.geometry.total_length
            )
        return observation, reward, terminated, truncated, info

    def _render_road(self, zoom, translation, angle):
        super()._render_road(zoom, translation, angle)
        # Make the exact completion line visible in live view and saved images.
        origin = np.asarray(self.track[0][2:4])
        forward = np.asarray(self.track[1][2:4]) - origin
        forward /= np.linalg.norm(forward)
        left = np.array([-forward[1], forward[0]])
        for row in range(2):
            for column in range(12):
                lateral_start = -TRACK_WIDTH + column * (2 * TRACK_WIDTH / 12)
                lateral_end = lateral_start + 2 * TRACK_WIDTH / 12
                depth_start, depth_end = (row - 1) * 0.6, row * 0.6
                polygon = [
                    origin + left * lateral + forward * depth
                    for lateral, depth in (
                        (lateral_start, depth_start),
                        (lateral_end, depth_start),
                        (lateral_end, depth_end),
                        (lateral_start, depth_end),
                    )
                ]
                color = (245, 245, 245) if (row + column) % 2 else (20, 20, 20)
                self._draw_colored_polygon(
                    self.surf, polygon, color, zoom, translation, angle
                )
