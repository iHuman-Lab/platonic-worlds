"""Episode buffers, image capture, and NPZ/CSV/JSON/PNG export."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image

from env.configuration import IMAGE_SIZES, SIMULATION_FPS
from env.random_road_agent import MODE_NAMES
from env.road_geometry import STATE_NAMES, STATE_UNITS

CONTROLLER_DETAIL_NAMES = (
    "target_speed",
    "target_heading_error",
    "preview_curvature_abs",
    "steering_noise",
)
EXTRA_NAMES = (
    "forward_velocity",
    "lateral_velocity_right",
    "steering_angle_mean",
    "wheel_contact_count",
    "progress",
    "center_distance",
    "nearest_segment",
    "beta_valid",
    "road_heading",
)


@dataclass
class EpisodeData:
    """Buffer numerical observations and transitions separately from images."""

    states: list[np.ndarray]
    extras: list[dict]
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    collector_cutoff: list[bool] = field(default_factory=list)
    controller_modes: list[int] = field(default_factory=list)
    controller_details: list[list[float]] = field(default_factory=list)

    def append_transition(
        self, action, reward, terminated, truncated, cutoff, mode, diagnostics
    ):
        self.actions.append(action)
        self.rewards.append(float(reward))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))
        self.collector_cutoff.append(cutoff)
        self.controller_modes.append(mode)
        self.controller_details.append(
            [diagnostics[name] for name in CONTROLLER_DETAIL_NAMES]
        )

    def to_arrays(self, geometry, seed):
        action_count = len(self.actions)
        times = np.arange(action_count + 1, dtype=np.float64) / SIMULATION_FPS
        return {
            "states": np.asarray(self.states, dtype=np.float64),
            "time_s": times,
            "extras": np.asarray(
                [[float(extra[name]) for name in EXTRA_NAMES] for extra in self.extras]
            ),
            "actions": np.asarray(self.actions, dtype=np.float32).reshape(
                action_count, 3
            ),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "terminated": np.asarray(self.terminated, dtype=bool),
            "truncated": np.asarray(self.truncated, dtype=bool),
            "collector_cutoff": np.asarray(self.collector_cutoff, dtype=bool),
            "controller_modes": np.asarray(self.controller_modes, dtype=np.uint8),
            "controller_details": np.asarray(
                self.controller_details, dtype=np.float32
            ).reshape(action_count, len(CONTROLLER_DETAIL_NAMES)),
            "track": geometry.track,
            "road_half_width": np.array(geometry.half_width),
            "seed": np.array(seed),
        }


def capture_image(env, observation, resolution):
    if resolution == "low":
        return observation
    # Render at native video resolution instead of upscaling the observation.
    rgb = env.unwrapped._render("rgb_array")
    if resolution == "middle":
        rgb = np.asarray(
            Image.fromarray(rgb).resize(
                IMAGE_SIZES[resolution], Image.Resampling.LANCZOS
            )
        )
    return rgb


def save_episode(
    output_dir, episode_id, recording, geometry, summary, image_file, save_png
):
    """Export the episode arrays, metadata, and optional PNG frames."""
    arrays = recording.to_arrays(geometry, summary["seed"])
    images = None
    try:
        if summary["collection_mode"] == "images":
            image_file.flush()
            images = np.memmap(
                image_file,
                dtype=np.uint8,
                mode="r",
                shape=(len(recording.states), *summary["image_shape"]),
            )
            arrays["images"] = images

        episode_dir = output_dir / f"episode_{episode_id:05d}"
        episode_dir.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(episode_dir / "trajectory.npz", **arrays)
        write_csv(
            episode_dir / "states.csv",
            ["observation_index", "time_s", *STATE_NAMES, *EXTRA_NAMES],
            state_rows(arrays),
        )
        write_csv(
            episode_dir / "transitions.csv",
            [
                "action_index",
                "time_s",
                "steering",
                "gas",
                "brake",
                "reward",
                "terminated",
                "truncated",
                "collector_cutoff",
                "controller_mode",
                *CONTROLLER_DETAIL_NAMES,
            ],
            transition_rows(arrays),
        )
        with (episode_dir / "metadata.json").open("w") as metadata_file:
            json.dump(summary, metadata_file, indent=2)
        if save_png:
            frames_dir = episode_dir / "frames"
            frames_dir.mkdir()
            for index, image in enumerate(arrays["images"]):
                Image.fromarray(image).save(frames_dir / f"frame_{index:05d}.png")
    finally:
        if images is not None:
            images._mmap.close()


def build_episode_summary(
    environment,
    args,
    episode_id,
    recording,
    speeds,
    total_reward,
    stop_reason,
    info,
):
    with_images = args.collection_mode == "images"
    action_count = len(speeds)
    image_shape = None
    if with_images:
        width, height = IMAGE_SIZES[args.image_resolution]
        image_shape = [height, width, 3]
    return {
        "episode_id": episode_id,
        "seed": args.seed + episode_id,
        "env_id": "CarRacing-v3",
        "gymnasium_version": gym.__version__,
        "fps": SIMULATION_FPS,
        "num_actions": action_count,
        "num_observations": action_count + 1,
        "total_reward": total_reward,
        "stop_reason": stop_reason,
        "lap_finished": bool(info.get("lap_finished", False)),
        "lap_progress": float(info.get("lap_progress", 0.0)),
        "visited_tiles": int(environment.tile_visited_count),
        "total_track_tiles": len(environment.geometry.track),
        "stop_on_spin": args.abort_spin,
        "collection_mode": args.collection_mode,
        "image_resolution": args.image_resolution if with_images else None,
        "image_shape": image_shape,
        "max_speed_target": args.max_speed,
        "corner_speed_target": args.corner_speed,
        "mean_speed": float(np.mean(speeds)) if action_count else 0.0,
        "peak_speed": max(speeds, default=0.0),
        "gym_terminated_last": bool(recording.terminated and recording.terminated[-1]),
        "gym_truncated_last": bool(recording.truncated and recording.truncated[-1]),
        "last_collector_cutoff": bool(
            recording.collector_cutoff and recording.collector_cutoff[-1]
        ),
        "state_names": list(STATE_NAMES),
        "extra_names": list(EXTRA_NAMES),
    }


def build_dataset_info(args):
    return {
        "env_id": "CarRacing-v3",
        "gymnasium_version": gym.__version__,
        "fps": SIMULATION_FPS,
        "episode_seed_rule": "base_seed + episode_index",
        "base_seed": args.seed,
        "state_names": list(STATE_NAMES),
        "state_units": list(STATE_UNITS),
        "state_reference": "car hull center of mass in Box2D world frame",
        "states_shape_rule": (
            "N+1 observations for N transitions; optional images indexed identically"
        ),
        "action_names": ["steering", "gas", "brake"],
        "steering_sign": "-1=left, +1=right",
        "clearance_definition": (
            "approximate hull clearance to local tangent-strip boundary"
        ),
        "road_geometry": "track array: (alpha,beta,x,y), world coordinates",
        "random_policy": (
            "privileged road-following exploratory behavior policy; not learned"
        ),
        "time_reference": "zero is the observation returned by reset()",
        "controller_modes": list(MODE_NAMES),
        "requested_episodes": args.episodes,
        "max_steps_per_episode": args.max_steps,
        "exploration": args.exploration,
        "completion_rule": (
            "forward start-line crossing after at least 95% net track progress"
        ),
        "default_gym_time_limit": False,
        "collection_mode": args.collection_mode,
        "image_resolution": args.image_resolution
        if args.collection_mode == "images"
        else None,
        "save_png": args.save_png,
        "episodes": [],
    }


def write_csv(path: Path, header: list[str], rows):
    with path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        writer.writerows(rows)


def state_rows(arrays):
    for index, state in enumerate(arrays["states"]):
        yield [
            index,
            float(arrays["time_s"][index]),
            *state.tolist(),
            *arrays["extras"][index].tolist(),
        ]


def transition_rows(arrays):
    for index, action in enumerate(arrays["actions"]):
        yield [
            index,
            float(arrays["time_s"][index]),
            *action.tolist(),
            float(arrays["rewards"][index]),
            bool(arrays["terminated"][index]),
            bool(arrays["truncated"][index]),
            bool(arrays["collector_cutoff"][index]),
            int(arrays["controller_modes"][index]),
            *arrays["controller_details"][index].tolist(),
        ]
