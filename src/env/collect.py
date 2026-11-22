"""Run CarRacing episodes and record synchronized observations and actions."""

from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from configuration import SIMULATION_FPS, parse_args
from episode_data import (
    EpisodeData,
    build_episode_summary,
    build_manifest,
    capture_image,
    save_episode,
)
from race_env import FinishLineCarRacing
from random_road_agent import RandomRoadAgent
from road_geometry import extract_state


def main():
    args = parse_args()
    run(args)


def run(args):
    """Initialize the simulation, collect episodes, and update the manifest."""
    if not args.render and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    collecting = args.collection_mode != "watch"
    output_dir = Path(args.output).expanduser().resolve()
    if collecting:
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "manifest.json").exists() or any(
            output_dir.glob("episode_*")
        ):
            raise SystemExit(
                f"Dataset already exists at {output_dir}. "
                "Choose a new --output folder."
            )

    manifest = build_manifest(args)
    environment = FinishLineCarRacing(
        continuous=True,
        render_mode="human" if args.render else None,
        domain_randomize=args.domain_randomize,
    )
    try:
        for episode_id in range(args.episodes):
            summary = collect_episode(
                environment, args, episode_id, output_dir
            )
            manifest["episodes"].append(summary)
            if collecting:
                with (output_dir / "manifest.json").open("w") as manifest_file:
                    json.dump(manifest, manifest_file, indent=2)
            print(
                f"[{episode_id + 1}/{args.episodes}] seed={summary['seed']} "
                f"steps={summary['num_actions']} "
                f"speed_mean={summary['mean_speed']:.1f} "
                f"peak={summary['peak_speed']:.1f} "
                f"lap={summary['lap_progress']:.1%} "
                f"stop={summary['stop_reason']}"
            )
            if summary["stop_reason"] == "user_closed":
                break
    finally:
        environment.close()
    if collecting:
        print(f"Saved dataset to: {output_dir}")


def collect_episode(environment, args, episode_id, output_dir):
    """Reset, drive, record, and save one episode in execution order."""
    seed = args.seed + episode_id
    observation, _ = environment.reset(seed=seed)
    geometry = environment.geometry
    controller = RandomRoadAgent(
        geometry,
        np.random.default_rng(seed + 900_001),
        exploration=args.exploration,
        max_speed=args.max_speed,
        corner_speed=args.corner_speed,
    )
    state, road, extras = extract_state(environment, geometry)
    recording = EpisodeData(states=[state], extras=[extras])
    collecting = args.collection_mode != "watch"
    with_images = args.collection_mode == "images"
    yaw_changes = deque(maxlen=50)
    speeds = []
    total_reward = 0.0
    stop_reason = "max_steps"
    info = {}

    # Keep the initial observation before recording any action.
    # Spool images to disk so full laps do not fill memory.
    with tempfile.TemporaryFile() as image_file:
        if with_images:
            image = capture_image(
                environment, observation, args.image_resolution
            )
            image_file.write(image.tobytes())

        for step in range(args.max_steps):
            if args.render and _user_requested_exit():
                stop_reason = "user_closed"
                break

            action, controller_mode, diagnostics = controller.act(state, road)
            observation, reward, terminated, truncated, info = (
                environment.step(action)
            )
            next_state, next_road, next_extras = extract_state(
                environment, geometry, road.nearest_segment
            )

            yaw_changes.append(float(next_state[5]) / SIMULATION_FPS)
            spin_cutoff = bool(
                args.abort_spin
                and (
                    abs(next_state[7]) > 2.35
                    or abs(next_state[5]) > 2.8
                    or abs(sum(yaw_changes)) > 2.70
                )
            )
            step_cap_reached = step + 1 == args.max_steps and not (
                terminated or truncated or spin_cutoff
            )

            if collecting:
                if with_images:
                    image = capture_image(
                        environment, observation, args.image_resolution
                    )
                    image_file.write(image.tobytes())
                recording.states.append(next_state)
                recording.extras.append(next_extras)
                recording.append_transition(
                    action,
                    reward,
                    terminated,
                    truncated,
                    cutoff=spin_cutoff or step_cap_reached,
                    mode=controller_mode,
                    diagnostics=diagnostics,
                )

            total_reward += float(reward)
            speeds.append(float(next_state[11]))
            state, road = next_state, next_road
            if args.verbose and step % args.log_every == 0:
                print(
                    f"episode={episode_id} step={step:4d} "
                    f"speed={state[11]:5.2f} "
                    f"lap={info.get('lap_progress', 0):.1%} "
                    f"mode={controller_mode}"
                )
            if terminated or truncated or spin_cutoff:
                if terminated:
                    stop_reason = (
                        "finish_line"
                        if info.get("lap_finished")
                        else "off_playfield"
                    )
                elif truncated:
                    stop_reason = "gym_truncated"
                else:
                    stop_reason = "spin_safety_cutoff"
                break

        if collecting and stop_reason == "user_closed" and speeds:
            recording.collector_cutoff[-1] = True
        summary = build_episode_summary(
            environment,
            args,
            episode_id,
            recording,
            speeds,
            total_reward,
            stop_reason,
            info,
        )
        if collecting:
            save_episode(
                output_dir,
                episode_id,
                recording,
                geometry,
                summary,
                image_file,
                args.save_png,
            )
    return summary


def _user_requested_exit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key in (
            pygame.K_ESCAPE,
            pygame.K_q,
        ):
            return True
    return False


if __name__ == "__main__":
    main()
