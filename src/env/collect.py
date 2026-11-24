"""Run CarRacing episodes and record synchronized observations and actions."""

from __future__ import annotations

import tempfile
from collections import deque

import numpy as np
import pygame

from env.configuration import SIMULATION_FPS
from env.episode_data import (
    EpisodeData,
    build_episode_summary,
    capture_image,
    save_episode,
)
from env.random_road_agent import RandomRoadAgent
from env.road_geometry import extract_state


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
            image = capture_image(environment, observation, args.image_resolution)
            image_file.write(image.tobytes())

        for step in range(args.max_steps):
            if args.render and user_requested_exit():
                stop_reason = "user_closed"
                break

            action, controller_mode, diagnostics = controller.act(state, road)
            observation, reward, terminated, truncated, info = environment.step(action)
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
            if terminated or truncated or spin_cutoff:
                if terminated:
                    stop_reason = (
                        "finish_line" if info.get("lap_finished") else "off_playfield"
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


def user_requested_exit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN and event.key in (
            pygame.K_ESCAPE,
            pygame.K_q,
        ):
            return True
    return False
