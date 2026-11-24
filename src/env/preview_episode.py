"""Replay saved RGB observations with synchronized state and action data.

Controls: SPACE=pause, left/right arrows=step, ESC or Q=quit.
"""

from __future__ import annotations

import numpy as np
import pygame

from env.random_road_agent import MODE_NAMES
from env.road_geometry import STATE_NAMES


def load_episode(filename):
    with np.load(filename, allow_pickle=False) as dataset:
        if "images" not in dataset:
            raise SystemExit(
                "This is a data-only episode. "
                "Collect with collect.collection_mode: images to enable replay."
            )
        episode = {
            name: dataset[name]
            for name in (
                "images",
                "states",
                "actions",
                "rewards",
                "time_s",
                "controller_modes",
            )
        }
        frames = episode["images"]
        states = episode["states"]
        actions = episode["actions"]
        times = episode["time_s"]
        if not (len(frames) == len(states) == len(times) == len(actions) + 1):
            raise ValueError("Dataset indices do not align")

    return episode


def handle_events(index, playing, last_index):
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False
            elif event.key == pygame.K_SPACE:
                playing = not playing
            elif event.key == pygame.K_RIGHT:
                playing = False
                index = min(index + 1, last_index)
            elif event.key == pygame.K_LEFT:
                playing = False
                index = max(index - 1, 0)
    return index, playing, running


def draw_frame(window, frames, index):
    window.fill((20, 25, 35))
    rgb = frames[index]
    surface = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    scale = min(576 / rgb.shape[1], 576 / rgb.shape[0])
    size = (round(rgb.shape[1] * scale), round(rgb.shape[0] * scale))
    surface = pygame.transform.scale(surface, size)
    window.blit(surface, (22 + (576 - size[0]) // 2, 55 + (576 - size[1]) // 2))


def overlay_lines(episode, index):
    states = episode["states"]
    actions = episode["actions"]
    rewards = episode["rewards"]
    times = episode["time_s"]
    modes = episode["controller_modes"]
    action_count = len(actions)
    lines = [
        (
            f"Observation {index:04d}/{action_count:04d}  t={times[index]:.2f}s",
            True,
        ),
        ("Vehicle / road physical state", True),
    ]
    for name, value in zip(STATE_NAMES, states[index]):
        lines.append((f"{name:>10s} : {value:+10.4f}", False))
    lines.append(("", False))
    if index == action_count:
        lines.append(("Final observation (no next action)", True))
        return lines

    action = actions[index]
    lines.extend(
        [
            ("Action at current observation", True),
            (
                f" steer={action[0]:+.3f} gas={action[1]:.3f} brake={action[2]:.3f}",
                False,
            ),
            (
                f" reward={rewards[index]:+.3f}  mode={MODE_NAMES[int(modes[index])]}",
                False,
            ),
        ]
    )
    return lines


def draw_overlay(window, font, title_font, episode, index):
    for line, (text, is_heading) in enumerate(overlay_lines(episode, index)):
        if not text:
            continue
        color = (230, 235, 245) if is_heading else (193, 208, 224)
        selected_font = title_font if is_heading else font
        window.blit(selected_font.render(text, True, color), (620, 23 + line * 25))
    hint = "SPACE pause | LEFT/RIGHT step | Q quit"
    window.blit(font.render(hint, True, (255, 210, 150)), (20, 637))


def play_episode(episode, filename, fps):
    pygame.init()
    try:
        window = pygame.display.set_mode((1190, 660))
        pygame.display.set_caption(f"Saved CarRacing episode: {filename.parent.name}")
        font = pygame.font.SysFont("monospace", 18)
        title_font = pygame.font.SysFont("monospace", 22, bold=True)
        clock = pygame.time.Clock()
        playing = True
        running = True
        index = 0
        action_count = len(episode["actions"])
        while running:
            index, playing, running = handle_events(index, playing, action_count)
            draw_frame(window, episode["images"], index)
            draw_overlay(window, font, title_font, episode, index)
            pygame.display.flip()
            clock.tick(max(fps, 1))
            if playing and index < action_count:
                index += 1
            elif index == action_count:
                playing = False
    finally:
        pygame.quit()
