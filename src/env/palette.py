"""Road and grass luminance of a domain-randomized CarRacing episode, from its seed."""

from __future__ import annotations

import gymnasium as gym

from dataset.pre_process import RGB_TO_GREY


def luminance(color) -> float:
    """Grayscale value (0-255) of an RGB color."""
    return float(sum(weight * channel for weight, channel in zip(RGB_TO_GREY, color)))


def road_grass_contrast(seeds: list[int]) -> dict[int, dict]:
    """Per seed: gap = |road - grass| luminance and reversed = road brighter than grass.

    The palette is drawn from the seeded generator at reset, so it can be recomputed
    without rendering. Only valid for episodes collected with domain_randomize on.
    """
    env = gym.make("CarRacing-v3", domain_randomize=True).unwrapped
    contrast = {}
    for seed in seeds:
        env.reset(seed=seed)
        road, grass = luminance(env.road_color), luminance(env.grass_color)
        contrast[seed] = {"gap": abs(road - grass), "reversed": road > grass}
    env.close()
    return contrast
