"""Physical constraints on the CarRacing state, as signed margins.

Each constraint is a scalar signal from (states, extras) compared to a threshold.
The margin is positive when the constraint is satisfied and its magnitude is how
far inside (or outside) the limit the car is, so it can serve directly as an
STL atomic-predicate robustness value.
"""

from __future__ import annotations

import torch

from env.configuration import EXTRA_NAMES
from env.road_geometry import STATE_NAMES

STATE_INDEX = {name: index for index, name in enumerate(STATE_NAMES)}
EXTRA_INDEX = {name: index for index, name in enumerate(EXTRA_NAMES)}


def clearance(states: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
    """Distance from the hull to the nearer road edge; negative once off the road."""
    return torch.minimum(
        states[..., STATE_INDEX["d_left"]], states[..., STATE_INDEX["d_right"]]
    )


def abs_heading(states: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
    """Absolute angle between the car's heading and the road direction."""
    return states[..., STATE_INDEX["e_theta"]].abs()


def speed(states: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
    """Speed of the hull center of mass."""
    return states[..., STATE_INDEX["speed"]]


def abs_sideslip(states: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
    """Absolute angle between the velocity direction and the car's heading."""
    return states[..., STATE_INDEX["beta"]].abs()


def road_speed(states: torch.Tensor, extras: torch.Tensor) -> torch.Tensor:
    """Forward velocity projected onto the road direction (progress along the track)."""
    forward = extras[..., EXTRA_INDEX["forward_velocity"]]
    return forward * torch.cos(states[..., STATE_INDEX["e_theta"]])


SIGNALS = {
    "clearance": clearance,
    "abs_heading": abs_heading,
    "speed": speed,
    "abs_sideslip": abs_sideslip,
    "road_speed": road_speed,
}


def constraint_names(config: dict) -> list[str]:
    """Constraint names in the order they are stacked in the output."""
    return list(config["constraints"])


def constraint_margins(
    states: torch.Tensor, extras: torch.Tensor, config: dict
) -> torch.Tensor:
    """Return scaled margins of shape (..., n_constraints); positive means satisfied.

    states: (..., 13) and extras: (..., 9), with any leading batch/time dimensions.
    """
    margins = []
    for cfg in config["constraints"].values():
        signal = SIGNALS[cfg["signal"]](states, extras)
        if cfg["bound"] == "lower":
            margin = signal - cfg["threshold"]
        elif cfg["bound"] == "upper":
            margin = cfg["threshold"] - signal
        else:
            raise ValueError(f"bound must be 'lower' or 'upper', got {cfg['bound']}")
        margins.append(margin / cfg["scale"])
    return torch.stack(margins, dim=-1)
