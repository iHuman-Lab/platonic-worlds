"""State representations for the dynamics experiments, and their constraint margins.

margins            the constraint margins only
augmented          margins + the physical variables in config['dynamics']['augment_states']
augmented_preview  augmented + the controller's preview curvature of the road ahead
raw_state          the 13-D simulator state

Every representation is mapped back to constraint margins, so they can all be
compared in the same physical constraint space.
"""

from __future__ import annotations

import torch

from stl.constraints import EXTRA_INDEX, STATE_INDEX, constraint_margins, constraint_names


def forward_velocity(states: torch.Tensor) -> torch.Tensor:
    """Velocity along the car's heading, from the state (as the simulator computes it)."""
    theta = states[..., STATE_INDEX["theta"]]
    vx, vy = states[..., STATE_INDEX["vx"]], states[..., STATE_INDEX["vy"]]
    return -vx * torch.sin(theta) + vy * torch.cos(theta)


def margins_from_state(states: torch.Tensor, config: dict) -> torch.Tensor:
    """Constraint margins computed from the state alone."""
    extras = torch.zeros(*states.shape[:-1], len(EXTRA_INDEX), dtype=states.dtype, device=states.device)
    extras[..., EXTRA_INDEX["forward_velocity"]] = forward_velocity(states)
    return constraint_margins(states, extras, config)


def represent(name: str, window: dict, config: dict) -> torch.Tensor:
    """Map a window of states/extras (B, T, ...) to the named representation (B, T, D)."""
    states = window["states"]
    if name == "raw_state":
        return states
    margins = constraint_margins(states, window["extras"], config)
    if name == "margins":
        return margins
    names = config["dynamics"]["augment_states"]
    augmented = torch.cat(
        [margins, torch.stack([states[..., STATE_INDEX[n]] for n in names], dim=-1)], dim=-1
    )
    if name == "augmented":
        return augmented
    if name == "augmented_preview":
        return torch.cat([augmented, window["preview"].unsqueeze(-1)], dim=-1)
    raise ValueError(f"Unknown representation: {name}")


def to_margins(name: str, representation: torch.Tensor, config: dict) -> torch.Tensor:
    """Constraint margins (B, T, C) of a (possibly predicted) representation."""
    if name == "raw_state":
        return margins_from_state(representation, config)
    return representation[..., : len(constraint_names(config))]
