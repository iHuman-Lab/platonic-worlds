"""State representations for the dynamics experiments, and their constraint margins.

margins            the constraint margins only
ego_state          the car-centered subset of the state a camera can observe (no world frame)
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


def margins_from_ego(ego: torch.Tensor, config: dict) -> torch.Tensor:
    """Constraint margins from the ego-state variables in config['dynamics']['ego_states']."""
    states = torch.zeros(*ego.shape[:-1], len(STATE_INDEX), dtype=ego.dtype, device=ego.device)
    for index, name in enumerate(config["dynamics"]["ego_states"]):
        states[..., STATE_INDEX[name]] = ego[..., index]
    extras = torch.zeros(*ego.shape[:-1], len(EXTRA_INDEX), dtype=ego.dtype, device=ego.device)
    extras[..., EXTRA_INDEX["forward_velocity"]] = states[..., STATE_INDEX["speed"]] * torch.cos(
        states[..., STATE_INDEX["beta"]]
    )
    return constraint_margins(states, extras, config)


def represent(name: str, window: dict, config: dict) -> torch.Tensor:
    """Map a window of states/extras (B, T, ...) to the named representation (B, T, D)."""
    states = window["states"]
    if name == "raw_state":
        return states
    if name == "ego_state":
        return torch.stack([states[..., STATE_INDEX[n]] for n in config["dynamics"]["ego_states"]], dim=-1)
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
    if name == "ego_state":
        return margins_from_ego(representation, config)
    return representation[..., : len(constraint_names(config))]
