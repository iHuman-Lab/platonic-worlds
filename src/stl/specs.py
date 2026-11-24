"""Build STL formulas from config and evaluate their robustness on constraint margins.

A spec is a nested dict with exactly one key per node:
    {pred: <constraint name>}          the constraint margin is >= 0
    {always: node}, {eventually: node}, {not: node}
    {and: [node, node]}, {or: [node, node]}, {implies: [node, node]}
Formulas are evaluated at the first frame of each window, over the whole window.
"""

from __future__ import annotations

from typing import Callable

import torch

from stl.constraints import constraint_names
from stl.stlcg import Always, And, Eventually, GreaterThan, Implies, Negation, Or

UNARY = {"always": Always, "eventually": Eventually, "not": Negation}
BINARY = {"and": And, "or": Or, "implies": Implies}


def build_formula(node: dict, names: list[str]) -> tuple[object, Callable]:
    """Return (formula, make_inputs); make_inputs maps reversed margins to stlcg inputs."""
    if len(node) != 1:
        raise ValueError(f"A spec node needs exactly one key, got {list(node)}")
    (op, arg), = node.items()
    if op == "pred":
        index = names.index(arg)
        return GreaterThan(lhs=arg, val=0.0), lambda margins: margins[..., index : index + 1]
    if op in UNARY:
        sub, make_inputs = build_formula(arg, names)
        return UNARY[op](subformula=sub), make_inputs
    if op in BINARY:
        left, make_left = build_formula(arg[0], names)
        right, make_right = build_formula(arg[1], names)
        formula = BINARY[op](left, right)
        return formula, lambda margins: (make_left(margins), make_right(margins))
    raise ValueError(f"Unknown spec operator: {op}")


class SpecSet:
    """All configured specs, built once; maps margins (B, T, C) to robustness (B, S)."""

    def __init__(self, config: dict) -> None:
        names = constraint_names(config)
        self.scale = config["stl"]["scale"]
        self.names = list(config["stl"]["specs"])
        built = [build_formula(node, names) for node in config["stl"]["specs"].values()]
        self.formulas = [formula for formula, _ in built]
        self.make_inputs = [make_inputs for _, make_inputs in built]

    def __call__(self, margins: torch.Tensor, smooth: bool) -> torch.Tensor:
        """Robustness per spec; smooth=True uses the configured logsumexp scale."""
        scale = self.scale if smooth else -1
        reversed_margins = torch.flip(margins, dims=[-2])  # stlcg expects newest first
        values = [
            formula.robustness(make(reversed_margins), time=0, scale=scale)
            for formula, make in zip(self.formulas, self.make_inputs)
        ]
        return torch.stack([value.reshape(margins.shape[0]) for value in values], dim=-1)
