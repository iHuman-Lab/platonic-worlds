"""Train and compare state-based dynamics models on different representations.

Every representation is rolled out autoregressively, mapped back to constraint margins,
and scored in that shared space: margin error per step, persistence and shuffled-action
baselines, exact STL robustness error, and spec satisfaction accuracy / AUROC.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torchmetrics.classification import BinaryAUROC

from dataset.torch_dataset import load_state_windows
from env.random_road_agent import MODE_BABBLE
from models.constraint_dynamics import ConstraintDynamics
from stl.constraints import constraint_margins, constraint_names
from stl.representation import represent, to_margins
from stl.specs import SpecSet

REPORT_STEPS = (1, 4, 8)
TABLE_METRICS = ("mse_step1", "mse_step8", "persistence_step8", "shuffled_step8", "stl_mse")


def build_model(name: str, train: dict, config: dict, device: str) -> ConstraintDynamics:
    """Dynamics model with the normalization of the named representation."""
    cfg = config["dynamics"]
    flat = represent(name, train, config).flatten(0, 1)
    std = flat.std(dim=0) + cfg["std_floor"]
    action_dim = train["actions"].shape[-1]
    return ConstraintDynamics(flat.mean(dim=0), std, action_dim, cfg["hidden"]).to(device)


def train_dynamics(
    name: str, train: dict, config: dict, train_rollout: int, seed: int, device: str
) -> ConstraintDynamics:
    """Fit by MSE over `train_rollout` autoregressive steps in the normalized space."""
    cfg = config["dynamics"]
    torch.manual_seed(seed)
    model = build_model(name, train, config, device)
    c = model.normalize(represent(name, train, config).to(device))
    actions = train["actions"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    for _ in range(cfg["epochs"]):
        order = torch.randperm(c.shape[0], device=device)
        for start in range(0, len(order), cfg["batch_size"]):
            idx = order[start : start + cfg["batch_size"]]
            pred = model.rollout(c[idx, 0], actions[idx, :train_rollout])
            loss = ((pred - c[idx, 1 : train_rollout + 1]) ** 2).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


@torch.no_grad()
def predict_margins(
    model: ConstraintDynamics, name: str, window: dict, config: dict, actions: torch.Tensor
) -> torch.Tensor:
    """Roll out from the true first frame; return margins (N, K + 1, C), frame 0 is true."""
    rep = represent(name, window, config).to(actions.device)
    pred = model.denormalize(model.rollout(model.normalize(rep[:, 0]), actions))
    return to_margins(name, torch.cat([rep[:, :1], pred], dim=1), config)


def window_errors(
    model: ConstraintDynamics, name: str, window: dict, config: dict, specs: SpecSet
) -> dict[str, torch.Tensor]:
    """Per-window squared margin errors and exact spec robustness, predicted and true."""
    device = model.mean.device
    actions = window["actions"].to(device)
    true = constraint_margins(window["states"], window["extras"], config).to(device)
    pred = predict_margins(model, name, window, config, actions)
    shuffled = predict_margins(model, name, window, config, actions.roll(1, dims=0))
    return {
        "squared": ((pred - true)[:, 1:] ** 2).cpu(),
        "persistence": ((true[:, :1] - true)[:, 1:] ** 2).cpu(),
        "shuffled": ((shuffled - true)[:, 1:] ** 2).cpu(),
        "rho_pred": specs(pred.cpu(), smooth=False),
        "rho_true": specs(true.cpu(), smooth=False),
    }


def spec_scores(errors: dict, mask: torch.Tensor, specs: SpecSet) -> dict[str, float]:
    """Accuracy and AUROC of violation detection from the predicted robustness."""
    scores = {}
    rho_pred, rho_true = errors["rho_pred"][mask], errors["rho_true"][mask]
    for index, spec in enumerate(specs.names):
        violated = rho_true[:, index] < 0
        scores[f"acc/{spec}"] = ((rho_pred[:, index] < 0) == violated).float().mean().item()
        both_classes = violated.any() and not violated.all()
        auroc = BinaryAUROC()(-rho_pred[:, index], violated.int()).item() if both_classes else float("nan")
        scores[f"auroc/{spec}"] = auroc
    return scores


def summarize(errors: dict, mask: torch.Tensor, specs: SpecSet, config: dict) -> dict[str, float]:
    """Scalar metrics over the windows selected by mask."""
    squared = errors["squared"][mask]
    metrics = {f"mse_step{step}": squared[:, step - 1].mean().item() for step in REPORT_STEPS}
    metrics["mse_mean"] = squared.mean().item()
    metrics["persistence_step8"] = errors["persistence"][mask][:, -1].mean().item()
    metrics["shuffled_step8"] = errors["shuffled"][mask][:, -1].mean().item()
    for index, constraint in enumerate(constraint_names(config)):
        metrics[f"mse8/{constraint}"] = squared[:, -1, index].mean().item()
    gap = errors["rho_pred"][mask] - errors["rho_true"][mask]
    metrics["stl_mse"] = (gap**2).mean().item()
    return {**metrics, **spec_scores(errors, mask, specs)}


def aggregate(per_seed: list[dict]) -> dict:
    """Mean and std over seeds of every metric, per subset."""
    result = {}
    for subset in per_seed[0]:
        result[subset] = {}
        for metric in per_seed[0][subset]:
            values = np.array([run[subset][metric] for run in per_seed])
            result[subset][metric] = {"mean": float(np.nanmean(values)), "std": float(np.nanstd(values))}
    return result


def run_dynamics_experiment(config: dict) -> dict:
    """Train every representation x training rollout x seed; save and return the results."""
    cfg = config["dynamics"]
    windows = load_state_windows(config, cfg["horizon"] + 1)
    val, specs = windows["val"], SpecSet(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    babble = (val["modes"] == MODE_BABBLE).any(dim=1)
    masks = {"all": torch.ones_like(babble), "babble": babble}
    results = {}
    for name in cfg["representations"]:
        for rollout in cfg["train_rollouts"]:
            per_seed = []
            for seed in cfg["seeds"]:
                model = train_dynamics(name, windows["train"], config, rollout, seed, device)
                errors = window_errors(model, name, val, config, specs)
                per_seed.append({s: summarize(errors, m, specs, config) for s, m in masks.items()})
            results[f"{name}|train_rollout={rollout}"] = aggregate(per_seed)
    path = Path(cfg["results_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return results


def format_results(results: dict) -> str:
    """Compact table of the headline metrics (mean over seeds) for all and babble windows."""
    lines = []
    for subset in ("all", "babble"):
        lines.append(f"[{subset} windows]  " + "  ".join(f"{m:>17s}" for m in TABLE_METRICS))
        for key, result in results.items():
            cells = "  ".join(f"{result[subset][m]['mean']:17.4f}" for m in TABLE_METRICS)
            lines.append(f"{key:34s}{cells}")
    return "\n".join(lines)
