"""Frozen-dynamics transfer: feed a camera encoder's output to a frozen dynamics model.

For each validation window the encoder reads the first `encoder.frames` frames and the
frozen dynamics model rolls out the next `dynamics.horizon` actions. The same rollout from
the true representation (the "oracle") gives the upper bound. Everything is scored in
constraint-margin space, so a constraint-space and a raw-state variant are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.torch_dataset import create_dataloader, list_episodes, split_episodes
from env.random_road_agent import MODE_BABBLE
from models.constraint_dynamics import ConstraintDynamics
from models.constraint_encoder import ConstraintEncoder
from stl.constraints import constraint_margins
from stl.representation import to_margins
from stl.specs import SpecSet
from trainers.dynamics_trainer import load_dynamics, summarize, window_errors
from trainers.encoder_trainer import build_encoder

TABLE_COLUMNS = (
    ("oracle", "mse_step8"),
    ("transfer", "mse_step0"),
    ("transfer", "mse_step8"),
    ("transfer", "shuffled_step8"),
    ("transfer", "est_persistence_step8"),
    ("transfer", "persistence_step8"),
    ("oracle", "stl_mse"),
    ("transfer", "stl_mse"),
)


def transfer_loader(config: dict, frames: int) -> DataLoader:
    """Validation windows of `frames + horizon` frames with images."""
    length = frames + config["dynamics"]["horizon"]
    dataset = {**config["dataset"], "window_length": length, "batch_size": config["encoder"]["batch_size"]}
    split = split_episodes(list_episodes(config["dataset"]["root"]), config)
    return create_dataloader([e["path"] for e in split["val"]], {**config, "dataset": dataset}, False)


def load_encoder(path: str, config: dict, variant_cfg: dict, out_dim: int, device: str) -> ConstraintEncoder:
    """Rebuild a saved camera encoder."""
    encoder = build_encoder(config, variant_cfg, out_dim)
    encoder.load_state_dict(torch.load(path, map_location=device, weights_only=False)["state_dict"])
    return encoder.to(device).eval()


def stack(batches: list[dict]) -> dict[str, torch.Tensor]:
    """Concatenate a list of dicts of tensors along the first dimension."""
    return {key: torch.cat([b[key] for b in batches]) for key in batches[0]}


@torch.no_grad()
def collect_errors(
    model: ConstraintDynamics,
    encoder: ConstraintEncoder,
    name: str,
    loader: DataLoader,
    config: dict,
    specs: SpecSet,
    frames: int,
) -> tuple[dict, dict, dict, torch.Tensor]:
    """Oracle errors, transfer errors, estimation extras and the babble mask over the loader."""
    start = frames - 1
    device = model.mean.device
    oracle, transfer, extras, babble = [], [], [], []
    for batch in loader:
        c_enc = encoder(batch["images"][:, :frames].squeeze(2).to(device))
        oracle.append(window_errors(model, name, batch, config, specs, start))
        transfer.append(window_errors(model, name, batch, config, specs, start, c0=c_enc))
        true = constraint_margins(batch["states"], batch["extras"], config)[:, start:].to(device)
        estimate = to_margins(name, model.denormalize(c_enc).unsqueeze(1), config)
        extras.append(
            {
                "est_step0": ((estimate[:, 0] - true[:, 0]) ** 2).cpu(),
                "est_persistence": ((estimate - true)[:, 1:] ** 2).cpu(),
            }
        )
        babble.append((batch["modes"][:, start:] == MODE_BABBLE).any(dim=1))
    return stack(oracle), stack(transfer), stack(extras), torch.cat(babble)


def evaluate_variant(config: dict, variant: str, specs: SpecSet, device: str) -> dict:
    """Oracle and transfer metrics for one variant, on all windows and on babble windows."""
    paths = config["transfer"]["variants"][variant]
    model, meta = load_dynamics(paths["dynamics_checkpoint"], device)
    model.requires_grad_(False)
    name = meta["representation"]
    frames = paths["frames"]
    encoder = load_encoder(paths["encoder_checkpoint"], config, paths, model.mean.shape[0], device)
    loader = transfer_loader(config, frames)
    oracle, transfer, extras, babble = collect_errors(model, encoder, name, loader, config, specs, frames)
    result = {}
    for subset, mask in {"all": torch.ones_like(babble), "babble": babble}.items():
        transfer_summary = summarize(transfer, mask, specs, config)
        transfer_summary["mse_step0"] = extras["est_step0"][mask].mean().item()
        transfer_summary["est_persistence_step8"] = extras["est_persistence"][mask][:, -1].mean().item()
        result[subset] = {"oracle": summarize(oracle, mask, specs, config), "transfer": transfer_summary}
    return result


def run_transfer_experiment(config: dict) -> dict:
    """Evaluate every variant and save the results."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = SpecSet(config)
    variants = [v for v, cfg in config["transfer"]["variants"].items() if cfg["active"]]
    results = {v: evaluate_variant(config, v, specs, device) for v in variants}
    path = Path(config["transfer"]["results_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))
    return results


def format_transfer(results: dict) -> str:
    """Table of the headline metrics per variant, for all and babble windows."""
    header = "  ".join(f"{source[:4]}.{metric[:14]:>14s}" for source, metric in TABLE_COLUMNS)
    lines = []
    for subset in ("all", "babble"):
        lines.append(f"[{subset} windows]  {header}")
        for variant, result in results.items():
            cells = "  ".join(f"{result[subset][source][metric]:19.4f}" for source, metric in TABLE_COLUMNS)
            lines.append(f"{variant:16s}{cells}")
    return "\n".join(lines)
