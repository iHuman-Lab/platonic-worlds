"""Frozen-dynamics transfer: feed a camera encoder's output to a frozen dynamics model.

For each validation window the encoder reads the first `encoder.frames` frames and the
frozen dynamics model rolls out the next `dynamics.horizon` actions. The same rollout from
the true representation (the "oracle") gives the upper bound. Everything is scored in
constraint-margin space, so a constraint-space and a raw-state variant are comparable.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset.torch_dataset import create_dataloader, list_episodes, split_episodes
from env.palette import road_grass_contrast
from env.random_road_agent import MODE_BABBLE
from models.constraint_dynamics import ConstraintDynamics
from models.constraint_encoder import ConstraintEncoder
from stl.constraints import constraint_margins
from stl.representation import to_margins
from stl.specs import SpecSet
from trainers.conv_jepa import ConvJEPA
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


def transfer_loader(config: dict, frames: int, files: list[str] | None = None) -> DataLoader:
    """Windows of `frames + horizon` frames with images: the validation split, or the given tar files."""
    length = frames + config["dynamics"]["horizon"]
    dataset = {**config["dataset"], "window_length": length, "batch_size": config["encoder"]["batch_size"]}
    if files is None:
        split = split_episodes(list_episodes(config["dataset"]["root"]), config)
        files = [e["path"] for e in split["val"]]
    return create_dataloader(files, {**config, "dataset": dataset}, False)


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


def evaluate_variant(config: dict, variant: str, specs: SpecSet, device: str, files: list[str] | None = None) -> dict:
    """Oracle and transfer metrics for one variant, on all windows and on babble windows."""
    paths = config["transfer"]["variants"][variant]
    model, meta = load_dynamics(paths["dynamics_checkpoint"], device)
    model.requires_grad_(False)
    name = meta["representation"]
    frames = paths["frames"]
    encoder = load_encoder(paths["encoder_checkpoint"], config, paths, model.mean.shape[0], device)
    loader = transfer_loader(config, frames, files)
    oracle, transfer, extras, babble = collect_errors(model, encoder, name, loader, config, specs, frames)
    result = {}
    for subset, mask in {"all": torch.ones_like(babble), "babble": babble}.items():
        transfer_summary = summarize(transfer, mask, specs, config)
        transfer_summary["mse_step0"] = extras["est_step0"][mask].mean().item()
        transfer_summary["est_persistence_step8"] = extras["est_persistence"][mask][:, -1].mean().item()
        result[subset] = {"oracle": summarize(oracle, mask, specs, config), "transfer": transfer_summary}
    return result


@torch.no_grad()
def conv_jepa_margins(model: ConvJEPA, batch: dict, shuffle_actions: bool = False) -> torch.Tensor:
    """ConvJEPA's own margins for frames start..end: the head on z now, then on each rolled latent."""
    device = next(model.parameters()).device
    context, _, taken, _ = model.split_batch(batch)
    taken = taken.to(device)
    z = torch.nn.functional.layer_norm(model.encoder(context.to(device)), (model.dim,))
    rolled = model.rollout(z, taken.roll(1, dims=0) if shuffle_actions else taken)
    return torch.cat([model.constraint_head(z).unsqueeze(1), model.constraint_head(rolled)], dim=1).cpu()


def conv_jepa_window_errors(model: ConvJEPA, batch: dict, specs: SpecSet) -> tuple[dict, dict]:
    """Errors in the layout of window_errors, plus the step-0 and persistence-from-estimate extras."""
    true = model.split_batch(batch)[3]
    pred = conv_jepa_margins(model, batch)
    shuffled = conv_jepa_margins(model, batch, shuffle_actions=True)
    errors = {
        "squared": ((pred - true)[:, 1:] ** 2),
        "persistence": ((true[:, :1] - true)[:, 1:] ** 2),
        "shuffled": ((shuffled - true)[:, 1:] ** 2),
        "rho_pred": specs(pred, smooth=False),
        "rho_true": specs(true, smooth=False),
    }
    extras = {"est_step0": (pred[:, 0] - true[:, 0]) ** 2, "est_persistence": (pred[:, :1] - true)[:, 1:] ** 2}
    return errors, extras


def evaluate_conv_jepa(config: dict, specs: SpecSet, device: str, files: list[str] | None = None) -> dict:
    """Score the saved ConvJEPA on the transfer validation windows with the transfer metrics."""
    model = ConvJEPA.load_from_checkpoint(config["conv_jepa_checkpoint"], config=config, map_location=device)
    model.eval()
    loader = transfer_loader(config, config["conv_jepa_frames"], files)
    pairs = [conv_jepa_window_errors(model, batch, specs) for batch in loader]
    errors, extras = stack([p[0] for p in pairs]), stack([p[1] for p in pairs])
    start = config["conv_jepa_frames"] - 1
    babble = torch.cat([(b["modes"][:, start:] == MODE_BABBLE).any(dim=1) for b in loader])
    result = {}
    for subset, mask in {"all": torch.ones_like(babble), "babble": babble}.items():
        summary = summarize(errors, mask, specs, config)
        summary["mse_step0"] = extras["est_step0"][mask].mean().item()
        summary["est_persistence_step8"] = extras["est_persistence"][mask][:, -1].mean().item()
        result[subset] = {"transfer": summary}
    return result


def evaluate_all(config: dict, files: list[str] | None = None) -> dict:
    """Every active variant, plus ConvJEPA when enabled, on the validation split or the given tars."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = SpecSet(config)
    variants = [v for v, cfg in config["transfer"]["variants"].items() if cfg["active"]]
    results = {v: evaluate_variant(config, v, specs, device, files) for v in variants}
    if config["transfer"]["include_conv_jepa"]:
        results["conv_jepa"] = evaluate_conv_jepa(config, specs, device, files)
    return results


def save_results(results: dict, path: str) -> None:
    """Write a results dict as JSON, creating the folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(results, indent=2))


def run_transfer_experiment(config: dict) -> dict:
    """Evaluate every variant on the validation split and save the results."""
    results = evaluate_all(config)
    save_results(results, config["transfer"]["results_file"])
    return results


def contrast_groups(config: dict, episodes: list[dict]) -> dict[str, list[dict]]:
    """Randomized episodes grouped by road-grass luminance gap (bins from the config) and by reversal."""
    low, high = config["transfer"]["adaptation"]["contrast_bins"]
    contrast = road_grass_contrast([e["seed"] for e in episodes])
    groups = {
        f"gap<{low}": lambda c: c["gap"] < low,
        f"gap {low}-{high}": lambda c: low <= c["gap"] < high,
        f"gap>={high}": lambda c: c["gap"] >= high,
        "road brighter than grass": lambda c: c["reversed"],
        "road darker than grass": lambda c: not c["reversed"],
        "all randomized": lambda c: True,
    }
    return {name: [e for e in episodes if keep(contrast[e["seed"]])] for name, keep in groups.items()}


def heldout_randomized(config: dict) -> list[dict]:
    """Randomized episodes no encoder trained on: the shift run plus the randomized set's val and test seeds."""
    cfg = config["transfer"]["adaptation"]
    shift = [e for e in list_episodes(cfg["shift_root"]) if e["run"] == cfg["randomized_run"]]
    split = split_episodes(list_episodes(cfg["train_root"]), config)
    return shift + split["val"] + split["test"]


def run_adaptation_experiment(config: dict) -> dict:
    """Original vs randomized-trained encoder, dynamics frozen, on standard and contrast-grouped randomized episodes."""
    cfg = config["transfer"]["adaptation"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    specs = SpecSet(config)
    standard = [e["path"] for e in list_episodes(cfg["shift_root"]) if e["run"] == cfg["standard_run"]]
    groups = {"standard colours": [{"path": path} for path in standard]}
    groups.update(contrast_groups(config, heldout_randomized(config)))
    results = {}
    for group, episodes in groups.items():
        files = [e["path"] for e in episodes]
        results[group] = {"episodes": len(files)}
        for variant in cfg["variants"]:
            results[group][variant] = evaluate_variant(config, variant, specs, device, files)
    save_results(results, cfg["results_file"])
    return results


def cell(subset_result: dict, source: str, metric: str) -> str:
    """One table cell; '-' where the variant has no such source (ConvJEPA has no oracle)."""
    if source not in subset_result:
        return f"{'-':>19s}"
    return f"{subset_result[source][metric]:19.4f}"


def format_transfer(results: dict) -> str:
    """Table of the headline metrics per variant, for all and babble windows."""
    header = "  ".join(f"{source[:4]}.{metric[:14]:>14s}" for source, metric in TABLE_COLUMNS)
    lines = []
    for subset in ("all", "babble"):
        lines.append(f"[{subset} windows]  {header}")
        for variant, result in results.items():
            cells = "  ".join(cell(result[subset], source, metric) for source, metric in TABLE_COLUMNS)
            lines.append(f"{variant:16s}{cells}")
    return "\n".join(lines)


def format_adaptation(results: dict) -> str:
    """One row per contrast group and encoder: oracle, step 0, step 8, STL error, mean violation AUROC."""
    lines = [f"{'group':26s}{'episodes':>9s}  {'encoder':36s}{'oracle8':>8s}{'step0':>8s}{'step8':>8s}{'stl':>8s}{'auroc':>8s}"]
    for group, group_results in results.items():
        for variant, result in group_results.items():
            if variant == "episodes":
                continue
            metrics, oracle = result["all"]["transfer"], result["all"]["oracle"]
            aurocs = [value for key, value in metrics.items() if key.startswith("auroc/")]
            auroc = float(np.nanmean(aurocs))
            lines.append(
                f"{group:26s}{group_results['episodes']:9d}  {variant:36s}{oracle['mse_step8']:8.3f}"
                f"{metrics['mse_step0']:8.3f}{metrics['mse_step8']:8.3f}{metrics['stl_mse']:8.3f}{auroc:8.3f}"
            )
    return "\n".join(lines)
