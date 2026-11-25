"""Train a camera encoder to output the constraint-space vector used by the dynamics model.

The target is the representation at the last frame of each window, normalized with the
saved dynamics model's mean/std, so the encoder output lives in the coordinates the
(frozen) dynamics model expects.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAUROC

from dataset.pre_process import augment_images
from dataset.torch_dataset import create_dataloader, list_episodes, split_episodes
from models.constraint_dynamics import ConstraintDynamics
from models.constraint_encoder import ConstraintEncoder
from stl.constraints import constraint_names
from env.road_geometry import STATE_NAMES
from stl.representation import represent, to_margins
from trainers.dynamics_trainer import load_dynamics


def encoder_loaders(config: dict, variant: str) -> dict[str, DataLoader]:
    """Train/val loaders; windows also span the rollout horizon when the variant uses a rollout loss."""
    variant_cfg = config["transfer"]["variants"][variant]
    horizon = config["dynamics"]["horizon"] if variant_cfg["rollout_weight"] > 0 else 0
    dataset = {
        **config["dataset"],
        "window_length": variant_cfg["frames"] + horizon,
        "batch_size": config["encoder"]["batch_size"],
    }
    split = split_episodes(list_episodes(variant_cfg["dataset_root"]), config)
    return {
        name: create_dataloader([e["path"] for e in split[name]], {**config, "dataset": dataset}, name == "train")
        for name in ("train", "val")
    }


def inputs_and_targets(
    batch: dict, dynamics: ConstraintDynamics, name: str, config: dict, frames: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Context frames, the normalized representation from the last context frame on, and the actions.

    targets: [B, 1 + horizon', D] (horizon' is 0 for windows without a rollout part);
    actions: [B, horizon', A], the actions taken from the last context frame onwards.
    """
    start = frames - 1
    images = batch["images"][:, :frames].squeeze(2).to(device)
    targets = dynamics.normalize(represent(name, batch, config)[:, start:].to(device))
    return images, targets, batch["actions"][:, start:].to(device)


def build_encoder(config: dict, variant_cfg: dict, out_dim: int) -> ConstraintEncoder:
    """Encoder with the variant's architecture, frames and pooling."""
    return ConstraintEncoder(
        variant_cfg["frames"],
        config["encoder"]["dim"],
        config["dataset"]["image_size"],
        out_dim,
        variant_cfg["arch"],
        variant_cfg["pool"],
    )


def lr_scale(step: int, variant_cfg: dict) -> float:
    """Learning-rate multiplier: linear warm-up, then constant or cosine decay to zero."""
    warmup = variant_cfg["warmup_steps"]
    if step < warmup:
        return (step + 1) / warmup
    if variant_cfg["lr_schedule"] == "cosine":
        progress = (step - warmup) / max(variant_cfg["max_steps"] - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return 1.0


def encoder_loss(
    model: ConstraintEncoder, dynamics: ConstraintDynamics, name: str, batch: dict, config: dict, variant_cfg: dict, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Alignment of the step-0 estimate plus the frozen dynamics' rollout error; and the per-dimension step-0 error."""
    images, targets, actions = inputs_and_targets(batch, dynamics, name, config, variant_cfg["frames"], device)
    if variant_cfg["augment"]:
        images = augment_images(images, config["encoder"]["augmentation"])
    c0 = model(images)
    loss = variant_cfg["alignment_weight"] * F.mse_loss(c0, targets[:, 0])
    if variant_cfg["rollout_weight"] > 0:
        loss = loss + variant_cfg["rollout_weight"] * F.mse_loss(dynamics.rollout(c0, actions), targets[:, 1:])
    return loss, ((c0 - targets[:, 0]) ** 2).mean(dim=0).detach()


def train_encoder(
    config: dict, variant_cfg: dict, dynamics: ConstraintDynamics, name: str, loader: DataLoader, device: str
) -> tuple[ConstraintEncoder, dict]:
    """Fit the encoder (the dynamics stay frozen); returns the model and the training logs."""
    cfg = config["encoder"]
    out_dim = dynamics.mean.shape[0]
    model = build_encoder(config, variant_cfg, out_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_scale(step, variant_cfg))
    max_norm = variant_cfg["grad_clip"] if variant_cfg["grad_clip"] > 0 else float("inf")
    logs = {"loss": [], "grad_norm_mean": [], "grad_norm_max": [], "per_dim_loss": []}
    running, norms, dims, step = [], [], [], 0
    while step < variant_cfg["max_steps"]:
        for batch in loader:
            loss, per_dim = encoder_loss(model, dynamics, name, batch, config, variant_cfg, device)
            optimizer.zero_grad()
            loss.backward()
            norms.append(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm).item())
            optimizer.step()
            scheduler.step()
            running.append(loss.item())
            dims.append(per_dim)
            step += 1
            if step % cfg["log_every"] == 0:
                logs["loss"].append(sum(running) / len(running))
                logs["grad_norm_mean"].append(sum(norms) / len(norms))
                logs["grad_norm_max"].append(max(norms))
                logs["per_dim_loss"].append(torch.stack(dims).mean(dim=0).tolist())
                running, norms, dims = [], [], []
            if step >= variant_cfg["max_steps"]:
                break
    return model, logs


def dimension_names(name: str, config: dict) -> list[str]:
    """Names of the representation's dimensions, in order."""
    names = constraint_names(config)
    if name == "raw_state":
        return list(STATE_NAMES)
    if name == "ego_state":
        return list(config["dynamics"]["ego_states"])
    if name == "margins":
        return names
    names = names + config["dynamics"]["augment_states"]
    return names + ["preview"] if name == "augmented_preview" else names


@torch.no_grad()
def evaluate_encoder(
    model: ConstraintEncoder, dynamics: ConstraintDynamics, name: str, loader: DataLoader, config: dict, frames: int, device: str
) -> dict:
    """R^2 per dimension (normalized space) and violation AUROC per constraint."""
    model.eval()
    preds, targets = [], []
    for batch in loader:
        images, target, _ = inputs_and_targets(batch, dynamics, name, config, frames, device)
        preds.append(model(images).cpu())
        targets.append(target[:, 0].cpu())
    pred, target = torch.cat(preds), torch.cat(targets)
    r2 = 1 - ((pred - target) ** 2).sum(0) / ((target - target.mean(0)) ** 2).sum(0)
    mean, std = dynamics.mean.cpu(), dynamics.std.cpu()
    margins_pred = to_margins(name, pred * std + mean, config)
    margins_true = to_margins(name, target * std + mean, config)
    names = dimension_names(name, config)
    auroc = {}
    for index, constraint in enumerate(constraint_names(config)):
        violated = (margins_true[:, index] < 0).int()
        auroc[constraint] = BinaryAUROC()(-margins_pred[:, index], violated).item()
    return {
        "windows": len(pred),
        "mse_normalized": ((pred - target) ** 2).mean().item(),
        "r2": {n: r2[i].item() for i, n in enumerate(names)},
        "auroc": auroc,
    }


def seed_everything(seed: int) -> None:
    """Seed torch, numpy and python's random (the loader shuffles with python's random)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_encoder_training(config: dict, variant: str) -> dict:
    """Train the camera encoder against the variant's saved dynamics model; save it and its scores."""
    cfg, variant_cfg = config["encoder"], config["transfer"]["variants"][variant]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(variant_cfg["seed"])
    dynamics, meta = load_dynamics(variant_cfg["dynamics_checkpoint"], device)
    dynamics.requires_grad_(False)
    name = meta["representation"]
    loaders = encoder_loaders(config, variant)
    model, logs = train_encoder(config, variant_cfg, dynamics, name, loaders["train"], device)
    metrics = evaluate_encoder(model, dynamics, name, loaders["val"], config, variant_cfg["frames"], device)
    metrics["train_loss_curve"] = logs["loss"]
    metrics["train_logs"] = logs
    checkpoint = Path(variant_cfg["encoder_checkpoint"])
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "representation": name, **cfg, "frames": variant_cfg["frames"]}, checkpoint)
    Path(variant_cfg["encoder_results"]).write_text(json.dumps(metrics, indent=2))
    return metrics
