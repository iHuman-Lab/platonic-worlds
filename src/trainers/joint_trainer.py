"""Constraint Space JEPA: image and state encoders trained jointly into one common space.

Per modality the loss decodes the constraint vector from C now and from the predictor's
rolled-out future, and matches the rolled-out embeddings to the EMA state-encoder targets of
the future states (JEPA). The anchored dims of the two modalities are aligned and a variance
hinge keeps every dimension from collapsing. With the prediction weights at 0 and
`future: False` this is Step 1 (no JEPA). predictor 'frozen' first trains the state side
(encoder, predictor, decoder) on state alone, freezes it, then trains only the image encoder.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from models.common_space import CommonSpace
from stl.constraints import STATE_INDEX
from stl.representation import to_margins
from stl.specs import SpecSet
from trainers.dynamics_trainer import load_dynamics
from trainers.encoder_trainer import (
    build_encoder, encoder_loaders, evaluate_encoder, inputs_and_targets, seed_everything, train_encoder,
)
from trainers.transfer_eval import evaluate_margin_predictor, score_encoder, transfer_loader
from utils import save_results

PHASES = {"joint": [("image", "state")], "frozen": [("state",), ("image",)]}


def state_extras(batch: dict, config: dict, frames: int, device: str) -> torch.Tensor:
    """Camera-observable extras the state encoder sees besides the constraint vector: [B, 1 + H, E]."""
    columns = [
        batch["preview"] if name == "preview" else batch["states"][..., STATE_INDEX[name]]
        for name in config["joint"]["state_extras"]
    ]
    return torch.stack(columns, dim=-1)[:, frames - 1 :].to(device)


def modality_loss(model: CommonSpace, c0, z, actions, target, cfg: dict) -> torch.Tensor:
    """Constraint loss now and in the future, JEPA loss against the EMA targets, variance hinge."""
    loss = F.mse_loss(model.decoder(c0), z[:, 0])
    loss = loss + cfg["variance_weight"] * F.relu(1.0 - c0.std(dim=0, correction=0)).mean()
    if z.shape[1] > 1:
        pred = model.predict(c0, actions)
        loss = loss + cfg["pred_constraint_weight"] * F.mse_loss(model.decoder(pred), z[:, 1:])
        loss = loss + cfg["jepa_weight"] * F.mse_loss(pred, target)
    return loss


def joint_loss(model: CommonSpace, dynamics, name, batch, config, variant_cfg, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Loss of the active modalities plus image-state alignment; also the first modality's per-dimension error."""
    cfg = config["joint"]
    images, z, actions = inputs_and_targets(batch, dynamics, name, config, variant_cfg["frames"], device)
    extra = state_extras(batch, config, variant_cfg["frames"], device)
    if "state" in model.active:
        model.update_target(cfg["ema"])
    with torch.no_grad():
        target = model.target(z[:, 1:], extra[:, 1:])
    codes = {"state": model.state(z[:, 0], extra[:, 0])}
    if "image" in model.active:
        codes = {"image": model.image(images), **codes}
    loss = sum(modality_loss(model, codes[m], z, actions, target, cfg) for m in model.active)
    if "image" in model.active:
        loss = loss + cfg["align_weight"] * F.mse_loss(codes["image"][:, : model.anchor], codes["state"][:, : model.anchor])
    first = codes[model.active[0]]
    return loss, (model.decoder(first) - z[:, 0]).pow(2).mean(dim=0).detach()


@torch.no_grad()
def embedding_report(model: CommonSpace, dynamics, name, loader, config, frames, device) -> dict:
    """Anchor-dim gap between the modalities and the std of every embedding dimension (collapse monitor)."""
    images_c, state_c = [], []
    for batch in loader:
        images, z, _ = inputs_and_targets(batch, dynamics, name, config, frames, device)
        images_c.append(model.image(images).cpu())
        state_c.append(model.state(z[:, 0], state_extras(batch, config, frames, device)[:, 0]).cpu())
    c_image, c_state = torch.cat(images_c), torch.cat(state_c)
    gap = (c_image - c_state)[:, : model.anchor].pow(2)
    return {
        "alignment_mse": gap.mean().item(),
        "alignment_mse_per_dim": gap.mean(dim=0).tolist(),
        "std_image_per_dim": c_image.std(dim=0).tolist(),
        "std_state_per_dim": c_state.std(dim=0).tolist(),
    }


def own_rollout_scores(model: CommonSpace, dynamics, name, config, specs, frames, device) -> dict:
    """The model's own predictor rolled from each modality's embedding, scored in margin space."""
    loader = transfer_loader(config, frames)
    scores = {}
    for modality in ("image", "state"):
        def predict(batch, shuffle, modality=modality):
            images, z, actions = inputs_and_targets(batch, dynamics, name, config, frames, device)
            c0 = model.image(images) if modality == "image" else model.state(z[:, 0], state_extras(batch, config, frames, device)[:, 0])
            future = model.predict(c0, actions.roll(1, dims=0) if shuffle else actions)
            vectors = dynamics.denormalize(model.decoder(torch.cat([c0.unsqueeze(1), future], dim=1)))
            return to_margins(name, vectors, config).cpu()
        with torch.no_grad():
            scores[modality] = evaluate_margin_predictor(predict, loader, frames - 1, specs, config)
    return scores


def build_common_space(config: dict, variant_cfg: dict, anchor: int) -> CommonSpace:
    """The common-space model with the recipe variant's image encoder."""
    cfg = config["joint"]
    return CommonSpace(build_encoder(config, variant_cfg, anchor + cfg["free_dims"]), anchor, cfg)


def run_joint_experiment(config: dict) -> dict:
    """Train the common space, then score it (frozen dynamics and its own predictor); save both."""
    cfg = config["joint"]
    variant_cfg = config["transfer"]["variants"][cfg["recipe_variant"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(cfg["seed"])
    dynamics, meta = load_dynamics(variant_cfg["dynamics_checkpoint"], device)
    dynamics.requires_grad_(False)
    name, anchor = meta["representation"], dynamics.mean.shape[0]
    model = build_common_space(config, variant_cfg, anchor).to(device)
    loaders = encoder_loaders(config, cfg["recipe_variant"], cfg["future"])
    curve = []
    for active in PHASES[cfg["predictor"]]:
        model.active = active
        if active == ("image",):
            model.requires_grad_(False)
            model.image.requires_grad_(True)
        steps = cfg["state_steps"] if active == ("state",) else variant_cfg["max_steps"]
        model, logs = train_encoder(config, {**variant_cfg, "max_steps": steps}, model, dynamics, name, loaders["train"], device, joint_loss)
        curve += logs["loss"]
    model.eval()
    frames, specs = variant_cfg["frames"], SpecSet(config)
    results = {
        "constraint_reconstruction": evaluate_encoder(model, dynamics, name, loaders["val"], config, frames, device),
        "embeddings": embedding_report(model, dynamics, name, loaders["val"], config, frames, device),
        "transfer_frozen_dynamics": score_encoder(config, variant_cfg, dynamics, name, model, specs),
        "own_predictor": own_rollout_scores(model, dynamics, name, config, specs, frames, device) if cfg["future"] else None,
        "train_loss_curve": curve,
    }
    Path(cfg["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, cfg["checkpoint"])
    save_results(cfg["results_file"], f"{cfg['experiment']}/seed{cfg['seed']}", results)
    return results
