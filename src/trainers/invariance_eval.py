"""Step 3 of Constraint Space JEPA: do the two modalities share one situation, geometry and predictor?

All tests use paired windows (the same timestep seen as an image and as a state) from the
validation split, and every number comes with a chance or shuffled-pairs baseline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from stl.representation import to_margins
from trainers.dynamics_trainer import load_dynamics
from trainers.encoder_trainer import inputs_and_targets
from trainers.joint_trainer import build_common_space, state_extras
from trainers.transfer_eval import transfer_loader
from utils import save_results


@torch.no_grad()
def collect_embeddings(model, dynamics, name: str, config: dict, frames: int, device: str) -> dict:
    """Both modalities' embeddings now and rolled out, the EMA targets and the constraint vector."""
    parts = {key: [] for key in ("c_image", "c_state", "z", "pred_image", "pred_state", "target")}
    for batch in transfer_loader(config, frames):
        images, z, actions = inputs_and_targets(batch, dynamics, name, config, frames, device)
        extra = state_extras(batch, config, frames, device)
        c_image, c_state = model.image(images), model.state(z[:, 0], extra[:, 0])
        values = {
            "c_image": c_image, "c_state": c_state, "z": z[:, 0],
            "pred_image": model.predict(c_image, actions), "pred_state": model.predict(c_state, actions),
            "target": model.target(z[:, 1:], extra[:, 1:]),
        }
        for key, value in values.items():
            parts[key].append(value.cpu())
    return {key: torch.cat(value) for key, value in parts.items()}


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm())).item()


def same_situation(e: dict, anchor: int, cfg: dict, gen: torch.Generator) -> dict:
    """Paired gap vs shuffled pairs, and retrieval of the true state embedding from the image embedding."""
    n = cfg["subset"]
    idx = torch.randperm(len(e["z"]), generator=gen)[:n]
    c_image, c_state = e["c_image"][idx, :anchor], e["c_state"][idx, :anchor]
    distances = torch.cdist(c_image, c_state)
    closer = (distances < distances.diag().unsqueeze(1)).sum(dim=1)
    return {
        "paired_mse": (c_image - c_state).pow(2).mean().item(),
        "shuffled_pairs_mse": (c_image - c_state[torch.randperm(n, generator=gen)]).pow(2).mean().item(),
        "retrieval_top1": (closer == 0).float().mean().item(),
        "retrieval_chance": 1.0 / n,
        "retrieval_median_rank": closer.float().median().item() + 1,
    }


def same_geometry(e: dict, anchor: int, cfg: dict, gen: torch.Generator) -> dict:
    """Correlation of pairwise distances: each modality's embeddings against the constraint vector's, and each other."""
    idx = torch.randperm(len(e["z"]), generator=gen)[: cfg["geometry_subset"]]
    upper = torch.triu_indices(len(idx), len(idx), offset=1)
    d = lambda x: torch.cdist(x, x)[upper[0], upper[1]]
    d_z = d(e["z"][idx])
    result = {}
    for label, dims in (("anchored", slice(0, anchor)), ("all", slice(None))):
        d_image, d_state = d(e["c_image"][idx][:, dims]), d(e["c_state"][idx][:, dims])
        result[label] = {
            "image_vs_constraint": pearson(d_image, d_z), "state_vs_constraint": pearson(d_state, d_z),
            "image_vs_state": pearson(d_image, d_state),
        }
    return result


def fit_probe(x_train, y_train, x_test, y_test, cfg: dict) -> float:
    """Test accuracy of a linear classifier on standardized features."""
    mean, std = x_train.mean(0), x_train.std(0) + 1e-6
    x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
    w, b = torch.zeros(x_train.shape[1], requires_grad=True), torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=cfg["probe_lr"])
    for _ in range(cfg["probe_steps"]):
        optimizer.zero_grad()
        F.binary_cross_entropy_with_logits(x_train @ w + b, y_train).backward()
        optimizer.step()
    return (((x_test @ w + b) > 0).float() == y_test).float().mean().item()


def modality_probe(e: dict, anchor: int, cfg: dict, gen: torch.Generator) -> dict:
    """Can a linear probe tell image embeddings from state embeddings of the same windows? 0.5 is chance."""
    n, result = len(e["z"]), {}
    split = int(cfg["probe_train_fraction"] * n)
    labels = lambda count: torch.cat([torch.zeros(count), torch.ones(count)])
    for label, dims in (("anchored", slice(0, anchor)), ("all", slice(None))):
        x = lambda lo, hi: torch.cat([e["c_image"][lo:hi, dims], e["c_state"][lo:hi, dims]])
        x_train, x_test, y_test = x(0, split), x(split, n), labels(n - split)
        y_train = labels(split)
        result[label] = {
            "accuracy": fit_probe(x_train, y_train, x_test, y_test, cfg),
            "shuffled_label_accuracy": fit_probe(x_train, y_train[torch.randperm(len(y_train), generator=gen)], x_test, y_test, cfg),
        }
    return result


def same_prediction(e: dict, model, dynamics, name: str, config: dict, anchor: int, gen: torch.Generator) -> dict:
    """Gap between the futures predicted from the two modalities, in embedding and in margin space."""
    steps = {"step1": 0, "step4": 3, "step8": 7}
    gap = lambda a, b, dims: {s: (a[:, i, dims] - b[:, i, dims]).pow(2).mean().item() for s, i in steps.items()}
    shuffled = e["pred_state"][torch.randperm(len(e["z"]), generator=gen)]
    margins = lambda pred: to_margins(name, dynamics.denormalize(model.decoder(pred.to(dynamics.mean.device))), config).cpu()
    m_image, m_state = margins(e["pred_image"]), margins(e["pred_state"])
    return {
        "embedding_gap_image_vs_state_anchored": gap(e["pred_image"], e["pred_state"], slice(0, anchor)),
        "embedding_gap_shuffled_pairs_anchored": gap(e["pred_image"], shuffled, slice(0, anchor)),
        "embedding_gap_image_vs_target_all": gap(e["pred_image"], e["target"], slice(None)),
        "embedding_gap_state_vs_target_all": gap(e["pred_state"], e["target"], slice(None)),
        "margin_gap_image_vs_state_step8": (m_image[:, 7] - m_state[:, 7]).pow(2).mean().item(),
    }


def run_invariance_experiment(config: dict) -> dict:
    """Run the three invariance tests on every trained seed of the configured joint experiment."""
    cfg, joint = config["invariance"], config["joint"]
    variant_cfg = config["transfer"]["variants"][joint["recipe_variant"]]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dynamics, meta = load_dynamics(variant_cfg["dynamics_checkpoint"], device)
    name, anchor, results = meta["representation"], dynamics.mean.shape[0], {}
    for seed in cfg["seeds"]:
        model = build_common_space(config, variant_cfg, anchor).to(device)
        model.load_state_dict(torch.load(cfg["checkpoint"].format(experiment=cfg["experiment"], seed=seed), map_location=device)["state_dict"])
        model.eval()
        gen = torch.Generator().manual_seed(config["random_state"])
        e = collect_embeddings(model, dynamics, name, config, variant_cfg["frames"], device)
        results[f"seed{seed}"] = {
            "windows": len(e["z"]),
            "same_situation": same_situation(e, anchor, cfg, gen),
            "same_geometry": same_geometry(e, anchor, cfg, gen),
            "modality_probe": modality_probe(e, anchor, cfg, gen),
            "same_prediction": same_prediction(e, model, dynamics, name, config, anchor, gen),
        }
        save_results(joint["results_file"], f"invariance/{cfg['experiment']}/seed{seed}", results[f"seed{seed}"])
    return results
