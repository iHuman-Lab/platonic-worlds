import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC, BinaryRecall

from models.conv_jepa import ConvEncoder
from stl.constraints import constraint_margins, constraint_names
from stl.specs import SpecSet

PROGRESS_BAR_METRICS = ("jepa", "jepa_last", "copy_last", "jepa_wrong_last", "stl")


class ConvJEPA(pl.LightningModule):
    """
    Action-conditioned conv JEPA for CarRacing with a constraint head and an STL loss.

    The student encodes `ctx` context frames into a latent z. A one-step predictor
    F(z, a) is rolled out `horizon` times with the actions taken; each rolled-out latent
    is trained to match the EMA-teacher latent of the frames at that step (L1 on
    layer-normed latents, plus a variance hinge on z across the batch against collapse).

    A constraint head reads z and every rolled-out latent and regresses the margins of
    stl.constraints. The STL loss compares the robustness of the predicted margin
    trajectory (current frame + horizon predicted frames) with that of the true one.
    With constraint_to_encoder False the head's gradient is detached, so it only probes
    what a JEPA-only encoder already encodes.

    Batches are the dicts from dataset.torch_dataset: images [B, ctx + horizon, 1, S, S]
    (grayscale) and continuous actions [B, ctx + horizon - 1, action_dim].
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.ctx = config["conv_jepa_frames"]
        self.horizon = config["conv_jepa_horizon"]
        self.ema = config["conv_jepa_ema"]
        self.reg_coeff = config["conv_jepa_reg"]
        self.lr = config["conv_jepa_lr"]
        self.weight_decay = config["conv_jepa_weight_decay"]
        self.constraint_weight = config["conv_jepa_constraint_weight"]
        self.constraint_to_encoder = config["conv_jepa_constraint_to_encoder"]
        self.violation_weight = config["conv_jepa_violation_weight"]
        self.stl_weight = config["conv_jepa_stl_weight"]
        self.constraint_names = constraint_names(config)
        self.specs = SpecSet(config)
        self.dim = config["conv_jepa_dim"]
        act_dim = config["conv_jepa_act_embed"]
        self.encoder = ConvEncoder(self.ctx, self.dim, config["dataset"]["image_size"])
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.act_embed = torch.nn.Linear(config["conv_jepa_action_dim"], act_dim)
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(self.dim + act_dim, 2 * self.dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * self.dim, self.dim),
        )
        hidden = config["conv_jepa_constraint_hidden"]
        self.constraint_head = torch.nn.Sequential(
            torch.nn.Linear(self.dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, len(self.constraint_names)),
        )
        keys = [f"{when}_{name}" for when in ("now", "last") for name in self.constraint_names]
        self.val_auroc = torch.nn.ModuleDict({key: BinaryAUROC() for key in keys})
        self.val_recall = torch.nn.ModuleDict({key: BinaryRecall() for key in keys})

    def on_train_batch_end(self, *args):
        with torch.no_grad():
            for s, t in zip(self.encoder.parameters(), self.target.parameters()):
                t.mul_(self.ema).add_(s, alpha=1 - self.ema)

    def split_batch(self, batch):
        """Return context frames, the future frame stacks, the actions and the true margins."""
        images, actions = batch["images"], batch["actions"]
        if images.shape[1] != self.ctx + self.horizon or images.shape[2] != 1:
            raise ValueError(
                f"Need windows of {self.ctx + self.horizon} grayscale frames "
                f"(dataset.window_length, preprocess), got images {tuple(images.shape)}"
            )
        frames = images.squeeze(2)  # [B, ctx + horizon, S, S]
        context = frames[:, : self.ctx]
        futures = torch.stack(  # step j sees the ctx frames ending at frame ctx - 1 + j
            [frames[:, j : j + self.ctx] for j in range(1, self.horizon + 1)], dim=1
        )
        taken = actions[:, self.ctx - 1 : self.ctx - 1 + self.horizon]  # a_t .. a_{t+horizon-1}
        margins = constraint_margins(batch["states"], batch["extras"], self.config)
        return context, futures, taken, margins[:, self.ctx - 1 :]

    def encode_targets(self, futures):
        """EMA-teacher latents of every future frame stack: [B, horizon, dim]."""
        batch_size = futures.shape[0]
        with torch.no_grad():
            tgt = self.target(futures.flatten(0, 1)).view(batch_size, self.horizon, -1)
            return F.layer_norm(tgt, (self.dim,))

    def rollout(self, z, taken):
        """Roll the one-step predictor forward with the actions taken: [B, horizon, dim]."""
        steps = []
        for j in range(self.horizon):
            delta = self.pred(torch.cat([z, self.act_embed(taken[:, j])], dim=-1))
            z = F.layer_norm(z + delta, (self.dim,))
            steps.append(z)
        return torch.stack(steps, dim=1)

    def weighted_mse(self, pred, target):
        """Squared error, counted violation_weight times more where the margin is below 0."""
        weights = 1.0 + (self.violation_weight - 1.0) * (target < 0).float()
        return (weights * (pred - target) ** 2).mean()

    def constraint_losses(self, z, rolled, margins):
        """Regress constraint margins from z (now) and from the rolled-out latents (future)."""
        if not self.constraint_to_encoder:
            z, rolled = z.detach(), rolled.detach()
        now, future = self.constraint_head(z), self.constraint_head(rolled)
        now_target, future_target = margins[:, 0], margins[:, 1:]
        trajectory = torch.cat([now.unsqueeze(1), future], dim=1)
        stl = F.mse_loss(self.specs(trajectory, smooth=True), self.specs(margins, smooth=True))
        losses = {
            "constraint_now": F.mse_loss(now, now_target),
            "constraint_future": F.mse_loss(future, future_target),
            "constraint_weighted": self.weighted_mse(now, now_target)
            + self.weighted_mse(future, future_target),
            "stl": stl,
        }
        with torch.no_grad():
            last_right = (future[:, -1] > 0) == (future_target[:, -1] > 0)
            losses["sign_acc_now"] = ((now > 0) == (now_target > 0)).float().mean()
            losses["sign_acc_last"] = last_right.float().mean()
        return losses, (now, now_target), (future[:, -1], future_target[:, -1])

    def losses(self, batch):
        context, futures, taken, margins = self.split_batch(batch)
        z = F.layer_norm(self.encoder(context), (self.dim,))
        tgt = self.encode_targets(futures)
        rolled = self.rollout(z, taken)
        jepa_steps = (rolled - tgt).abs().mean(dim=(0, 2))  # L1 per rollout step
        between = z.std(dim=0)
        reg = F.relu(1.0 - between).mean()
        with torch.no_grad():
            # Do the actions matter? Rolling out with another sample's actions should hurt.
            wrong = (self.rollout(z, taken.roll(1, dims=0)) - tgt).abs().mean(dim=(0, 2))
            copy_steps = (z.detach().unsqueeze(1) - tgt).abs().mean(dim=(0, 2))
        constraint, now_pair, last_pair = self.constraint_losses(z, rolled, margins)
        losses = {
            "loss": jepa_steps.mean()
            + self.reg_coeff * reg
            + self.constraint_weight * constraint["constraint_weighted"]
            + self.stl_weight * constraint["stl"],
            **constraint,
            "jepa": jepa_steps.mean(),
            "jepa_last": jepa_steps[-1],
            "copy_last": copy_steps[-1],
            "jepa_wrong": wrong.mean(),
            "jepa_wrong_last": wrong[-1],
            "reg": reg,
            "between_std": between.mean(),
        }
        return losses, now_pair, last_pair

    def log_losses(self, stage, losses, batch_size):
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_epoch=True,
                prog_bar=name in PROGRESS_BAR_METRICS,
                batch_size=batch_size,
            )

    def update_violation_metrics(self, when, pair):
        """Violation = margin below 0; the AUROC score is the negated predicted margin."""
        predicted, target = pair
        for index, name in enumerate(self.constraint_names):
            key = f"{when}_{name}"
            violated = (target[:, index] < 0).int()
            self.val_auroc[key].update(-predicted[:, index], violated)
            self.val_recall[key].update((predicted[:, index] < 0).int(), violated)
            self.log(f"val/auroc_{when}/{name}", self.val_auroc[key], on_epoch=True)
            self.log(f"val/recall_{when}/{name}", self.val_recall[key], on_epoch=True)

    def training_step(self, batch, batch_idx):
        losses, _, _ = self.losses(batch)
        self.log_losses("train", losses, batch["images"].shape[0])
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses, now_pair, last_pair = self.losses(batch)
        self.log_losses("val", losses, batch["images"].shape[0])
        self.update_violation_metrics("now", now_pair)
        self.update_violation_metrics("last", last_pair)

    def configure_optimizers(self):
        params = (
            list(self.encoder.parameters())
            + list(self.act_embed.parameters())
            + list(self.pred.parameters())
            + list(self.constraint_head.parameters())
        )
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


def train_conv_jepa(config: dict, loaders: dict) -> None:
    """Train ConvJEPA on the train loader and validate on the val loader."""
    trainer = pl.Trainer(
        logger=TensorBoardLogger(config["log_dir"], name="car_racing/conv_jepa"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=config["epochs"],
        log_every_n_steps=config["log_every_n_steps"],
    )
    trainer.fit(ConvJEPA(config), loaders["train"], loaders["val"])
