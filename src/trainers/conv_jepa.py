import copy

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger
from torchmetrics.classification import BinaryAUROC, BinaryRecall

from models.conv_jepa import ConvEncoder
from stl.constraints import constraint_margins, constraint_names

PROGRESS_BAR_METRICS = ("jepa", "copy_baseline", "jepa_wrong_actions", "constraint_now")


class ConvJEPA(pl.LightningModule):
    """
    Action-conditioned conv JEPA for CarRacing: from the latent of `ctx` context frames
    and the `horizon` actions taken from the last context frame onwards, predict the
    EMA-teacher latent of the frames `horizon` steps later. Loss: L1 on layer-normed
    latents + a variance hinge on the student latent across the batch (keeps samples
    apart, so no collapse).

    A constraint head reads the latent (and the predicted future latent) and regresses the
    physical-constraint margins of stl.constraints. With constraint_to_encoder False its
    gradient is detached, so it only probes what a JEPA-only encoder already encodes.

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
        dim = config["conv_jepa_dim"]
        act_dim = config["conv_jepa_act_embed"]
        self.constraint_weight = config["conv_jepa_constraint_weight"]
        self.constraint_to_encoder = config["conv_jepa_constraint_to_encoder"]
        self.violation_weight = config["conv_jepa_violation_weight"]
        self.constraint_names = constraint_names(config)
        self.encoder = ConvEncoder(self.ctx, dim, config["dataset"]["image_size"])
        self.target = copy.deepcopy(self.encoder).requires_grad_(False)
        self.act_embed = torch.nn.Linear(config["conv_jepa_action_dim"], act_dim)
        self.pred = torch.nn.Sequential(
            torch.nn.Linear(dim + self.horizon * act_dim, 2 * dim),
            torch.nn.GELU(),
            torch.nn.Linear(2 * dim, dim),
        )
        self.constraint_head = torch.nn.Sequential(
            torch.nn.Linear(dim, config["conv_jepa_constraint_hidden"]),
            torch.nn.GELU(),
            torch.nn.Linear(config["conv_jepa_constraint_hidden"], len(self.constraint_names)),
        )
        self.val_auroc = torch.nn.ModuleDict({n: BinaryAUROC() for n in self.constraint_names})
        self.val_recall = torch.nn.ModuleDict({n: BinaryRecall() for n in self.constraint_names})

    def predict(self, z, actions):
        return self.pred(torch.cat([z, self.act_embed(actions).flatten(1)], dim=-1))

    def on_train_batch_end(self, *args):
        with torch.no_grad():
            for s, t in zip(self.encoder.parameters(), self.target.parameters()):
                t.mul_(self.ema).add_(s, alpha=1 - self.ema)

    def split_batch(self, batch):
        """Return context frames, future frames and the actions taken in between."""
        images, actions = batch["images"], batch["actions"]
        if images.shape[1] != self.ctx + self.horizon or images.shape[2] != 1:
            raise ValueError(
                f"Need windows of {self.ctx + self.horizon} grayscale frames "
                f"(dataset.window_length, preprocess), got images {tuple(images.shape)}"
            )
        frames = images.squeeze(2)  # [B, ctx + horizon, S, S]
        context = frames[:, : self.ctx]
        future = frames[:, self.horizon : self.horizon + self.ctx]
        taken = actions[:, self.ctx - 1 : self.ctx - 1 + self.horizon]  # a_t .. a_{t+horizon-1}
        return context, future, taken

    def weighted_mse(self, pred, target):
        """Squared error, counted violation_weight times more where the margin is below 0."""
        weights = 1.0 + (self.violation_weight - 1.0) * (target < 0).float()
        return (weights * (pred - target) ** 2).mean()

    def constraint_losses(self, z, pred, batch):
        """Regress constraint margins from z (now) and from pred (future latent)."""
        margins = constraint_margins(batch["states"], batch["extras"], self.config)
        now_target = margins[:, self.ctx - 1]
        future_target = margins[:, self.ctx + self.horizon - 1]
        if not self.constraint_to_encoder:
            z, pred = z.detach(), pred.detach()
        now = self.constraint_head(z)
        future = self.constraint_head(pred)
        losses = {
            "constraint_now": F.mse_loss(now, now_target),
            "constraint_future": F.mse_loss(future, future_target),
            "constraint_weighted": self.weighted_mse(now, now_target)
            + self.weighted_mse(future, future_target),
        }
        with torch.no_grad():
            right_now = (now > 0) == (now_target > 0)
            right_future = (future > 0) == (future_target > 0)
            losses["sign_acc_now"] = right_now.float().mean()
            losses["sign_acc_future"] = right_future.float().mean()
            for index, name in enumerate(self.constraint_names):
                losses[f"sign_acc_now/{name}"] = right_now[:, index].float().mean()
        return losses, now, now_target

    def losses(self, batch):
        context, future, taken = self.split_batch(batch)
        z = self.encoder(context)
        z = F.layer_norm(z, (z.size(-1),))
        with torch.no_grad():
            tgt = self.target(future)
            tgt = F.layer_norm(tgt, (tgt.size(-1),))
        pred = self.predict(z, taken)
        jepa = F.l1_loss(pred, tgt)
        between = z.std(dim=0)
        reg = F.relu(1.0 - between).mean()
        with torch.no_grad():
            # Does the predictor use the actions? (equal to jepa = it ignores them)
            wrong = F.l1_loss(self.predict(z, taken.roll(1, dims=0)), tgt)
            # Does the predictor beat copying the current latent as its guess for the future?
            copy_baseline = F.l1_loss(z.detach(), tgt)
        constraint, now, now_target = self.constraint_losses(z, pred, batch)
        losses = {
            "loss": jepa
            + self.reg_coeff * reg
            + self.constraint_weight * constraint["constraint_weighted"],
            **constraint,
            "jepa": jepa,
            "reg": reg,
            "copy_baseline": copy_baseline,
            "jepa_wrong_actions": wrong,
            "between_std": between.mean(),
        }
        return losses, now, now_target

    def log_losses(self, stage, losses, batch_size):
        for name, value in losses.items():
            self.log(
                f"{stage}/{name}",
                value,
                on_epoch=True,
                prog_bar=name in PROGRESS_BAR_METRICS,
                batch_size=batch_size,
            )

    def update_violation_metrics(self, now, now_target):
        """Violation = margin below 0; the AUROC score is the negated predicted margin."""
        for index, name in enumerate(self.constraint_names):
            violated = (now_target[:, index] < 0).int()
            self.val_auroc[name].update(-now[:, index], violated)
            self.val_recall[name].update((now[:, index] < 0).int(), violated)
            self.log(f"val/auroc/{name}", self.val_auroc[name], on_epoch=True)
            self.log(f"val/recall/{name}", self.val_recall[name], on_epoch=True)

    def training_step(self, batch, batch_idx):
        losses, _, _ = self.losses(batch)
        self.log_losses("train", losses, batch["images"].shape[0])
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses, now, now_target = self.losses(batch)
        self.log_losses("val", losses, batch["images"].shape[0])
        self.update_violation_metrics(now, now_target)

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
