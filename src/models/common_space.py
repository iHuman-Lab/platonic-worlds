import copy

import torch
from torch import nn

from models.constraint_encoder import ConstraintEncoder


class StateEncoder(nn.Module):
    """The constraint vector plus camera-observable extras (standardized) into C."""

    def __init__(self, anchor: int, extra: int, hidden: int, dim: int):
        super().__init__()
        self.norm = nn.BatchNorm1d(extra, affine=False)
        self.net = nn.Sequential(nn.Linear(anchor + extra, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, z: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        extra = self.norm(extra.reshape(-1, extra.shape[-1])).reshape(extra.shape)
        return self.net(torch.cat([z, extra], dim=-1))


class CommonSpace(nn.Module):
    """Image and state encoders into one space C of `anchor` constraint dims plus free dims.

    The anchor dims are the dynamics model's normalized constraint vector. One shared linear
    decoder reads the constraint vector back from either modality, and one shared
    action-conditioned predictor rolls C forward. The EMA copy of the state encoder (`target`)
    embeds future states as JEPA targets. forward(images) is the decoded image estimate, so the
    module can be scored like any camera encoder. `active` lists the modalities being trained.
    """

    def __init__(self, image_encoder: ConstraintEncoder, anchor: int, cfg: dict):
        super().__init__()
        dim = anchor + cfg["free_dims"]
        self.anchor, self.active = anchor, ("image", "state")
        self.image = image_encoder
        self.state = StateEncoder(anchor, len(cfg["state_extras"]), cfg["state_hidden"], dim)
        self.target = copy.deepcopy(self.state).requires_grad_(False)
        self.decoder = nn.Linear(dim, anchor)
        hidden = cfg["predictor_hidden"]
        self.predictor = nn.Sequential(
            nn.Linear(dim + cfg["action_dim"], hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.image(images))

    def predict(self, c0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Roll C forward with the actions taken: [B, steps, dim]."""
        steps, c = [], c0
        for step in range(actions.shape[1]):
            c = c + self.predictor(torch.cat([c, actions[:, step]], dim=-1))
            steps.append(c)
        return torch.stack(steps, dim=1)

    @torch.no_grad()
    def update_target(self, decay: float) -> None:
        """EMA of the state encoder into the target encoder."""
        for online, target in zip(self.state.parameters(), self.target.parameters()):
            target.mul_(decay).add_(online, alpha=1 - decay)
