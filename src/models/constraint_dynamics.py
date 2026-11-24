import torch
from torch import nn


class ConstraintDynamics(nn.Module):
    """One-step residual dynamics c_{t+1} = c_t + MLP(c_t, u_t) in a normalized space.

    The per-dimension mean/std come from the training windows and are stored as buffers,
    so the model can be applied to raw representations and rolled out autoregressively.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, action_dim: int, hidden: int):
        super().__init__()
        dim = mean.shape[0]
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.net = nn.Sequential(
            nn.Linear(dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def normalize(self, c: torch.Tensor) -> torch.Tensor:
        return (c - self.mean) / self.std

    def denormalize(self, c: torch.Tensor) -> torch.Tensor:
        return c * self.std + self.mean

    def step(self, c: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """One step in the normalized space: [B, D], [B, A] -> [B, D]."""
        return c + self.net(torch.cat([c, action], dim=-1))

    def rollout(self, c0: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Feed each prediction back in; [B, D], [B, K, A] -> [B, K, D] (normalized)."""
        steps = []
        c = c0
        for k in range(actions.shape[1]):
            c = self.step(c, actions[:, k])
            steps.append(c)
        return torch.stack(steps, dim=1)
