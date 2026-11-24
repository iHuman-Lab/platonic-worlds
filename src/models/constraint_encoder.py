import torch
from torch import nn

from models.conv_jepa import ConvEncoder

GROUPS = 8  # GroupNorm groups (all channel counts below are multiples of 8)
STAGES = ((32, 1), (64, 2), (128, 2), (256, 2))  # (channels, stride of the first block)
STEM_CHANNELS = 32


class ResidualBlock(nn.Module):
    """Two 3x3 convs with GroupNorm and a skip connection (1x1 conv when the shape changes)."""

    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.norm1 = nn.GroupNorm(GROUPS, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.norm2 = nn.GroupNorm(GROUPS, out_channels)
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + self.skip(x))


class ResidualEncoder(nn.Module):
    """Residual CNN trunk: [B, frames, S, S] -> [B, dim].

    pool='global' averages the final feature map; pool='flatten' keeps its spatial layout.
    """

    def __init__(self, frames: int, dim: int, image_size: int, pool: str):
        super().__init__()
        layers = [
            nn.Conv2d(frames, STEM_CHANNELS, 5, 2, 2, bias=False),
            nn.GroupNorm(GROUPS, STEM_CHANNELS),
            nn.GELU(),
        ]
        channels, size = STEM_CHANNELS, (image_size + 1) // 2
        for out_channels, stride in STAGES:
            layers += [ResidualBlock(channels, out_channels, stride), ResidualBlock(out_channels, out_channels, 1)]
            channels, size = out_channels, (size + stride - 1) // stride
        if pool == "global":
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, dim)]
        elif pool == "flatten":
            layers += [nn.Flatten(), nn.Linear(channels * size * size, dim)]
        else:
            raise ValueError(f"pool must be 'global' or 'flatten', got {pool}")
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConstraintEncoder(nn.Module):
    """Camera frames -> constraint-space vector, in the dynamics model's normalized coordinates.

    arch='small' reuses the conv trunk of ConvEncoder (Atari-style CNN, always flattened);
    arch='resnet' uses ResidualEncoder. Both end in a small readout head with `out_dim` outputs.
    """

    def __init__(self, frames: int, dim: int, image_size: int, out_dim: int, arch: str, pool: str):
        super().__init__()
        if arch == "small":
            self.trunk = ConvEncoder(frames, dim, image_size)
        elif arch == "resnet":
            self.trunk = ResidualEncoder(frames, dim, image_size, pool)
        else:
            raise ValueError(f"arch must be 'small' or 'resnet', got {arch}")
        self.head = nn.Sequential(nn.GELU(), nn.Linear(dim, out_dim))

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(frames))
