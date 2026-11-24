"""Turn streamed CarRacing tar samples into preprocessed training windows.

Each episode is its own tar, so windows never cross episodes. Within a window of
T frames, actions[i] is the action taken between frame i and frame i + 1, so
there are T - 1 actions.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

RGB_TO_GREY = (0.299, 0.587, 0.114)


class Grayscale:
    """Convert (T, 3, H, W) RGB to (T, 1, H, W) luminance."""

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        weights = torch.tensor(RGB_TO_GREY, dtype=images.dtype).view(1, 3, 1, 1)
        return (images * weights).sum(dim=1, keepdim=True)


class Resize:
    """Resize (T, C, H, W) to (T, C, size, size); unchanged if already that size."""

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        if tuple(images.shape[-2:]) == (self.size, self.size):
            return images
        return F.interpolate(
            images,
            size=(self.size, self.size),
            mode="bilinear",
            antialias=True,
            align_corners=False,
        )


class ComposePreprocessor:
    """Apply image steps in order to a window of frames."""

    def __init__(self, steps: list) -> None:
        self.steps = steps

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        for step in self.steps:
            images = step(images)
        return images


STEP_BUILDERS = {
    "grayscale": lambda cfg: Grayscale(),
    "resize": lambda cfg: Resize(cfg["image_size"]),
}


def build_preprocessor(config: dict) -> ComposePreprocessor:
    """Build the image pipeline from the ordered config['dataset']['preprocess'] list."""
    cfg = config["dataset"]
    return ComposePreprocessor([STEP_BUILDERS[name](cfg) for name in cfg["preprocess"]])


def sliding_windows(
    samples: Iterator[dict], window: int, stride: int, skip_intro: int
) -> Iterator[list[dict]]:
    """Yield lists of `window` consecutive samples, restarting at each new tar."""
    current_tar = None
    frames: list[dict] = []
    for sample in samples:
        if sample["__url__"] != current_tar:
            current_tar = sample["__url__"]
            frames = []
        step = int(sample["__key__"])
        if step < skip_intro:
            continue
        frames = (frames + [sample])[-window:]
        first_step = step - window + 1
        if len(frames) == window and (first_step - skip_intro) % stride == 0:
            yield frames


def window_to_tensors(
    frames: list[dict], preprocessor: ComposePreprocessor
) -> dict[str, torch.Tensor]:
    """Stack one window into float tensors, preprocessing the images."""
    images = np.stack([frame["png"] for frame in frames])
    images = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
    return {
        "images": preprocessor(images),
        "states": torch.from_numpy(np.stack([f["state.npy"] for f in frames])).float(),
        "extras": torch.from_numpy(np.stack([f["extras.npy"] for f in frames])).float(),
        "actions": torch.from_numpy(
            np.stack([f["action.npy"] for f in frames[:-1]])
        ).float(),
    }
