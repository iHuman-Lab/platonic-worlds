"""Write one CarRacing episode as a WebDataset tar, one sample per observation.

Sample t holds the image (images mode only), the privileged state, and the
action taken *from* observation t. The final observation has no action, so it
carries a zero action and has_action=False.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import webdataset as wds

from env.configuration import CONTROLLER_DETAIL_NAMES


def build_samples(arrays: dict) -> Iterator[dict]:
    """Yield one WebDataset sample per observation, in time order."""
    transitions = len(arrays["actions"])
    for step in range(transitions + 1):
        has_action = step < transitions
        action = arrays["actions"][step] if has_action else np.zeros(3, np.float32)
        sample = {
            "__key__": f"{step:06d}",
            "state.npy": arrays["states"][step],
            "extras.npy": arrays["extras"][step],
            "action.npy": action,
            "json": step_metadata(arrays, step, has_action),
        }
        if "images" in arrays:
            sample["png"] = np.asarray(arrays["images"][step])
        yield sample


def step_metadata(arrays: dict, step: int, has_action: bool) -> dict:
    """Per-step scalars; zeros/False on the final observation."""
    meta = {"has_action": has_action, "time_s": float(arrays["time_s"][step])}
    if not has_action:
        return {**meta, "reward": 0.0, "terminated": False, "truncated": False,
                "collector_cutoff": False, "controller_mode": 0}  # fmt: skip
    meta.update(
        reward=float(arrays["rewards"][step]),
        terminated=bool(arrays["terminated"][step]),
        truncated=bool(arrays["truncated"][step]),
        collector_cutoff=bool(arrays["collector_cutoff"][step]),
        controller_mode=int(arrays["controller_modes"][step]),
    )
    details = arrays["controller_details"][step]
    for name, value in zip(CONTROLLER_DETAIL_NAMES, details):
        meta[name] = float(value)
    return meta


def write_episode_tar(arrays: dict, tar_path: Path) -> None:
    """Write one episode to tar_path."""
    with wds.TarWriter(str(tar_path)) as sink:
        for sample in build_samples(arrays):
            sink.write(sample)
