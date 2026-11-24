"""Find CarRacing episode tars, split them by seed, and build DataLoaders."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import numpy as np
import webdataset as wds
from torch.utils.data import DataLoader

from dataset.pre_process import (
    build_preprocessor,
    sliding_windows,
    window_to_tensors,
)


def list_episodes(root: str) -> list[dict]:
    """Collect every episode under root/<run>/ using each run's data_info.json."""
    episodes = []
    frame_skips = set()
    for info_path in sorted(Path(root).glob("*/data_info.json")):
        with info_path.open() as info_file:
            info = json.load(info_file)
        frame_skips.add(info["frame_skip"])
        for episode in info["episodes"]:
            tar_path = info_path.parent / f"episode_{episode['episode_id']:05d}.tar"
            if not tar_path.exists():
                raise FileNotFoundError(tar_path)
            episodes.append(
                {
                    "run": info_path.parent.name,
                    "seed": int(episode["seed"]),
                    "path": str(tar_path),
                }
            )
    if not episodes:
        raise FileNotFoundError(f"No episodes found under {root}")
    if len(frame_skips) > 1:
        raise ValueError(f"Runs use different frame_skip values: {sorted(frame_skips)}")
    return episodes


def split_episodes(episodes: list[dict], config: dict) -> dict[str, list[dict]]:
    """Hold out whole seeds per run, so no track appears in two splits."""
    cfg = config["dataset"]
    rng = np.random.default_rng(config["random_state"])
    split = {"train": [], "val": [], "test": []}
    for run in sorted({episode["run"] for episode in episodes}):
        run_episodes = [e for e in episodes if e["run"] == run]
        order = rng.permutation(len(run_episodes))
        n_test, n_val = cfg["test_per_run"], cfg["val_per_run"]
        for rank, index in enumerate(order):
            if rank < n_test:
                name = "test"
            elif rank < n_test + n_val:
                name = "val"
            else:
                name = "train"
            split[name].append(run_episodes[index])
    return split


def write_split(split: dict[str, list[dict]], path: str) -> None:
    """Save which seeds went to which split."""
    with Path(path).open("w") as split_file:
        json.dump(split, split_file, indent=2)


def create_dataloader(files: list[str], config: dict, train: bool) -> DataLoader:
    """Build a DataLoader of windows; shuffles shards and windows when training."""
    cfg = config["dataset"]
    pipeline = wds.WebDataset(
        files,
        shardshuffle=100 if train else False,
        empty_check=False,
    ).decode("rgb8")
    pipeline = pipeline.compose(
        partial(
            sliding_windows,
            window=cfg["window_length"],
            stride=cfg["stride"],
            skip_intro=cfg["skip_intro"],
        )
    ).map(partial(window_to_tensors, preprocessor=build_preprocessor(config)))
    if train:
        pipeline = pipeline.shuffle(cfg["shuffle_buffer"])
    return DataLoader(
        pipeline, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"]
    )


def get_car_racing_dataloaders(config: dict) -> dict[str, DataLoader]:
    """Split episodes by seed, save the split, and return train/val/test loaders."""
    cfg = config["dataset"]
    split = split_episodes(list_episodes(cfg["root"]), config)
    write_split(split, cfg["split_file"])
    return {
        name: create_dataloader([e["path"] for e in episodes], config, name == "train")
        for name, episodes in split.items()
    }
