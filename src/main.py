import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
import yaml
from dataset.pre_process import ComposePreprocessor, Resize, StackWithLabels
from dataset.torch_dataset import get_torch_dataloaders
from lightning.pytorch.loggers import TensorBoardLogger

from env.collect import collect_episode
from env.episode_data import build_dataset_info
from env.preview_episode import load_episode, play_episode
from env.race_env import FinishLineCarRacing
from trainers.conv_jepa import ConvJEPA
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

LOG_DIR = config["log_dir"]


with skip_run("skip", "conv_jepa_pretraining") as check, check():
    frames, horizon = int(config["conv_jepa_frames"]), int(config["conv_jepa_horizon"])
    jepa_config = {
        **config,
        "stack_length": frames + horizon,
    }  # context + future frames
    game = config["games"]
    games_name = "_".join(game)
    preprocessor = ComposePreprocessor(
        [Resize(jepa_config), StackWithLabels(jepa_config)]
    )
    data_loaders = get_torch_dataloaders(game, jepa_config, preprocessor)

    model = ConvJEPA(
        jepa_config,
        dim=int(config["conv_jepa_dim"]),
        ctx=frames,
        horizon=horizon,
        num_actions=int(config["num_actions"]),
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(LOG_DIR, name=f"{games_name}/conv_jepa"),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        max_epochs=int(config["epochs"]),
        log_every_n_steps=10,
    )
    trainer.fit(model, data_loaders["train"])


with skip_run("skip", "collect_env_data") as check, check():
    # Options live in the `collect` block of configs/config.yaml
    args = SimpleNamespace(**config["collect"])
    args.collection_mode = "images" if args.save_png else args.collection_mode
    args.render = args.render or args.collection_mode == "watch"
    collecting = args.collection_mode != "watch"
    output_dir = Path(args.output).resolve()
    if collecting:
        output_dir.mkdir(parents=True)  # fails if the dataset already exists

    # Headless machines have no display, so use SDL's dummy video driver
    if not args.render and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
    ):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    dataset_info = build_dataset_info(args)
    environment = FinishLineCarRacing(
        continuous=True,
        render_mode="human" if args.render else None,
        domain_randomize=args.domain_randomize,
    )
    try:
        for episode_id in range(args.episodes):
            summary = collect_episode(environment, args, episode_id, output_dir)
            dataset_info["episodes"].append(summary)
            if collecting:
                with (output_dir / "manifest.json").open("w") as info_file:
                    json.dump(dataset_info, info_file, indent=2)
            print(
                f"[{episode_id + 1}/{args.episodes}] seed={summary['seed']} "
                f"steps={summary['num_actions']} lap={summary['lap_progress']:.1%} "
                f"stop={summary['stop_reason']}"
            )
            if summary["stop_reason"] == "user_closed":
                break
    finally:
        environment.close()
    if collecting:
        print(f"Saved dataset to: {output_dir}")


with skip_run("skip", "preview_env_episode") as check, check():
    # Replay a saved episode; needs images mode and a desktop display
    preview_path = Path(config["preview"]["episode"])
    if preview_path.is_dir():
        preview_path = preview_path / "trajectory.npz"
    episode = load_episode(preview_path)
    play_episode(episode, preview_path, int(config["preview"]["fps"]))
