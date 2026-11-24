import json
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from dataset.torch_dataset import get_car_racing_dataloaders
from env.collect import collect_episode
from env.episode_data import build_dataset_info
from env.preview_episode import load_episode, play_episode
from env.race_env import FinishLineCarRacing
from trainers.conv_jepa import train_conv_jepa
from trainers.dynamics_trainer import format_results, run_dynamics_experiment
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)



with skip_run("run", "collect_env_data") as check, check():
    # Options live in the `collect` block of configs/config.yaml
    args = SimpleNamespace(**config["collect"])
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
                with (output_dir / "data_info.json").open("w") as info_file:
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


with skip_run("skip", "car_racing_dataloaders") as check, check():
    data_loaders = get_car_racing_dataloaders(config)
    batch = next(iter(data_loaders["train"]))


with skip_run("skip", "train_conv_jepa") as check, check():
    data_loaders = get_car_racing_dataloaders(config)
    train_conv_jepa(config, data_loaders)


with skip_run("skip", "constraint_dynamics") as check, check():
    results = run_dynamics_experiment(config)
    print(format_results(results))


with skip_run("skip", "preview_env_episode") as check, check():
    # Replay a saved episode; needs images mode and a desktop display
    preview_path = Path(config["preview"]["episode"])
    episode = load_episode(preview_path)
    play_episode(episode, preview_path, int(config["preview"]["fps"]))
