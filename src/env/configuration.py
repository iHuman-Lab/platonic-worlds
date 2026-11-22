"""Watch full CarRacing laps or collect synchronized state/action/image data.

Run `python collect.py --interactive` for a setup menu. See README.md.
"""

import argparse

import numpy as np

IMAGE_SIZES = {"low": (96, 96), "middle": (300, 200), "high": (600, 400)}
SIMULATION_FPS = 50


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Choose live view and collection options in a menu",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Emergency cap; unfinished runs are labeled max_steps",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="dataset")
    parser.add_argument("--exploration", type=float, default=0.08)
    parser.add_argument(
        "--max-speed",
        type=float,
        default=48.0,
        help="Straight speed target in simulated world units/s",
    )
    parser.add_argument(
        "--corner-speed",
        type=float,
        default=16.0,
        help="Minimum curvature-based speed target; recovery may slow further",
    )
    parser.add_argument(
        "--render",
        "--live",
        action="store_true",
        help="Watch live with any collection mode",
    )
    parser.add_argument(
        "--collection-mode",
        choices=("watch", "data", "images"),
        default="images",
        help=(
            "watch=no files; data=states/actions/diagnostics; "
            "images=images plus all data"
        ),
    )
    parser.add_argument(
        "--image-resolution", choices=tuple(IMAGE_SIZES), default="low"
    )
    parser.add_argument(
        "--save-png",
        "--save-images",
        action="store_true",
        help="Save PNGs plus all state/action data; enables images mode",
    )
    parser.add_argument("--domain-randomize", action="store_true")
    spin = parser.add_mutually_exclusive_group()
    spin.add_argument(
        "--abort-spin",
        dest="abort_spin",
        action="store_true",
        help="Opt into early extreme-spin cutoff",
    )
    spin.add_argument(
        "--no-abort-spin", dest="abort_spin", action="store_false"
    )
    parser.set_defaults(abort_spin=False)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=100)
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    if args.interactive:
        configure_interactively(args)
    if args.save_png:
        args.collection_mode = "images"
    if args.collection_mode == "watch":
        args.render = True
    return args


def configure_interactively(args):
    args.collection_mode = _choose_option(
        "Collection: watch only, data only, or images + all data",
        ("watch", "data", "images"),
        args.collection_mode,
    )
    args.render = _choose_option("Watch live", ("yes", "no"), "yes") == "yes"
    if args.collection_mode == "watch":
        args.render = True
    if args.collection_mode == "images":
        args.image_resolution = _choose_option(
            "Image resolution: low 96x96, middle 300x200, high 600x400",
            tuple(IMAGE_SIZES),
            args.image_resolution,
        )
        args.save_png = (
            _choose_option(
                "Also export individual PNG files",
                ("yes", "no"),
                "yes" if args.save_png else "no",
            )
            == "yes"
        )
    else:
        args.save_png = False


def _choose_option(prompt, choices, default):
    while True:
        answer = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ")
        answer = answer.strip().lower() or default
        if answer in choices:
            return answer
        print("Choose one of: " + ", ".join(choices))


def _validate_args(parser, args):
    if args.episodes < 1 or args.max_steps < 1 or args.log_every < 1:
        parser.error(
            "--episodes, --max-steps and --log-every must be positive"
        )
    if not 0.0 <= args.exploration <= 1.0:
        parser.error("--exploration must be between 0.0 and 1.0")
    if not (
        np.isfinite(args.max_speed)
        and np.isfinite(args.corner_speed)
        and 0 < args.corner_speed <= args.max_speed
    ):
        parser.error(
            "Speed targets must be finite and "
            "0 < --corner-speed <= --max-speed"
        )
