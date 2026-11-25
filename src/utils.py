import sys
from contextlib import contextmanager
from pathlib import Path

import yaml


class SkipWith(Exception):
    pass


@contextmanager
def skip_run(flag, f):
    """To skip a block of code.

    Parameters
    ----------
    flag : str
        skip or run.

    Returns
    -------
    None

    """

    @contextmanager
    def check_active():
        deactivated = ["skip"]
        p = ColorPrint()  # printing options
        if flag in deactivated:
            p.print_skip("{:>12}  {:>2}  {:>12}".format("Skipping the block", "|", f))
            raise SkipWith()
        else:
            p.print_run("{:>12}  {:>3}  {:>12}".format("Running the block", "|", f))
            yield

    try:
        yield check_active
    except SkipWith:
        pass


class ColorPrint:
    @staticmethod
    def print_skip(message, end="\n"):
        sys.stderr.write("\x1b[88m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_run(message, end="\n"):
        sys.stdout.write("\x1b[1;32m" + message.strip() + "\x1b[0m" + end)

    @staticmethod
    def print_warn(message, end="\n"):
        sys.stderr.write("\x1b[1;33m" + message.strip() + "\x1b[0m" + end)


def load_config(config_dir: str) -> dict:
    """Merge every YAML file in config_dir into one config; a key in two files is an error."""
    config = {}
    for path in sorted(Path(config_dir).glob("*.yaml")):
        with path.open() as config_file:
            section = yaml.safe_load(config_file)
        repeated = set(section) & set(config)
        if repeated:
            raise ValueError(f"{path.name} repeats keys from another config file: {sorted(repeated)}")
        config.update(section)
    return config
