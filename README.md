# platonic-worlds

A constraint-space representation learning which can provide a modality-independent description of the world, enabling heterogeneous observations to be mapped into a shared representation that preserves physical constraints and their temporal semantics.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📁 Project Structure

```{=bash}
├── src/
│   ├── main.py         # Entry point
│   ├── utils.py        # Utility functions including skip_run context manager
│   ├── env/            # CarRacing data collection and replay
│   ├── data/           # Data loading and saving scripts
│   ├── dataset/        # Dataset preparation and loaders
│   ├── stl/            # Physical constraints and STL robustness
│   ├── models/         # Model training and evaluation
│   ├── visualization/  # Plots and visualizations
│   └── **init**.py
│
├── configs/
│   └── config.yaml     # YAML configuration file
│
├── project.toml        # Dependencies and tool configuration
├── .pre-commit-config.yaml
├── README.md
└── tests/              # Unit and integration tests

```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/iHuman-Lab/platonic-worlds.git
cd platonic-worlds
```

### 2. Always use a conda virtual environment

```bash
   conda activate myenv
```

   *(Replace `myenv` and Python version as needed)*

---

## 🛠️ Pre-commit Setup

This project uses [pre-commit](https://pre-commit.com/) to automatically check your code before commits and pushes, ensuring consistent code quality and style.

### How to install and enable pre-commit hooks:

1. **Ayour conda environment**

   ```bash
   conda activate myenv
   ```

   *(Replace `myenv` and Python version as needed)*

2. **Install `pre-commit` in your conda environment**

   ```bash
   conda install -c conda-forge pre-commit
   ```

   or if you prefer:

   ```bash
   pip install pre-commit
   ```

3. **Run pre-commit checks manually (optional)**

   To manually run all pre-commit checks on all files:

   ```bash
   pre-commit run --all-files
   ```

---

### Notes:

* Always activate your conda environment before running Git commands to ensure hooks work properly.
* If you update `.pre-commit-config.yaml`, run `pre-commit install` again to update the hooks.
* The `pre-push` hook will run all pre-commit checks before pushing. If any check fails, the push will be aborted.



---

## ⚙️ Configuration

The project uses a YAML config file at `configs/config.yaml`.

```python
import yaml

with open("configs/config.yaml") as f:
    config = yaml.safe_load(f)
```

Modify this file to adjust parameters, paths, and flags.

---

## ⏯️ Conditional Execution with `skip_run`

Use the `skip_run` context manager from `utils.py` to control which pipeline blocks run.

Example from `main.py`:

```python
from utils import skip_run

with skip_run("skip", "Data") as check, check():
    # This block will be skipped if flag is "skip"
    ...
```

* Pass `"skip"` to skip blocks
* Pass `"run"` (or anything else) to execute blocks

The utility prints colored console messages indicating running/skipping status.

---

## 🧹 Code Quality

This project uses pre-commit hooks for automated linting and formatting:

* `ruff` for linting and enforcing naming conventions
* `codespell` for typo detection

### Install pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files
```


---

## 🏎️ Environment Data Collection

`src/env/` collects randomized road-following CarRacing-v3 episodes (states, actions, rewards, optional images). Everything is run from `src/main.py` and configured in `configs/config.yaml`.

```text
src/env/
  collect.py           # collect_episode(): one full simulation and recording loop
  configuration.py     # Shared constants (image sizes, simulation fps)
  episode_data.py      # Recording buffers, images, metadata, file export
  race_env.py          # Gym environment, lap detection, finish-line rendering
  random_road_agent.py # Road following, exploration, recovery controls
  road_geometry.py     # Track projection and physical-state extraction
  preview_episode.py   # Saved-episode replay
```

### Setup

```bash
conda create -n gym python=3.11 pip swig=4 -c conda-forge
conda activate gym
pip install -e .
python -c "import gymnasium as gym; import Box2D; print('Gym:', gym.__version__)"
```

### Collect

Set the `collect_env_data` block in `src/main.py` to `"run"`, adjust the `collect` block in `configs/config.yaml`, then run `python main.py` from `src/`.

| Option             | Default   | Description                                                                        |
| ------------------ | --------- | ---------------------------------------------------------------------------------- |
| `episodes`         | `5`       | Number of generated tracks                                                         |
| `max_steps`        | `6000`    | Emergency step cap per episode                                                     |
| `seed`             | `42`      | First track seed; incremented per episode                                          |
| `output`           | `dataset` | Destination folder; must not exist yet           |
| `collection_mode`  | `images`  | `watch` (no files, live window), `data` (numbers only), `images` (numbers + RGB)   |
| `image_resolution` | `low`     | `low` 96x96, `middle` 300x200, `high` 600x400                                      |
| `render`           | `False`   | Show the live simulator (always on in `watch` mode)                                |
| `save_png`         | `False`   | Also export PNG frames; forces `images` mode                                       |
| `domain_randomize` | `False`   | Randomize road/background colors                                                   |
| `exploration`      | `0.08`    | Steering noise strength, 0 to 1                                                    |
| `max_speed`        | `48.0`    | Straight speed target (world units/s)                                              |
| `corner_speed`     | `16.0`    | Minimum curvature-based speed target       |
| `abort_spin`       | `False`   | Stop early on extreme spin                                                         |

Output: one `episode_XXXXX.tar` per episode (WebDataset: `png`, `state.npy`, `extras.npy`, `action.npy`, `json` per observation). The dataset folder also holds `data_info.json`, which lists the settings and one summary per episode.

For `N` actions an episode has `N+1` states and images: index `t` is the observation before action `t`, and `t+1` is its result.

One step applies one action and advances the physics by 0.02 s (50 steps per second). A race ends when the car crosses the start/finish line forward after a lap (`stop_reason=finish_line`). Other stop reasons are `max_steps`, `off_playfield`, `gym_truncated`, `spin_safety_cutoff` and `user_closed`. Closing the window, ESC or Q stops a live run and saves the partial episode.

### Replay

Set the `preview_env_episode` block in `src/main.py` to `"run"` and set `preview.episode` and `preview.fps` in the config. It needs a desktop display and an episode collected with `collection_mode: images`. Controls: SPACE pause, LEFT/RIGHT step, ESC/Q quit.

---

## 📝 Running the Project

Start the main pipeline:

```bash
python main.py
```

---

## 📜 License

MIT License © Your name (or your organization/company/team)

---

## 🙏 Acknowledgements

Project scaffolded using a custom Cookiecutter template for Python data science projects.

