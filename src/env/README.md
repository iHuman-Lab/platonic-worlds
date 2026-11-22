# Randomized road-following CarRacing-v3 dataset collector

## Code structure and execution flow

```text
env/
  collect.py           # Program entry point and complete simulation loop
  configuration.py     # CLI options, input validation, interactive menu
  episode_data.py      # Recording buffers, images, metadata, file export
  race_env.py          # Gym environment, lap detection, finish-line rendering
  random_road_agent.py # Road following, exploration, recovery controls
  road_geometry.py     # Track projection and physical-state extraction
  preview_episode.py   # Saved-episode replay command
  environment.yml      # Runtime environment and dependencies
  README.md            # Documentation
```


1. `main()` calls `configuration.parse_args()`. It parses and validates the
   existing options, applies the interactive menu when requested, and resolves
   watch mode and PNG-export options.
2. `run()` checks the output directory, creates the manifest, and initializes
   `FinishLineCarRacing`. It then calls `collect_episode()` for each episode.
3. `collect_episode()` resets the environment with `base_seed + episode_id`.
   Reset generates the track and initializes its geometry and finish tracker.
   A new `RandomRoadAgent` uses `episode_seed + 900001` for controller randomness.
4. `extract_state()` reads the initial physical state and road measurements.
   The recording starts with this state and its diagnostics. In images mode,
   the reset image is written to the temporary image file before action 0.
5. Each iteration checks for a live-window exit, then calls
   `controller.act(state, road)` to calculate steering, gas, brake, and
   controller diagnostics.
6. `environment.step(action)` advances the Box2D simulation and checks the
   physical finish-line crossing. It returns the next observation, reward,
   termination/truncation flags, and lap information.
7. `extract_state()` reads the resulting physical state. The loop evaluates
   the optional spin cutoff and step cap, then records the next state,
   diagnostics, action, reward, flags, and optional image. It updates speed and
   reward totals and uses the resulting state for the next action.
8. The loop stops for a completed lap, off-playfield termination, Gym
   truncation, optional spin cutoff, step cap, or user exit. A user exit after
   an action marks the last recorded transition as a collector cutoff.
9. When collecting, `episode_data.save_episode()` converts the buffers to
   arrays, reads the temporary images through a memory map, and writes
   `trajectory.npz`, both CSVs, `metadata.json`, and optional PNGs. It closes
   the memory map, and the episode function closes the temporary file.
10. `run()` appends the summary to `manifest.json` after each saved episode.
    It starts the next episode unless the user exited, then closes the
    environment in `finally`. Watch mode runs the same simulation without
    writing dataset files.

For `N` actions, the recording contains `N+1` states and timestamps, plus
`N+1` images when enabled. Index `t` describes the observation before action
`t`; index `t+1` describes its result. See the dataset-format section for all
fields and their shapes.

## 1. Create the Conda environment

With Miniconda or Anaconda:

```bash
conda env create -f environment.yml
conda activate gym
unset PYTHONPATH PYTHONHOME  # avoid Python paths inherited from a other shell
python -c "import gymnasium as gym; import Box2D; print('Gym:', gym.__version__)"
```

This project targets Gymnasium 1.3.0 with Python 3.11 and `CarRacing-v3`.
If the installation fails on some systems, try:

```bash
conda install -c conda-forge swig
python -m pip install 'gymnasium[box2d]==1.3.0' pillow
```

## 2. Run command guide

Run commands from the project directory with the `gym` environment active.
Options can be combined; these examples cover every collection mode, image
resolution, and supported command-line option. 
Choose one example at a time.

Use a new `--output` directory for each collection run: existing datasets are
not overwritten. Paths in the validation/replay examples refer to datasets
created by the corresponding collection examples.

### Help and interactive setup

```bash
# Show all collector options and their descriptions.
python collect.py --help

# Choose collection mode, live viewing, image resolution, and PNG export in a menu.
python collect.py --interactive --episodes 3 --output dataset_menu

# Default run: five tracks, seed 42, low-resolution images + all data, no live window.
python collect.py
```

The default run saves to `dataset/`. The menu changes the options for that run;
collection mode is selected before driving starts.

### Watch live and choose what to collect

| Mode | What is saved | Live window |
| --- | --- | --- |
| `watch` | No dataset files | Always enabled |
| `data` | States, actions, rewards, diagnostics, track, CSVs, metadata | Add `--render` |
| `images` (default) | All data above plus RGB frames inside `trajectory.npz` | Add `--render` |

```bash
# Watch one complete race without saving a dataset.
python collect.py --collection-mode watch --episodes 1

# Collect three tracks of numerical data without a live window or saved images.
python collect.py --collection-mode data --episodes 3 --output dataset_data

# Watch live while collecting numerical data only.
python collect.py --render --collection-mode data --episodes 3 --output dataset_live_data

# Collect images + all numerical data without a live window.
python collect.py --collection-mode images --episodes 3 --output dataset_images

# Watch live while collecting images + all numerical data.
python collect.py --render --collection-mode images --episodes 3 --output dataset_live_images

# --live is an alias for --render; this also watches and records images + all data.
python collect.py --live --episodes 1 --output dataset_live_alias
```

Closing the live window, ESC, or Q stops the run and saves the partial episode
when collection is enabled. Without live viewing, collection runs as fast as
processing and storage allow. On a server without a display, the script selects
the SDL dummy video driver automatically; use `data` or `images` mode there.

### Choose image resolution

| Resolution | Width × height | Source |
| --- | --- | --- |
| `low` (default) | 96 × 96 | Native square Gym observation |
| `middle` | 300 × 200 | Half-size native video render, preserving its proportions |
| `high` | 600 × 400 | Native video render |

```bash
# Save low-resolution frames in NPZ, along with all numerical data.
python collect.py --collection-mode images --image-resolution low --episodes 1 --output dataset_npz_low

# Save middle-resolution frames in NPZ, along with all numerical data.
python collect.py --collection-mode images --image-resolution middle --episodes 1 --output dataset_npz_middle

# Save high-resolution frames in NPZ, along with all numerical data.
python collect.py --collection-mode images --image-resolution high --episodes 1 --output dataset_npz_high
```



### Export individual PNG images too

```bash
# Save every low-resolution frame as PNG, plus images in NPZ and all numerical data.
python collect.py --save-images --image-resolution low --episodes 1 --output dataset_png_low

# Save every middle-resolution frame as PNG, plus images in NPZ and all numerical data.
python collect.py --save-images --image-resolution middle --episodes 1 --output dataset_png_middle

# Save every high-resolution frame as PNG, plus images in NPZ and all numerical data.
python collect.py --save-images --image-resolution high --episodes 1 --output dataset_png_high

# Watch live while saving high-resolution PNGs and all other data.
python collect.py --render --save-images --image-resolution high --episodes 3 --output dataset_live_png_high

# --save-png is an alias for --save-images and saves the same types of data.
python collect.py --save-png --image-resolution high --episodes 1 --output dataset_png_alias
```

`--save-images` / `--save-png` always enable `images` mode, even if a different
`--collection-mode` was supplied. Images mode alone saves frames in NPZ; these
flags additionally write matching PNGs to `episode_XXXXX/frames/`.
Frames are spooled to a temporary disk file while driving to limit memory usage.
High-resolution laps require substantial temporary and output disk space.

### Episode count, reproducible tracks, and output directory

```bash
# Collect 30 reproducible tracks, using seeds 42 through 71.
python collect.py --episodes 30 --seed 42 --output dataset_30

# Start a different reproducible sequence: seeds 100 through 104.
python collect.py --episodes 5 --seed 100 --output dataset_seed100

# Select a custom destination; quote paths that contain spaces.
python collect.py --episodes 1 --output "runs/my dataset"
```

Each episode uses `--seed + episode_index`. Reusing a seed with the same
settings and software versions reproduces the track and controller randomness.

### Speed, exploration, colors, and progress logging

```bash
# Increase straight and nominal corner speed targets (simulated world units/s).
python collect.py --max-speed 55 --corner-speed 18 --output dataset_faster

# Use lower speed targets.
python collect.py --max-speed 35 --corner-speed 12 --output dataset_slower

# Disable randomized steering exploration; road following remains active.
python collect.py --exploration 0 --output dataset_no_noise

# Use the default exploration strength explicitly.
python collect.py --exploration 0.08 --output dataset_default_noise

# Increase randomized steering exploration.
python collect.py --exploration 0.20 --output dataset_more_noise

# Randomize road/background colors for the generated episodes.
python collect.py --domain-randomize --output dataset_random_colors

# Print step, speed, lap progress, and controller mode every 100 steps (default interval).
python collect.py --verbose --output dataset_verbose

# Print progress every 50 steps; --log-every takes effect with --verbose.
python collect.py --verbose --log-every 50 --output dataset_logs_50
```

Speed targets must satisfy `0 < --corner-speed <= --max-speed`; actual speed
varies and recovery may slow the car further. Exploration must be between 0 and
1. Larger speed/exploration settings can make driving less stable.

### Step limits and optional spin cutoff

```bash
# Short partial-run smoke test; 200 steps are at most four simulated seconds.
python collect.py --episodes 1 --max-steps 200 --seed 42 --output dataset_smoke --verbose

# Allow up to 10,000 steps per track; a completed lap still ends at the finish line.
python collect.py --episodes 3 --max-steps 10000 --output dataset_long_cap

# Opt into early stopping when extreme spin indicators develop.
python collect.py --abort-spin --output dataset_spin_cutoff

# Explicitly keep spin-based early stopping disabled (the default).
python collect.py --no-abort-spin --output dataset_no_spin_cutoff
```

`--abort-spin` and `--no-abort-spin` are mutually exclusive. A low step cap or
an enabled spin cutoff can stop a run before the finish line; these runs are
labeled as partial runs rather than completed laps.

### Complete collector option reference

| Option | Default | Description |
| --- | --- | --- |
| `-h`, `--help` | — | Show help and exit. |
| `--interactive` | Off | Open the text setup menu before driving. |
| `--episodes N` | `5` | Number of generated tracks; positive integer. |
| `--max-steps N` | `6000` | Emergency action-step cap per episode; positive integer. |
| `--seed N` | `42` | First track/controller seed; incremented for each episode. |
| `--output PATH` | `dataset` | Destination directory; use a new dataset path. |
| `--collection-mode MODE` | `images` | Choose `watch`, `data`, or `images`. |
| `--render`, `--live` | Off | Show the live simulator while collecting. Watch mode enables this automatically. |
| `--image-resolution SIZE` | `low` | Choose `low`, `middle`, or `high` in images mode. |
| `--save-png`, `--save-images` | Off | Export standalone PNGs in addition to NPZ images and all numerical data; selects images mode. |
| `--max-speed VALUE` | `48.0` | Maximum nominal straight speed target in world units/s. |
| `--corner-speed VALUE` | `16.0` | Minimum nominal curvature-based corner target; recovery can drive slower. |
| `--exploration VALUE` | `0.08` | Smooth randomized steering strength between 0 and 1. |
| `--domain-randomize` | Off | Randomize road/background colors. |
| `--abort-spin` | Off | Allow early extreme-spin cutoff. |
| `--no-abort-spin` | On | Disable early extreme-spin cutoff explicitly. |
| `--verbose` | Off | Print periodic driving progress. |
| `--log-every N` | `100` | Progress-print interval in steps; requires `--verbose`, positive integer. |

## 3. What a step and a completed lap mean

One **step** applies one steering/gas/brake action, advances the simulator's
physics by **0.02 simulated seconds**, and produces the next observation,
physical state, and reward. There are **50 steps per simulated second**:
1,000 steps represent 20 seconds.
Wall-clock time can differ because of live display pacing, rendering, and saving.
For `N` steps, a collected episode has `N` actions and `N+1` observations/states,
including the initial observation before action 0.

By default, a race continues until the car crosses the visible checkered
start/finish line **forward after completing a lap**. Tile coverage alone does
not end a race. Gym's 1,000-step TimeLimit is removed, and the optional spin
cutoff is off. The explicit emergency cap is 6,000 steps.

The controller targets up to 48 world units/s on straights and plans corner
speeds using road curvature and braking distance. Its minimum nominal corner
target is 16 world units/s; actual speed varies and recovery can slow it further.

A completed lap has `stop_reason=finish_line` and `lap_finished=true` in metadata.
A step-capped run has `stop_reason=max_steps`, `lap_finished=false`, and a final
`collector_cutoff=true`. User exits, off-playfield failures, and optional spin
cutoffs have separate stop reasons. Increasing a step limit allows more time;
it does not force the car to drive that many steps after finishing.

## 4. Validate, replay, and test

### Validate saved datasets

```bash
# Show validator usage.
python check_dataset.py --help

# Validate all recorded episodes: alignment, shapes, ranges, and completion metadata.
python check_dataset.py dataset

# Validate one episode folder instead of the entire dataset.
python check_dataset.py dataset/episode_00000
```

The validator supports data-only episodes and all three image resolutions.

### Replay saved images with synchronized state/action data

```bash
# Show replay options.
python preview_episode.py --help

# Replay an episode folder at the default 50 frames/s.
python preview_episode.py dataset/episode_00000

# Open the NPZ file directly and replay at 25 frames/s (half normal simulation speed).
python preview_episode.py dataset/episode_00000/trajectory.npz --fps 25

# Replay at 10 frames/s for slower inspection.
python preview_episode.py dataset/episode_00000 --fps 10
```

Replay needs a desktop display and an episode containing images. It supports
all image resolutions and overlays the 13 states and the corresponding action.
SPACE toggles pause, LEFT/RIGHT step through observations, and ESC/Q closes the
window. Data-only episodes have no visual replay. `--fps` changes recorded
playback speed; it does not change collection frequency or saved timestamps.


