# TEDI

## Environment

The experiments are run with:

```text
Python 3.8.20
PyTorch 1.13.1
CUDA 11.7
mujoco-py 2.1.2.14
MuJoCo 2.1.0
```

Create the Conda environment:

```bash
conda create -n TEDI python=3.8.20 -y
conda activate TEDI
```

Install the required packages:

```bash
pip install -r requirements.txt
```

MuJoCo 2.1.0 should be installed at:

```text
~/.mujoco/mujoco210
```

Set the MuJoCo environment variables:

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin"
```

The commands above target Linux. The dependency file uses Gymnasium's `mujoco-py` backend for the MuJoCo v2 environments. A working compiler and OpenGL development libraries are also required to build `mujoco-py`.

## Run

Activate the environment before training:

```bash
conda activate TEDI
```

Run commands from the repository root.

Example:

```bash
python main.py --env HalfCheetah-v2 --steps 1000000 --seed 10
```

Other supported environments can be selected with `--env`:

```text
Ant-v2
HalfCheetah-v2
Hopper-v2
Swimmer-v2
```

Existing run directories are protected. Use a new `--run_name` for each new experiment.

## Repository structure

```text
main.py                  Training entry point
tedi/                    Core TEDI implementation
scripts/                 Diagnostics, analysis, and video utilities
assets/                  Existing demonstration videos
requirements.txt         Training dependencies
requirements-analysis.txt Optional analysis and video dependencies
```

## Analysis and diagnostics

Install optional analysis and video dependencies:

```bash
pip install -r requirements-analysis.txt
```

Examples:

```bash
python -m scripts.read_metrics /path/to/runs
python -m scripts.frozen_reward_diag --runs /path/to/probe_run --verify
python -m scripts.reeval_actor --help
python -m scripts.record_video --help
```

The comparison scripts export statistics and CSV data without generating figures. The native `mujoco` Python package is additionally required by `scripts/render_from_qpos.py`; it is separate from the MuJoCo 2.1.0 training backend.


## Comparison videos

- [TEDI vs matched learner — seed 10](assets/seed10_TEDI_vs_matched.mp4)
- [TEDI vs matched learner — seed 45](assets/seed45_TEDI_vs_matched.mp4)




Schedule and simulated-rollout tests passed; real MuJoCo training was not tested
locally because `mujoco_py` was unavailable.
