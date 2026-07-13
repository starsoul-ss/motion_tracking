# Motion Tracking

[![HEFT Website](https://img.shields.io/badge/Website-heft.axell.top-0A66C2)](https://heft.axell.top/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)
![mjlab 1.3.0](https://img.shields.io/badge/mjlab-1.3.0-orange.svg)

This repository contains training, evaluation, export, and deployment-facing assets for
humanoid whole-body motion tracking.

> **Branch guide**
>
> - The **main branch** is the official implementation of
>   [HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion
>   Guidance and Windowed Payload Curriculum](https://heft.axell.top/).
> - The **compliance branch** contains a G1 tracking + compliance framework inspired by
>   Gentle humanoid. [Demo](https://motion-tracking.axell.top/).
> - The **sim2real branch** provides the deployment runtime and available checkpoints.

The simulation and training utilities are based on **mjlab**.

## Repository Structure

To train HEFT G1/L7 policies, clone the default `main` branch:

```bash
git clone https://github.com/Axellwppr/motion_tracking.git
cd motion_tracking
```

To deploy released policies, clone the `sim2real` branch:

```bash
git clone -b sim2real --single-branch https://github.com/Axellwppr/motion_tracking.git motion_tracking_sim2real
cd motion_tracking_sim2real
```

To use the G1 compliance framework, clone the `compliance` branch:

```bash
git clone -b compliance --single-branch https://github.com/Axellwppr/motion_tracking.git motion_tracking_compliance
cd motion_tracking_compliance
```

## Release Status

HEFT consists of an efficient humanoid motion tracking framework, PMG
(Privileged Motion Guidance), and WPC (Windowed Payload Curriculum).

- [x] Efficient motion tracking training framework
- [x] PMG (Privileged Motion Guidance) support
- [x] WPC (Windowed Payload Curriculum) support
- [x] Released checkpoints on the `sim2real` branch
- [x] Deployment runtime on the `sim2real` branch
- [ ] Public training datasets and WPC window-payload labels
- [ ] VR data recording, reference reconstruction, and paired-dataset generation workflow

## Installation

This project uses `uv` for Python dependency and environment management. If you do not
have `uv` installed, follow the
[uv installation guide](https://docs.astral.sh/uv/getting-started/installation/).

```bash
uv sync
```

## Dataset Layout

Training loads motion memdatasets from `dataset/` by default. You can override the
dataset root with the `MEMPATH` environment variable:

```bash
export MEMPATH=/path/to/motion_tracking_dataset
```

The profile files reference paths relative to `MEMPATH`, for example:

```text
dataset/
  g1/
    lafan/
    100style/
    seed/all/
    vr_paired/
      clean/
      raw/
  l7/
    lafan/
    100style/
    seed/all/
    vr_paired/
      clean/
      raw/
```

The paired `vr_paired/{clean,raw}` folders are used by PMG. The clean side is used as
the teacher/reference target, while the raw side represents deployable student input.

We provide some preprocessed G1 memdatasets for smoke-testing the
training pipeline:
[Google Drive dataset](https://drive.google.com/drive/folders/1-FBUxllaYwqGIUSCaWg_4inD-u5Tdvi9?usp=sharing).
After downloading, place the extracted folders under `dataset/g1/` or set `MEMPATH` to
the extracted dataset root. These samples are not the exact HEFT training set, but they
are useful for validating the framework and obtaining similar training behavior. The
full dataset will be released later.

## Build A Motion Dataset

The dataset builder consumes retargeted robot `npz` files and writes a memdataset:

Each `npz` should follow the GMR-style robot motion format at 50 Hz. The required
fields are `root_pos`, `root_rot` in `xyzw` quaternion order, `dof_pos`,
`local_body_pos`, `joint_names`, and `body_names`. You can use the
[modified GMR exporter](https://github.com/Axellwppr/GMR) to generate compatible
retargeted motions.

```bash
uv run python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/retargeted_npz \
  --mem-path dataset/g1/my_motion_set
```

For paired PMG data, provide both clean teacher motions and raw student motions:

```bash
uv run python scripts/data_process/generate_dataset.py \
  --dataset-root /path/to/clean_npz \
  --mem-path dataset/g1/vr_paired/clean \
  --student-root /path/to/raw_npz \
  --student-mem-path dataset/g1/vr_paired/raw
```

If you write to `dataset/g1/my_motion_set`, use `g1/my_motion_set` in the YAML profile.
If you write under another root, set `MEMPATH` before training.

## Configure Training

Robot-specific training settings live in:

- `cfg/task/profile/G1_tracking.yaml`
- `cfg/task/profile/L7_tracking.yaml`

The most common changes are:

- dataset groups under `command.dataset.groups`
- dataset weights
- whether the `vr_paired` group is enabled
- `student_motion_randomization.enable`
- action/noise/randomization ranges for a specific robot

For PMG training with paired clean/raw data:

1. Enable the `vr_paired` dataset group.
2. Set `student_motion_randomization.enable: false`.

If paired PMG data is unavailable:

1. Leave the paired group commented out.
2. Keep `student_motion_randomization.enable: true`.

L7 WPC support is configured through `cfg/load/l7_wpc.yaml`. The training datasets and
window-cap labels are external artifacts and are planned for a later release.

## L7 Expert And WPC Labels

Train the clean-reference expert as a single-stage policy:

```bash
LOAD_CONFIG=l7_expert TAG=l7_expert bash train.sh
```

Or run expert training, eight-GPU rollout, infeasible-motion filtering, and label
construction end to end:

```bash
bash scripts/run_l7_expert_label_pipeline.sh
```

This defaults to 8 GPUs with 12,288 environments per training process. Only training
uses W&B, and checkpoints remain local. The original datasets are preserved; filtered
copies and the final NPZ are written under `outputs/expert_label_pipeline/<run>/`.

Generate 5 s window caps from that expert using the paper protocol grid (30 kg down to
0 kg in 5 kg steps):

```bash
CFG_PATH=/path/to/expert/cfg.yaml \
CHECKPOINT_PATH=/path/to/expert/checkpoint_final.pt \
OUTPUT_DIR=outputs/window_caps \
bash scripts/run_l7_window_caps_8gpu.sh
```

Rows whose status is not `success` must not enter a training label.
`scripts/finalize_l7_window_caps.py` removes every motion containing such a row, applies
the same removal manifest to paired VR clean/raw datasets, and builds labels only from
the retained motions' existing successful rollout rows. A second rollout is unnecessary:
filtering removes complete motions while preserving the retained motions' frame data and
metadata.

## Training

Use `train.sh` as the main entry point. L7 WPC requires the filtered dataset and its
matching final label file:

```bash
export MEMPATH=/absolute/path/to/dataset_filtered
export L7_WINDOW_LOAD_CAP_LABEL_PATH=/absolute/path/to/window_caps_5s.npz
bash train.sh
```

`train.sh` runs the full three-stage pipeline:

1. `+exp=train`
2. `+exp=adapt`, initialized from the train run
3. `+exp=finetune`, initialized from the adapt run

Override `PROFILE`, `LOAD_CONFIG`, `NPROC`, `TAG`, `OUTPUT_ROOT`, or `WANDB_MODE` as
environment variables. Expert training defaults to the `train` stage only; WPC training
defaults to all three stages. If GPU memory is constrained, reduce `NPROC` and
`task.num_envs`. Reducing the total environments or budget may affect performance.

## Evaluation And Export

Each stage saves a checkpoint every 150 iterations. During or after training, you can
use `eval.py` to evaluate checkpoint performance or export a deployment policy.

Play a trained policy:

```bash
uv run python scripts/eval.py --run_path ${wandb_run_path} -p
```

Export a deployment policy:

```bash
uv run python scripts/eval.py --run_path ${wandb_run_path} -p --export
```

Evaluate with an explicit profile:

```bash
uv run python scripts/eval.py --run_path ${wandb_run_path} --task tracking --profile L7_tracking -p
```

Exported policies are written under:

```text
scripts/exports/<task-name>-<timestamp>/
```

The exported folder contains the policy artifacts needed by the deployment runtime,
including `policy.onnx`, `policy.pt`, and `policy.json`.

## Deployment

Deployment assets are provided on the `sim2real` branch:

```bash
git clone -b sim2real --single-branch https://github.com/Axellwppr/motion_tracking.git motion_tracking_sim2real
```

That branch contains the deployment runtime, sim2sim/sim2real configuration, and
released policy assets.
