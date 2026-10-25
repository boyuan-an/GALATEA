# GALATEA reinforcement-learning tracking

> [!NOTE]
> This public code was obtained by reorganizing our development code with vibe coding.
> If you run into any problems, please [open an issue](https://github.com/boyuan-an/GALATEA/issues).

[Installation](#installation) · [Data and checkpoints](#data-and-checkpoints) · [Training](#training) · [Evaluation](#evaluation) · [Preprocessing](#preprocessing)

This directory contains the reinforcement-learning tracking component of
[GALATEA](../README.md). In Isaac Gym, a Franka Research 3 arm with a Sharpa
Wave right hand learns to track hand–object trajectories. A single policy
commands the arm and hand joints. It is trained with PPO using multiple
exploration blocks and an asymmetric actor–critic, under randomized PD gains,
friction, object mass, object pose noise and action delay.

## Requirements

- Linux, Python 3.8, CUDA 12.1
- Isaac Gym Preview 4
- Training: 4 × RTX 4090 with 6144 environments per GPU, or 2 × RTX 4090 with 12288 environments per GPU
- Evaluation: 1 × RTX 4090

## Installation

```bash
git clone https://github.com/boyuan-an/GALATEA.git
cd GALATEA/rl
conda create -n galatea-rl python=3.8
conda activate galatea-rl

# Isaac Gym Preview 4: download from https://developer.nvidia.com/isaac-gym
pip install -e /path/to/isaacgym/python

pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"
pip install "git+https://github.com/lixiny/manotorch.git@a2a70c591f91551078b7bb2af9b5d9f275b626e0"
pip install "git+https://github.com/otaheri/chamfer_distance@f86f6f7cadd3aca642704573d1626c67ca2e2846"
pip install "git+https://github.com/KailinLi/bps_torch.git@4a2eb59dfbf8982de1af878552e561d688dced28"
```

## Data and checkpoints

The pretrained checkpoints and the datasets are attached to the
[rl-v1.0 release](https://github.com/boyuan-an/GALATEA/releases/tag/rl-v1.0).
Download them into `rl/`:

```bash
mkdir -p checkpoints data
for object in power_drill dog old_camera2; do
  wget -P checkpoints https://github.com/boyuan-an/GALATEA/releases/download/rl-v1.0/${object}.pth
  wget -qO- https://github.com/boyuan-an/GALATEA/releases/download/rl-v1.0/${object}.tar.gz | tar -xz -C data
done
```

Each dataset archive holds the trajectories of one object: per-frame object poses
and MANO hand parameters, the object mesh and its convex collision model, and the
arm reset configurations used by the environment. The release also contains the
recorded trajectories from which these datasets were built, `<object>_raw.tar.gz`,
which are the input of the [preprocessing](#preprocessing).

Download `MANO_RIGHT.pkl` from the official [MANO website](https://mano.is.tue.mpg.de/)
(registration required) and place it at `data/mano_v1_2/models/MANO_RIGHT.pkl`.
The datasets store hand motion as MANO parameters, and the hand mesh is built
with this model when the data is loaded.

| Object | Checkpoint |
|---|---|
| `power_drill` | `checkpoints/power_drill.pth` |
| `dog` | `checkpoints/dog.pth` |
| `old_camera2` | `checkpoints/old_camera2.pth` |

All commands run from `rl/` and use the following layout:

```
data/
  mano_v1_2/models/MANO_RIGHT.pkl
  power_drill/
    data/manifest.json
    data/<trajectory>/motion.npy                 object pose and MANO hand parameters
    data/<trajectory>/object.obj                 object mesh
    data/<trajectory>/object_collision.obj       convex collision model
    data/<trajectory>/object_collision.urdf
    ik/<trajectory>_fr3_retarget.npz             arm reset configurations
  dog/ ...
  old_camera2/ ...
checkpoints/
assets/robot/fr3_sharpa_right.urdf
```

`cfg/object/<object>.yaml` sets the data directories, trajectory ids, seed and
hand self-collision of each object. To use data stored elsewhere, pass
`object.data_dir=/path/to/data object.ik_dir=/path/to/ik`.

## Training

```bash
bash scripts/train_4x6144.sh power_drill
bash scripts/train_2x12288.sh dog
```

Both scripts use the same global batch and minibatch size. Extra arguments are
passed to Hydra, for example `full_experiment_name=run0`.

Each run writes to `runs/<object>__<timestamp>/` (or `runs/<full_experiment_name>/`
when it is set): `config.yaml`, checkpoints in `nn/`, TensorBoard summaries in
`summaries/`, and `frame0_done_reasons.jsonl` with the per-trajectory results of
the periodic evaluation.

To resume, pass a checkpoint; the network, optimizer, normalization statistics,
epoch counter and environment state are restored:

```bash
bash scripts/train_4x6144.sh power_drill checkpoint=runs/<run>/nn/<checkpoint>.pth
```

## Evaluation

```bash
bash scripts/eval.sh power_drill checkpoints/power_drill.pth
```

The policy runs with deterministic actions from the first frame of every
trajectory under the training randomization, with the same settings as the
periodic evaluation during training. The success rate is printed, and
per-trajectory results are written to
`runs/power_drill_eval__<timestamp>/frame0_done_reasons.jsonl`.

## Preprocessing

`preprocess/` builds a dataset in the layout above from recorded hand–object
trajectories, such as those in `<object>_raw.tar.gz`. The input is a directory
with a `manifest.json` that maps each sequence name to a motion file and an
object mesh:

```json
{
  "power_drill_00_grasp_move": {"motion": "motion/power_drill_00_grasp_move.npz", "mesh": "meshes/power_drill.obj"}
}
```

Each motion file is an `.npz` archive in a z-up frame, in meters:

| Key | Contents |
|---|---|
| `rhand` | `params` with the right-hand MANO `global_orient` (T×3), `fullpose` (T×45, axis-angle relative to the flat hand) and `transl` (T×3), and `betas` (10) |
| `object` | `params` with `global_orient` (T×3) and `transl` (T×3); object points are `v @ R + transl`, where `R` is the rotation matrix of `global_orient` |
| `framerate` | frames per second |
| `contact` | optional, `object` (T) hand–object contact flags |

The hand mesh uses the same MANO model as above. With a CUDA GPU, run from `rl/`:

```bash
pip install hpp-fcl==2.4.4 coacd
mkdir -p raw
wget -qO- https://github.com/boyuan-an/GALATEA/releases/download/rl-v1.0/power_drill_raw.tar.gz | tar -xz -C raw
bash preprocess/run.sh raw/power_drill output/power_drill
```

`run.sh` runs these stages in order:

| Stage | What it does |
|---|---|
| `quality.py` | Repairs short rotation glitches of the wrist and fingers, trims corrupted endings and drops sequences with object-tracking jumps |
| `prepare.py` | Builds the hand mesh, resamples to 30 Hz and puts the first-frame object on the table |
| `trim_tail.py` | Trims hand jumps that remain at the end of a sequence |
| `wrist_targets.py` | Computes the wrist pose targets from the hand mesh |
| `placement.py` | Places each trajectory in the robot workspace |
| `solve_ik.py` | Chooses the scene orientation and first arm configuration, then solves a continuous arm trajectory and the reset configurations |
| `filter_bank.py` | Keeps the trajectories that have an arm solution |
| `slow_reference.py` | Slows the trajectories down 8× |
| `smooth_ik.py` | Solves the arm again on the slowed trajectories |
| `reset_safety.py` | Checks the reset states for collisions, lifts resets that touch the table and drops trajectories without a safe reset or with discontinuous arm motion |
| `retime.py` | Retimes each trajectory so that arm joint speeds stay within 10% of their limits |
| `finalize.py` | Runs the final collision and continuity checks |
| `collision_models.py` | Decomposes the object mesh into convex parts for simulation |

`WORKERS` (default 32) sets the number of parallel processes. `IK_TIMEOUT`
optionally limits the time of each arm solution in seconds; trajectories that
exceed it are dropped. The dataset is written to `output/power_drill/final/data`
and `output/power_drill/final/ik`. It stores the hand mesh vertices in place of
the MANO parameters; the loader accepts both. To train on it, add a
`cfg/object/<name>.yaml` like the existing ones with these two directories and
the trajectory ids listed in `final/data/manifest.json`, then pass `<name>` to
the training script.

## License

This component is released under the GNU General Public License v3.0
([LICENSE](LICENSE)), separately from the rest of the repository, because it is
modified from GPL-3.0 code. Its sources, the robot model licenses and the
separately obtained dependencies are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
