# GALATEA reinforcement-learning tracking

> [!NOTE]
> This public code was obtained by reorganizing our development code with vibe coding.
> If you run into any problems, please [open an issue](https://github.com/boyuan-an/GALATEA/issues).

[Installation](#installation) · [Data and checkpoints](#data-and-checkpoints) · [Training](#training) · [Evaluation](#evaluation)

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

The datasets will be added to the same release.

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
    data/<trajectory>/102_sv_dict.npy      object pose and MANO hand parameters
    data/<trajectory>/102_obj.obj
    data/<trajectory>/102_obj_coacd.obj
    data/<trajectory>/102_obj_coacd.urdf
    ik/<trajectory>_fr3_retarget.npz       arm reset configurations
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

## License

This component is released under the GNU General Public License v3.0
([LICENSE](LICENSE)), separately from the rest of the repository, because it is
modified from GPL-3.0 code. Its sources, the robot model licenses and the
separately obtained dependencies are listed in [THIRD_PARTY.md](THIRD_PARTY.md).
