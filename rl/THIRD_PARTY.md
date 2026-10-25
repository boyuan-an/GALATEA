# Third-party components and model assets

This component is distributed under the GNU General Public License v3.0 in
[LICENSE](LICENSE), separately from the rest of the repository. The notices
below identify the code it is modified from and the robot models it includes;
they do not replace any component's license.

GALATEA releases for this component contain source code, robot model files,
pretrained checkpoints and datasets, **not Isaac Gym or MANO model data**.
Obtain those directly from their providers.

## Code and provenance

This component contains code modified in 2026 from the following projects.

| Component and role | Source | License |
|---|---|---|
| ManipTrans: simulation environment, tracking reward, demonstration loading and training loop in `train.py`, `armhand/` and `preprocess/reference.py` | [ManipTrans/ManipTrans](https://github.com/ManipTrans/ManipTrans) | GPL-3.0; this component is distributed under the same license. |
| SimToolReal: rl_games-based SAPG training that `armhand/rl/` follows | [tylerlum/simtoolreal](https://github.com/tylerlum/simtoolreal) | MIT, Copyright (c) 2026 Tyler Lum |
| SAPG: block-structured PPO in `armhand/rl/agent.py`, `blocks.py`, `model.py` and `network.py` | [jayeshs999/sapg](https://github.com/jayeshs999/sapg) | MIT, Copyright (c) 2024 Jayesh Singla |
| rl_games: PPO training loop in `armhand/rl/base.py`, `agent.py`, `model.py`, `network.py` and `running_stats.py` | [Denys88/rl_games](https://github.com/Denys88/rl_games) | MIT, Copyright (c) 2019 Denys88 |
| Isaac Gym Envs: task base class and utilities in `train.py`, `armhand/envs/base.py` and `armhand/utils/` | [isaac-sim/IsaacGymEnvs](https://github.com/isaac-sim/IsaacGymEnvs) | BSD-3-Clause; the NVIDIA notice is kept in the file headers. |

## Robot models

| Model | Files | Source | License |
|---|---|---|---|
| Franka Research 3 | `assets/robot/meshes_isaac/fr3_*` | [frankaemika/franka_description](https://github.com/frankaemika/franka_description), Copyright 2023 Franka Robotics GmbH | Apache-2.0. Collision meshes are unchanged; visual meshes are converted to OBJ. |
| Sharpa Wave hand | `assets/robot/meshes_isaac_wave14_hb1/`, `assets/robot/meshes_isaac/sharpa_*` | [sharpa-robotics/sharpa-urdf-usd-xml](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml), Copyright 2025 Sharpa Group | Apache-2.0. Meshes are unchanged. |

`assets/robot/fr3_sharpa_right.urdf` combines both models with a connector.

## Datasets

The released datasets contain object poses, MANO hand parameters, object meshes
and arm reset configurations; the raw trajectory archives contain the recorded
MANO hand parameters, object poses, contact flags and object meshes. The MANO
model itself is not included.

The `power_drill` mesh derives from **035_power_drill** in the
**Yale-CMU-Berkeley (YCB) Object and Model Set**, by Berk Calli, Arjun Singh,
Aaron Walsman, James Bruce, Konrad Konolige, Siddhartha Srinivasa,
Pieter Abbeel, and Aaron M. Dollar and their collaborators
([YCB object models](https://www.ycbbenchmarks.com/object-models/)), licensed under
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/).
It was converted and modified for simulation; the CC BY 4.0 license continues to
apply to this derivative geometry. No endorsement by the YCB authors is implied.

## Separately obtained software and model data

| Asset | Official source | Terms |
|---|---|---|
| Isaac Gym Preview 4 | [NVIDIA Isaac Gym](https://developer.nvidia.com/isaac-gym) | NVIDIA software license; downloaded separately. |
| MANO_RIGHT.pkl | [MANO registration/download](https://mano.is.tue.mpg.de/), [MANO license](https://mano.is.tue.mpg.de/license.html) | Non-commercial research license; register, accept the terms and place the file locally. |
| manotorch | [lixiny/manotorch](https://github.com/lixiny/manotorch) | GPL-3.0; installed separately. |
| bps_torch | [KailinLi/bps_torch](https://github.com/KailinLi/bps_torch) | Max Planck license of the original bps_torch; installed separately. |
| chamfer_distance | [otaheri/chamfer_distance](https://github.com/otaheri/chamfer_distance) | Terms of the repository; installed separately. |
| PyTorch3D | [facebookresearch/pytorch3d](https://github.com/facebookresearch/pytorch3d) | BSD-3-Clause; installed separately. |
| HPP-FCL (preprocessing) | [humanoid-path-planner/hpp-fcl](https://github.com/humanoid-path-planner/hpp-fcl) | BSD-3-Clause; installed separately. |
| CoACD (preprocessing) | [SarahWeiii/CoACD](https://github.com/SarahWeiii/CoACD) | MIT; installed separately. |

## Retained texts

`third_party_licenses/` contains the Apache-2.0 text, the NOTICE files of
franka_description and sharpa-urdf-usd-xml, and the MIT text with the copyright
lines of rl_games, SAPG and SimToolReal. Keep these texts and the notices above
when redistributing this component.
