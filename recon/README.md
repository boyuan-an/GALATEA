# GALATEA HOI reconstruction

> [!NOTE]
> This public code was obtained by reorganizing our development code with vibe coding.
> If you run into any problems, please [open an issue](https://github.com/boyuan-an/GALATEA/issues).

[Installation](#installation) · [Quick start](#quick-start) · [Baselines](baselines/README.md) · [Documentation](#documentation)

This directory contains the hand–object interaction (HOI) reconstruction
component of [GALATEA](../README.md): given a video and a known object mesh, it
estimates the object's 3D position and orientation together with an articulated
MANO hand mesh. MANO is the parametric hand model used to represent hand shape
and finger pose.

The entry point documented here runs perception, joint hand–object optimization,
trajectory export, and visual inspection. Paper-specific benchmark drivers,
ground-truth scoring scripts, and result-table generators are not distributed
with this reconstruction package. Use the bundled example and RGB overlays
to check installation and reconstruction quality. GALATEA's simulation training,
policy distillation, and real-world controller deployment are separate parts of
the system; see the [project page](https://boyuan-an.github.io/GALATEA/) for the
overall method.

## Reconstruction pipeline

The pipeline first estimates depth, segmentation, hand motion, object motion,
and contact intervals. It then refines the hand and object together so their
geometry agrees with image evidence and estimated contact.

| Stage | What it does |
|---|---|
| Depth and camera reference | MoGe-2 predicts per-frame depth; the first-frame reference anchors depth alignment. |
| Object segmentation | GroundingDINO locates the named object, and SAM 2 propagates its mask through the video. |
| Object tracking | FoundationPose estimates the pose of the supplied metric CAD mesh. Low mask coverage can trigger a color-adapted retry. |
| Hand reconstruction | WiLoR estimates the right hand. Temporal cleanup repairs short, bounded detection gaps and jumps. |
| Contact estimation | HOI-DETR estimates when the hand contacts the object; manually supplied intervals take priority. |
| Joint optimization | Image silhouettes, keypoints, contact, and temporal constraints refine hand–object placement. |
| Output alignment | The joint result is retained in camera coordinates and transformed once into a Z-up world frame. |

The jointly optimized hand and object are the final camera-frame result.
FoundationPose provides the initial track and soft constraints during optimization.
Small depth corrections are allowed; image evidence can relax the penalty on
larger corrections. The optimized trajectory is retained for visualization and export. World alignment applies one constant
coordinate transform to the whole sequence and preserves the optimized motion.

### Input modes

Both modes require a **fixed camera**, a **known rigid object mesh in meters**,
and a video with **one interacting right hand**. No ground-truth hand or object
poses are required.

| Mode | Required inputs | First-frame depth and intrinsics |
|---|---|---|
| Provided RGB-D calibration | Video, CAD mesh, reference RGB image, depth image, and camera intrinsics | Supplied by the user; this is the input configuration described in the paper, using stereo depth. |
| RGB-only | Video and CAD mesh | Predicted from the first video frame with MoGe-2. This additional mode supports inputs without measured calibration. |

**RGB-only predictions are not measured RGB-D.** Their source, input hashes,
resolution, frame count, and FPS are recorded in `prep_metadata.json`. Keep
the input modes separate when reporting experiments or comparing against the
paper. Video preparation preserves the input frame rate without resampling.

## Installation

Use Linux x86_64, Conda, and an NVIDIA GPU. The installer supplies its own
CUDA compiler (12.1 by default), Eigen, and C/C++ toolchain, detects the target GPU
architecture, and pins the reconstruction dependencies. See the
[installation guide](docs/installation.md) for OS libraries, GPU compatibility,
cluster builds, and troubleshooting.

```bash
git clone https://github.com/boyuan-an/GALATEA.git
cd GALATEA/recon
bash scripts/setup/install.sh
```

Obtain and place the model files described in [docs/model-assets.md](docs/model-assets.md),
then check the installation (allocate a GPU first on Slurm):

```bash
bash scripts/setup/check_install.sh --gpu
```

For RTX 50 / Blackwell, add `--cuda128` to the installation command. This selects
CUDA 12.8 / PyTorch 2.7.1 and Python 3.10 for FoundationPose. When switching from
an existing CUDA 12.1 installation, use fresh environments as described in the
[installation guide](docs/installation.md#cuda-128--rtx-50-profile).

Installation prepares dependencies only; it does not download checkpoints and
needs no model-access token. Follow the [official model links and placement guide](docs/model-assets.md)
to obtain FoundationPose, HOI-DETR, WiLoR, and MANO assets under their original
terms. MANO requires registration with its provider. Use
`check_install.sh --skip-weights` to check dependencies before assets are ready.

The pipeline switches between its environments automatically. Package and model
caches stay in `recon/.cache/` by default. Model weights are separate
from the bundled example data and retain their [own terms](THIRD_PARTY.md).
The first-run commands include HOI-DETR for automatic contact detection in the
bundled example. If it was previously skipped, run
`bash scripts/setup/install_env_hoidetr.sh` (add `--cuda128` for the CUDA 12.8 profile).
See the [installation guide](docs/installation.md) for all options.

## Quick start

### Run the included example

The repository includes a reviewed **7-second power-drill manipulation video**,
a matching mesh, first-frame reference depth, and calibrated intrinsics.
The hand grasps, lifts, tilts, and moves the drill. No external dataset is needed.

```bash
bash scripts/run_example.sh --gpu 0
```

The default inputs follow [Section III-A.2 of the paper](https://arxiv.org/html/2609.10050v2#S3.SS1.SSS2):
video, metric object mesh, calibrated intrinsics, and first-frame reference
depth. The object name guides detection. MoGe predicts subsequent depth maps
and aligns them to the supplied reference; HOI-DETR estimates contact intervals.
No contact annotations are supplied. Normal optimization settings and inspection
outputs are enabled. See the
[example guide](examples/power_drill/README.md) for outputs, provenance, units,
and asset attribution, or [preview the automatic result](examples/power_drill/preview.mp4).
Prepare the [required local model assets](docs/model-assets.md) first. MoGe,
SAM 2, and GroundingDINO may download from their original model hosts on first use.

### Use your own inputs

Both modes need a fixed camera, one interacting right hand, and a rigid mesh
with the correct physical dimensions in **meters**. For the paper's input
configuration, place `rgb.png`, `depth.png`, and `cam_intrinsics.txt` beside
the video, then run:

```bash
bash scripts/run_pipeline_custom.sh \
    --video /path/to/video.mp4 --mesh /path/to/object.obj \
    --object-name bottle \
    --tag bottle_demo --gpu 0 --with-qc
```

The English object name guides detection. Contact is estimated automatically.
For a deliberate manual override, add `--contact-windows START-END`; frame
indices start at zero and include both endpoints.

If reference depth and intrinsics are unavailable, the optional `--rgb-only`
mode predicts both with MoGe. That extension uses different inputs from the paper.
See the [custom-video guide](docs/pipeline-quickstart.md) for exact formats,
contact review, perception-only runs, and cache behavior.

## Results and inspection

For the bundled example, results are under `data/custom/`:

| Output | Contents |
|---|---|
| `_phase3/power_drill_auto/gr_hsz_world.npy` | Object-to-world transforms, `(169, 4, 4)` |
| `_phase3/power_drill_auto/gr_hsz_handverts_world.npy` | MANO hand vertices, `(169, 778, 3)` |
| `_bench/power_drill_auto/hoidetr_contact.json` | Target/hand associations, automatic contact intervals, input hashes, parameters, and inference errors |
| `_viz/power_drill_auto/` | Object overlays, contact checks, and a four-view reconstruction video |

Coordinates use meters, +Z up, and an estimated support plane at z=0. Camera
coordinates also retain the same joint optimization result; see the
[output reference](docs/pipeline-quickstart.md#outputs).

Inspect object tracking and hand placement before using the trajectories.
Successful execution can still leave gaps, penetration, or tracking drift.
The outputs are motion references that need adaptation to a robot's coordinate
frame and controller.

Use a fresh `--tag` when changing inputs or settings. See
[reruns](docs/pipeline-quickstart.md#reruns-and-optional-outputs) before reusing
cached results. Full options: `bash scripts/run_pipeline_custom.sh --help`.

## Failure cases

Most failure cases come from FoundationPose tracking failures. The joint
hand–object optimization can recover some of them, but not all. We found the
camera viewpoint of the video to be important. Fortunately, our videos are
generated in controlled settings, so the viewpoint can also be controlled during
generation; this controllable generation allows the whole reconstruction
pipeline to perform stably.

## Documentation

- [Baseline implementations](baselines/README.md) — VideoManip, EgoInfinity + FP++, and DO AS I DO adapters, pinned sources, setup and shared outputs.
- [Installation and checks](docs/installation.md) — dependencies, GPU support, and troubleshooting.
- [Included example](examples/power_drill/README.md) — a complete first run without a dataset download.
- [Custom-video walkthrough](docs/pipeline-quickstart.md) — inputs, contact review, outputs, and reruns.
- [HOI-DETR setup](docs/hoidetr-setup.md) — installation, hardware support, and verification.
- [Reconstruction correctness guards](docs/reconstruction-correctness.md) — missing observations and dense-mesh rendering.
- [Third-party components and assets](THIRD_PARTY.md) — provenance and licensing boundaries.

## Citation

If you use this code in your research, please cite [GALATEA](../README.md#citation).

## Acknowledgments

The reconstruction pipeline integrates MoGe-2, GroundingDINO, SAM 2,
FoundationPose, WiLoR, and HOI-DETR. Component sources and terms are listed in
[THIRD_PARTY.md](THIRD_PARTY.md); see also the
[project acknowledgments](../README.md#acknowledgments).

## License

This code is released under the [NVIDIA License](../LICENSE), which limits use
to non-commercial research and evaluation. Third-party code, checkpoints, and
MANO data retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
We publish no third-party weight bundle or combined environment/container.
Use the [original providers' links and acquisition instructions](docs/model-assets.md).
Downloading an asset does not grant additional redistribution rights.
