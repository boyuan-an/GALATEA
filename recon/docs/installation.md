# Installing GALATEA reconstruction

Use Linux x86_64, Conda (Miniforge is suitable), and an NVIDIA GPU. Installation
builds its own CUDA compiler (12.1 by default, 12.8 with `--cuda128`) and C/C++
toolchain inside Conda. It does not require system CUDA or system Eigen, and
does not change shell startup files.

On Ubuntu 22.04, install these OS libraries if they are missing:

```bash
sudo apt-get update
sudo apt-get install -y git ca-certificates libgl1 libegl1 libglib2.0-0 libgomp1 libxrender1
```

Other distributions need the equivalent OpenGL/EGL, GLib, and OpenMP libraries.
The NVIDIA driver remains a host prerequisite; CUDA 12.1 requires a compatible
driver. Linux 530.30.02 or newer is recommended for CUDA 12.1 builds/JIT; see
[NVIDIA's driver compatibility table](https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html). Allow ample disk space for three environments,
compiled extensions, caches, and model weights; 60 GB free is a useful starting
budget, not an upper bound.

## Install

```bash
bash scripts/setup/install.sh
```

This installs dependencies only, with no checkpoint download or model-access
token required. Next, follow the [original providers' model instructions](model-assets.md)
to obtain and place FoundationPose, HOI-DETR, WiLoR, and MANO files before
checking or running reconstruction. GALATEA publishes no third-party weight
bundle. Model assets retain their own terms; see [THIRD_PARTY.md](../THIRD_PARTY.md).

The default CUDA 12.1 profile uses:

| Environment | Python / PyTorch / CUDA | Role |
|---|---|---|
| `sam3d-objects` | 3.11 / 2.5.1 / 12.1 | Depth, segmentation, hands, optimization, rendering |
| `factory` | 3.8 / 2.4.1 / 12.1 | FoundationPose |
| `codetr-cu121` | 3.10 / 2.4.1 / 12.1 | Automatic contact detection |

The installer pins source revisions and direct dependency versions. It checks
an existing environment's Python version. When a third-party source revision
changes, it fetches the new revision first and saves the entire previous
checkout, including local edits, in a sibling `*.backup.*/source` directory.
FoundationPose/HOI-DETR weights are copied into the new checkout. A failed
fetch leaves existing sources untouched. For an isolated installation, set
`CONDA_ENVS_PATH=/your/disk/galatea-envs` before installation **and before running**
GALATEA. Installation, checks and inference all use the first directory in this
variable; they do not fall back to a same-named environment elsewhere.
Conda packages come from conda-forge without depending on your configured
default channels. Set `CACHE_ROOT` before installation **and before running**
to put model/package caches on another disk.

Runtime entry points support `HF_TOKEN_FILE` when a model provider requires
authentication; an explicit `HF_TOKEN` takes priority. Keep tokens outside the
repository. Token contents are never written into generated cache settings.
The runtime disables Python user-site packages to keep packages installed with
`pip --user` from overriding the selected environment. It only reads the
selected project cache, not machine-specific cache or credential paths.

WiLoR and MANO files are stored under `.cache/wilor/pretrained_models/` (or
`$CACHE_ROOT/wilor/pretrained_models/`). `WILOR_CACHE_DIR` overrides the WiLoR
asset root (the parent of `pretrained_models/`). Perception and optimization
resolve the same MANO file. Missing/empty assets stop execution before
WiLoR-mini's automatic downloader can run. Older package-local assets are not
migrated automatically; explicitly select or place assets you are authorized
to use, following the [model guide](model-assets.md#wilor-and-mano).

The installer uses explicit dependency versions for GALATEA: WiLoR-mini's
upstream metadata requests older Torch/YOLO versions, while SAM 2 needs
Torch 2.5.1. MoGe's standalone Gradio application is not installed. As a result,
`pip check` can report upstream metadata conflicts or optional application
dependencies. Validate this reconstruction workflow with `check_install.sh`
and the bundled example rather than installing a different Torch version.

## CUDA 12.8 / RTX 50 profile

Use `--cuda128` for RTX 50 / Blackwell GPUs. This keeps the same pipeline and
dependency pins except for the Python/Torch/CUDA versions below and the
corresponding torchvision versions:

| Environment | Python | PyTorch / torchvision | CUDA |
|---|---|---|---|
| `sam3d-objects` | 3.11 | 2.7.1 / 0.22.1 | 12.8 |
| `factory` | 3.10 | 2.7.1 / 0.22.1 | 12.8 |
| `codetr-cu121` | 3.10 | 2.7.1 / 0.22.1 | 12.8 |

The contact environment is named `codetr-cu121` in both profiles. The installer forwards `--cuda128` to the optional contact installer.

When changing profiles, use a fresh `CONDA_ENVS_PATH` to rebuild the native
extensions against the selected stack; do not upgrade an existing environment
in place. Keep this setting for installation, checks, and inference:

```bash
export CONDA_ENVS_PATH=/your/disk/galatea-cu128-envs
MAX_JOBS=4 bash scripts/setup/install.sh --cuda128
# Obtain and place model files using docs/model-assets.md before checking.
bash scripts/setup/check_install.sh --gpu
bash scripts/run_example.sh --gpu 0
```

These commands expect a compatible host driver. The CUDA 12.8 profile has been
tested through the full example on Ubuntu 22.04 with an RTX 5070 Ti Laptop GPU
(compute capability 12.0) and driver 570.211.01. This is a tested driver version,
not a minimum-driver requirement.

The checker detects the installed profile; it needs no `--cuda128` flag. For
the standalone contact installer, use
`bash scripts/setup/install_env_hoidetr.sh --cuda128`.

## GPU architectures and clusters

With an allocated GPU, compilation detects its compute capability. To compile
without a visible GPU, explicitly specify the GPU you will use:

```bash
# A100/A800
TORCH_CUDA_ARCH_LIST=8.0 bash scripts/setup/install.sh
# RTX 3090: 8.6; RTX 4090 / RTX 5880 / RTX 6000 Ada: 8.9
# RTX 50, without a visible GPU:
TORCH_CUDA_ARCH_LIST=12.0 bash scripts/setup/install.sh --cuda128
```

Use a semicolon-separated list such as `"8.0;8.9"` for multiple targets.
Activation does not force a GPU architecture or leak build settings into the
next environment. On Slurm, allocate a GPU before inference or `--gpu` checks.

All three environments use the selected CUDA profile but remain separate
because their Python and framework dependencies differ. The contact installer
builds MMCV from source for the selected GPU, replacing the legacy CUDA 11.3 wheel. A successful
Python import is not proof that CUDA extensions work on your GPU: run the check
below. Hopper kernels can be compiled with `9.0`; inference must be checked on
the target hardware. The CUDA 12.8 profile additionally accepts `10.0` and
`12.0` compilation targets; only `12.0` has been tested at runtime here.
See [contact environment setup](hoidetr-setup.md) for migration and hardware limits.

## Verify installation without a dataset

```bash
bash scripts/setup/check_install.sh
# On an allocated GPU: test CUDA operations and the EGL rendering context.
bash scripts/setup/check_install.sh --gpu
# Then run the included video and mesh.
bash scripts/run_example.sh --gpu 0
```

The first check imports the pipeline, checks Python/PyTorch versions, loads
native extensions, and verifies FoundationPose/HOI-DETR/WiLoR/MANO file paths. It fails
with a nonzero exit code if a required check fails. It does not download or
install anything. MoGe, SAM2, and GroundingDINO fetch from their original model
repositories on first use, so the example's first run needs model-host access
unless those models are cached. WiLoR/MANO must already be present. No external
dataset is required. See the [example guide](../examples/power_drill/README.md).

## Optional installation modes

- `--cuda128`: select CUDA 12.8 / PyTorch 2.7.1, including the optional contact
  environment. See the profile and migration instructions above.
- `--skip-hoidetr`: install only the main and FoundationPose environments.
  This mode requires explicitly supplied `--contact-windows`; it cannot run
  the default automatic-contact example. Also use
  `--skip-hoidetr` with the installation checker.
- `check_install.sh --skip-weights`: skip local asset checks until the files
  are available. Obtain them using the [model guide](model-assets.md), then
  check again without this flag.
- `--skip-cache-redirect`: retain your configured cache locations.
- `MAX_JOBS=4`: limit native compilation parallelism when memory is limited.

## Troubleshooting

| Failure | Action |
|---|---|
| No visible CUDA GPU during installation | Allocate one, or explicitly set `TORCH_CUDA_ARCH_LIST` for the target GPU. |
| RTX 50 / compute capability 12.0 rejected | Use `--cuda128` with fresh environments; the default CUDA 12.1 profile cannot build for this target. |
| Existing environment has the wrong Python | Use a new `CONDA_ENVS_PATH`; do not mix stacks in one environment. |
| Source revision changed after an update | Rerun installation; old sources are backed up automatically. The installer prints the backup path. |
| Missing FoundationPose/HOI-DETR/WiLoR/MANO files | Follow [official model acquisition and placement](model-assets.md); reinstalling dependencies does not fetch them. |
| HTTP 401 or 403 from a model provider | Complete the provider's access process; use your authorized token if required. |
| Missing `libGL.so.1`, EGL, or GLib | Install the OS libraries listed above. |
| Missing libXrender / `XRenderFindVisualFormat` error | Install `libxrender1`; it is also needed on headless servers. |
| Compilation killed | Lower `MAX_JOBS` and check available memory and disk space. |
| Network download fails | Check connectivity to the original provider and its access instructions; GALATEA provides no model mirror. |
| Imports resolve to unrelated ROS or Python packages | Remove conflicting `PYTHONPATH` entries, or run the command with `env -u PYTHONPATH`. |
