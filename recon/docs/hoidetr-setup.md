# Automatic contact detection

HOI-DETR estimates the frames in which a hand contacts the object. The standard
installer prepares its separate `codetr-cu121` environment. The bundled example
uses this component by default. Explicit manual intervals can bypass it.

## Install and check

```bash
# Only the contact component:
bash scripts/setup/install_env_hoidetr.sh
# For RTX 50 / Blackwell, use this installation command instead:
bash scripts/setup/install_env_hoidetr.sh --cuda128
# Obtain epoch_5.pth using docs/model-assets.md#hoi-detr before checking.
# Read-only import/config/weight check:
bash scripts/setup/install_env_hoidetr.sh --check-only
# On an allocated GPU, also check CUDA NMS, deformable attention, and input transfer:
bash scripts/setup/install_env_hoidetr.sh --check-only --gpu
```

The HOI-DETR source revision is pinned to
`1b367292f3833afd64a204bd4d9d84519541d035`. This installer follows the
[upstream PyTorch 2.4 migration](https://github.com/AhmadDarKhalil/HOI-DETR/blob/main/INSTALL_HOPPER.md),
adapted to Linux x86_64. Both installation profiles use Python 3.10 and
MMCV-full 1.7.2:

| Profile | PyTorch | torchvision | CUDA |
|---|---|---|---|
| Default | 2.4.1+cu121 | 0.19.1+cu121 | 12.1 |
| `--cuda128` | 2.7.1+cu128 | 0.22.1+cu128 | 12.8 |

MMCV is built from the pinned v1.7.2 source against the selected Torch/CUDA
combination, using a Conda-provided
compiler. Two narrow compatibility patches update MMCV's CUDA stream handling
and its optional DDP flag for these PyTorch versions. Reapplying them is safe; unexpected
source layouts fail explicitly. Additional dependency versions are fixed in
`scripts/setup/hoidetr-constraints.txt` and `scripts/setup/hoidetr-cu128-constraints.txt`.
The installer does not download weights. Obtain the approximately 5.9 GB
checkpoint from the [author's model repository](https://huggingface.co/ahmaddarkhalil/hoi-detr)
using the [upstream instructions and local placement guide](model-assets.md#hoi-detr).
It is a model checkpoint, not a dataset download or a GALATEA weight bundle.

Defaults use the `codetr-cu121` environment and the `3rd/HOI-DETR-cu121`
source directory. Without this component, supply manual contact intervals.
`HOIDETR_ENV`, `HOIDETR_ROOT`, and `HOIDETR_MMCV_ROOT` override the environment,
HOI-DETR checkout, and MMCV build-source directory respectively.
An existing different source revision is backed up to a sibling
`*.backup.*/source` directory before the new checkout is activated; downloaded
checkpoints are retained. A failed fetch leaves existing sources untouched.
`CACHE_ROOT` configures cache storage.
Keep `CONDA_ENVS_PATH` and `CACHE_ROOT` consistent between installation and runs.
When changing CUDA profiles, select a fresh `CONDA_ENVS_PATH`; see the
[migration instructions](installation.md#cuda-128--rtx-50-profile).
The `codetr-cu121` name is retained with `--cuda128`. Check-only commands detect
the installed profile and do not need this flag.
Installation is dependency-only. With `--check-only`, `--skip-weights` skips the
checkpoint check until you have obtained the file from its provider.

## Hardware

The contact environment uses the selected CUDA profile, matching the other two
environments when installed through `install.sh`. Keep these Python environments separate.
Compilation detects the allocated GPU; for a machine without a visible GPU,
set `TORCH_CUDA_ARCH_LIST` (for example `8.0` for A100/A800 or `8.9` for Ada).
For multiple targets, use a quoted list such as `"8.9;9.0"`.

Run the GPU check on the actual inference device. It verifies CUDA NMS,
deformable-attention forward/backward against a reference, and CPU-to-GPU input
transfer. Import-only checks do not establish GPU compatibility. This stack can
compile Hopper kernels with `TORCH_CUDA_ARCH_LIST=9.0`; that is not a claim of
Hopper inference validation. The CUDA 12.8 profile also accepts `10.0` and `12.0`
targets. ARM is outside this installer; see the upstream migration
instructions linked above.

GPU operator checks and full detector inference have been tested on RTX 5880
Ada with CUDA 12.1 and RTX 5070 Ti Laptop with CUDA 12.8. Compute capabilities
9.0 and 10.0 have not been tested at runtime. These compatibility checks do not
establish contact accuracy on other videos.

## Use manual intervals instead

Install GALATEA with `--skip-hoidetr`, then supply `--contact-windows 24-100`
for your video (replace these frame numbers). This is a manual override, not
the default example workflow. For a perception-only first pass, use
`--contact-mode manual --prep-only` and then supply the interval for the full run.
`--prep-only` alone still uses automatic detection by default.

Automatic boundaries now default to zero start/end trim. Median filtering and
brief-gap merging remain; contact reaching the start or end of a clip is kept.
Nonzero `HOIDETR_START_TRIM`/`HOIDETR_END_TRIM` values
are explicit user overrides in frames.

The pipeline runs contact detection after SAM/object recovery and WiLoR. It
selects one object box matching the reconstructed target and one hand box
matching the primary observed right hand, independently of contact confidence.
Other pairs are excluded. Missing or ambiguous matches remain unmatched, and
interpolated WiLoR frames are not treated as observed hands. The output records
the target boxes, selected detection indices, IoUs, and match statuses.

Automatic reports are reused only when content hashes of the RGB timeline,
target tracks, model weights/configuration/source, and inference parameters
match a complete report. `--force` refreshes automatic contact even with an
existing auto sidecar. Manual CLI intervals and deliberate sidecar overrides
retain priority; see [contact intervals](pipeline-quickstart.md#contact-intervals).

For model-level inference verification, run a video without a manual contact
interval and inspect the generated `hoidetr_contact.json`, error frames, and
contact montage. The kernel check is not a full detector-quality test.

## Attribution

The retained [HOI-DETR license](../third_party_licenses/HOI-DETR.LICENSE) and
[third-party notice](../THIRD_PARTY.md) describe the component and asset terms.
