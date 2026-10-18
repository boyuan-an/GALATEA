# Model assets from the original providers

GALATEA distributes source code, license notices, and the attributed example
inputs/preview. **We do not publish third-party checkpoints, MANO model data,
model mirrors, or a combined weight/environment/container archive.** Each user
obtains model assets from the original provider under that provider's terms.
Links and installation commands do not grant additional rights or accept terms
on the user's behalf. See [THIRD_PARTY.md](../THIRD_PARTY.md) for component
licenses and attribution.

`scripts/setup/install.sh` and `install_env_hoidetr.sh` install dependencies only.
After installation, prepare the local files below before running the example.
All paths are relative to the `recon/` directory unless otherwise specified.

## FoundationPose

Follow the original [FoundationPose data preparation instructions](https://github.com/NVlabs/FoundationPose#data-prepare)
and [NVIDIA license](https://github.com/NVlabs/FoundationPose/blob/main/LICENSE).
Download the scorer/refiner directories from the
[authors' Google Drive folder linked in that README](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing).
Keep each checkpoint with its matching configuration:

```text
imports/FoundationPose/weights/
├── 2023-10-28-18-33-37/       # refiner
│   ├── model_best.pth
│   └── config.yml
└── 2024-01-11-20-02-45/       # scorer
    ├── model_best.pth
    └── config.yml
```

## HOI-DETR

Follow [the author's checkpoint instructions](https://github.com/AhmadDarKhalil/HOI-DETR#1-download-checkpoint)
and download `epoch_5.pth` from
[ahmaddarkhalil/hoi-detr](https://huggingface.co/ahmaddarkhalil/hoi-detr/tree/main).
Place it at `3rd/HOI-DETR-cu121/checkpoints/epoch_5.pth`, or
`$HOIDETR_ROOT/checkpoints/epoch_5.pth` when that source root is overridden.
The upstream repository's code license is MIT. Follow the provider's published
model terms and consult the author for permissions not explicitly covered;
GALATEA does not relicense or redistribute this checkpoint.

## WiLoR and MANO

Follow [WiLoR's original installation instructions](https://github.com/rolpotamias/WiLoR#original-installation).
WiLoR's [model terms](https://github.com/rolpotamias/WiLoR/blob/main/license.txt)
are CC BY-NC-ND 4.0; the detector also depends on
[Ultralytics](https://github.com/ultralytics/ultralytics#license).
Obtain these files directly from the original providers:

| File | Original source and required action |
|---|---|
| `wilor_final.ckpt` | [Author-hosted checkpoint](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/wilor_final.ckpt), linked by the original WiLoR README. |
| `detector.pt` | [Author-hosted detector](https://huggingface.co/spaces/rolpotamias/WiLoR/resolve/main/pretrained_models/detector.pt), linked by the original WiLoR README. |
| `mano_mean_params.npz` | [Original WiLoR repository file](https://github.com/rolpotamias/WiLoR/blob/main/mano_data/mano_mean_params.npz) under `mano_data/`; obtain it from that checkout/download. |
| `MANO_RIGHT.pkl` | Register at the [MANO website](https://mano.is.tue.mpg.de/), review and accept the [MANO terms](https://mano.is.tue.mpg.de/license.html), then download the MANO model archive through its [download page](https://mano.is.tue.mpg.de/download.php) and extract the right-hand model. This file is not supplied by GALATEA. |

For the WiLoR-mini runtime used here, place these files in this layout (different
from the original WiLoR demo's `mano_data/` layout):

```text
.cache/wilor/pretrained_models/
├── wilor_final.ckpt
├── detector.pt
├── mano_mean_params.npz
└── MANO_RIGHT.pkl
```

With `CACHE_ROOT` set, use `$CACHE_ROOT/wilor/pretrained_models/`. To use an
existing directory of assets you are authorized to use, set `WILOR_CACHE_DIR`
to the **parent of `pretrained_models/`**, before checking and running. Local
symlinks to those files are supported. Perception and optimization resolve the
same MANO file.

GALATEA checks for all four nonempty files **before constructing WiLoR-mini**,
so missing files cannot trigger its mirror downloader. It does not automatically
migrate package-local assets from older installations. MANO registration and
acceptance must happen with its provider; an existing file or a successful
installation check is not proof of permission. Do not share MANO data in a
release, cache archive, container, or environment export containing files.

## Models fetched by upstream libraries on first use

The following public models still use the original libraries' download/cache
mechanisms. Review the linked terms before first use. These are upstream
repositories, not GALATEA model packages:

| Component | Original code / model source | Published model terms |
|---|---|---|
| MoGe-2 | [microsoft/MoGe](https://github.com/microsoft/MoGe), [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | MIT in the model card. |
| SAM 2 | [facebookresearch/sam2](https://github.com/facebookresearch/sam2), [facebook/sam2-hiera-large](https://huggingface.co/facebook/sam2-hiera-large) | [Apache-2.0 for checkpoints](https://github.com/facebookresearch/sam2#license). |
| GroundingDINO | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | Apache-2.0 in the model card. |

Keep `CACHE_ROOT` consistent between installation and inference. The default HF
cache is `.cache/huggingface/`. These models need access to their original model
hosts on the first run unless already cached. No GALATEA-hosted fallback is
configured. If you override a model ID or endpoint, check its provenance and
terms separately.

## Verify and run

```bash
# Dependencies only, before local assets have been placed:
bash scripts/setup/check_install.sh --skip-weights --gpu
# After placing FoundationPose, HOI-DETR, WiLoR and MANO files:
bash scripts/setup/check_install.sh --gpu
bash scripts/run_example.sh --gpu 0
```

The checker does not download assets. It checks required local file paths and
basic completeness, imports, and (with `--gpu`) native CUDA operations. It does
not validate licenses or fully load every checkpoint. The example exercises
model loading/inference, including the three upstream first-use downloads.
