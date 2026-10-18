# Third-party components and model assets

This repository is distributed under the research/evaluation, non-commercial
terms in [LICENSE](../LICENSE). The notices below identify the direct components
of the reconstruction pipeline; they do not replace any component's license or
grant commercial/redistribution rights.

GALATEA releases contain source and attributed example inputs/preview, **not
third-party weights, MANO data, or a combined environment/container**. Obtain
each asset directly from its original provider, following the
[official links, access requirements, and local placement guide](docs/model-assets.md).
Dependency installation does not download checkpoints or accept model terms.

## Code and provenance

| Component and role | Source / pinned revision | License and distribution boundary |
|---|---|---|
| GRAIL, code framework that parts of this codebase build on | [NVlabs/GRAIL](https://github.com/NVlabs/GRAIL) | NVIDIA License, retained at the repository root. |
| FoundationPose, object registration/tracking | [NVlabs/FoundationPose](https://github.com/NVlabs/FoundationPose), pinned `a1b694b83e633c2cb6115b9063d940a687759392` | NVIDIA research/evaluation-only license. Complete [license copy](third_party_licenses/FoundationPose.LICENSE) retained; source is fetched separately into `imports/FoundationPose`. The installer adjusts the CUDA extension C++ standard for the selected torch version. |
| MoGe-2, depth and RGB-only camera bootstrap | [microsoft/MoGe](https://github.com/microsoft/MoGe), pinned `07444410f1e33f402353b99d6ccd26bd31e469e8` | MIT; complete [license copy](third_party_licenses/MoGe.LICENSE). MoGe's bundled DINOv2 code retains its separate Apache-2.0 license. Source is installed separately. |
| HOI-DETR, automatic contact detection | [AhmadDarKhalil/HOI-DETR](https://github.com/AhmadDarKhalil/HOI-DETR), installer pins `1b367292f3833afd64a204bd4d9d84519541d035` | Repository-level MIT; complete [license copy](third_party_licenses/HOI-DETR.LICENSE). Its Co-DETR/MMDetection/MMCV components keep their upstream notices. HOI-DETR source is installed separately; the installer applies two documented MMCV/PyTorch compatibility patches ([setup details](docs/hoidetr-setup.md)). |
| SAM 2, mask propagation | [facebookresearch/sam2](https://github.com/facebookresearch/sam2#license) | Apache-2.0 for model/code; optional cc_torch extension has its own BSD-3-Clause notice. Installed separately. |
| GroundingDINO via transformers, text-guided object detection | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO), [model card](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | Refer to the Apache-2.0 license and model card distributed by the authors; installed/downloaded separately. |
| WiLoR-mini, packaged hand inference | [warmshao/WiLoR-mini](https://github.com/warmshao/WiLoR-mini), pinned `ebec42f94c389070cdd7dda6fd1bf0b4a659c960` | Recommended by the original WiLoR README. The wrapper has no independent license declaration; consult its authors for applicable permissions. Installed separately and not bundled. |
| Original WiLoR | [rolpotamias/WiLoR](https://github.com/rolpotamias/WiLoR), [license.txt](https://github.com/rolpotamias/WiLoR/blob/main/license.txt) | The README identifies the models as CC BY-NC-ND 4.0. External dependencies and wrapper contributions retain their own terms. |
| Ultralytics, WiLoR's hand detector runtime | [Ultralytics 8.3.40 LICENSE](https://github.com/ultralytics/ultralytics/blob/v8.3.40/LICENSE) | AGPL-3.0; installed separately. Its obligations apply independently of this repository's license, including when distributing a combined work. |
| PyTorch3D, differentiable rendering | [PyTorch3D license](https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE) | BSD-3-Clause; installed separately. Preserve the installed package's notices. |
| nvdiffrast, FoundationPose's differentiable renderer | [nvdiffrast license](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt) | NVIDIA Source Code License (1-Way Commercial), including non-commercial research/evaluation use limitations for recipients other than NVIDIA. Installed separately; not Apache-2.0. |

The MoGe/FoundationPose revisions above are pinned in
`scripts/setup/source-revisions.sh`; the HOI-DETR installer also pins its source.
These pins identify the installed sources, not permission to redistribute their
weights or a claim that every transitive dependency is version-locked.

## Checkpoints, body models and input assets

| Asset | Official source / access | Terms and handling |
|---|---|---|
| MoGe-2 ViT-L normal checkpoint | [Ruicheng/moge-2-vitl-normal](https://huggingface.co/Ruicheng/moge-2-vitl-normal) | Model card declares MIT. Downloaded on first use; not committed. |
| FoundationPose scorer/refiner | [Original FoundationPose instructions and author-hosted download](https://github.com/NVlabs/FoundationPose#data-prepare) | Follow NVIDIA's terms and place the matching checkpoint/config directories locally. Not downloaded by the GALATEA installer or redistributed here. |
| HOI-DETR epoch_5.pth | [ahmaddarkhalil/hoi-detr](https://huggingface.co/ahmaddarkhalil/hoi-detr) | Author-hosted checkpoint linked from the official README. Follow the provider's published terms; consult the author for model permissions not explicitly covered. GALATEA does not relicense or redistribute this checkpoint. |
| SAM 2 checkpoints | [SAM 2 license statement](https://github.com/facebookresearch/sam2#license) | Authors license checkpoints under Apache-2.0. Downloaded separately. |
| GroundingDINO tiny checkpoint | [IDEA-Research/grounding-dino-tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | Model card declares Apache-2.0; fetched by the upstream library on first use, not bundled. |
| WiLoR hand/detector checkpoints and mean parameters | [Original WiLoR instructions](https://github.com/rolpotamias/WiLoR#original-installation), [author-hosted checkpoints](https://huggingface.co/spaces/rolpotamias/WiLoR/tree/main/pretrained_models) | Follow the original model and detector terms, including WiLoR's non-commercial/no-derivatives restriction and Ultralytics' terms. Obtain files through these original sources; no GALATEA weight bundle or WiLoR-mini mirror download. |
| MANO_RIGHT.pkl / MANO model data | [MANO registration/download](https://mano.is.tue.mpg.de/), [MANO license](https://mano.is.tue.mpg.de/license.html) | Separate non-commercial research/education/artistic license. Register, accept terms and obtain the model from its provider. GALATEA requires a local file and stops before WiLoR-mini can auto-download it. Do not include these files in this repository or release archives. |
| Input videos, CAD meshes and textures | Provided by the user or their dataset provider | No rights to third-party datasets or private captures are granted by this repository. Redistribute examples only when their original terms allow it. |

## Retained texts

`third_party_licenses/` contains unmodified license texts copied from the listed
source checkouts. Keep these texts and the component notices when redistributing
the applicable source. Package managers install further dependencies with their
own licenses; preserve those packages' notices when creating binary/container
distributions.

## Optional baseline adapters

`baselines/` contains GALATEA's execution/input/output adapters and source patches.
Exact source revisions and installation order are recorded in
[`baselines/sources.json`](baselines/sources.json). Upstream repositories are
fetched separately into ignored `third_party/`; this release includes no baseline
checkpoints or dataset captures. Preserve the applicable upstream notices when
distributing patched dependencies.

| Method / dependency | Original source | Attribution |
|---|---|---|
| VideoManip | [hychen-naza/VideoManip](https://github.com/hychen-naza/VideoManip) | Its reconstruction wrappers supply separate SAM2, HaMeR and FoundationPose patches. Consult each component's original terms; no blanket license for their code or models is implied. |
| EgoInfinity | [Rice-RobotPI-Lab/EgoInfinity](https://github.com/Rice-RobotPI-Lab/EgoInfinity) | MIT source notice retained at `baselines/egoinfinity/UPSTREAM_LICENSE`; models and external components keep their own terms. |
| FoundationPose++ | [teal024/FoundationPose-plus-plus](https://github.com/teal024/FoundationPose-plus-plus) | Installed separately at the adapter's pinned revision. FoundationPose, Cutie and the remaining components retain their upstream terms. |
| DO AS I DO | [malik-group/do-as-i-do](https://github.com/malik-group/do-as-i-do) | MIT source notice retained at `baselines/do_as_i_do/UPSTREAM_LICENSE`; its Fast-SAM3D, SAM3D, SAM3, HaWoR and TAPIR submodules retain their upstream terms. |

Installation, model-acquisition links and the differences from the original
inference settings are documented in each [baseline guide](baselines/README.md).

## Bundled power-drill example

The included mesh derives from YCB 035_power_drill (CC BY 4.0); it has been
scaled and simplified. Its source, changes, and attribution are recorded in
[examples/power_drill/ATTRIBUTION.md](examples/power_drill/ATTRIBUTION.md). The
generated video and reference capture are provided by the GALATEA project.
