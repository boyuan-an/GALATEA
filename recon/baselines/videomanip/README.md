# VideoManip

[Official source](https://github.com/hychen-naza/VideoManip) ·
[Paper](https://arxiv.org/abs/2602.09013) · [Shared interface](../README.md)

This adapter runs VideoManip's reconstruction stages with a supplied metric
object mesh: **MoGe depth → HaMeR hand → SAM2 mask → FoundationPose object**.
Object-mesh generation and robot retargeting are outside this comparison.

## Install

1. Run `python -m baselines.setup --method videomanip` from GALATEA. This fetches
   VideoManip at `9d0f286af35d73f4252d115d968e0a9c06542b9c`, SAM2, HaMeR and
   FoundationPose at the revisions in `../sources.json`. The source patches and
   wrapper files are applied by this command; do not run the upstream
   `setup_externals.sh` afterward, as it can replace the adapted entrypoints.
2. Follow the pinned [reconstruction setup](https://github.com/hychen-naza/VideoManip/blob/9d0f286af35d73f4252d115d968e0a9c06542b9c/reconstruction/README.md#3-setup)
   for the Python environments and model files. Use its component guides for
   [HaMeR](https://github.com/hychen-naza/VideoManip/tree/9d0f286af35d73f4252d115d968e0a9c06542b9c/reconstruction/hamer),
   [SAM2](https://github.com/hychen-naza/VideoManip/tree/9d0f286af35d73f4252d115d968e0a9c06542b9c/reconstruction/sam2), and
   [FoundationPose](https://github.com/hychen-naza/VideoManip/tree/9d0f286af35d73f4252d115d968e0a9c06542b9c/reconstruction/foundationpose).
   FoundationPose needs its compiled extensions and scorer/refiner weights.
   Obtain MANO and HaMeR/ViTPose weights from their original providers.
3. Copy `../configs/videomanip.example.json` to `videomanip.local.json` in the
   same directory and set the four interpreter commands. A single compatible
   environment can serve several stages; the adapter does not require that.

Use the shared `python -m baselines.run --method videomanip ...` command.
The manifest requires video, geometry, intrinsics and one frame-0 positive point.
It preserves supplied K after MoGe predicts depth. Source UVs, textures and
materials are discarded. FoundationPose uses **scale 1.0 and refinement 5/2**;
the historical fast 2/1 setting is not the release default. Both iteration
counts are explicit config fields and affect the run fingerprint.

## Adaptations

The source recipe first applies VideoManip's own SAM2/HaMeR/FoundationPose
patches and added scripts. Our small extra patches make manual-click files
per-run, retain headless operation, fix early CLI parsing of refinement flags,
and preserve the PyTorch/ViTPose and FoundationPose import compatibility fixes
used by the local implementation. No per-user paths or node-local cache bundles
are required. Hand conversion retains the original closest-hand-to-object rule;
missing hand detections remain missing.

The source and models retain their upstream terms; VideoManip's wrappers do not
grant a new license for SAM2, HaMeR, MANO or FoundationPose. Third-party source
and checkpoints are downloaded separately, not copied into GALATEA releases.
