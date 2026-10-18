# DO AS I DO

[Official source](https://github.com/malik-group/do-as-i-do) ·
[Paper](https://arxiv.org/abs/2606.19333) · [Shared interface](../README.md)

This is the **coordinate-aligned known-mesh adaptation** used in our comparison.
The upstream method normally reconstructs its own mesh from RGB. Our wrapper
first runs that same RGB frontend, then aligns the supplied geometry into its
local frame before the original guided tracking and final optimization.

## Install

1. Run `python -m baselines.setup --method do_as_i_do`. This fetches
   `b5a617060a970ba2d9af7b2e216903c643cda185` with its pinned submodules and applies
   the runtime, Fast-SAM3D and HaWoR patches from `../sources.json`.
2. Follow the pinned [reconstruction README](https://github.com/malik-group/do-as-i-do/blob/b5a617060a970ba2d9af7b2e216903c643cda185/reconstruction/README.md)
   and [environment guide](https://github.com/malik-group/do-as-i-do/blob/b5a617060a970ba2d9af7b2e216903c643cda185/reconstruction/env/README.md)
   for the SAM3, SAM3D, HaWoR and TAPIR environments. Complete the upstream
   Mip-Splatting renderer build and obtain SAM checkpoints, TAPIR and MANO assets
   through their original providers. The upstream guide recommends ≥32 GB VRAM.
3. Copy `../configs/do_as_i_do.example.json` to `do_as_i_do.local.json` in the
   same directory. Set the four `ENV_*` names or absolute environment prefixes,
   and the CPU `adapter_python` command. `conda` must be available on PATH for
   the upstream driver. Set `DOAS_SAM3D_CONFIG` or `SAM3_CKPT` only when storing
   checkpoints outside the upstream default locations.

Use `python -m baselines.run --method do_as_i_do ...` with one frame-0 RGB-only
positive point. Source dimensions are in meters. This adapter never passes
supplied calibration, sensor depth, GT masks, hand labels or pose labels into
the frontend or tracking. There is no interactive display requirement when a
point is supplied.

## Execution and coordinates

1. Run upstream stages 0–2.5 on the copied input video: segmentation, SAM3D
   reconstruction, MoGe, static-camera HaWoR, gravity, and TAPIR motion.
2. Compute the existing geometry-only PCA/normalized-Chamfer alignment from
   known mesh to reconstructed mesh. Bake its rotation, translation and scale
   into a separate tracking mesh and record both source hashes and the transform.
3. Run upstream stages 3–4 with the aligned mesh: guided pose tracking,
   camera-frame conversion and translation/scale optimization.
4. Use the frozen alignment metadata to express the result in the supplied
   metric mesh's coordinates. Respect the optimized top-level mesh scale and
   HaWoR's right-hand validity mask; preserve missing frames as NaN.

Passing the unaligned CAD mesh directly into the SAM3D frame is incorrect.
`mesh.py` preserves the validated conversion and its round-trip regression test.
Symmetric or poor reconstructions can still give ambiguous geometric alignment.

The patches add headless point input, frontend/track stage boundaries, supplied
mesh routing and compatibility fixes. They make environment paths configurable
and preserve an inherited GPU allocation. Native inference steps and optimizer
defaults are retained. Node-local cache bundles, timing experiments, recovery
arrays and method-ablation launchers remain outside this interface.

The upstream MIT notice is retained in `UPSTREAM_LICENSE`; submodules and weights
remain under their respective original licenses.
