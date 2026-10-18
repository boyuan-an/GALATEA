# EgoInfinity + FP++

[Official source](https://github.com/Rice-RobotPI-Lab/EgoInfinity) ·
[FP++ source](https://github.com/teal024/FoundationPose-plus-plus) ·
[Shared interface](../README.md)

The released path runs **EgoInfinity's complete frontend/post-tracking chain →
known-mesh FP++ → upstream rotation-only bake → camera-space export**. An
intermediate phase-D rotation track is not accepted as the final baseline.

## Install

1. Run `python -m baselines.setup --method egoinfinity`. It fetches EgoInfinity
   `de59610531c08010d63d0493d6a72973eabd402e` and FP++
   `58aa7154de23ae86d542366858fc6bcca3b747b0`, then applies the recorded adaptations.
2. Follow EgoInfinity's [setup and optional-component instructions](https://github.com/Rice-RobotPI-Lab/EgoInfinity/tree/de59610531c08010d63d0493d6a72973eabd402e).
   Install its main environment, WiLoR sources/assets, SAM2, MEMFOF and infiller.
   Set up separate SAM3 and SAM 3D Objects environments and obtain access to
   their gated checkpoints. Obtain MANO from its provider.
3. Follow the pinned [FP++ installation guide](https://github.com/teal024/FoundationPose-plus-plus/blob/58aa7154de23ae86d542366858fc6bcca3b747b0/Install.md).
   FP++ needs its FoundationPose scorer/refiner and compiled `mycpp` extension,
   Cutie weights and dependencies. Install the adapter requirements in the
   environment that runs `baselines.egoinfinity.fpplusplus`. If using its
   official container, configure `fp_python` as a container command prefix and
   mount source, outputs and weights at matching absolute paths.
4. Copy `../configs/egoinfinity.example.json` to `egoinfinity.local.json` in the
   same directory. Set `ego_python`, `adapter_python`, `fp_python`, and the SAM3/
   SAM3D environment variables. `adapter_python` needs NumPy, SciPy, trimesh and
   OpenCV, but does not need to load a GPU model. `fp_env` can add FP++-specific
   settings separately from the frontend's `env`.

Use `python -m baselines.run --method egoinfinity ...` with the shared manifest.
Intrinsics are predicted by EgoInfinity. The optional `frame0_mask` selects
the RGB-reviewed initialization variant; without it, detection uses the supplied
noun phrase and its final noun token. Later masks always come from the method.

## Supplied-mesh protocol

Known geometry is oriented using the native reconstruction's axes while keeping
its true meter-scale dimensions. FP++ receives this geometry, a uniform solid
color, EgoInfinity's RGB, predicted depth, predicted K and predicted masks.
It runs the recorded upstream settings: refinement **10/5**, 2D tracking and
Kalman filter enabled, measurement-noise scale **0.05**. Its outputs must pass
shape, finiteness and rigid-transform checks before bake. Bake runs on a copied
PKL and verifies that translations remain unchanged; the upstream interaction
logic incorporates the new rotations.

The patch also exports the surface mesh from the same SAM3D inference, fails
when a required worker disappears, supports reviewed first-frame initialization,
and retains the local Open3D/CPU-thread compatibility fixes. A manifest option
preserves the input frame order instead of the upstream default 15 FPS extraction.
Worker sockets are unique to a run. Checkpoint staging to a specific cluster's
NVMe and historical experiment schedulers are not required.

EgoInfinity's license notice is retained in `UPSTREAM_LICENSE`. Its dependencies,
FP++, and all model assets retain their own terms. Use the official repository's
citation metadata when citing EgoInfinity and FP++.
