# Power-drill example

This example contains a **7.04-second generated video (169 frames at 24 FPS)**.
A right hand approaches the drill, grasps its handle, lifts it, tilts it, and
moves it forward and back. The clip was selected from the project's reviewed
generated videos. It is an end-to-end usage example, not a benchmark result.

The inputs follow [Section III-A.2 of the paper](https://arxiv.org/html/2609.10050v2#S3.SS1.SSS2):
generated video, metric object mesh, calibrated camera intrinsics, and
first-frame reference depth. The reference RGB image is included for alignment.
Contact intervals are estimated by HOI-DETR; no contact annotation is supplied.
**No external dataset download is needed.** Model weights are still required;
see [installation](../../docs/installation.md) and
[official model acquisition](../../docs/model-assets.md).

The [preview](preview.mp4) shows input and reconstruction side by side from
the default automatic-contact run, all 169 frames at 24 FPS. It uses the final
jointly optimized trajectories without manual edits. Small contact gaps and
intersections can remain; inspect the output before using it.

## Run

From the `recon/` directory, after installation and placement of the required model files:

```bash
bash scripts/setup/check_install.sh --gpu
bash scripts/run_example.sh --gpu 0
```

The check covers all three environments, including HOI-DETR. The run keeps the
supplied depth and intrinsics, estimates contact from the video, and processes
all frames with normal optimizer settings and inspection images/videos enabled.
For this asymmetric crop, the example also corrects the depth model's centered
camera assumption and reuses the first frame's depth scale and offset throughout
the clip. Each frame still gets its own depth prediction; the input video and
output camera coordinates are unchanged. The settings are
`DEPTH_RECTIFY_CAMERA=1` and `DEPTH_CALIBRATION=reference`, and can be overridden
through the environment. See [cropped inputs](../../docs/pipeline-quickstart.md#cropped-calibrated-inputs).
Contact probabilities, target/right-hand matches, post-processing parameters,
input hashes, and any inference errors are saved in `hoidetr_contact.json`.

The default tag is `power_drill_auto`. For a fresh run with no reconstruction
or contact cache, choose a new output directory:

```bash
CUSTOM_DATA_ROOT=/path/to/new-output-directory bash scripts/run_example.sh --gpu 0
```

Model weights and compiled kernels can be reused; the output directory must
not contain prepared frames, `contact.yaml`, or previous stage outputs.

The first run may download MoGe/SAM 2/GroundingDINO weights from their original
model hosts and compile CUDA kernels. WiLoR/MANO, FoundationPose, and HOI-DETR
files must already be present; GALATEA does not distribute them.
Subsequent runs validate completed stages and automatic contact caches before
reuse. Changed inputs/settings invalidate the contact estimate; `--force`
recomputes it. Use a fresh `--tag` or output directory for an independent run. `--prep-only`
stops after perception for inspection. RGB-only prediction of depth and intrinsics
is an optional extension (`--rgb-only`), not this example's default input mode.

The included preview uses the automatically estimated contact interval 16–168
with zero start/end trim. This is a model prediction to inspect alongside the
visible interaction; it is not an input annotation or ground truth.

## Inspect the result

The default tag is `power_drill_auto`. Under `data/custom/`:

| File | What to check |
|---|---|
| `_prep/power_drill_auto/cam0/prep_metadata.json` | Input mode is `rgbd`; the reference hashes identify the supplied calibration files. |
| `_bench/power_drill_auto/hoidetr_contact.json` | Contact source is `hoidetr-auto`; inspect target/hand associations, boundaries, and error frames. |
| `_bench/power_drill_auto/depth/depth_calibration.json` | Camera rectification is enabled; calibration mode is `reference` with the same scale/offset in all frames. |
| `_viz/power_drill_auto/fp_overlay_20.png` | Projected object edges should follow the video through lifting and rotation. |
| `_viz/power_drill_auto/contact_montage.png` | Check model-estimated contact against the visible hand/object interaction. |
| `_viz/power_drill_auto/4view/solid_4view.mp4` | Inspect hand placement from multiple views, including the final held pose. |
| `_phase3/power_drill_auto/gr_hsz.npy` | Final jointly optimized object poses in camera coordinates, shape `(169, 4, 4)`. |
| `_phase3/power_drill_auto/gr_hsz_world.npy` | Final object-to-world transforms, shape `(169, 4, 4)`. |
| `_phase3/power_drill_auto/gr_hsz_handverts_world.npy` | Final hand vertices, shape `(169, 778, 3)`. |

World output applies a single fixed coordinate transform to the joint result.
Coordinates are meters, +Z is up, and the estimated support plane is z=0. Trajectories preserve 24 FPS. The diagnostic videos use slower playback
(15 FPS for the four-view video). These are reconstruction results; do not assume
they are collision-free or directly executable robot commands.

## Inputs and provenance

- `video.mp4`: the project's Seedance-generated clip, 648×496 pixels.
- `rgb.png`, `depth.png`, `cam_intrinsics.txt`: reference capture supplied by
  the project, at 636×480. Preparation resizes the video to this calibrated
  resolution. The depth image is uint16 millimeters; the mesh is in meters.
- The video and reference capture are cropped on the left side (216 and 212
  pixels, the same field of view); the principal point `cx` in
  `cam_intrinsics.txt` is shifted by the calibration crop.
- `object.obj`: a 5,000-triangle version of the matching YCB power-drill model.
  It uses closed geometry, simplified to the pipeline's normal triangle budget,
  with texture removed; no separate material or texture download is needed.
- `manifest.json`: input hashes, frame count, FPS, dimensions, and automatic
  contact mode. It contains no contact interval.

The reference depth/calibration belong to the capture used to generate the
video; they are not per-frame ground truth for the generated motion.
See [ATTRIBUTION.md](ATTRIBUTION.md) for asset sources and terms.
