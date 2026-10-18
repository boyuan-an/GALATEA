# Reconstructing your own video

Start with [installation](installation.md) and the
[included power-drill example](../examples/power_drill/README.md). No benchmark
dataset or ground-truth poses are needed.

The default input follows the paper: a **fixed-camera video of one interacting
right hand**, a **known rigid object mesh in meters**, **calibrated camera
intrinsics**, and **first-frame reference depth**. The pipeline estimates depth,
object masks, object motion, hand motion, and contact, then optimizes hand–object
placement and exports trajectories. An English object name guides detection.

## Input layouts

### Supplied first-frame depth and calibration (default example)

Place these files next to the video:

```text
capture/
├── video.mp4
├── rgb.png
├── depth.png
└── cam_intrinsics.txt
```

`rgb.png` is the capture reference image. Depth must be aligned to it, use uint16
millimeters, and have the same dimensions. Intrinsics contain one row:
`fx fy cx cy width height`, in pixels. The camera must remain fixed, and the
mesh must match the physical object's dimensions in meters.

```bash
bash scripts/run_pipeline_custom.sh \
  --video /path/to/capture/video.mp4 --mesh /path/to/object.obj \
  --object-name bottle --tag bottle_run --gpu 0 --with-qc
```

Complete sidecars select the provided mode automatically. Partial calibration
raises an error; supplying only intrinsics is not currently supported. Video
frames are resized to match the reference resolution. `prep_metadata.json`
records the input mode, reference hashes, and unchanged frame rate.

This command uses automatic contact detection. If you installed with
`--skip-hoidetr`, install the [automatic contact environment](hoidetr-setup.md)
first, or deliberately override it with `--contact-windows START-END`.

### RGB-only (optional extension)

When measured depth and intrinsics are unavailable, add `--rgb-only` to the
command above. MoGe-2 then predicts both from the first video frame and ignores
any calibration sidecars. These are estimated references, not the paper's
supplied calibration inputs. The metadata explicitly records their provenance.

### Already prepared frames

Use `--cam /path/to/cam0` instead of `--video` for this layout:

```text
cam0/
├── rgb/000000.png, 000001.png, ...
├── depth/000000.png
├── cam_intrinsics.txt
└── contact.yaml                  # optional saved contact interval
```

Prefer `--video` for new inputs: it validates calibration and records FPS,
frame count, input hashes, and reference provenance automatically.

## Contact intervals

By default, HOI-DETR estimates contact. You can instead provide reviewed
intervals with `--contact-windows 24-100` or `--contact-windows 24-100,108-120`.
Frame indices start at zero; both ends are included. A supplied interval skips
automatic detection. The included example uses automatic detection.

Resolution priority is CLI interval, then a manual sidecar, then a validated
automatic estimate. Automatic `contact.yaml` files do not bypass cache checks:
changing RGB frames, target tracks, model/configuration, or detection settings
invalidates the estimate; `--force` recomputes it. To retain a manual interval,
set `source: manual` in the sidecar. New automatic sidecars also keep
`auto_window`, so editing only `window` is recognized as a manual override.
Legacy automatic sidecars are recomputed unless explicitly marked manual.

Detection runs after object/hand perception. Each frame matches HOI-DETR boxes
to the SAM object track and the observed WiLoR primary right hand. Unmatched or
ambiguous detections cannot use another hand/object pair as a substitute. The
default median filter and brief-gap merging preserve contact at clip boundaries;
there is no fixed start/end trim. `HOIDETR_START_TRIM` and `HOIDETR_END_TRIM` are
optional explicit adjustments in frames, both defaulting to zero. An empty
automatic interval still stops the run. See [HOI-DETR setup](hoidetr-setup.md).

## Inspect before continuing

Add `--prep-only` to run perception and generate inspection images, then rerun
without that flag to continue. To avoid automatic contact detection during
this first phase, also use `--contact-mode manual`.

Check the object edges in `fp_overlay_20.png`, the masks, and the contact
montage. A mesh overlapping the object is insufficient: its edges should fit
throughout the motion. Incorrect segmentation, heavy occlusion, and symmetric
objects can still produce unreliable tracks. The color-adapted retry selects
between tracking candidates; it is not a quality certificate.

`--with-qc` generates contact views, a four-view reconstruction video, and
world-alignment checks for the completed run. Inspect both hand placement and
object motion. Successful execution does not guarantee accurate contact or
absence of penetration.

## Outputs

For `--tag bottle_run`, outputs are under `data/custom/` by default:

| Directory | Contents |
|---|---|
| `_prep/bottle_run/cam0/` | Prepared video, reference calibration, metadata, contact sidecar |
| `_bench/bottle_run/` | Depth, masks, object tracks, tracking/contact diagnostics |
| `_phase3/bottle_run_wilor_mano/` | Raw/cleaned hand estimates and cleanup records |
| `_phase3/bottle_run/` | Optimization and final trajectories |
| `_viz/bottle_run/` | Runtime log and requested inspection media |

The final world-frame object transforms are `gr_hsz_world.npy` with shape
`(frames, 4, 4)`. The corresponding hand vertices are
`gr_hsz_handverts_world.npy`, shape `(frames, 778, 3)`. Units are meters, +Z is
up, and the estimated support plane is z=0. `gravity.json` records alignment.

The final camera-frame results are the jointly optimized `gr_hsz.npy`,
`gr_hsz_hand.npy`, and `gr_hsz_handverts.npy`. FoundationPose supplies the
initialization and soft constraints during optimization. FP image errors are
summed over observed frames. A soft depth prior allows small corrections and
penalizes large shifts more when image evidence does not support them. See
[FP constraint settings](reconstruction-correctness.md#soft-foundationpose-constraints).
Before export, a default continuity guard checks contact on/off boundaries.
For anomalous steps it blends a rigid correction over up to 0.5 seconds on the
non-contact side, keeping every contact-frame hand and object sample exact.
Visualization and export use this protected joint trajectory. See
[contact-boundary protection](reconstruction-correctness.md#contact-boundary-continuity).
World alignment applies one constant camera-to-world transform to this result.

Existing output directories automatically refresh downstream videos and world
exports when their source marker identifies the previous output policy. Old
trajectory files are preserved on disk but are not selected by the pipeline.

Optimization saves the best evaluated parameters within each stage and checks
for non-finite losses, gradients, and outputs. `gr_hsz_optimization.json` reports
which iteration was selected and whether numerical recovery occurred. An old
cache without this completed report, without the boundary guard policy, or with
different FP settings, contact labels, frame limit, or FPS is recomputed. This does not certify
geometric accuracy; inspect the output as described below.

Temporary hand/object absence is completed automatically. Interior gaps use
neighboring observations; leading/trailing gaps hold the closest observed pose.
Object tracking re-registers on return. The companion `gr_hsz_observations.npz`
marks observed and filled frames and applies to camera/world outputs. Filled
values must not be interpreted as measured training references. At least one
usable observation of each target is required. See the policies and limits in
[correctness guards](reconstruction-correctness.md#leaving-and-returning-to-the-field-of-view).

## Cropped calibrated inputs

An asymmetric crop can move the camera principal point away from the image
center. Updating the intrinsics is necessary, but the depth model also needs
to see rays consistent with its centered-camera assumption. Set
`DEPTH_RECTIFY_CAMERA=1` to infer depth in a centered virtual camera and map it
back to the original image and camera coordinates using the supplied intrinsics.
No uncropped image is required.

For a fixed-camera clip whose generated background changes, set
`DEPTH_CALIBRATION=reference` to fit the depth scale and offset once against the
first-frame reference background, then reuse them for each frame's prediction.
This avoids unstable per-frame background fits; it does not freeze object depth
or motion. It can retain model depth drift, so inspect the resulting trajectory.
The bundled `scripts/run_example.sh` enables both settings by default. Custom
pipeline calls retain `DEPTH_RECTIFY_CAMERA=0` and
`DEPTH_CALIBRATION=per-frame` unless explicitly overridden.

`_bench/<tag>/depth/depth_calibration.json` records the selected mode, camera
rectification flag, fitted coefficients, and background support. See
[the calibration limits](reconstruction-correctness.md#cropped-camera-depth-and-reference-calibration).
When changing these settings on an existing run, use a fresh output directory
or `--force` to refresh the downstream diagnostic videos as well.

## Reruns and optional outputs

Use a **fresh tag when changing inputs or parameters**. Some stages currently
reuse results based on file existence, so changing a setting does not always
invalidate downstream caches. `--force` recomputes stages, but saved contact
intervals remain authoritative: check them when replacing the input.

Depth and SAM caches validate input content, calibration, frame count, and array
integrity before reuse. Depth completion is recorded only after all raw and
calibrated frames and masks pass validation. Interrupted or damaged exports are
recomputed on the next run. `--force` also refreshes SAM masks and their JPEG
inputs. Depth caches also include the camera-rectification and calibration modes.
Older depth/SAM caches without matching completion metadata recompute once.

WiLoR also validates the RGB timeline, intrinsics, handedness, detection/cleanup
settings, and complete output files. Shared frame-zero predictions are reused
only for the matching image and detector threshold. Joint optimization checks
all parsed optimization options (including loss weights and early-stop settings),
`CONTACT_ITERS`, and hashes of its inputs and outputs. Changing these settings
invalidates the corresponding cache automatically. Completion is published only after all
required outputs have been validated; an interrupted refresh cannot reuse an
older completion report.

Use `--force` or a fresh tag after changing the segmentation
implementation or model weights; these are not part of the input fingerprint.

`CUSTOM_DATA_ROOT` changes the output root. `--max-frames N` shortens a run.

Full options: `bash scripts/run_pipeline_custom.sh --help`.


## Automatic object recovery

Object recovery is enabled by default before joint optimization. The pipeline
checks several image-derived segmentation and FP orientation candidates over
multiple frames, and retries registration when a visible object loses alignment.
Original SAM outputs stay intact; selected masks, poses and decisions are saved
under `_bench/<tag>/object_recovery/`.

Low-confidence pose frames are completed between reliable poses (nearest reliable
pose at sequence ends) and explicitly marked as estimates. Their FP projection
and depth priors are disabled, while usable image masks still supervise the joint
optimization. Final exports remain the joint result, with no FP re-anchoring.

No additional model, manual prompt, or evaluation dataset is needed. Recovery
adds bounded segmentation/registration work on the first run; matching completed
outputs are cached. `--no-object-recovery` selects the original FP path for a
comparison. `--no-fp` explicitly reuses existing poses, including an existing
recovery result when enabled; it does not revalidate perception inputs.

Inspect `quality.json` for selected candidates, recovery events, estimated frames
and per-frame image scores. These scores describe image support, not calibrated
3D accuracy. Symmetric objects, sustained occlusion and ambiguous scenes may
still need review. See [reconstruction correctness](reconstruction-correctness.md).
