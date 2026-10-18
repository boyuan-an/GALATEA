# Reconstruction correctness guards

The custom pipeline completes missing trajectories while keeping an independent
record of which frames were observed. Completed frames are estimates, not valid
detections or measured training targets.

## Cropped camera depth and reference calibration

The optional `DEPTH_RECTIFY_CAMERA=1` path uses the full pinhole intrinsics to
rotate image rays into a centered virtual camera before MoGe inference. It maps
the predicted depth back and converts virtual-camera Z to original-camera Z.
Hand estimation, object tracking and exported trajectories keep the original
calibration. Small virtual-canvas margins use replicated RGB borders; output
depth is sampled only at rays belonging to the supplied image. This corrects
the camera-model mismatch, but cannot make a learned prediction invariant to
cropping, resampling or loss of image context.

`DEPTH_CALIBRATION=reference` fits one trimmed affine scale/offset using the
first-frame reference background outside the hand/object masks. The same pair
is applied to every independently predicted depth map. Unlike the default
`per-frame` mode, later background changes cannot refit that pair. This is useful
for the bundled fixed-camera generated clip; it does not correct subsequent
model scale drift, an incorrect reference, moving cameras or bad foreground masks.
If fewer than 100 first-frame background pixels are usable, calibration falls
back to identity and records that the reference fit was unsupported. A fitted
non-positive or non-finite scale fails explicitly. Invalid raw depths remain zero
in reference or rectified mode.

`depth_calibration.json` records the mode, rectification flag, background counts
and coefficients. Both settings participate in depth cache validation. The
bundled example enables both; general custom calls retain their original defaults.

## Leaving and returning to the field of view

The single-right-hand pipeline enables gap handling automatically:

- SAM2 can initialize from a later detection if the object or hand starts outside
  the image. A lost single object or previously prompted hand is detected again
  and re-prompted. Recovery re-detects the single target object and the hand independently.
- FoundationPose skips missing or clipped object observations. When a complete
  observation returns, it registers again instead of tracking from a stale pose.
- WiLoR hand gaps are completed before optimization, including gaps longer than
  the existing three-frame jump cleanup. Raw detections are retained.
- The optimizer excludes completed frames from their corresponding keypoint,
  silhouette, depth, and object-image tracking terms. Contact constraints require
  observed hand and object geometry. Temporal regularization still applies.
- After optimization, object gaps use linear translation and quaternion SLERP.
  Within a continuous contact interval, loss of either target triggers shared
  hand gap completion in object coordinates between jointly observed endpoints,
  so the hand moves with the object. Other hand gaps interpolate in camera
  coordinates. If there are no jointly observed anchors, hand completion falls
  back to its own camera-frame observations.

An object mask with fewer than 20 pixels, or touching an image boundary, is treated
as unavailable. After loss, a small mask remnant cannot trigger return until it
reaches 25% of the last complete mask's area. A hand prediction is unavailable if
required values are non-finite, it lies behind the camera, or any of its 21 image
keypoints lies outside the image. These conservative heuristics also reject
partially clipped observations; they cannot identify every plausible but incorrect
detector prediction. A fully occluded object may be handled in the same way as an
out-of-view object.

Interior gaps use both neighboring anchors. Leading/trailing gaps hold the nearest
anchor because there is no second observation. At least one usable observation
of each target is required; otherwise reconstruction stops with an explicit error.
Long gaps have no hard length limit, but interpolation cannot recover unseen
motions, infer a hidden contact change, or resolve a symmetric object's orientation.
It provides position-continuous estimates, not guaranteed continuous velocity.

Observation files retain the original timeline:

| File | Meaning |
|---|---|
| `_bench/<tag>/bg_cal_observations.npz` | Boolean `object_observed` and indices of re-registrations |
| `_phase3/<tag>_wilor_mano/wilor_hands.npz` | Hand parameters and boolean `hand_observed` |
| `_phase3/<tag>/gr_hsz_observations.npz` | Boolean `object_observed`, `hand_observed`, `object_filled`, `hand_filled` |

`hand_filled` can be true while `hand_observed` is also true: during contact, an
observed hand may be replaced to follow the completed object motion. For original
observation coverage use `*_observed`; for excluding estimated output frames use
`*_filled`.

The final validity file also applies to the camera- and world-coordinate
arrays, which preserve frame order. Filter by the observed flags when computing
detection coverage or selecting measured references; finite values alone no longer
mean observed. Color-candidate coverage excludes unavailable mask frames rather
than counting their absence as a tracking failure.

Standalone stage commands opt in with `--fill-missing` on `fp_runner`,
`wilor_runner` (with `--hand-side right`), and `hand_obj_optimizer`. The custom shell
entry point passes these flags automatically. `--no-hand-cleanup` disables short
jump correction but keeps missing-frame completion. The legacy multi-hand mode
does not enable hand completion.

## Missing hand silhouettes

The joint optimizer removes the object and forearm from the observed foreground
before using it as a hand-silhouette target. If fewer than 20 hand pixels remain,
that frame contributes no hand-silhouette loss. Other optimization terms still
apply. An empty segmentation is treated as an unknown observation, rather than
evidence that the reconstructed hand should disappear from the image.

## Dense object meshes

The shared object-mask renderer used by coverage scoring and FP overlays sets
the rasterizer's coarse-bin capacity to at least the complete mesh face count
(with a minimum of 10,000). This prevents dropped faces on dense native CADs.
It preserves the supplied mesh and camera; it does not simplify geometry.
The larger bin capacity can increase GPU memory use for dense meshes.

## Optimization safeguards

Each optimization stage returns its lowest-loss finite parameter state, including
the initialization and the final update. Geometry and contact stages have separate
objectives and separate checkpoints. Early stopping controls how long a stage runs;
turning it off does not disable checkpoint restoration or numerical checks.

Non-finite parameters, losses, or gradients stop optimization and restore that
stage's best evaluated state. Later stages are skipped after numerical recovery.
If no finite state exists, the run fails instead of exporting a new trajectory.
Final object poses, hand joints, and hand vertices must also be finite.

`gr_hsz_optimization.json` records completion/failure, initial/best/last losses,
the selected iteration, and numerical recovery. It is written atomically. The
pipeline only reuses optimization caches with a completed current-version report;
older caches are recomputed. A recovered run is explicitly marked in this report.

This guards numerical execution, not geometric correctness. A lower objective can
still leave drift, hand-object gaps, or penetration; `geometry_quality` remains
`not_assessed`. Inspect the RGB alignment and multi-view output. World coordinates apply a single fixed transform to the final joint result.

## Soft FoundationPose constraints

FoundationPose (FP) supplies both the initial object track and a soft image-space
reference during both optimization stages. Each observed frame averages its
vertex projection L1 errors; these frame losses are **summed**, with coefficient
0.3. Repeating frames therefore increases this term in the same way as the summed
hand-keypoint/contact terms. Other loss reductions are unchanged.

A separate soft prior discourages unsupported motion towards or away from the
camera. It compares the camera-space depth of the mesh bounding-box center with
that center under FP, so mesh-origin offsets and residual rotations are handled
correctly. Corrections within `max(0.01 m, 0.05 * mesh diagonal)` are unpenalized.
Beyond that allowance, squared excess divided by the squared allowance is summed
over observed frames, with default weight 1.

The FP silhouette's soft IoU with the observed object mask scales trust in the
reference. Pixels hidden by the hand are excluded. If the current silhouette
improves over FP, the penalty is divided by `(1 + IoU gain / 0.05)^2`: a gain of
0.05 reduces it by four. This evidence term remains differentiable. Worse image
agreement does not relax the prior. Without silhouette targets, a usable FP
observation receives the unrelaxed soft prior; missing, empty, and clipped object
observations contribute neither FP projection nor FP depth loss.

These are engineering defaults, not calibrated uncertainty or proof of correct
3D depth. Image overlap can still be misleading with occlusion, inaccurate masks,
or symmetry. Contact alone does not count as evidence for relaxing this prior.
There is no hard depth limit or pose reset after optimization. MoGe depth loss
remains separately controlled and disabled in the default custom pipeline.

The custom entry point accepts `W_FP_DEPTH` (default 1), `FP_DEPTH_FREE_M` (0.01),
`FP_DEPTH_FREE_FRACTION` (0.05), and `FP_DEPTH_EVIDENCE_GAIN` (0.05) as environment
settings. Their optimizer CLI equivalents are `--w_fp_depth`, `--fp_depth_free_m`,
`--fp_depth_free_fraction`, and `--fp_depth_evidence_gain`. Setting the depth weight
to 0 keeps frame-summed projection tracking and disables only this new depth term.

The optimization report records the versioned constraint settings, observation
counts, actual allowance, evidence source, and final depth corrections. An older
loss policy or changed FP constraint settings invalidate the optimization cache.

## Contact-boundary continuity

Joint optimization can place the same target at different depths just inside
and outside a contact window. The default output guard detects unusually large
translation or rotation steps at these transitions and absorbs the discontinuity
into the nearby non-contact trajectory. The optimized contact frames stay exactly
unchanged, including their hand-object relationship. No FP re-anchoring occurs.

The guard estimates approach/release velocity from nearby non-contact samples.
The excess over the estimated normal step must exceed both the local motion threshold (four times median speed)
and an absolute floor (20 mm or 20 degrees per frame at 24 FPS). The absolute
floor scales inversely with FPS. The correction uses smoothstep weights over
up to 0.5 seconds; object rotations and hand rigid orientations interpolate on
SO(3). A hand's vertices and joints receive the same rigid correction, preserving
its shape. Finger articulation itself is not interpolated.

Objects are checked in camera coordinates, hands in object coordinates. Large
object steps supported by matching FP input motion are retained. The guard does
not globally smooth motion, adjust contact labels, or extend a grasp to the end
of a clip. A wrong contact interval still needs correction at its source.

Each non-contact interval is processed jointly at both ends, so nearby windows
cannot overwrite one another or change contact frames. Missing-observation
boundaries and boundaries without enough outside samples are skipped. Candidates
that increase the interval's largest translation/rotation step, fail to reduce
the selected seam sufficiently, or introduce non-finite/behind-camera output
are rejected. Such cases are recorded rather than reported as repaired; very
short gaps between incompatible grasp poses may remain discontinuous.

`gr_hsz_optimization.json` includes `contact_boundary_guard` with per-boundary
before/after steps, decisions, and changed frame indices. Hand detection runs
after object correction; its `input_step_m` / `output_step_m` additionally compare
the original and final relative hand motion. Optimization losses
and FP diagnostics in the same report describe the selected optimizer state
before this export guard. Prepared-video FPS comes from `prep_metadata.json`;
legacy `--cam` inputs without metadata use 24 FPS.

The guard is enabled in both the optimizer and the single-video pipeline.
`--no-contact-boundary-guard` disables it for comparison. Its policy, contact
label contents, FPS and frame limit participate in the joint cache identity.
Older joint caches recompute once; subsequent world exports and requested videos
refresh from the new camera-frame outputs.


## Segmentation, orientation and tracking recovery

The default custom pipeline verifies object perception before joint optimization.
It generates independent SAM2 tracks from neutral object descriptions, distinct
image detector boxes and a bounded expanded box. Original segmentation is kept
as a candidate and never overwritten. CAD projections are not segmentation labels.

FoundationPose's existing rotation search supplies diverse pose hypotheses.
Candidates must improve an occlusion-aware two-sided silhouette score, agree with
RGB edges, and remain supported over subsequent frames. Actual sensor depth adds
weak evidence when metadata identifies supplied RGB-D/sensor input; legacy depth
without provenance is ignored here. Predicted MoGe depth is not independent validation.
Both absolute quality and improvement are required, so the best of several bad
candidates is not automatically accepted. This is an engineering heuristic,
not a learned probability or a proof of unique 3D orientation.

Sustained image misalignment triggers bounded re-registration. Recent reliable
orientation and image-centered translation/scale supply additional hypotheses;
large motion alone never triggers clamping. A new track must overlap the same
target at the previous reliable frame. Accepted recovery can track backward up
to one second to repair detection delay. Each pass allows at most eight local
registration retries; one further SAM pass can use at most three requested
frames. Candidate tracks and short confirmation windows are bounded.

Pose confidence and mask validity are independent. A visible target can still
provide silhouette evidence when its pose is wrong. Unconfirmed pose samples
are interpolated on translation and SO(3), or held from the nearest anchor at the
ends, and receive zero FP prior weight. Long invisible intervals cannot recover
true unseen motion. If no image-supported pose exists, the stage stops with a
failed report instead of fabricating a complete reconstruction.

`object_recovery/quality.json` records candidate decisions, image scores, retries
and estimated indices. `poses_observations.npz` stores mask validity, pose anchors,
estimated flags and FP confidence. These confidence weights scale the frame-summed
FP projection and soft depth losses; image supervision remains active on valid
masks. The final output is still the joint optimization result, followed by the
existing contact-boundary guard. No output FP re-anchoring is introduced.

The recovery cache fingerprints inputs, parameters and recovery source code,
validates output hashes, and publishes its completion stamp last. The joint cache
also fingerprints selected poses, observation flags/confidence and selected masks.
Changing a candidate or confidence timeline therefore recomputes joint outputs.
Recovery artifacts are reconstruction outputs; no benchmark runner is required.
