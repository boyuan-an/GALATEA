# Baselines

Adapters for the **VideoManip**, **EgoInfinity + FP++**, and **DO AS I DO**
comparisons used in GALATEA. Each method runs in its own upstream environments.
The adapters prepare explicit inputs, run the complete reconstruction path, and
export comparable camera-space hand/object trajectories.

These are **supplied-mesh adaptations**, not claims to reproduce the upstream
methods' original RGB-only settings. Dataset selection, ground-truth loaders,
metric aggregation and cluster-specific experiment launchers are separate from
this release. The default GALATEA reconstruction pipeline does not import or
install these optional methods.

## Organization

| Location | Responsibility |
|---|---|
| [`videomanip/`](videomanip/README.md) | MoGe / HaMeR / SAM2 / known-mesh FoundationPose runner and output adapter |
| [`egoinfinity/`](egoinfinity/README.md) | EgoInfinity runner, known-mesh FP++ preparation/tracking, rotation bake |
| [`do_as_i_do/`](do_as_i_do/README.md) | RGB frontend, coordinate-safe mesh substitution, tracking and optimization |
| `common/` | Input preparation, geometry-only mesh export, coordinate conversion and execution logs |
| `configs/` | Input-manifest and per-method runtime templates; copy runtime files to `*.local.json` |
| [`sources.json`](sources.json) | Exact upstream revisions and ordered patch/copy recipes |
| `tests/` | CPU regression tests for geometry, input isolation and orchestration |

The organization follows the independent baseline area used by
[SABER](https://github.com/sssr-lab/saber/tree/main/baselines), with the
**pinned upstream + local wrapper/patch** approach described by
[VideoManip](https://github.com/hychen-naza/VideoManip/blob/9d0f286af35d73f4252d115d968e0a9c06542b9c/reconstruction/EXTERNAL_REPOS.md).
Upstream code is fetched into ignored `third_party/`; weights and datasets are
obtained separately from their providers.

## Setup

Run these commands from the `recon/` directory. Use Python 3.10 or newer and
install `ffmpeg` on the host. The adapter environment needs only:

```bash
python -m pip install -r baselines/requirements.txt
python -m baselines.setup --method videomanip --dry-run
python -m baselines.setup --method videomanip
```

Replace `videomanip` with `egoinfinity`, `do_as_i_do`, or `all`.
The setup command fetches pinned **source**, initializes required submodules,
applies upstream patches first and our compatibility/adaptation patches second,
and records file hashes. It does not install GPU environments or fetch weights.
Follow the selected method's README for those steps. A repeated setup verifies
the recorded files. It refuses an existing checkout with a different revision
or unrecorded edits; use `--destination /new/path` for a separate installation.

```bash
python -m baselines.setup --method videomanip --check
cp baselines/configs/videomanip.example.json baselines/configs/videomanip.local.json
```

Edit the copied configuration with your repository and interpreter locations.
Repository paths are relative to that JSON file; Python commands are argument
lists, such as `["/opt/envs/hamer/bin/python"]`. They may also use a container
launcher ending in `python`; mount the GALATEA checkout, dependency source and
output directory at the **same absolute paths**, and forward `PYTHONPATH` and
the allocated GPU into the container. `env` supplies upstream runtime settings.
Keep access tokens in the process environment, outside configuration files.

## Inputs and execution

Copy [`configs/clips.example.json`](configs/clips.example.json) and replace its
paths and frame-0 point with your own data. Relative input paths are resolved
against the manifest's directory. Every clip has a unique `clip_id`, a local
`video`, a `mesh` in **meters**, and an `object_name`. These adapters currently
export the interacting **right hand** and one rigid target object.

| Method | Additional input | Supplied-mesh use |
|---|---|---|
| VideoManip | `intrinsics` and one RGB-only frame-0 `point: [x, y]` | Geometry-only mesh, fixed scale 1.0; supplied intrinsics restored after MoGe depth prediction |
| EgoInfinity | Optionally an RGB-reviewed `frame0_mask` PNG | Metric mesh is aligned into the reconstructed local axes for FP++; predicted depth and camera intrinsics remain upstream outputs |
| DO AS I DO | One RGB-only frame-0 `point: [x, y]` | RGB frontend reconstructs its own mesh first; supplied geometry is then baked into that frame before tracking |

Intrinsics accept either a 3×3 matrix or `fx fy cx cy width height`. Points must
lie on the target object in the first frame. Optional EgoInfinity masks must
match the input resolution; this is explicitly a **reviewed initialization**
variant, with subsequent frames tracked by EgoInfinity's SAM2. Without a mask,
its text detector uses the supplied noun phrase and its final noun token.

No GT pose, GT hand, GT contact, later sensor depth or GT mask fields are
accepted. The caller is responsible for obtaining the provided click/mask from
RGB inspection, not GT annotations. Source appearance is removed from the mesh.
The adapters do not infer or repair mesh units. Input preparation copies only
the named video and explicit calibration/geometry, never a whole dataset folder.

```bash
# CPU: validate inputs, extract the original frames, and inspect planned commands.
python -m baselines.run --method videomanip \
  --manifest /path/to/clips.json \
  --config baselines/configs/videomanip.local.json \
  --output outputs/baselines --dry-run

# GPU: run after configuring the upstream environments and acquiring a GPU.
python -m baselines.run --method videomanip \
  --manifest /path/to/clips.json \
  --config baselines/configs/videomanip.local.json \
  --output outputs/baselines --resume
```

The same interface works for all three methods. `--only clip_a,clip_b` selects
clips. `--prepare-only` stops after input preparation; `--dry-run` also records
the method commands. Neither executes a model. `--resume` requires identical
input bytes and configuration. For changed inputs or settings, choose a new
output directory. Stage failures produce nonzero exit status and per-clip
failure records; they are not counted as successful predictions.

On Slurm, obtain a GPU allocation first and run the GPU command inside that
allocation. The runners preserve `CUDA_VISIBLE_DEVICES`; they do not pick a
cluster partition, node, or GPU for you. DO AS I DO's upstream guide recommends
at least 32 GB VRAM. See each upstream guide for the full environment requirements.

## Outputs

Each run writes `OUTPUT/METHOD/CLIP_ID/`:

| File or directory | Contents |
|---|---|
| `inputs.json`, `inputs/` | Prepared input locations, content fingerprint, geometry-only mesh and calibration |
| `native/` | Isolated upstream working directory and intermediate products |
| `logs/`, `stages.json`, `status.json` | Stage commands, return codes, elapsed time, and run state |
| `predictions/_phase3/CLIP_ID/gr_hsz.npy` | `(T, 4, 4)` transforms from supplied mesh coordinates to OpenCV camera coordinates |
| `predictions/_phase3/CLIP_ID/gr_hsz_hand.npy` | `(T, 21, 3)` right-hand joints, meters |
| `predictions/_phase3/CLIP_ID/gr_hsz_handverts.npy` | `(T, 778, 3)` right-hand MANO vertices, meters |
| `predictions/_phase3/CLIP_ID/mano_right_faces.npy` | MANO triangle indices; empty if no hand mesh was recovered |
| `predictions/summary.json` | Valid-frame counts, coordinate convention and conversion details |

OpenCV camera axes are x-right, y-down, z-forward. Frames keep the original
video order, including fractional frame rates; there is no 15 FPS resampling.
Missing predictions remain NaN at their original indices. Geometry-only
PCA/Chamfer alignment uses the supplied mesh and reconstructed shape; it does
not read pose annotations and cannot resolve every symmetry or shape error.

## Validation and attribution

```bash
python -m pip install pytest
python -m pytest -q baselines/tests
```

CPU tests cover mesh appearance removal, DO AS I DO scale/coordinate conversion,
MANO validity masks, FP++ metric scale, input validation, and runner behavior.
Passing these checks is not an end-to-end GPU benchmark or a guarantee of the
paper's numerical results. A fresh environment still needs the upstream model
assets, compiled extensions and a real-video GPU run.

See the per-method READMEs and [THIRD_PARTY.md](../THIRD_PARTY.md) for upstream
sources and attribution. Cite each baseline's paper when using its results.
