#!/usr/bin/env bash
set -eo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CUSTOM_DATA_ROOT="${CUSTOM_DATA_ROOT:-$REPO/data/custom}"

usage() {
  cat <<'EOF'
Reconstruct a right-hand MANO mesh and the 6-DoF trajectory of a rigid object
from a fixed-camera RGB video and a metric object mesh.

Usage:
  bash scripts/run_pipeline_custom.sh --video FILE --mesh FILE --object-name NAME [options]
  bash scripts/run_pipeline_custom.sh --cam DIR --mesh FILE --object-name NAME [options]

Inputs:
  --video FILE        RGB video, extracted into _prep/<tag>/cam0. rgb.png, depth.png and
                      cam_intrinsics.txt next to the video select provided calibration;
                      otherwise the first-frame depth and intrinsics are predicted.
  --cam DIR           prepared capture with rgb/, depth/000000.png and cam_intrinsics.txt
  --rgb-only          predict first-frame depth and intrinsics with MoGe-2 (requires --video)
  --mesh FILE         rigid object mesh in meters (required)
  --object-name STR   object noun for detection, e.g. "wine glass" (required)
  --tag NAME          output tag under $CUSTOM_DATA_ROOT/_{bench,phase3,viz}/<tag>
                      (default: derived from the capture directory)
  --max-frames N      frames to process (default: all frames)
  --gpu ID            CUDA device (default: the GPU with the most free memory)

Contact:
  --contact-windows S-E[,S-E...]  manual right-hand contact windows, inclusive frame
                      indices from zero. Without them, HOI-DETR estimates the window and
                      the estimate is stored in contact.yaml.
  --contact-start N --contact-end N   shorthand for a single window
  --contact-mode NAME auto (default, HOI-DETR) | manual (wait for a filled window)
  --no-contact-boundary-guard  disable local protection at contact boundaries

Object tracking:
  --objroot DIR       object model root (default: $CUSTOM_DATA_ROOT/object_models/object)
  --obj-id N          placeholder object id (default: 0)
  --max-faces N       decimate meshes denser than this (default: 20000)
  --decim-target N    face count after decimation (default: 5000)
  --fp-dir DIR        FoundationPose source (default: <repo>/imports/FoundationPose)
  --fp-backend measured  plain FoundationPose (the only supported backend)
  --no-fp-color-fallback  disable the color retry used when the gray FoundationPose track
                      covers less than --fp-color-trigger of the SAM 2 masks
  --fp-color-trigger X    mean-coverage threshold for the color retry (default: 0.75)
  --fp-color-min-gain X   coverage gain required to keep the color track (default: 0.05)
  --no-object-recovery    disable automatic segmentation and pose recovery
  --no-fp             reuse an existing bg_cal.npy instead of running FoundationPose

Hands and optimization:
  --hand-side NAME    WiLoR hand side: right (default) | all (with --no-hand-cleanup)
  --no-hand-cleanup   disable short jump cleanup (missing frames are still filled)
  --iters N           geometry iterations of the joint optimization (default: 150)
  --no-joint-early-stop  always run the full iteration limits

Outputs:
  --with-qc           write FoundationPose/contact/gravity images and contact/four-view videos
  --sam-mask-viz      also write whole-person SAM 2 panels (runs SAM 2 a second time)
  --with-all-outputs  enable all optional outputs
  --no-render         disable optional renders
  --prep-only         stop after perception and contact estimation
  --show-output-plan  print the optional-output settings and exit
  --force             recompute every stage

Environment:
  CUSTOM_DATA_ROOT    root of all generated data (default: <repo>/data/custom)
  HOIDETR_ENV         HOI-DETR conda environment (default: codetr-cu121)
  HOIDETR_ROOT        HOI-DETR source with checkpoints/epoch_5.pth
                      (default: <repo>/3rd/HOI-DETR-cu121)
EOF
}

CAM=""
VIDEO=""
INPUT_MODE=auto
MESH=""
ONAME=""
CONTACT_BOUNDARY_GUARD=true
OBJECT_RECOVERY=true
CSTART=""; CEND=""
WINDOWS=""

TAG=""
MAXF=""
GPU=""
OID=0
OBJROOT="$CUSTOM_DATA_ROOT/object_models/object"
FP_DIR="$REPO/imports/FoundationPose"; ITERS=150
FP_BACKEND="${FP_BACKEND:-measured}"
MAXFACES=20000; DECIM_TARGET=5000
FORCE=false; NO_FP=false; PREP_ONLY=false
PRODUCE_QC=false
SHOW_OUTPUT_PLAN=false
HAND_SIDE=right; CLEAN_HANDS=true
SAM_MASK_VIZ=false
JOINT_EARLY_STOP=true
FP_COLOR_FALLBACK=true
FP_COLOR_TRIGGER="${FP_COLOR_TRIGGER:-0.75}"
FP_COLOR_MIN_GAIN="${FP_COLOR_MIN_GAIN:-0.05}"
CONTACT_MODE="${CONTACT_MODE:-auto}"
HOIDETR_ENV="${HOIDETR_ENV:-codetr-cu121}"
HOIDETR_ROOT="${HOIDETR_ROOT:-$REPO/3rd/HOI-DETR-cu121}"

while [ $# -gt 0 ]; do case "$1" in
  --cam) CAM="$2"; shift 2;;
  --video) VIDEO="$2"; shift 2;;
  --rgb-only) INPUT_MODE=rgb-only; shift 1;;
  --mesh) MESH="$2"; shift 2;;
  --object-name) ONAME="$2"; shift 2;;
  --tag) TAG="$2"; shift 2;;
  --objroot) OBJROOT="$2"; shift 2;;
  --contact-start) CSTART="$2"; shift 2;;
  --contact-end) CEND="$2"; shift 2;;
  --contact-windows) WINDOWS="$2"; shift 2;;
  --contact-mode) CONTACT_MODE="$2"; shift 2;;
  --no-object-recovery) OBJECT_RECOVERY=false; shift;;
  --no-contact-boundary-guard) CONTACT_BOUNDARY_GUARD=false; shift;;
  --max-frames) MAXF="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  --obj-id) OID="$2"; shift 2;;
  --fp-dir) FP_DIR="$2"; shift 2;;
  --fp-backend) FP_BACKEND="$2"; shift 2;;
  --no-fp-color-fallback) FP_COLOR_FALLBACK=false; shift 1;;
  --fp-color-trigger) FP_COLOR_TRIGGER="$2"; shift 2;;
  --fp-color-min-gain) FP_COLOR_MIN_GAIN="$2"; shift 2;;
  --iters) ITERS="$2"; shift 2;;
  --sam-mask-viz) SAM_MASK_VIZ=true; shift 1;;
  --no-joint-early-stop) JOINT_EARLY_STOP=false; shift 1;;
  --hand-side) HAND_SIDE="$2"; shift 2;;
  --no-hand-cleanup) CLEAN_HANDS=false; shift 1;;
  --max-faces) MAXFACES="$2"; shift 2;;
  --decim-target) DECIM_TARGET="$2"; shift 2;;
  --no-fp) NO_FP=true; shift 1;;
  --with-qc) PRODUCE_QC=true; shift 1;;
  --with-all-outputs) PRODUCE_QC=true; SAM_MASK_VIZ=true; shift 1;;
  --no-render) PRODUCE_QC=false; SAM_MASK_VIZ=false; shift 1;;
  --show-output-plan) SHOW_OUTPUT_PLAN=true; shift 1;;
  --prep-only) PREP_ONLY=true; shift 1;;
  --force) FORCE=true; shift 1;;
  -h|--help) usage; exit 0;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done

[ "$PREP_ONLY" = true ] && PRODUCE_QC=true
if [ "$SHOW_OUTPUT_PLAN" = true ]; then
  echo "qc=$PRODUCE_QC"
  echo "sam_mask_qc=$SAM_MASK_VIZ"
  exit 0
fi

if [ "$FP_BACKEND" != measured ]; then
  echo "ERROR: only plain FoundationPose (--fp-backend measured) is supported"
  exit 1
fi
if [ "$INPUT_MODE" = rgb-only ] && [ -z "$VIDEO" ]; then
  echo "ERROR: --rgb-only requires --video; --cam already supplies its own depth and intrinsics"
  exit 1
fi

case "$HAND_SIDE" in right|all) ;; *) echo "ERROR: --hand-side must be right or all"; exit 1;; esac
case "$CONTACT_MODE" in auto|manual) ;; *) echo "ERROR: --contact-mode must be auto or manual"; exit 1;; esac
if [ "$CLEAN_HANDS" = true ] && [ "$HAND_SIDE" != right ]; then
  echo "ERROR: automatic hand cleanup requires --hand-side right"
  echo "       use --hand-side all --no-hand-cleanup for the legacy multi-hand output"
  exit 1
fi

if [ -n "$VIDEO" ]; then
  VIDEO="$(cd "$(dirname "$VIDEO")" && pwd)/$(basename "$VIDEO")"
  STEM="$(basename "$VIDEO" .mp4)"
  if [ -z "$TAG" ]; then
    base="$(basename "$(dirname "$VIDEO")")"; base="${base%_}"
    [ "$STEM" = "seq1" ] && TAG="$base" || TAG="${base}_${STEM}"
  fi
  [ -z "$CAM" ] && CAM="$CUSTOM_DATA_ROOT/_prep/$TAG/cam0"
fi

WINDOW_SOURCE=""
if [ -n "$WINDOWS" ]; then
  WINDOW_SOURCE="cli-manual"
elif [ -n "$CSTART" ] && [ -n "$CEND" ]; then
  WINDOWS="$CSTART-$CEND"
  WINDOW_SOURCE="cli-manual"
fi
SIDECAR=""
[ -n "$CAM" ] && SIDECAR="$CAM/contact.yaml"
miss=""
{ [ -z "$CAM" ] && [ -z "$VIDEO" ]; } && miss="$miss --cam(or --video)"
[ -z "$MESH" ]    && miss="$miss --mesh"
[ -z "$ONAME" ]   && miss="$miss --object-name"
if [ -n "$miss" ]; then
  echo "ERROR: missing required args:$miss"
  echo "e.g.: bash scripts/run_pipeline_custom.sh \\"
  echo "        --cam data/videos/<seq>/cam0 --mesh /path/to/obj.obj \\"
  echo "        --object-name 'wine glass'"
  echo "      (manual override: --contact-windows 80-194,219-285)"
  echo "(run with --help for all options)"; exit 1
fi
if [ -z "$VIDEO" ]; then
  CAM="$(cd "$(dirname "$CAM")" && pwd)/$(basename "$CAM")"
  [ -d "$CAM/rgb" ] || { echo "ERROR: no rgb/ under $CAM"; exit 1; }
fi
[ -f "$MESH" ]    || { echo "ERROR: mesh not found: $MESH"; exit 1; }
[ -z "$TAG" ] && TAG="$(basename "$(dirname "$CAM")")"
[ -z "$MAXF" ] && [ -z "$VIDEO" ] && MAXF="$(ls "$CAM"/rgb/*.png 2>/dev/null | wc -l)"
[ -z "$GPU" ] && GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
                        2>/dev/null | sort -t',' -k2 -nr | head -1 | cut -d',' -f1 | tr -d ' ')"
[ -z "$GPU" ] && GPU=0

GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
step() { echo -e "\n${CYAN}== $* ==${NC}"; }
ok()   { echo -e "  ${GREEN}[OK]${NC} $*"; }
skip() { echo -e "  ${YELLOW}[skip]${NC} $* (exists; --force to redo)"; }

FILT='FutureWarning|warnings.warn|UserWarning|it/s\]|propagate|Warp |Kernel cache|Supported fully|CUDA Toolkit|Devices:|free\(\): invalid'
runpy() {
  local n0; n0=$(wc -l < "$LIVE" 2>/dev/null || echo 0)
  set +e
  "$@" 2>&1 | tee -a "$LIVE" >/dev/null
  local rc=${PIPESTATUS[0]}
  set -e
  tail -n "+$((n0 + 1))" "$LIVE" | grep -avE "$FILT" | tail -4 || true
  if [ "$rc" -ne 0 ]; then echo -e "  ${RED}[FAIL rc=$rc]${NC} tail of $LIVE:"; tail -20 "$LIVE"; return "$rc"; fi
}

source "$REPO/scripts/setup/runtime_env.sh"
galatea_load_conda "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if [ -f "${CACHE_ROOT:-$REPO/.cache}/env.sh" ]; then
  source "${CACHE_ROOT:-$REPO/.cache}/env.sh"
fi
galatea_load_hf_token
export CUDA_VISIBLE_DEVICES="$GPU"; export EGL_DEVICE_ID="$GPU"
export PYTHONUNBUFFERED=1; export PRINT_EVERY="${PRINT_EVERY:-20}"
use() { conda deactivate 2>/dev/null || true; galatea_activate "$1"; }

MESH="$(cd "$(dirname "$MESH")" && pwd)/$(basename "$MESH")"
B="$CUSTOM_DATA_ROOT/_bench/$TAG"; DEPTH="$B/depth"
P="$CUSTOM_DATA_ROOT/_phase3/$TAG"; W="$CUSTOM_DATA_ROOT/_phase3/${TAG}_wilor_mano"
VIZ="$CUSTOM_DATA_ROOT/_viz/$TAG"
SEG="$P/seg_labels.json"; MANUAL="$P/manual_fingers.json"
mkdir -p "$DEPTH" "$P" "$W" "$VIZ" "$OBJROOT/$TAG"
LIVE="$VIZ/live.log"; : > "$LIVE"

have() { [ "$FORCE" = false ] && [ -e "$1" ]; }

if [ -n "$VIDEO" ]; then
  step "Step -1  extract video -> cam0 ($VIDEO)"; use sam3d-objects
  PREP_VIDEO_ARGS=(--video "$VIDEO" --out "$CAM" --input-mode "$INPUT_MODE")
  [ -n "$MAXF" ] && PREP_VIDEO_ARGS+=(--max_frames "$MAXF")
  [ "$FORCE" = true ] && PREP_VIDEO_ARGS+=(--force)
  runpy python -m galatea_recon.pipeline.prep_video "${PREP_VIDEO_ARGS[@]}"
  ok "cam0/rgb + depth/000000.png + cam_intrinsics.txt + prep_metadata.json"
  [ -z "$MAXF" ] && MAXF="$(ls "$CAM"/rgb/*.png 2>/dev/null | wc -l)"
fi

use sam3d-objects
if [ -z "$WINDOWS" ] && [ -n "$SIDECAR" ]; then
  WINDOWS="$(python -m galatea_recon.pipeline.contact_cache "$SIDECAR")"
  [ -z "$WINDOWS" ] || WINDOW_SOURCE="sidecar-manual"
fi
if [ -z "$WINDOWS" ] && [ "$CONTACT_MODE" = manual ] \
    && [ "$PREP_ONLY" = false ]; then
  echo "ERROR: manual mode requires --contact-windows or a manual sidecar window"
  exit 1
fi
AUTOCONTACT="$B/hoidetr_contact.json"

step "Step 0  scaffold (mesh -> objroot, obj_name sidecar + obj_pose_rt)"
use sam3d-objects
echo "$TAG" > "$CAM/obj_name.txt"
echo "$ONAME" > "$CAM/obj_noun.txt"
mkdir -p "$CAM/obj_pose_rt"
printf '%s 1 0 0 0 0 1 0 0 0 0 1 0.4 0 0 0 1\n' "$OID" > "$CAM/obj_pose_rt/000000.txt"
WMESH="$OBJROOT/$TAG/$TAG.obj"
NFACE="$(python -c 'import sys,trimesh; print(len(trimesh.load(sys.argv[1],force="mesh").faces))' "$MESH")"
if [ "$NFACE" -gt "$MAXFACES" ]; then
  python - "$MESH" "$WMESH" "$DECIM_TARGET" "$NFACE" <<'PY'
import sys
import open3d as o3d
m = o3d.io.read_triangle_mesh(sys.argv[1])
m = m.simplify_quadric_decimation(int(sys.argv[3]))
m.remove_unreferenced_vertices()
m.compute_vertex_normals()
if not o3d.io.write_triangle_mesh(sys.argv[2], m):
    raise SystemExit("Could not write decimated mesh: " + sys.argv[2])
print(f"  decimated dense mesh: {sys.argv[4]} -> {len(m.triangles)} faces")
PY
  MESH="$WMESH"
else
  cp -f "$MESH" "$WMESH"
  echo "  mesh: $NFACE faces (<= $MAXFACES, no decimation)"
fi
write_contact_labels() {
python - "$P" "$WINDOWS" "$MAXF" <<'PY'
import json, sys
from galatea_recon.pipeline.contact_cache import validate_windows
P, win = sys.argv[1], sys.argv[2]
if win:
    validate_windows(win, int(sys.argv[3]))
wins = [tuple(int(x) for x in w.split("-")) for w in win.split(",") if w]
fingers = ["thumb", "index", "middle", "ring", "pinky"]
manual = {"0": [{"hand": "right", "finger": f, "start": s, "end": e}
                for f in fingers for (s, e) in wins]}
seg = {"0": [{"hand": "right", "finger": f, "seg": k, "start": s, "end": e}
             for f in fingers for (s, e) in wins for k in (1, 2, 3)]}
json.dump(manual, open(f"{P}/manual_fingers.json", "w"), indent=2)
json.dump(seg, open(f"{P}/seg_labels.json", "w"))
print(f"  contact windows {wins} -> {len(manual['0'])} manual / {len(seg['0'])} seg intervals")
PY
}
python -c 'import sys; from galatea_recon.pipeline.h2o_loader import H2OSequence as S; s=S(*sys.argv[1:]);
print("  loader: obj_id",s.obj_id,"->",s.obj_name,"| frames",s.num_frames)' "$CAM" "$OBJROOT" \
  || { echo -e "  ${RED}obj name unresolved or mesh missing${NC}"; exit 1; }
echo -e "${GREEN}$TAG${NC} | object='$ONAME' | frames=$MAXF | gpu=$GPU | contact=[$WINDOWS]"

step "Step 1/10  MoGe depth"; use sam3d-objects
DEPTH_ARGS=(--cam_dir "$CAM" --object_models_root "$OBJROOT" --out_dir "$DEPTH" --max_frames "$MAXF")
if [ "$FORCE" = false ] && python -m galatea_recon.pipeline.export_depth "${DEPTH_ARGS[@]}" --check-cache; then
  skip "moge_cal (validated complete cache)"
else
  [ "$FORCE" = true ] && DEPTH_ARGS+=(--force)
  runpy python -m galatea_recon.pipeline.export_depth "${DEPTH_ARGS[@]}"
  ok "moge_cal ($MAXF frames; completion verified)"
fi

if [ "$SAM_MASK_VIZ" = true ]; then
  step "Optional SAM2 whole-person QC panels (second SAM2 pass)"
  if have "$DEPTH/sam_000000.png"; then skip sam_masks
  else runpy python -m galatea_recon.pipeline.sam_masks --cam_dir "$CAM" \
         --object_models_root "$OBJROOT" --out_dir "$DEPTH" --n_frames "$MAXF" \
         --whole_person --object_name "$ONAME"; ok "sam_*.png (optional QC only)"; fi
fi

step "Step 3/10  object pose ($FP_BACKEND backend, register init)"
if [ "$NO_FP" = true ]; then echo "  (--no-fp) reuse $B/bg_cal.npy"
elif [ "$OBJECT_RECOVERY" = true ] && [ "${SAM_REVIEW_ONLY:-false}" = false ]; then
  echo "  segmentation hypotheses and FP verification are handled by object recovery"
elif have "$B/bg_cal.npy" && [ -f "$B/bg_cal_observations.npz" ]; then skip bg_cal.npy
else
  use factory
  FP_COMMON_ARGS=(--cam_dir "$CAM" --mesh "$MESH" --fp_dir "$FP_DIR"
      --init_mode register --fill-missing --sam2_masks "$DEPTH/sam2_masks.npz"
      --init_depth_mode "${INIT_DEPTH_MODE:-npy}"
      --est_refine_iter "${EST_REFINE_ITER:-5}"
      --track_refine_iter "${TRACK_REFINE_ITER:-2}"
      --depth_mode npy --depth_dir "$DEPTH/moge_cal" --max_frames "$MAXF")
  if [ "$FP_COLOR_FALLBACK" = false ]; then
    runpy python -m galatea_recon.pipeline.fp_runner "${FP_COMMON_ARGS[@]}" --out "$B/bg_cal.npy"
    use sam3d-objects; ok "bg_cal.npy (gray; color fallback disabled)"
  else
    GRAY_POSES="$B/bg_cal_gray.npy"; GRAY_SCORE="$B/fp_coverage_gray.json"
    COLOR_POSES="$B/bg_cal_color.npy"; COLOR_SCORE="$B/fp_coverage_color.json"
    COLOR_DECISION="$B/fp_color_fallback.json"

    rm -f "$COLOR_POSES" "$COLOR_SCORE"

    runpy python -m galatea_recon.pipeline.fp_runner "${FP_COMMON_ARGS[@]}" \
      --bake_color none --out "$GRAY_POSES"
    use sam3d-objects
    runpy python -m galatea_recon.pipeline.fp_color_fallback score \
      --cam-dir "$CAM" --mesh "$MESH" --poses "$GRAY_POSES" \
      --masks "$DEPTH/sam2_masks.npz" --out "$GRAY_SCORE"

    COLOR_SELECT_ARGS=()
    if python -m galatea_recon.pipeline.fp_color_fallback needs-retry \
         --score "$GRAY_SCORE" --trigger-coverage "$FP_COLOR_TRIGGER"; then
      use factory
      if runpy python -m galatea_recon.pipeline.fp_runner "${FP_COMMON_ARGS[@]}" \
           --bake_color auto --out "$COLOR_POSES"; then
        use sam3d-objects
        if runpy python -m galatea_recon.pipeline.fp_color_fallback score \
             --cam-dir "$CAM" --mesh "$MESH" --poses "$COLOR_POSES" \
             --masks "$DEPTH/sam2_masks.npz" --out "$COLOR_SCORE"; then
          COLOR_SELECT_ARGS=(--color-score "$COLOR_SCORE" --color-poses "$COLOR_POSES")
        else
          echo "  [WARN] color retry scoring failed; retaining the gray candidate"
        fi
      else
        echo "  [WARN] color retry failed; retaining the gray candidate"
        use sam3d-objects
      fi
    fi

    runpy python -m galatea_recon.pipeline.fp_color_fallback select \
      --gray-score "$GRAY_SCORE" --gray-poses "$GRAY_POSES" \
      "${COLOR_SELECT_ARGS[@]}" \
      --trigger-coverage "$FP_COLOR_TRIGGER" --min-gain "$FP_COLOR_MIN_GAIN" \
      --out-poses "$B/bg_cal.npy" --out-decision "$COLOR_DECISION"
    ok "bg_cal.npy (coverage-gated gray/color; audit=fp_color_fallback.json)"
  fi
fi
ACTIVE_FP_POSES="$B/bg_cal.npy"
ACTIVE_SAM2_MASKS="${ACTIVE_SAM2_MASKS:-$DEPTH/sam2_masks.npz}"
if [ "$OBJECT_RECOVERY" = true ] && [ "$NO_FP" = false ]; then
  use factory
  RECOVERY_FP_PYTHON="$(command -v python)"
  use sam3d-objects
  RECOVERY_ARGS=()
  [ "$FORCE" = true ] && RECOVERY_ARGS+=(--force)
  [ "$FP_COLOR_FALLBACK" = false ] && RECOVERY_ARGS+=(--no-color-fallback)
  runpy python -m galatea_recon.pipeline.object_recovery \
    --cam-dir "$CAM" --mesh "$MESH" --masks "$ACTIVE_SAM2_MASKS" \
    --depth-dir "${TRACKING_DEPTH_DIR:-$DEPTH/moge_cal}" --baseline-poses "$B/bg_cal.npy" \
    --fp-dir "$FP_DIR" --fp-python "$RECOVERY_FP_PYTHON" --object-name "$ONAME" \
    --max-frames "$MAXF" --init-depth-mode "${INIT_DEPTH_MODE:-npy}" \
    --est-refine-iter "${EST_REFINE_ITER:-5}" --track-refine-iter "${TRACK_REFINE_ITER:-2}" \
    --color-trigger "$FP_COLOR_TRIGGER" --color-min-gain "$FP_COLOR_MIN_GAIN" \
    --output-dir "$B/object_recovery" "${RECOVERY_ARGS[@]}"
  ACTIVE_FP_POSES="$B/object_recovery/poses.npy"
  ACTIVE_SAM2_MASKS="$B/object_recovery/masks.npz"
elif [ "$OBJECT_RECOVERY" = true ] && [ -f "$B/object_recovery/recovery.json" ]; then
  ACTIVE_FP_POSES="$B/object_recovery/poses.npy"
  ACTIVE_SAM2_MASKS="$B/object_recovery/masks.npz"
fi

if [ -f "$ACTIVE_FP_POSES" ] && [ "$PRODUCE_QC" = true ]; then
  use sam3d-objects
  if have "$VIZ/fp_overlay_20.png" && [ ! "$ACTIVE_FP_POSES" -nt "$VIZ/fp_overlay_20.png" ]; then skip fp_overlay_20.png
  else runpy python -m galatea_recon.pipeline.render_fp_overlay --cam_dir "$CAM" \
         --object_models_root "$OBJROOT" --mesh "$MESH" --poses "$ACTIVE_FP_POSES" \
         --out "$VIZ/fp_overlay_20.png" --max_frames "$MAXF"; ok "fp_overlay_20.png (FP self-check)"; fi
fi

step "Step 4/10  WiLoR hands (side=$HAND_SIDE, auto-clean=$CLEAN_HANDS)"; use sam3d-objects
HAND_ARGS=(--cam_dir "$CAM" --out "$W" --max_frames "$MAXF" --hand-side "$HAND_SIDE"
  --frame0-predictions "$DEPTH/wilor_frame0_predictions.npz")
if [ "$HAND_SIDE" = right ]; then HAND_ARGS+=(--fill-missing); fi
HAND_AUDIT_MSG=""
if [ "$CLEAN_HANDS" = true ]; then
  HAND_ARGS+=(--clean-jumps \
    --max-repair-span "${HAND_MAX_REPAIR_SPAN:-3}" \
    --jump-position-mm "${HAND_JUMP_POSITION_MM:-80}" \
    --jump-rotation-deg "${HAND_JUMP_ROTATION_DEG:-75}")
  HAND_AUDIT_MSG=", audit=wilor_cleanup.json"
fi
if [ "$FORCE" = false ] && python -m galatea_recon.pipeline.wilor_runner "${HAND_ARGS[@]}" --check-cache; then
  skip "wilor_hands.npz (validated complete cache)"
else
  runpy python -m galatea_recon.pipeline.wilor_runner "${HAND_ARGS[@]}"
  ok "wilor_hands.npz (raw=wilor_hands_raw.npz$HAND_AUDIT_MSG)"
fi

if [ -z "$WINDOWS" ] && [ "$CONTACT_MODE" = auto ]; then
  step "Automatic contact window (HOI-DETR, target object + right hand)"
  HOIDETR_ARGS=(--cam-dir "$CAM" --max-frames "$MAXF" --out "$AUTOCONTACT"
    --hoidetr-root "$HOIDETR_ROOT" --target-masks "$ACTIVE_SAM2_MASKS"
    --target-hands "$W/wilor_hands.npz"
    --interaction-threshold "${HOIDETR_INTERACTION_THR:-0.5}"
    --start-trim "${HOIDETR_START_TRIM:-0}" --end-trim "${HOIDETR_END_TRIM:-0}")
  [ "$PRODUCE_QC" = true ] && HOIDETR_ARGS+=(--viz-dir "$VIZ/hoidetr_contact")
  if [ "$FORCE" = false ] && python -m galatea_recon.pipeline.hoidetr_contact "${HOIDETR_ARGS[@]}" --check-cache; then
    skip "hoidetr_contact.json (validated complete cache)"
  else
    use "$HOIDETR_ENV"
    [ "$FORCE" = true ] && HOIDETR_ARGS+=(--force)
    runpy python -m galatea_recon.pipeline.hoidetr_contact "${HOIDETR_ARGS[@]}"
    use sam3d-objects
  fi
  WINDOWS="$(python - "$AUTOCONTACT" <<'PYCONTACT'
import json
import sys
with open(sys.argv[1]) as stream:
    print(json.load(stream)["window"])
PYCONTACT
)"
  [ -n "$WINDOWS" ] || { echo "ERROR: HOI-DETR produced an empty contact window"; exit 1; }
  WINDOW_SOURCE="hoidetr-auto"
  ok "contact window [$WINDOWS] (audit: $AUTOCONTACT)"
fi

write_contact_labels

CMONT=""
if [ "$PRODUCE_QC" = true ]; then
  step "Contact montage + sidecar (source=$WINDOW_SOURCE)"
  use sam3d-objects
  CMONT="$VIZ/contact_montage.png"
  if [ -n "$WINDOWS" ] || ! have "$CMONT"; then
    runpy python -m galatea_recon.pipeline.render_contact_montage --cam_dir "$CAM" \
      --out "$CMONT" --stride "${CSTRIDE:-3}" --fps "${FPS:-24}" --max_frames "$MAXF" \
      --object-name "$ONAME" --tag "$TAG" ${WINDOWS:+--window "$WINDOWS"}
    ok "contact_montage.png (every ${CSTRIDE:-3}f${WINDOWS:+, VERIFY [$WINDOWS]})"
  else skip contact_montage.png; fi
else
  echo -e "\n  (lean default) skipping contact montage; use --with-qc to produce it"
fi
if [ -n "$SIDECAR" ] && { [ ! -f "$SIDECAR" ] \
    || [ "$WINDOW_SOURCE" = cli-manual ] || [ "$WINDOW_SOURCE" = hoidetr-auto ]; }; then
  python - "$SIDECAR" "$TAG" "$ONAME" "$MAXF" "${FPS:-24}" "$CMONT" \
    "${WINDOWS:-}" "${WINDOW_SOURCE:-manual-pending}" "$AUTOCONTACT" <<'PY'
import json
import os
import sys

sidecar, tag, oname, frames, fps, montage, window, source, audit = sys.argv[1:10]
raw_window = ""
if source == "hoidetr-auto" and os.path.isfile(audit):
    with open(audit) as input_file:
        raw_window = json.load(input_file).get("raw_window", "")
lines = [
    "# Right-hand contact window. The pipeline reads `window` on every re-run.",
    "# Edit `window` or set source: manual for an override; ranges are inclusive.",
    "object: " + json.dumps(oname),
    "frames: " + frames,
    "fps: " + fps,
    "montage: " + json.dumps(montage),
    "source: " + json.dumps(source),
]
if raw_window:
    lines.append("raw_window: " + json.dumps(raw_window))
if source == "hoidetr-auto":
    lines.append("audit: " + json.dumps(audit))
    lines.append("auto_window: " + json.dumps(window))
lines.append("window: " + json.dumps(window))
temporary = sidecar + ".tmp"
with open(temporary, "w") as output_file:
    output_file.write("\n".join(lines) + "\n")
os.replace(temporary, sidecar)
print("  wrote sidecar: " + sidecar + ("" if window else "  (fill `window:` then re-run)"))
PY
  [ -z "$WINDOWS" ] && echo -e "  ${YELLOW}ACTION${NC}: open $CMONT, set window in $SIDECAR, then re-run"
fi

if [ "$PREP_ONLY" = true ]; then
  echo -e "\n${GREEN}=== $TAG ('$ONAME') PREP DONE (Steps 0-4) ===${NC}"
  echo "  depth      : $DEPTH/moge_cal  (+ sam2_masks.npz)"
  echo "  fp pose    : $ACTIVE_FP_POSES    (self-check: $VIZ/fp_overlay_20.png)"
  echo "  hands      : $W/wilor_hands.npz"
  if [ -n "$WINDOWS" ]; then
    echo "  contact    : [$WINDOWS] ($WINDOW_SOURCE; verify in $CMONT)"
    echo "  (re-run without --prep-only; the persisted window is read from $SIDECAR)"
  else
    echo "  contact    : $CMONT  ->  fill window in $SIDECAR"
    echo "  (manual mode: re-run after filling the sidecar)"
  fi
  exit 0
fi

step "Step 6/10  contact labels ($WINDOW_SOURCE, no GT)"
[ -f "$SEG" ] && [ -f "$MANUAL" ] && ok "seg_labels.json + manual_fingers.json (windows [$WINDOWS])"

step "Step 7/10  joint optimization (★ final camera-frame result)"
EARLY_STOP_ARGS=()
if [ "$JOINT_EARLY_STOP" = true ]; then
  EARLY_STOP_ARGS=(
    --early-stop
    --early-stop-check-every "${JOINT_EARLY_STOP_CHECK_EVERY:-10}"
    --early-stop-patience "${JOINT_EARLY_STOP_PATIENCE:-3}"
    --early-stop-min-geom "${JOINT_EARLY_STOP_MIN_GEOM:-80}"
    --early-stop-min-contact "${JOINT_EARLY_STOP_MIN_CONTACT:-80}"
    --early-stop-rel-total "${JOINT_EARLY_STOP_REL_TOTAL:-0.002}"
    --early-stop-rel-contact "${JOINT_EARLY_STOP_REL_CONTACT:-0.005}"
  )
fi
BOUNDARY_ARGS=()
[ "$CONTACT_BOUNDARY_GUARD" = false ] && BOUNDARY_ARGS=(--no-contact-boundary-guard)
GAP_ARGS=()
if [ "$HAND_SIDE" = right ]; then GAP_ARGS=(--fill-missing); fi
JOINT_ARGS=(--cam_dir "$CAM" --mesh "$MESH" \
       --fp_poses "$ACTIVE_FP_POSES" --wilor_hands "$W/wilor_hands.npz" \
       --depth_dir "$DEPTH/moge_cal" --sam2_masks "$ACTIVE_SAM2_MASKS" \
       --full_objective --silhouette_z --sil_scale "${SIL_SCALE:-0.35}" --w_silhouette "${W_SIL:-1000000}" --w_depth_override "${W_DEPTH_OVERRIDE:-0}" \
       --w_fp_depth "${W_FP_DEPTH:-1}" --fp_depth_free_m "${FP_DEPTH_FREE_M:-0.01}" \
       --fp_depth_free_fraction "${FP_DEPTH_FREE_FRACTION:-0.05}" --fp_depth_evidence_gain "${FP_DEPTH_EVIDENCE_GAIN:-0.05}" \
       --contact_delta 0.05 --use_mano --seg_labels "$SEG" --manual_json "$MANUAL" \
       --w_hand_reg_override 0 --w_contact_override "${W_CONTACT:-200}" --hand_silhouette_z --w_hand_silhouette 200000 \
       --seq_id 0 --max_frames "$MAXF" --iters "$ITERS" "${EARLY_STOP_ARGS[@]}" "${GAP_ARGS[@]}" "${BOUNDARY_ARGS[@]}" \
       --out "$P/gr_hsz.npy")
if [ "$FORCE" = false ] && python -m galatea_recon.pipeline.hand_obj_optimizer "${JOINT_ARGS[@]}" --check-cache; then
  skip "gr_hsz.npy (validated complete cache)"
else
  runpy python -m galatea_recon.pipeline.hand_obj_optimizer "${JOINT_ARGS[@]}"
  ok "gr_hsz.npy + hand + handverts + faces (final joint result)"
fi

FINAL_POSES="$P/gr_hsz.npy"
FINAL_HAND_JOINTS="$P/gr_hsz_hand.npy"
FINAL_HANDVERTS="$P/gr_hsz_handverts.npy"
FINAL_MODE="joint"

source_matches() {
  [ -f "$1" ] && [ "$(head -1 "$1")" = "$FINAL_MODE" ] \
    && [ ! "$FINAL_POSES" -nt "$1" ] && [ ! "$FINAL_HANDVERTS" -nt "$1" ] \
    && [ ! "$FINAL_HAND_JOINTS" -nt "$1" ]
}
write_source_marker() {
  mkdir -p "$(dirname "$1")"
  printf '%s\n' "$FINAL_MODE" > "$1.tmp"
  mv "$1.tmp" "$1"
}
RENDER_SOURCE_MARKER="$VIZ/pipeline_render_source.txt"
WORLD_SOURCE_MARKER="$P/pipeline_world_source.txt"
RENDER_UPDATED=false; source_matches "$RENDER_SOURCE_MARKER" || RENDER_UPDATED=true
WORLD_UPDATED=false; source_matches "$WORLD_SOURCE_MARKER" || WORLD_UPDATED=true

if [ "$PRODUCE_QC" = false ]; then echo -e "\n  (lean default) skipping steps 9-10 QC videos; use --with-qc"
else
  export PYOPENGL_PLATFORM=egl
  step "Step 9/10  contact-highlight videos"
  if [ "$RENDER_UPDATED" = false ] && have "$VIZ/contact_3d.mp4" && [ -f "$VIZ/contact_overlay.mp4" ]; then skip contact videos
  else runpy python -m galatea_recon.pipeline.render_grasp_contact --cam_dir "$CAM" --mesh "$MESH" \
         --obj_poses "$FINAL_POSES" --hand_verts "$FINAL_HANDVERTS" \
         --seg_labels "$SEG" --seq_id 0 \
         --out_overlay "$VIZ/contact_overlay.mp4" \
         --out_3d "$VIZ/contact_3d.mp4" --max_frames "$MAXF"; ok "contact_overlay.mp4 + contact_3d.mp4"; fi
  step "Step 10/10  4-view solid video"
  if [ "$RENDER_UPDATED" = false ] && have "$VIZ/4view/solid_4view.mp4"; then skip 4view
  else runpy python -m galatea_recon.pipeline.render_4view --obj_poses "$FINAL_POSES" \
         --hand_verts "$FINAL_HANDVERTS" --obj_mesh "$MESH" --cam_dir "$CAM" \
         --out_dir "$VIZ/4view" --fps 15 --max_frames "$MAXF"; ok 4view/solid_4view.mp4; fi
  write_source_marker "$RENDER_SOURCE_MARKER"
fi

if [ -f "$FINAL_POSES" ]; then
  step "Step 11/11  gravity-align (Z-up world: support plane z=0, +Z=up)"; use sam3d-objects
  GRAVITY_READY=true
  for output in "$P/gr_hsz_world.npy" "$P/gr_hsz_handverts_world.npy" "$P/gravity.json"; do
    [ -f "$output" ] || GRAVITY_READY=false
  done
  [ "$PRODUCE_QC" = true ] && [ ! -f "$VIZ/gravity_align.png" ] && GRAVITY_READY=false
  GRAVITY_VIZ_ARGS=()
  [ "$PRODUCE_QC" = true ] && GRAVITY_VIZ_ARGS=(--out_viz "$VIZ/gravity_align.png")
  if [ "$WORLD_UPDATED" = false ] && have "$P/gr_hsz_world.npy" \
      && [ "$GRAVITY_READY" = true ]; then skip gr_hsz_world.npy
  else runpy python -m galatea_recon.pipeline.align_gravity --cam_dir "$CAM" \
         --poses "$FINAL_POSES" --handverts "$FINAL_HANDVERTS" \
         --out_poses "$P/gr_hsz_world.npy" --out_handverts "$P/gr_hsz_handverts_world.npy" \
         --out_json "$P/gravity.json" "${GRAVITY_VIZ_ARGS[@]}" --tag "$TAG"
       ok "gr_hsz_world.npy + gr_hsz_handverts_world.npy + gravity.json${GRAVITY_VIZ_ARGS:+ + gravity_align.png}"; fi
  write_source_marker "$WORLD_SOURCE_MARKER"
fi

echo -e "\n${GREEN}=== $TAG ('$ONAME') DONE ===${NC}"
echo "  final cam  : $FINAL_POSES  +  $(basename "$FINAL_HANDVERTS")  (mode=$FINAL_MODE)"
echo "  world traj : $P/gr_hsz_world.npy  +  gr_hsz_handverts_world.npy  (Z-up; gravity.json)"
[ "$PRODUCE_QC" = true ] && echo "  QC outputs : $VIZ/  (FP/contact/gravity images + contact/4-view videos)"
exit 0
