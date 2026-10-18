import argparse
import os
from pathlib import Path

import numpy as np

from galatea_recon.pipeline.h2o_loader import H2OSequence
from galatea_recon.pipeline.perception_cache import (
    atomic_numpy,
    depth_cache_complete,
    depth_identity,
    depth_settings,
    publish_depth_cache,
)
from galatea_recon.pipeline.object_quality import atomic_json
from galatea_recon.pipeline.step34_calibrate import get_sam2_masks, trimmed_affine_fit


def run(
    cam_dir,
    object_models_root,
    out_dir,
    depth_scale=1000.0,
    device="cuda",
    trim_pct=0.2,
    max_frames=None,
    moge_model=None,
    wilor_pipeline=None,
    object_detector=None,
    sam2_predictor=None,
    frame0_wilor_cache=None,
    force=False,
    calibration_mode=None,
    rectify_camera=None,
):
    seq = H2OSequence(cam_dir, object_models_root, depth_scale)
    n = seq.num_frames if max_frames is None else min(max_frames, seq.num_frames)
    calibration_mode, rectify_camera = depth_settings(calibration_mode, rectify_camera)
    inputs = depth_identity(seq, n, depth_scale, trim_pct, calibration_mode, rectify_camera)
    (Path(out_dir) / "depth_complete.json").unlink(missing_ok=True)
    frames = list(range(n))
    raw_dir = os.path.join(out_dir, "moge_raw")
    cal_dir = os.path.join(out_dir, "moge_cal")
    for d in (raw_dir, cal_dir):
        os.makedirs(d, exist_ok=True)
    print(f"[export] {seq.obj_name} {n} frames -> {out_dir}")

    if frame0_wilor_cache is None:
        frame0_wilor_cache = os.path.join(out_dir, "wilor_frame0_predictions.npz")
    sam_fg, _ = get_sam2_masks(
        cam_dir,
        object_models_root,
        out_dir,
        n - 1,
        depth_scale,
        device,
        wilor_pipeline=wilor_pipeline,
        object_detector=object_detector,
        sam2_predictor=sam2_predictor,
        frame0_wilor_cache=frame0_wilor_cache,
        force=force,
    )
    fg_anchor = sam_fg[0]
    measured_0 = seq.load_depth(0)

    rgb_paths = [seq.rgb_files[i] for i in frames]
    print(f"[export] running MoGe-2 on {n} frames...")
    from galatea_recon.adapters.depth import est_depth

    moge = est_depth(
        rgb_paths,
        device=device,
        intrinsics=seq.K,
        model=moge_model,
        rectify_intrinsics=rectify_camera,
    )
    moge = [d.detach().cpu().numpy().astype(np.float32) for d in moge]

    reference_affine = (1.0, 0.0)
    records = []
    for i in frames:
        m = moge[i]
        atomic_numpy(os.path.join(raw_dir, f"{i:06d}.npy"), m)
        fg_i = sam_fg[i]
        bg = ~((fg_anchor > 0) | (fg_i > 0)) & (measured_0 > 0.1) & (measured_0 < 8.0) & (m > 0.05)
        if calibration_mode == "reference" and i > 0:
            s, b = reference_affine
        elif bg.sum() >= 100:
            s, b = trimmed_affine_fit(m[bg], measured_0[bg], trim_pct=trim_pct)
        else:
            s, b = 1.0, 0.0
        if calibration_mode == "reference" and (not np.isfinite([s, b]).all() or s <= 0):
            raise ValueError("first-frame depth calibration must have a finite positive scale")
        if i == 0:
            reference_affine = (s, b)
        cal = np.clip(s * m + b, 0, None).astype(np.float32)
        if calibration_mode == "reference" or rectify_camera:
            cal[~np.isfinite(m) | (m <= 0)] = 0
        records.append(
            dict(frame=i, scale=float(s), offset_m=float(b), background_pixels=int(bg.sum()))
        )
        atomic_numpy(os.path.join(cal_dir, f"{i:06d}.npy"), cal)
        if i % 50 == 0:
            print(f"  frame {i}/{n}")
    atomic_json(
        Path(out_dir) / "depth_calibration.json",
        {
            "state": "complete",
            "mode": calibration_mode,
            "rectify_camera": rectify_camera,
            "first_frame_background_supported": records[0]["background_pixels"] >= 100,
            "frames": records,
        },
    )
    publish_depth_cache(out_dir, inputs)
    print(f"[export] done -> {raw_dir} , {cal_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--object_models_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true", help="also recompute cached SAM2 masks")
    ap.add_argument(
        "--check-cache",
        action="store_true",
        help="validate without model inference; exit 1 if stale/incomplete",
    )
    args = ap.parse_args()
    if args.check_cache:
        seq = H2OSequence(args.cam_dir, args.object_models_root, args.depth_scale)
        n = seq.num_frames if args.max_frames is None else min(args.max_frames, seq.num_frames)
        raise SystemExit(
            0 if depth_cache_complete(args.out_dir, depth_identity(seq, n, args.depth_scale)) else 1
        )
    run(
        args.cam_dir,
        args.object_models_root,
        args.out_dir,
        args.depth_scale,
        args.device,
        max_frames=args.max_frames,
        force=args.force,
    )
