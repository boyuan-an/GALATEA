import argparse
import os
from glob import glob
from pathlib import Path

import cv2
import numpy as np

from galatea_recon.pipeline.clean_wilor_hands import (
    clean_wilor_hands,
    save_wilor_npz,
    write_report,
)
from galatea_recon.pipeline.observation_gaps import complete_hands, hand_observed
from galatea_recon.pipeline.trajectory_cache import hand_cache_complete, hand_inputs, publish_hand_cache


def load_K(cam_dir):
    p = os.path.join(cam_dir, "cam_intrinsics.txt")
    fx, fy, cx, cy, W, H = np.loadtxt(p)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    return K, int(W), int(H)


def place_hand_h2o(kp3d, kp2d, verts, K):
    if not all(np.isfinite(value).all() for value in (kp3d, kp2d, verts, K)):
        return None, None, None, None
    root = kp3d[0:1]
    kp3d_rel = (kp3d - root).astype(np.float64)
    verts_rel = (verts - root).astype(np.float64)
    try:
        ok, rvec, tvec = cv2.solvePnP(
            kp3d_rel, kp2d.astype(np.float64), K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
    except cv2.error:
        return None, None, None, None
    if not ok:
        return None, None, None, None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3)
    kp3d_cam = (R @ kp3d_rel.T).T + t
    verts_cam = (R @ verts_rel.T).T + t
    return (
        verts_cam.astype(np.float32),
        kp3d_cam.astype(np.float32),
        R.astype(np.float32),
        t.astype(np.float32),
    )


def predict_requested_hands(pipe, rgb, hand_side, hand_conf=0.3):
    if hand_side == "all":
        return pipe.predict(rgb, hand_conf=hand_conf), 0

    detections = pipe.hand_detector(rgb, conf=hand_conf, verbose=pipe.verbose)[0]
    requested_is_right = 1.0 if hand_side == "right" else 0.0
    bboxes = []
    is_rights = []
    discarded = 0
    for detection in detections:
        is_right = float(detection.boxes.cls.cpu().detach().reshape(-1)[0].item())
        if (is_right > 0.5) != (requested_is_right > 0.5):
            discarded += 1
            continue
        box = detection.boxes.data.cpu().detach().reshape(-1).numpy()
        bboxes.append(box[:4])
        is_rights.append(requested_is_right)
    if not bboxes:
        return [], discarded
    predictions = pipe.predict_with_bboxes(
        rgb,
        np.asarray(bboxes, dtype=np.float32),
        np.asarray(is_rights, dtype=np.float32),
    )
    return predictions, discarded


def load_frame0_predictions(path, rgb=None, hand_conf=0.3):
    if not path or not os.path.isfile(path):
        return None
    from galatea_recon.pipeline.frame0_cache import load_predictions

    return load_predictions(path, rgb, hand_conf)


def select_cached_predictions(predictions, hand_side):
    if hand_side == "all":
        return list(predictions), 0
    requested_is_right = hand_side == "right"
    selected = [
        prediction
        for prediction in predictions
        if (float(prediction.get("is_right", 0.0)) > 0.5) == requested_is_right
    ]
    return selected, len(predictions) - len(selected)


def load_wilor_pipeline(device="cuda"):
    import torch
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline,
    )

    from galatea_recon.adapters.wilor_assets import wilor_asset_dir

    return WiLorHandPose3dEstimationPipeline(
        device=device, dtype=torch.float16, wilor_pretrained_dir=wilor_asset_dir(),
    )


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--viz_frames", type=int, nargs="+", default=[0, 30, 59])
    ap.add_argument("--hand-side", choices=("right", "left", "all"), default="all")
    ap.add_argument("--hand-conf", type=float, default=0.3)
    ap.add_argument("--clean-jumps", action="store_true")
    ap.add_argument("--fill-missing", action="store_true",
                    help="complete missing right-hand spans and save observation validity")
    ap.add_argument("--max-repair-span", type=int, default=3)
    ap.add_argument("--jump-position-mm", type=float, default=80.0)
    ap.add_argument("--jump-rotation-deg", type=float, default=75.0)
    ap.add_argument("--check-cache", action="store_true", help="validate cached hands without inference")
    ap.add_argument(
        "--frame0-predictions",
        default=None,
        help="raw frame-0 WiLoR cache produced during SAM2 prompt generation",
    )
    return ap


def run(args, pipe=None):
    if args.clean_jumps and args.hand_side != "right":
        raise ValueError("--clean-jumps currently requires --hand-side right")
    if getattr(args, "fill_missing", False) and args.hand_side != "right":
        raise ValueError("--fill-missing requires --hand-side right")

    os.makedirs(args.out, exist_ok=True)
    (Path(args.out) / "wilor_complete.json").unlink(missing_ok=True)
    inputs = hand_inputs(args)
    K, W, H = load_K(args.cam_dir)
    rgb_files = sorted(glob(os.path.join(args.cam_dir, "rgb", "*.png")))[: args.max_frames]
    if pipe is None:
        pipe = load_wilor_pipeline()
    rgb0 = cv2.cvtColor(cv2.imread(rgb_files[0]), cv2.COLOR_BGR2RGB)
    frame0_predictions = load_frame0_predictions(args.frame0_predictions, rgb0, args.hand_conf)
    marker = Path(args.out) / "gap_completion_v1"
    if marker.exists():
        marker.unlink()

    hands_per_frame = []
    discarded_side_detections = 0
    for i, f in enumerate(rgb_files):
        bgr = cv2.imread(f)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if i == 0 and frame0_predictions is not None:
            preds, discarded = select_cached_predictions(frame0_predictions, args.hand_side)
            print(f"  frame 0: reused {len(preds)} cached WiLoR prediction(s)")
        else:
            preds, discarded = predict_requested_hands(
                pipe, rgb, args.hand_side, hand_conf=args.hand_conf
            )
        discarded_side_detections += discarded
        frame_hands = []
        for p in preds:
            wp = p["wilor_preds"]
            kp3d = wp["pred_keypoints_3d"][0]
            kp2d = wp["pred_keypoints_2d"][0]
            verts = wp["pred_vertices"][0]
            vc, kc, R, t = place_hand_h2o(kp3d, kp2d, verts, K)
            if vc is None:
                continue
            frame_hands.append(
                {
                    "is_right": float(p["is_right"]),
                    "verts": vc,
                    "kp3d": kc,
                    "kp2d": kp2d.astype(np.float32),
                    "global_orient": wp["global_orient"][0].astype(np.float32),
                    "hand_pose": wp["hand_pose"][0].astype(np.float32),
                    "betas": wp["betas"][0].astype(np.float32),
                    "place_R": R,
                    "place_t": t,
                    "root": kp3d[0].astype(np.float32),
                }
            )
        hands_per_frame.append(frame_hands)

        if i % 20 == 0:
            print(f"  frame {i}/{len(rgb_files)}: {len(frame_hands)} hand(s)")

    raw_hands = np.empty(len(hands_per_frame), dtype=object)
    for frame, frame_hands in enumerate(hands_per_frame):
        raw_hands[frame] = frame_hands
    save_wilor_npz(os.path.join(args.out, "wilor_hands_raw.npz"), hands=raw_hands, camera=K)
    final_hands = raw_hands
    observed = None
    report = {}
    if getattr(args, "fill_missing", False):
        final_hands = np.empty(len(raw_hands), dtype=object)
        for i, frame in enumerate(raw_hands):
            final_hands[i] = [h for h in frame if hand_observed(h, (W, H), True)]
        observed = np.array([bool(len(frame)) for frame in final_hands], dtype=bool)
    if args.clean_jumps:
        final_hands, report = clean_wilor_hands(
            final_hands,
            max_repair_span=args.max_repair_span,
            position_floor_mm=args.jump_position_mm,
            rotation_floor_deg=args.jump_rotation_deg,
        )
        if observed is not None:
            for span in report["interpolated_jump_spans"]:
                observed[span["frames"]] = False
        report.update(
            {
                "requested_hand_side": args.hand_side,
                "detector_confidence": args.hand_conf,
                "discarded_non_requested_detections": discarded_side_detections,
            }
        )
    if observed is not None:
        final_hands, observed = complete_hands(
            final_hands, observed, (W, H), require_mano=True)
        report["filled_frames"] = np.flatnonzero(~observed).tolist()
        report["gap_policy"] = "linear translation / SLERP; nearest hold at boundaries"
        report["unresolved_missing_frames"] = []
    if report:
        write_report(os.path.join(args.out, "wilor_cleanup.json"), report)
    save_wilor_npz(os.path.join(args.out, "wilor_hands.npz"), hands=final_hands,
                   camera=K, observed=observed)
    if observed is not None:
        marker.touch()

    for frame in args.viz_frames:
        if not 0 <= frame < len(rgb_files):
            continue
        bgr = cv2.imread(rgb_files[frame])
        _overlay(
            bgr,
            final_hands[frame],
            K,
            os.path.join(args.out, f"overlay_{frame:06d}.png"),
        )

    nfh = [len(h) for h in final_hands]
    print(
        f"[wilor] {len(rgb_files)} frames, hands/frame min={min(nfh)} max={max(nfh)} "
        f"side={args.hand_side}, cleaned={args.clean_jumps} -> {args.out}/wilor_hands.npz"
    )
    publish_hand_cache(args, inputs)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.clean_jumps and args.hand_side != "right":
        parser.error("--clean-jumps currently requires --hand-side right")
    if args.check_cache:
        raise SystemExit(0 if hand_cache_complete(args) else 1)
    run(args)


def _overlay(bgr, frame_hands, K, out_png):
    img = bgr.copy()
    for h in frame_hands:
        uv = (K @ h["verts"].T).T
        uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
        for u, v in h["kp2d"].astype(int):
            cv2.circle(img, (u, v), 3, (0, 0, 255), -1)
    cv2.imwrite(out_png, img)


if __name__ == "__main__":
    main()
