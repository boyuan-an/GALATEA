import argparse
import os
import sys
from pathlib import Path

import numpy as np

from galatea_recon.pipeline._paths import REPO
from galatea_recon.pipeline.contact_cache import (
    PARAMETERS, cached_contact, contact_inputs, frame_paths, publish_contact,
)


DEFAULT_HOIDETR_ROOT = os.path.join(REPO, "3rd", "HOI-DETR-cu121")
DEFAULT_CONFIG = os.path.join(
    "projects",
    "configs",
    "co_dino_vit",
    "co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py",
)
DEFAULT_CHECKPOINT = os.path.join("checkpoints", "epoch_5.pth")


def median_filter(bits, width=5):
    if width < 1 or width % 2 == 0:
        raise ValueError("median width must be a positive odd integer")
    half = width // 2
    output = []
    for index in range(len(bits)):
        neighborhood = bits[max(0, index - half) : index + half + 1]
        output.append(1 if sum(neighborhood) * 2 > len(neighborhood) else 0)
    return output


def binary_segments(bits, min_length=5, max_gap=4):
    segments = []
    start = None
    for index, active in enumerate(list(bits) + [0]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            segments.append([start, index - 1])
            start = None

    merged = []
    for segment in segments:
        if merged and segment[0] - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = segment[1]
        else:
            merged.append(segment)
    return [segment for segment in merged if segment[1] - segment[0] + 1 >= min_length]


def estimate_windows(
    signal,
    threshold=0.5,
    median_width=5,
    min_length=5,
    max_gap=4,
    start_trim=0,
    end_trim=0,
):
    if not 0 <= threshold <= 1 or min_length < 1 or max_gap < 0:
        raise ValueError("invalid contact threshold or segment lengths")
    if start_trim < 0 or end_trim < 0:
        raise ValueError("contact trims must be nonnegative")
    bits = [1 if value >= threshold else 0 for value in signal]
    filtered = median_filter(bits, median_width)
    raw_segments = binary_segments(filtered, min_length=min_length, max_gap=max_gap)
    calibrated = []
    for start, end in raw_segments:
        corrected = [start + start_trim, end - end_trim]
        if corrected[0] <= corrected[1]:
            calibrated.append(corrected)
    return raw_segments, calibrated, filtered


def format_windows(segments):
    return ",".join("{}-{}".format(start, end) for start, end in segments)


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite([a, b]).all():
        return 0.0
    intersection = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:] - a[:2]).prod() + np.maximum(0, b[2:] - b[:2]).prod() - intersection
    return float(intersection / union) if union > 0 else 0.0


def match_detection(detections, class_id, target_box, min_iou, margin):
    if target_box is None:
        return None, 0.0, "unobserved"
    matches = sorted(((box_iou(item["box"], target_box), i)
                      for i, item in enumerate(detections) if item["class_id"] == class_id),
                     reverse=True)
    if not matches or matches[0][0] < min_iou:
        return None, 0.0, "unmatched"
    score, index = matches[0]
    if len(matches) > 1 and matches[1][0] >= min_iou and score - matches[1][0] < margin:
        return None, score, "ambiguous"
    return index, score, "matched"


def load_targets(mask_path, hand_path, n):
    with np.load(hand_path, allow_pickle=True) as data:
        hands = data["hands"]
        observed = data["hand_observed"] if "hand_observed" in data else np.ones(len(hands), bool)
    if hands.shape != (n,) or observed.shape != (n,) or observed.dtype != np.bool_:
        raise ValueError("target hands must match the RGB timeline")
    targets = []
    with np.load(mask_path, allow_pickle=False) as masks:
        for i in range(n):
            mask = masks["obj_{}".format(i)]
            if mask.ndim != 2 or mask.dtype != np.bool_:
                raise ValueError("target object mask must be a 2D boolean array")
            y, x = np.nonzero(mask)
            object_box = [int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1] if len(x) >= 20 else None
            right = [h for h in hands[i] if float(h["is_right"]) > .5]
            hand_box = None
            if observed[i] and len(right) == 1:
                points = np.asarray(right[0]["kp2d"], float)
                if points.shape != (21, 2) or not np.isfinite(points).all():
                    raise ValueError("invalid target hand keypoints")
                hand_box = np.r_[points.min(axis=0), points.max(axis=0)].tolist()
            targets.append({"object_box": object_box, "hand_box": hand_box})
    return targets


def associate_targets(detections, target, args):
    obj, oiou, ostatus = match_detection(detections, 1, target["object_box"],
                                       args.object_match_iou, args.match_margin)
    hand, hiou, hstatus = match_detection(detections, 0, target["hand_box"],
                                        args.hand_match_iou, args.match_margin)
    return dict(target, object_index=obj, hand_index=hand, object_iou=oiou,
                hand_iou=hiou, object_status=ostatus, hand_status=hstatus)


def _pair_probabilities(torch, branch, embeddings, group_a, group_b):
    output = []
    for item_a in group_a:
        for item_b in group_b:
            with torch.no_grad():
                features = torch.cat(
                    [embeddings[item_a["query_idx"]], embeddings[item_b["query_idx"]]],
                    dim=0,
                ).unsqueeze(0)
                logits = branch.mlp(features)
                prediction = int(logits.argmax(dim=1).item())
                probability = float(torch.softmax(logits, dim=-1)[0, 1].item())
            output.append((item_a, item_b, prediction, probability))
    return output


def estimate_contact(args):
    inputs = contact_inputs(args)
    if not args.force:
        previous = cached_contact(args.out, inputs)
        if previous is not None:
            print("[hoidetr-contact] using complete matching cache", flush=True)
            return previous
    Path(args.out).unlink(missing_ok=True)
    frames = frame_paths(args)
    targets = load_targets(args.target_masks, args.target_hands, len(frames))
    hoidetr_root = os.path.abspath(args.hoidetr_root)
    model_config = os.path.join(hoidetr_root, args.model_config)
    checkpoint = os.path.join(hoidetr_root, args.checkpoint)
    if not os.path.isfile(model_config):
        raise FileNotFoundError("HOI-DETR config not found: {}".format(model_config))
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError("HOI-DETR checkpoint not found: {}".format(checkpoint))

    sys.path.insert(0, hoidetr_root)
    sys.path.insert(0, os.path.join(hoidetr_root, "demo"))

    import mmcv
    import torch
    from mmdet.apis import init_detector
    from mmdet.datasets.pipelines import Compose

    from projects import models as _projects_models

    from configs import CLASS_NAMES
    from helpers import (
        compute_style,
        draw_ui,
        find_interaction_branch,
        run_inference as run_frame_inference,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)

    model = init_detector(model_config, checkpoint, device=args.device)
    model.eval()
    test_pipeline = Compose(model.cfg.data.test.pipeline)
    branch = find_interaction_branch(model.query_head)
    print("[hoidetr-contact] model ready; {} frames".format(len(frames)), flush=True)

    records = []
    signal = []
    error_frames = []
    for frame_index, image_path in enumerate(frames):
        try:
            detections, embeddings = run_frame_inference(
                model,
                test_pipeline,
                str(image_path),
                device=args.device,
                class_names=CLASS_NAMES,
                score_thr=args.score_threshold,
                nms_iou=args.nms_iou,
            )
        except Exception as error:
            print(
                "[hoidetr-contact] frame {} failed: {}".format(frame_index, error),
                flush=True,
            )
            detections, embeddings = [], None
            error_frames.append(frame_index)

        association = associate_targets(detections, targets[frame_index], args)
        hand_index, object_index = association["hand_index"], association["object_index"]
        interactions = (
            _pair_probabilities(torch, branch, embeddings,
                                [detections[hand_index]], [detections[object_index]])
            if hand_index is not None and object_index is not None
            else []
        )
        probability = max([item[3] for item in interactions] or [0.0])
        signal.append(probability)

        detection_indices = {id(item): index for index, item in enumerate(detections)}
        records.append(
            {
                "frame_idx": frame_index,
                "file_name": os.path.basename(image_path),
                "interaction_probability": round(probability, 4),
                "association": association,
                "detections": [
                    {
                        "box": [round(float(value), 1) for value in item["box"]],
                        "score": round(float(item["score"]), 4),
                        "class_id": int(item["class_id"]),
                    }
                    for item in detections
                ],
                "hf": [
                    {
                        "a": detection_indices[id(item_a)],
                        "b": detection_indices[id(item_b)],
                        "pred": prediction,
                        "prob": round(probability_item, 4),
                    }
                    for item_a, item_b, prediction, probability_item in interactions
                ],
            }
        )

        if args.viz_dir and args.viz_every and frame_index % args.viz_every == 0:
            image = mmcv.imread(str(image_path))
            visualization = image.copy()
            style = compute_style(visualization.shape)
            drawn_interactions = [
                (item_a, item_b, probability_item)
                for item_a, item_b, prediction, probability_item in interactions
                if prediction == 1
            ]
            draw_ui(
                visualization,
                detections,
                drawn_interactions,
                [],
                style,
                verbose_labels=True,
            )
            mmcv.imwrite(
                visualization,
                os.path.join(args.viz_dir, os.path.basename(image_path)),
            )

        if frame_index % 20 == 0:
            print(
                "[hoidetr-contact] {}/{}".format(frame_index, len(frames)),
                flush=True,
            )

    raw_segments, calibrated, filtered = estimate_windows(
        signal,
        threshold=args.interaction_threshold,
        median_width=args.median_width,
        min_length=args.min_length,
        max_gap=args.max_gap,
        start_trim=args.start_trim,
        end_trim=args.end_trim,
    )
    if not calibrated:
        raise RuntimeError(
            "HOI-DETR found no usable contact window; provide --contact-windows manually"
        )

    report = {
        "source": "hoidetr-auto",
        "state": "complete",
        "inputs": inputs,
        "window": format_windows(calibrated),
        "raw_window": format_windows(raw_segments),
        "num_frames": len(frames),
        "error_frames": error_frames,
        "parameters": {name: getattr(args, name) for name in PARAMETERS},
        "filtered_contact": filtered,
        "frames": records,
    }
    publish_contact(args.out, report)
    print(
        "[hoidetr-contact] raw={} calibrated={}".format(report["raw_window"], report["window"]),
        flush=True,
    )
    return report


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Estimate contact windows with HOI-DETR")
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--target-masks", required=True, help="SAM object track used by reconstruction")
    parser.add_argument("--target-hands", required=True, help="WiLoR primary right-hand track")
    parser.add_argument("--out", required=True)
    parser.add_argument("--viz-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--hoidetr-root", default=DEFAULT_HOIDETR_ROOT)
    parser.add_argument("--model-config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--interaction-threshold", type=float, default=0.5)
    parser.add_argument("--median-width", type=int, default=5)
    parser.add_argument("--min-length", type=int, default=5)
    parser.add_argument("--max-gap", type=int, default=4)
    parser.add_argument("--start-trim", type=int, default=0, help="optional explicit trim in frames")
    parser.add_argument("--end-trim", type=int, default=0, help="optional explicit trim in frames")
    parser.add_argument("--object-match-iou", type=float, default=0.25)
    parser.add_argument("--hand-match-iou", type=float, default=0.25)
    parser.add_argument("--match-margin", type=float, default=0.05)
    parser.add_argument("--viz-every", type=int, default=10)
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not all(0 < value <= 1 for value in (args.object_match_iou, args.hand_match_iou)) or not 0 <= args.match_margin <= 1:
        parser.error("target IoU thresholds must lie in (0, 1] and margin in [0, 1]")
    estimate_windows([], args.interaction_threshold, args.median_width,
                     args.min_length, args.max_gap, args.start_trim, args.end_trim)
    if args.check_cache:
        try:
            hit = not args.force and cached_contact(args.out, contact_inputs(args)) is not None
        except (OSError, ValueError, KeyError):
            hit = False
        raise SystemExit(0 if hit else 1)
    estimate_contact(args)


if __name__ == "__main__":
    main()
