import os
import re
import shutil

import cv2
import numpy as np
import torch

_HAND_PROMPT_ID = 1

from galatea_recon.pipeline.h2o_loader import H2OSequence
from galatea_recon.pipeline.masks import project_points, render_object_mask

try:
    import trimesh
except ImportError:
    trimesh = None


def _write_jpg_frames(seq, jpg_dir, n_frames=None):
    os.makedirs(jpg_dir, exist_ok=True)
    n = seq.num_frames if n_frames is None else min(n_frames, seq.num_frames)
    for name in os.listdir(jpg_dir):
        if re.fullmatch(r"\d+\.jpg", name) and int(name[:-4]) >= n:
            os.unlink(os.path.join(jpg_dir, name))
    for i in range(n):
        out = os.path.join(jpg_dir, f"{i:05d}.jpg")
        bgr = cv2.cvtColor(seq.load_rgb(i), cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(out, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise OSError(f"failed to write SAM2 input frame: {out}")
    return n


def _bbox_from_mask(mask, pad=0.05):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    H, W = mask.shape
    return np.array([max(0, x1 - px), max(0, y1 - py),
                     min(W, x2 + px), min(H, y2 + py)], dtype=np.float32)


def _person_points(hand_uv, width, height, head_xy=(0.5, 0.18), ladder=6):
    pts = [p for uv in hand_uv for p in np.asarray(uv).tolist()]
    if pts:
        c = np.mean(pts, axis=0)
    else:
        c = np.array([width * 0.5, height * 0.6])
    head = np.array([width * head_xy[0], height * head_xy[1]])
    for a in np.linspace(0.15, 1.0, ladder):
        pts.append((c * (1 - a) + head * a).tolist())
    return np.array(pts, dtype=np.float32)


_OBJ_NOUN = {"book": "book", "espresso": "box", "lotion": "bottle",
             "spray": "bottle", "milk": "carton", "cocoa": "box",
             "chips": "can", "cappuccino": "box"}


def load_wilor_pipeline(device="cuda"):
    from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
        WiLorHandPose3dEstimationPipeline)
    from galatea_recon.adapters.wilor_assets import wilor_asset_dir

    return WiLorHandPose3dEstimationPipeline(
        device=device, dtype=torch.float16, wilor_pretrained_dir=wilor_asset_dir(),
    )


def save_wilor_frame0_predictions(path, predictions, rgb=None):
    from galatea_recon.pipeline.frame0_cache import save_predictions

    save_predictions(path, predictions, rgb)


def load_wilor_frame0_predictions(path):
    with np.load(path, allow_pickle=True) as data:
        return list(data["predictions"])


def hand_points_from_predictions(predictions):
    return [np.asarray(p["wilor_preds"]["pred_keypoints_2d"][0], np.float32)
            for p in predictions if "wilor_preds" in p]


def detect_hand_points(rgb, device="cuda", pipeline=None, cache_path=None,
                       predictions=None):
    if predictions is None:
        pipe = pipeline if pipeline is not None else load_wilor_pipeline(device)
        predictions = pipe.predict(rgb)
    if cache_path:
        save_wilor_frame0_predictions(cache_path, predictions, rgb)
    return hand_points_from_predictions(predictions)


def load_object_detector(model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return processor, model


def detect_object_box(rgb, object_name, hand_box=None, device="cuda",
                      model_id="IDEA-Research/grounding-dino-tiny", thresh=0.2,
                      detector=None):
    if np.ptp(np.asarray(rgb), axis=(0, 1)).max() < 2:
        return None
    import torch
    from PIL import Image
    noun = _OBJ_NOUN.get(object_name, object_name)
    text = f"a small {noun} in the hand. the {object_name} in the hand."
    pil = Image.fromarray(rgb)
    proc, model = detector if detector is not None else load_object_detector(model_id, device)
    inputs = proc(images=pil, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
    kw = dict(target_sizes=[pil.size[::-1]], text_threshold=0.15)
    try:
        res = proc.post_process_grounded_object_detection(out, inputs.input_ids,
                                                          threshold=thresh, **kw)
    except TypeError:
        res = proc.post_process_grounded_object_detection(out, inputs.input_ids,
                                                          box_threshold=thresh, **kw)
    boxes = res[0]["boxes"].cpu().numpy()
    scores = res[0]["scores"].cpu().numpy()
    if len(boxes) == 0:
        return None
    labels = res[0].get("text_labels")
    if not labels:
        labels = res[0].get("labels")
    if labels is None or len(labels) != len(boxes):
        labels = [""] * len(boxes)
    labels = [str(s).lower() for s in labels]
    obj_kw = noun.split()[-1].lower()
    _BODY = ("hand", "arm", "person", "finger", "wrist", "palm", "fist", "glove")

    def _has_word(s, w):
        return re.search(rf"\b{re.escape(w)}\b", s) is not None

    obj_idx = [i for i, s in enumerate(labels) if _has_word(s, obj_kw)]
    if not obj_idx:
        obj_idx = [i for i, s in enumerate(labels)
                   if not any(_has_word(s, w) for w in _BODY)]
    cand = np.array(obj_idx) if obj_idx else np.arange(len(boxes))
    if hand_box is not None:
        hw, hh = hand_box[2] - hand_box[0], hand_box[3] - hand_box[1]
        ex = np.array([hand_box[0] - hw, hand_box[1] - hh,
                       hand_box[2] + hw, hand_box[3] + hh])
        cx, cy = (boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2
        near = [i for i in cand
                if ex[0] <= cx[i] <= ex[2] and ex[1] <= cy[i] <= ex[3]]
        if near:
            cand = np.array(near)
    return boxes[int(cand[scores[cand].argmax()])].astype(np.float32)


def _frame0_prompts(seq, prompt_source, device, object_name=None, wilor_pipeline=None,
                    object_detector=None, frame0_wilor_cache=None,
                    frame0_wilor_predictions=None):
    if prompt_source == "gt":
        mesh = trimesh.load(seq.mesh_path, force="mesh")
        V, F = np.asarray(mesh.vertices, np.float64), np.asarray(mesh.faces, np.int64)
        _, pose0 = seq.load_obj_pose(0)
        obj_sil = render_object_mask(V, F, seq.K, pose0, (seq.width, seq.height),
                                     device=device, dilate_px=3)
        obj_box = _bbox_from_mask(obj_sil)
        hand_uv = []
        for J in seq.load_hand_joints(0):
            uv, z = project_points(J, seq.K)
            hand_uv.append(uv[z > 0])
        return obj_box, hand_uv
    rgb0 = seq.load_rgb(0)
    hand_uv = detect_hand_points(rgb0, device=device, pipeline=wilor_pipeline,
                                 cache_path=frame0_wilor_cache,
                                 predictions=frame0_wilor_predictions)
    hand_box = None
    if hand_uv:
        allk = np.concatenate(hand_uv, 0)
        hand_box = np.array([allk[:, 0].min(), allk[:, 1].min(),
                             allk[:, 0].max(), allk[:, 1].max()], np.float32)
    obj_box = detect_object_box(rgb0, object_name or seq.obj_noun, hand_box=hand_box,
                                device=device, detector=object_detector)
    return obj_box, hand_uv


def load_sam2_predictor(model_id="facebook/sam2-hiera-large", device="cuda"):
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    return SAM2VideoPredictor.from_pretrained(model_id, device=device)


def propagate_with_recovery(predictor, state, n, object_ids, first_object_frame,
                            first_hand_frame, detect_boxes, detect_points):
    from galatea_recon.pipeline.observation_gaps import mask_observed

    masks = {}
    start = 0
    while start < n:
        restart = False
        for fidx, ids, logits in predictor.propagate_in_video(state, start_frame_idx=start):
            current = {int(oid): (logits[k] > 0).squeeze().cpu().numpy()
                       for k, oid in enumerate(ids)}
            for oid in object_ids:
                if fidx < first_object_frame:
                    current[oid] = np.zeros_like(current[oid])
            if first_hand_frame is not None and fidx < first_hand_frame:
                current[_HAND_PROMPT_ID] = np.zeros_like(current[_HAND_PROMPT_ID])
            prompts = []
            if (len(object_ids) == 1 and fidx >= first_object_frame
                    and not mask_observed(current.get(object_ids[0]))):
                boxes = detect_boxes(fidx)
                if len(boxes):
                    prompts.append(dict(obj_id=object_ids[0], box=boxes[0]))
            if (first_hand_frame is not None and fidx >= first_hand_frame
                    and not mask_observed(current.get(_HAND_PROMPT_ID))):
                points = detect_points(fidx)
                if len(points):
                    prompts.append(dict(obj_id=_HAND_PROMPT_ID, points=points,
                                        labels=np.ones(len(points), dtype=np.int32)))
            for prompt in prompts:
                _, new_ids, new_logits = predictor.add_new_points_or_box(
                    state, frame_idx=fidx, **prompt)
                k = list(new_ids).index(prompt["obj_id"])
                current[prompt["obj_id"]] = (new_logits[k] > 0).squeeze().cpu().numpy()
            masks[fidx] = current
            if prompts:
                print(f"[sam2] re-prompt frame {fidx}: {[p['obj_id'] for p in prompts]}")
                start = fidx + 1
                restart = True
                break
        if not restart:
            break
    return masks


def compute_sam2_masks(cam_dir, object_models_root, work_dir, n_frames=None,
                       depth_scale=1000.0, device="cuda", extra_points=None,
                       whole_person=False, model_id="facebook/sam2-hiera-large",
                       prompt_source="auto", object_name=None, wilor_pipeline=None,
                       object_detector=None, sam2_predictor=None,
                       frame0_wilor_cache=None, frame0_prompts=None,
                       frame0_wilor_predictions=None):
    seq = H2OSequence(cam_dir, object_models_root, depth_scale)
    jpg_dir = os.path.join(work_dir, "jpg")
    n = _write_jpg_frames(seq, jpg_dir, n_frames)
    print(f"[sam2] {n} jpg frames -> {jpg_dir} (prompts={prompt_source})")

    if frame0_prompts is None:
        obj_box, hand_uv = _frame0_prompts(
            seq, prompt_source, device, object_name, wilor_pipeline=wilor_pipeline,
            object_detector=object_detector, frame0_wilor_cache=frame0_wilor_cache,
            frame0_wilor_predictions=frame0_wilor_predictions)
    else:
        obj_box, hand_uv = frame0_prompts

    if whole_person:
        hand_pts = _person_points(hand_uv, seq.width, seq.height)
        tag = "person"
    else:
        hand_pts = np.array([p for uv in hand_uv for p in np.asarray(uv).tolist()],
                            dtype=np.float32).reshape(-1, 2)
        tag = "hand/arm"
    if extra_points:
        hand_pts = np.concatenate([hand_pts, np.array(extra_points, np.float32)], 0)
    hand_lbl = np.ones(len(hand_pts), dtype=np.int32)
    print(f"[sam2] prompts: object box={None if obj_box is None else obj_box.round(0)}, "
          f"{tag} points={len(hand_pts)}")

    object_boxes = [] if obj_box is None else [obj_box]
    first_object_frame = 0
    first_hand_frame = 0 if len(hand_pts) else None
    if prompt_source == "auto":
        recovery_detector = object_detector
        recovery_wilor = wilor_pipeline

        def detect_boxes(frame):
            nonlocal recovery_detector
            if recovery_detector is None:
                recovery_detector = load_object_detector(device=device)
            box = detect_object_box(seq.load_rgb(frame), object_name or seq.obj_noun,
                                    thresh=0.3, device=device, detector=recovery_detector)
            return [] if box is None else [box]

        def detect_points(frame):
            nonlocal recovery_wilor
            if recovery_wilor is None:
                recovery_wilor = load_wilor_pipeline(device=device)
            arrays = detect_hand_points(seq.load_rgb(frame), device=device,
                                        pipeline=recovery_wilor)
            return np.asarray([p for uv in arrays for p in np.asarray(uv).tolist()],
                              dtype=np.float32).reshape(-1, 2)

        if not object_boxes:
            for first_object_frame in range(1, n):
                object_boxes = detect_boxes(first_object_frame)
                if len(object_boxes):
                    break
            if not len(object_boxes):
                raise ValueError("object was never detected; cannot initialize SAM2 tracking")
        if first_hand_frame is None:
            for frame in range(1, n):
                points = detect_points(frame)
                if len(points):
                    hand_pts = points
                    hand_lbl = np.ones(len(points), dtype=np.int32)
                    first_hand_frame = frame
                    break

    predictor = (sam2_predictor if sam2_predictor is not None
                 else load_sam2_predictor(model_id=model_id, device=device))
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=jpg_dir)
        predictor.reset_state(state)
        for object_id, box in enumerate(object_boxes):
            predictor.add_new_points_or_box(state, frame_idx=first_object_frame,
                                            obj_id=object_id, box=box)
        if len(hand_pts):
            predictor.add_new_points_or_box(state, frame_idx=first_hand_frame, obj_id=_HAND_PROMPT_ID,
                                            points=hand_pts, labels=hand_lbl)
        if prompt_source == "auto":
            masks = propagate_with_recovery(
                predictor, state, n, list(range(len(object_boxes))), first_object_frame,
                first_hand_frame, detect_boxes, detect_points)
        else:
            masks = {}
            for fidx, obj_ids, logits in predictor.propagate_in_video(state):
                masks[fidx] = {}
                for k, oid in enumerate(obj_ids):
                    masks[fidx][int(oid)] = (logits[k] > 0.0).squeeze().cpu().numpy()
    print(f"[sam2] tracked {len(masks)} frames")
    return seq, masks


def run_viz(cam_dir, object_models_root, out_dir, n_frames=None, viz_step=20,
            depth_scale=1000.0, device="cuda", whole_person=False,
            prompt_source="auto", object_name=None):
    work = os.path.join(out_dir, "_work")
    seq, masks = compute_sam2_masks(cam_dir, object_models_root, work, n_frames,
                                    depth_scale, device, whole_person=whole_person,
                                    prompt_source=prompt_source, object_name=object_name)
    os.makedirs(out_dir, exist_ok=True)
    frames = sorted(masks.keys())[::viz_step]
    for i in frames:
        bgr = cv2.cvtColor(seq.load_rgb(i), cv2.COLOR_RGB2BGR)
        om = masks[i].get(0, np.zeros((seq.height, seq.width), bool))
        hm = masks[i].get(1, np.zeros((seq.height, seq.width), bool))
        fg = (om | hm)
        ov = bgr.copy()
        ov[om] = (ov[om] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
        ov[hm] = (ov[hm] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        cv2.putText(ov, f"f{i} SAM2 obj(red)+fg(blue) fg={100*fg.mean():.1f}%",
                    (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cut = bgr.copy(); cut[fg] = 0
        cv2.imwrite(os.path.join(out_dir, f"sam_{i:06d}.png"), np.hstack([ov, cut]))
        print(f"  f{i}: fg={100*fg.mean():.1f}%")
    print(f"\nWritten -> {out_dir}/sam_*.png")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--object_models_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=None, help="limit frames (speed)")
    ap.add_argument("--viz_step", type=int, default=20)
    ap.add_argument("--whole_person", action="store_true",
                    help="mask the whole person instead of just hand+arm")
    ap.add_argument("--prompt_source", choices=["auto", "gt"], default="auto",
                    help="auto=detectors (no GT, default); gt=legacy GT-pose/joint prompts")
    ap.add_argument("--object_name", default=None,
                    help="override the detector's object name (auto mode)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_viz(args.cam_dir, args.object_models_root, args.out_dir,
            args.n_frames, args.viz_step, device=args.device,
            whole_person=args.whole_person, prompt_source=args.prompt_source,
            object_name=args.object_name)
