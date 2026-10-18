import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np

from galatea_recon.pipeline.object_quality import mask_box


def box_iou(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    lo, hi = np.maximum(a[:2], b[:2]), np.minimum(a[2:], b[2:])
    overlap = float(np.prod(np.maximum(hi-lo, 0)))
    return overlap / max(float(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-overlap), 1)


def proposal_boxes(rgb, object_name, detector, original=None, limit=5):
    import torch
    from PIL import Image
    processor, model = detector
    prompts = [f"{object_name}. 3d printed {object_name} model.",
               "object. handheld object. model."]
    candidates = []
    for prompt in prompts:
        inputs = processor(images=Image.fromarray(rgb), text=prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        kwargs = dict(target_sizes=[rgb.shape[:2]], text_threshold=.12)
        try:
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=.15, **kwargs)[0]
        except TypeError:
            result = processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, box_threshold=.15, **kwargs)[0]
        boxes = result["boxes"].detach().cpu().numpy()
        scores = result["scores"].detach().cpu().numpy()
        labels = result.get("text_labels", result.get("labels", [""]*len(boxes)))
        for box, score, label in zip(boxes, scores, labels):
            area = float(np.prod(box[2:]-box[:2])) / (rgb.shape[0]*rgb.shape[1])
            if .0003 < area < .65 and str(label).lower().strip() not in ("hand", "arm", "person"):
                candidates.append((float(score), box, prompt))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for score, box, prompt in candidates:
        if all(box_iou(box, old["box"]) < .85 for old in selected):
            selected.append(dict(box=box.tolist(), detector_score=score, prompt=prompt))
        if len(selected) >= limit-1:
            break
    base = (np.asarray(selected[0]["box"]) if selected else original)
    if base is not None:
        base = np.asarray(base, float)
        pad = (base[2:]-base[:2])*.22
        enlarged = np.r_[np.maximum(base[:2]-pad, 0),
                         np.minimum(base[2:]+pad, [rgb.shape[1], rgb.shape[0]])]
        selected.append(dict(box=enlarged.tolist(), detector_score=None, prompt="expanded_image_box"))
    return selected[:limit]


def generate(cam_dir, masks_path, object_name, frames, output, max_frames=999999,
             max_tracks=5):
    import torch
    from galatea_recon.pipeline.sam_masks import load_object_detector, load_sam2_predictor
    cam = Path(cam_dir)
    files = sorted((cam / "rgb").glob("*.png"))[:max_frames]
    with np.load(masks_path) as archive:
        base = np.stack([archive[f"obj_{f}"] for f in range(len(files))]).astype(bool)
    n, height, width = base.shape
    frames = sorted(set(min(max(int(f), 0), n-1) for f in frames))[:3]
    detector = load_object_detector(device="cuda")
    specs = []
    for frame in frames:
        rgb = cv2.cvtColor(cv2.imread(str(files[frame])), cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (width, height))
        limit = max(2, max_tracks // len(frames))
        specs += [dict(spec, frame=frame, source="automatic_image_prompt")
                  for spec in proposal_boxes(rgb, object_name, detector, mask_box(base[frame]), limit)]
    del detector
    gc.collect()
    torch.cuda.empty_cache()
    specs = specs[:max_tracks]
    work = Path(output).parent / "proposal_frames"
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("*.jpg"):
        if stale.stem.isdigit():
            stale.unlink()
    for f, path in enumerate(files):
        image = cv2.resize(cv2.imread(str(path)), (width, height))
        cv2.imwrite(str(work / ("%05d.jpg" % f)), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    predictor = load_sam2_predictor(device="cuda")
    tracks, metadata = [], []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=str(work), offload_video_to_cpu=True,
                                     offload_state_to_cpu=True)
        for spec in specs:
            predictor.reset_state(state)
            frame = spec["frame"]
            predictor.add_new_points_or_box(state, frame_idx=frame, obj_id=0,
                                            box=np.asarray(spec["box"], np.float32))
            track = np.zeros_like(base)
            for f, ids, logits in predictor.propagate_in_video(state, start_frame_idx=frame):
                track[f] = (logits[0] > 0).squeeze().cpu().numpy()
            if frame > 0:
                for f, ids, logits in predictor.propagate_in_video(state, start_frame_idx=frame, reverse=True):
                    track[f] = (logits[0] > 0).squeeze().cpu().numpy()
            existing = [base] + tracks
            if any(np.count_nonzero(track[frame] & old[frame]) /
                   max(1, np.count_nonzero(track[frame] | old[frame])) > .96 for old in existing):
                continue
            tracks.append(track)
            metadata.append(spec)
            print("[proposals] kept frame=%d area=%d source=%s" %
                  (frame, track[frame].sum(), spec["prompt"]), flush=True)
    stored = dict(shape=np.array(base.shape, np.int64), metadata=np.array(json.dumps(metadata)))
    stored.update({f"track_{i}": np.packbits(track) for i, track in enumerate(tracks)})
    np.savez_compressed(output, **stored)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Alternative image-derived SAM2 tracks for bounded object recovery.")
    parser.add_argument("--cam-dir", required=True)
    parser.add_argument("--masks", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--frames", default="0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-frames", type=int, default=999999)
    parser.add_argument("--max-tracks", type=int, default=5)
    args = parser.parse_args()
    generate(args.cam_dir, args.masks, args.object_name, args.frames.split(","),
             args.out, args.max_frames, args.max_tracks)


if __name__ == "__main__":
    main()
