import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MOGE_MODEL = "Ruicheng/moge-2-vitl-normal"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def predict_first_frame(rgb, model_name, device):
    import torch
    from moge.model.v2 import MoGeModel

    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    tensor = torch.from_numpy(rgb.copy()).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1) / 255.0
    with torch.inference_mode():
        prediction = model.infer(
            tensor, resolution_level=9, use_fp16=str(device).startswith("cuda")
        )
    return (
        prediction["depth"].float().cpu().numpy(),
        prediction["intrinsics"].float().cpu().numpy(),
    )


def encode_prediction(depth, normalized_k, height, width):
    depth = np.asarray(depth)
    k = np.asarray(normalized_k, dtype=np.float64).copy()
    if depth.shape != (height, width):
        raise ValueError(f"MoGe depth shape {depth.shape} != image {(height, width)}")
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("MoGe returned invalid camera intrinsics")
    if k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
        raise ValueError("MoGe returned invalid focal length or homogeneous row")
    valid = np.isfinite(depth) & (depth >= 0.0005) & (depth <= 65.535)
    if not valid.any():
        raise ValueError("MoGe returned no representable positive depth")
    encoded = np.zeros((height, width), dtype=np.uint16)
    encoded[valid] = np.rint(depth[valid] * 1000).astype(np.uint16)
    k[0] *= width
    k[1] *= height
    return encoded, np.array([k[0, 0], k[1, 1], k[0, 2], k[1, 2], width, height])


def prep(
    video,
    out_cam0,
    rgb0=None,
    depth0=None,
    intrinsics=None,
    max_frames=None,
    input_mode="auto",
    moge_model=DEFAULT_MOGE_MODEL,
    device="cuda",
    force=False,
):
    video = Path(video).resolve()
    out = Path(out_cam0).resolve()
    if not video.is_file():
        raise FileNotFoundError(f"missing video: {video}")
    if input_mode not in ("auto", "rgbd", "rgb-only"):
        raise ValueError(f"unknown input mode: {input_mode}")
    if max_frames is not None and int(max_frames) <= 0:
        raise ValueError("max_frames must be positive")
    rgb0 = Path(rgb0 or video.parent / "rgb.png").resolve()
    depth0 = Path(depth0 or video.parent / "depth.png").resolve()
    intrinsics = Path(intrinsics or video.parent / "cam_intrinsics.txt").resolve()
    if input_mode == "auto":
        if depth0.is_file() != intrinsics.is_file():
            raise ValueError(
                "partial calibration: supply depth.png + cam_intrinsics.txt "
                "or explicitly choose --input-mode rgb-only"
            )
        mode = "rgbd" if depth0.is_file() else "rgb-only"
    else:
        mode = input_mode
    refs = [rgb0, depth0, intrinsics] if mode == "rgbd" else []
    for path in refs:
        if not path.is_file():
            raise FileNotFoundError(f"missing calibration input: {path}")
    signature = {
        "video_sha256": _sha256(video),
        "input_mode": mode,
        "max_frames": int(max_frames) if max_frames is not None else None,
        "references": {str(p): _sha256(p) for p in refs},
        "moge_model": moge_model if mode == "rgb-only" else None,
    }
    metadata_path = out / "prep_metadata.json"
    managed = [out / name for name in ("rgb", "depth", "cam_intrinsics.txt", "prep_metadata.json")]
    if any(p.is_symlink() for p in managed):
        raise ValueError("video preparation refuses symlinked outputs; use a fresh --tag")
    if any(p.exists() for p in managed) and not force:
        if metadata_path.is_file():
            old = json.loads(metadata_path.read_text())
            complete = (
                old.get("input_signature") == signature
                and len(list((out / "rgb").glob("*.png"))) == old.get("num_frames")
                and (out / "rgb/000000.png").is_file()
                and (out / "depth/000000.png").is_file()
                and (out / "cam_intrinsics.txt").is_file()
            )
            if complete:
                print(f"[prep-video] reuse {old['num_frames']} verified {mode} frames in {out}")
                return old["num_frames"]
        raise ValueError(
            "existing preparation is incomplete, legacy, or belongs to different "
            "inputs; use a fresh --tag or --force to rebuild"
        )

    cap = cv2.VideoCapture(str(video))
    try:
        ok, first = cap.read()
        if not ok:
            raise ValueError(f"no frames decoded from {video}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("video has no valid FPS; re-encode with an explicit frame rate")
        if mode == "rgbd":
            reference = cv2.imread(str(rgb0))
            depth = cv2.imread(str(depth0), cv2.IMREAD_UNCHANGED)
            camera = np.loadtxt(intrinsics).reshape(-1)
            if reference is None or depth is None:
                raise ValueError("cannot decode frame-zero RGB/depth calibration")
            height, width = reference.shape[:2]
            if depth.dtype != np.uint16 or depth.shape != (height, width):
                raise ValueError("depth.png must be uint16 millimeters at rgb.png resolution")
            if (
                camera.shape != (6,)
                or not np.isfinite(camera).all()
                or np.any(camera[:2] <= 0)
                or not np.array_equal(camera[4:], [width, height])
            ):
                raise ValueError("intrinsics must contain fx fy cx cy W H matching rgb.png")
        else:
            height, width = first.shape[:2]
            predicted_depth, normalized_k = predict_first_frame(
                cv2.cvtColor(first, cv2.COLOR_BGR2RGB), moge_model, device
            )
            depth, camera = encode_prediction(predicted_depth, normalized_k, height, width)
            reference = first

        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".prep-video-", dir=out.parent) as scratch:
            stage = Path(scratch)
            (stage / "rgb").mkdir()
            (stage / "depth").mkdir()
            count = 0
            frame = first
            while True:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(stage / "rgb" / f"{count:06d}.png"), frame):
                    raise OSError("could not write extracted RGB frame")
                count += 1
                if max_frames is not None and count >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
            if not cv2.imwrite(str(stage / "depth/000000.png"), depth):
                raise OSError("could not write first-frame depth")
            np.savetxt(stage / "cam_intrinsics.txt", camera[None], fmt="%.10f")
            metadata = {
                "schema_version": 1,
                "input_signature": signature,
                "video": str(video),
                "input_mode": mode,
                "num_frames": count,
                "source_fps": fps,
                "output_fps": fps,
                "resampled": False,
                "width": width,
                "height": height,
                "depth_source": "moge2_prediction" if mode == "rgb-only" else "provided",
                "intrinsics_source": "moge2_prediction" if mode == "rgb-only" else "provided",
                "depth_units": "uint16 millimeters; 0 is invalid",
                "depth_valid_fraction": float((depth > 0).mean()),
                "moge_model": moge_model if mode == "rgb-only" else None,
                "note": (
                    "Predicted calibration is not measured RGB-D or ground truth."
                    if mode == "rgb-only"
                    else "Uses caller-provided calibration."
                ),
            }
            (stage / "prep_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
            first_prepared = cv2.imread(str(stage / "rgb/000000.png"))
            mad = float(np.abs(first_prepared.astype(float) - reference.astype(float)).mean())
            if mad > 12:
                print(f"[WARN] video frame zero differs from rgb.png (meanAbs={mad:.2f})")
            out.mkdir(parents=True, exist_ok=True)
            for name in ("rgb", "depth", "cam_intrinsics.txt", "prep_metadata.json"):
                target = out / name
                if target.is_dir():
                    shutil.rmtree(target)
                os.replace(stage / name, target)
    finally:
        cap.release()
    print(
        f"[prep-video] {video.name} -> {out}: {count} {mode} frames @ {width}x{height}, {fps:g} FPS"
    )
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare RGB video with provided calibration or a MoGe-2 predicted reference.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True, help="output cam0 directory")
    parser.add_argument("--rgb0", default=None)
    parser.add_argument("--depth0", default=None)
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--input-mode", choices=("auto", "rgbd", "rgb-only"), default="auto")
    parser.add_argument(
        "--moge-model", default=DEFAULT_MOGE_MODEL, help="HF model ID or local checkpoint"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prep(
        args.video,
        args.out,
        args.rgb0,
        args.depth0,
        args.intrinsics,
        args.max_frames,
        args.input_mode,
        args.moge_model,
        args.device,
        args.force,
    )
