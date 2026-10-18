from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from galatea_recon.pipeline.observation_gaps import object_observations, observation_path


DEFAULT_TRIGGER_COVERAGE = 0.75
DEFAULT_MIN_GAIN = 0.05


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coverage_summary(coverage: np.ndarray) -> dict[str, Any]:
    coverage = np.asarray(coverage, dtype=np.float64)
    if coverage.ndim != 1 or len(coverage) == 0:
        raise ValueError("coverage must be a non-empty 1D array")
    if not np.isfinite(coverage).all():
        raise ValueError("coverage contains NaN or Inf")
    return {
        "n": int(len(coverage)),
        "mean_coverage": float(coverage.mean()),
        "min_coverage": float(coverage.min()),
        "frames_coverage_lt_0_5": int((coverage < 0.5).sum()),
        "coverage": coverage.tolist(),
    }


def needs_color_retry(
    gray_mean_coverage: float,
    trigger_coverage: float = DEFAULT_TRIGGER_COVERAGE,
) -> bool:
    return float(gray_mean_coverage) < float(trigger_coverage)


def choose_candidate(
    gray_mean_coverage: float,
    color_mean_coverage: float | None,
    *,
    trigger_coverage: float = DEFAULT_TRIGGER_COVERAGE,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> tuple[str, str]:
    gray = float(gray_mean_coverage)
    if not needs_color_retry(gray, trigger_coverage):
        return "gray", "gray_coverage_above_retry_threshold"
    if color_mean_coverage is None:
        return "gray", "color_retry_not_available"
    gain = float(color_mean_coverage) - gray
    if gain > float(min_gain) + 1e-12:
        return "color", "color_coverage_gain_above_threshold"
    return "gray", "color_coverage_gain_not_high_enough"


def score_pose_coverage(
    cam_dir: str | Path,
    mesh_path: str | Path,
    poses_path: str | Path,
    masks_path: str | Path,
    *,
    device: str = "cuda",
) -> dict[str, Any]:
    from galatea_recon.pipeline.masks import render_object_mask

    cam_dir = Path(cam_dir).resolve()
    mesh_path = Path(mesh_path).resolve()
    poses_path = Path(poses_path).resolve()
    masks_path = Path(masks_path).resolve()

    intrinsics = np.loadtxt(cam_dir / "cam_intrinsics.txt").reshape(-1)
    if len(intrinsics) != 6:
        raise ValueError("cam_intrinsics.txt must contain fx fy cx cy width height")
    fx, fy, cx, cy, width, height = intrinsics.tolist()
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    width, height = int(width), int(height)

    poses = np.load(poses_path)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4) or not np.isfinite(poses).all():
        raise ValueError(f"expected finite (T,4,4) poses, got {poses.shape}")

    mesh = trimesh.load(mesh_path, force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError(f"empty mesh: {mesh_path}")

    coverage = []
    scored_frames = []
    excluded_frames = []
    with np.load(masks_path) as masks:
        observed = object_observations(masks, len(poses))
        for frame in range(len(poses)):
            key = f"obj_{frame}"
            if key not in masks.files:
                raise ValueError(f"missing {key} in {masks_path}")
            reference = np.asarray(masks[key], dtype=bool)
            if reference.shape != (height, width):
                raise ValueError(
                    f"{key} mask shape {reference.shape} does not match {(height, width)}"
                )
            if not observed[frame]:
                excluded_frames.append(frame)
                continue
            projected = render_object_mask(
                vertices,
                faces,
                K,
                poses[frame],
                (width, height),
                device=device,
                dilate_px=0,
            ).astype(bool)
            reference_area = int(reference.sum())
            intersection = int((reference & projected).sum())
            coverage.append(intersection / reference_area if reference_area else 0.0)
            scored_frames.append(frame)

    result = coverage_summary(np.asarray(coverage))
    result.update(
        {
            "version": 1,
            "cam_dir": str(cam_dir),
            "mesh_path": str(mesh_path),
            "poses_path": str(poses_path),
            "masks_path": str(masks_path),
            "scored_frames": scored_frames,
            "excluded_unobserved_frames": excluded_frames,
        }
    )
    return result


def select_and_write(
    gray_score_path: str | Path,
    gray_poses_path: str | Path,
    output_poses_path: str | Path,
    decision_path: str | Path,
    *,
    color_score_path: str | Path | None = None,
    color_poses_path: str | Path | None = None,
    trigger_coverage: float = DEFAULT_TRIGGER_COVERAGE,
    min_gain: float = DEFAULT_MIN_GAIN,
) -> dict[str, Any]:
    gray_score_path = Path(gray_score_path).resolve()
    gray_poses_path = Path(gray_poses_path).resolve()
    output_poses_path = Path(output_poses_path).resolve()
    decision_path = Path(decision_path).resolve()
    gray_score = json.loads(gray_score_path.read_text())
    gray_mean = float(gray_score["mean_coverage"])

    color_score = None
    color_mean = None
    if color_score_path is not None or color_poses_path is not None:
        if color_score_path is None or color_poses_path is None:
            raise ValueError("color score and color poses must be provided together")
        color_score_path = Path(color_score_path).resolve()
        color_poses_path = Path(color_poses_path).resolve()
        color_score = json.loads(color_score_path.read_text())
        color_mean = float(color_score["mean_coverage"])

    selected, reason = choose_candidate(
        gray_mean,
        color_mean,
        trigger_coverage=trigger_coverage,
        min_gain=min_gain,
    )
    source_poses = gray_poses_path if selected == "gray" else color_poses_path
    if source_poses is None:
        raise RuntimeError("selected color without a color pose candidate")
    _atomic_copy(source_poses, output_poses_path)
    source_observations = observation_path(source_poses)
    target_observations = observation_path(output_poses_path)
    if source_observations.exists():
        _atomic_copy(source_observations, target_observations)
    elif target_observations.exists():
        target_observations.unlink()

    decision = {
        "version": 1,
        "mode": "coverage_gated_color_fallback",
        "selected": selected,
        "reason": reason,
        "trigger_coverage": float(trigger_coverage),
        "min_color_gain": float(min_gain),
        "gray": {
            "mean_coverage": gray_mean,
            "n": int(gray_score["n"]),
            "score_path": str(gray_score_path),
            "poses_path": str(gray_poses_path),
        },
        "color": (
            None
            if color_score is None
            else {
                "mean_coverage": color_mean,
                "gain_over_gray": color_mean - gray_mean,
                "n": int(color_score["n"]),
                "score_path": str(color_score_path),
                "poses_path": str(color_poses_path),
            }
        ),
        "output_poses_path": str(output_poses_path),
        "output_sha256": _sha256(output_poses_path),
    }
    _atomic_json(decision_path, decision)
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coverage-gated solid-color fallback for plain FoundationPose.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score pose coverage against SAM2 masks")
    score_parser.add_argument("--cam-dir", required=True)
    score_parser.add_argument("--mesh", required=True)
    score_parser.add_argument("--poses", required=True)
    score_parser.add_argument("--masks", required=True)
    score_parser.add_argument("--device", default="cuda")
    score_parser.add_argument("--out", required=True)

    retry_parser = subparsers.add_parser(
        "needs-retry", help="exit 0 only when color retry is needed"
    )
    retry_parser.add_argument("--score", required=True)
    retry_parser.add_argument("--trigger-coverage", type=float, default=DEFAULT_TRIGGER_COVERAGE)

    select_parser = subparsers.add_parser(
        "select", help="atomically promote the selected pose candidate"
    )
    select_parser.add_argument("--gray-score", required=True)
    select_parser.add_argument("--gray-poses", required=True)
    select_parser.add_argument("--color-score")
    select_parser.add_argument("--color-poses")
    select_parser.add_argument("--trigger-coverage", type=float, default=DEFAULT_TRIGGER_COVERAGE)
    select_parser.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN)
    select_parser.add_argument("--out-poses", required=True)
    select_parser.add_argument("--out-decision", required=True)

    args = parser.parse_args()
    if args.command == "score":
        result = score_pose_coverage(
            args.cam_dir,
            args.mesh,
            args.poses,
            args.masks,
            device=args.device,
        )
        _atomic_json(Path(args.out).resolve(), result)
        print(
            f"[fp-color] coverage mean={result['mean_coverage']:.3f} "
            f"min={result['min_coverage']:.3f} "
            f"frames<0.5={result['frames_coverage_lt_0_5']}/{result['n']}"
        )
    elif args.command == "needs-retry":
        score = json.loads(Path(args.score).read_text())
        retry = needs_color_retry(score["mean_coverage"], args.trigger_coverage)
        print(
            f"[fp-color] gray coverage={float(score['mean_coverage']):.3f}, "
            f"trigger<{args.trigger_coverage:.3f}: {'retry color' if retry else 'keep gray'}"
        )
        raise SystemExit(0 if retry else 1)
    else:
        decision = select_and_write(
            args.gray_score,
            args.gray_poses,
            args.out_poses,
            args.out_decision,
            color_score_path=args.color_score,
            color_poses_path=args.color_poses,
            trigger_coverage=args.trigger_coverage,
            min_gain=args.min_gain,
        )
        color = decision["color"]
        suffix = "" if color is None else f", color={color['mean_coverage']:.3f}"
        print(
            f"[fp-color] selected={decision['selected']} "
            f"gray={decision['gray']['mean_coverage']:.3f}{suffix} "
            f"reason={decision['reason']}"
        )


if __name__ == "__main__":
    main()
