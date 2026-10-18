import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from galatea_recon.pipeline.object_quality import POLICY, atomic_json, file_hash
from galatea_recon.pipeline.observation_gaps import observation_path


def perception_identity(poses, masks=None):
    inputs = {"poses": file_hash(poses)}
    sidecar = observation_path(poses)
    inputs["observations"] = file_hash(sidecar) if sidecar.exists() else None
    inputs["masks"] = file_hash(masks) if masks and Path(masks).exists() else None
    return inputs


def load_fp_confidence(poses, frames):
    path = observation_path(poses)
    if not path.exists():
        return np.ones(frames, np.float32)
    with np.load(path) as data:
        if "fp_confidence" not in data:
            return np.ones(frames, np.float32)
        confidence = np.asarray(data["fp_confidence"][:frames], np.float32)
    if confidence.shape != (frames,) or not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("invalid FP confidence timeline")
    return confidence


def merge_proposals(first, second, destination, limit=9):
    shape, metadata, tracks = None, [], []
    for path in (first, second):
        if not Path(path).exists():
            continue
        with np.load(path) as bank:
            current_shape = bank["shape"]
            if shape is not None and not np.array_equal(shape, current_shape):
                raise ValueError("proposal timelines disagree")
            shape = current_shape
            items = json.loads(str(bank["metadata"].item()))
            for i, item in enumerate(items):
                if len(metadata) >= limit:
                    break
                metadata.append(item)
                tracks.append(bank[f"track_{i}"].copy())
    if shape is None:
        return False
    np.savez_compressed(destination, shape=shape, metadata=np.array(json.dumps(metadata)),
                        **{f"track_{i}": track for i, track in enumerate(tracks)})
    return True


def run(args):
    start = time.monotonic()
    root = Path(__file__).resolve().parents[2]
    for key in ("cam_dir", "mesh", "masks", "depth_dir", "baseline_poses", "fp_dir", "output_dir", "fp_python"):
        setattr(args, key, str(Path(getattr(args, key)).resolve()))
    cam, out = Path(args.cam_dir), Path(args.output_dir)
    if args.max_frames < 1 or args.max_retries < 0 or min(args.est_refine_iter, args.track_refine_iter) < 1:
        raise ValueError("frame/refinement counts must be positive and retry count nonnegative")
    if not 0 <= args.color_trigger <= 1 or not 0 <= args.color_min_gain <= 1:
        raise ValueError("color thresholds must lie in [0, 1]")
    published = {out / name for name in ("poses.npy", "masks.npz", "poses_observations.npz")}
    if Path(args.masks) in published or Path(args.baseline_poses) in published:
        raise ValueError("recovery outputs must be isolated from original input files")
    out.mkdir(parents=True, exist_ok=True)
    inputs = {"policy": POLICY, "source": (perception_identity(args.baseline_poses, args.masks)
              if Path(args.baseline_poses).exists() else dict(poses=None, masks=file_hash(args.masks))),
              "mesh": file_hash(args.mesh), "K": file_hash(cam / "cam_intrinsics.txt"),
              "object_name": args.object_name, "max_frames": args.max_frames,
              "init_depth_mode": args.init_depth_mode, "max_retries": args.max_retries,
              "est_refine_iter": args.est_refine_iter, "track_refine_iter": args.track_refine_iter,
              "color_fallback": not args.no_color_fallback,
              "color_trigger": args.color_trigger, "color_min_gain": args.color_min_gain,
              "modules": {name: file_hash(Path(__file__).with_name(name+".py")) for name in
                          ("object_recovery", "object_quality", "object_recovery_fp", "object_proposals",
                           "fp_runner", "fp_color_fallback", "observation_gaps")}}
    rgb = sorted((cam / "rgb").glob("*.png"))[:args.max_frames]
    if not rgb:
        raise ValueError("no RGB PNG frames found")
    depth = sorted(Path(args.depth_dir).glob("*.npy"))[:len(rgb)]
    inputs["rgb"] = [file_hash(p) for p in rgb]
    inputs["depth"] = [file_hash(p) for p in depth]
    inputs["sensor_depth"] = {p.name: file_hash(p) for p in sorted((cam / "depth").glob("*.png"))}
    metadata_path = cam / "prep_metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    inputs["prep_metadata"] = metadata
    fps = float(metadata.get("output_fps", metadata.get("fps", 24)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid video FPS")
    report_path = out / "recovery.json"
    if report_path.exists() and not args.force:
        previous = json.loads(report_path.read_text())
        if previous.get("inputs") == inputs:
            if all((out / key).exists() and file_hash(out / key) == value
                   for key, value in previous.get("outputs", {}).items()) and previous.get("outputs"):
                print("[object-recovery] using complete matching cache", flush=True)
                return
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    def command(python, module, arguments, log=None):
        if log is not None:
            with open(log, "w") as stream:
                return subprocess.run([python, "-m", "galatea_recon.pipeline."+module] + list(map(str, arguments)),
                                      cwd=str(root), env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([python, "-m", "galatea_recon.pipeline."+module] + list(map(str, arguments)),
                       cwd=str(root), env=env, check=True)

    with np.load(args.masks) as masks:
        first = next((f for f in range(len(rgb)) if np.count_nonzero(masks[f"obj_{f}"]) >= 20), 0)
    proposal = out / "initial_proposals.npz"
    command(sys.executable, "object_proposals", ["--cam-dir", cam, "--masks", args.masks,
            "--object-name", args.object_name, "--frames", first,
            "--out", proposal, "--max-frames", len(rgb)])
    baseline = Path(args.baseline_poses)
    if not baseline.exists():
        baseline = out / "baseline.npy"
        baseline.unlink(missing_ok=True)
        baseline_log = out / "baseline.log"
        try:
            command(args.fp_python, "fp_runner", ["--cam_dir", cam, "--mesh", args.mesh,
                "--fp_dir", args.fp_dir, "--init_mode", "register", "--fill-missing",
                "--sam2_masks", args.masks, "--depth_mode", "npy", "--depth_dir", args.depth_dir,
                "--init_depth_mode", args.init_depth_mode, "--max_frames", len(rgb),
                "--est_refine_iter", args.est_refine_iter, "--track_refine_iter", args.track_refine_iter,
                "--out", baseline], log=baseline_log)
        except subprocess.CalledProcessError:
            if "object is unobserved in every frame" not in baseline_log.read_text():
                raise
            print("[object-recovery] original segmentation cannot initialize FP; trying alternatives", flush=True)
        if baseline.exists() and not args.no_color_fallback:
            from galatea_recon.pipeline.fp_color_fallback import score_pose_coverage, needs_color_retry, choose_candidate
            import torch
            gray = score_pose_coverage(cam, args.mesh, baseline, args.masks)
            torch.cuda.empty_cache()
            if needs_color_retry(gray["mean_coverage"], args.color_trigger):
                color_path = out / "baseline_color.npy"
                color_path.unlink(missing_ok=True)
                color = None
                try:
                    command(args.fp_python, "fp_runner", ["--cam_dir", cam, "--mesh", args.mesh,
                        "--fp_dir", args.fp_dir, "--init_mode", "register", "--fill-missing",
                        "--sam2_masks", args.masks, "--depth_mode", "npy", "--depth_dir", args.depth_dir,
                        "--init_depth_mode", args.init_depth_mode, "--max_frames", len(rgb),
                        "--est_refine_iter", args.est_refine_iter, "--track_refine_iter", args.track_refine_iter,
                        "--bake_color", "auto", "--out", color_path], log=out / "baseline_color.log")
                    color = score_pose_coverage(cam, args.mesh, color_path, args.masks)
                except (subprocess.CalledProcessError, ValueError, RuntimeError) as error:
                    if "out of memory" in str(error).lower():
                        raise
                    print("[object-recovery] optional color retry failed; retaining gray: %s" % error, flush=True)
                finally:
                    torch.cuda.empty_cache()
                color_score = color["mean_coverage"] if color is not None else None
                selected, reason = choose_candidate(gray["mean_coverage"], color_score,
                    trigger_coverage=args.color_trigger, min_gain=args.color_min_gain)
                atomic_json(out / "baseline_color_selection.json", dict(selected=selected, reason=reason,
                            gray=gray["mean_coverage"], color=color_score))
                if selected == "color":
                    baseline = color_path
    cfg = dict(cam_dir=args.cam_dir, mesh=args.mesh, depth_dir=args.depth_dir, masks=args.masks,
               baseline_poses=str(baseline), fp_dir=args.fp_dir, proposals=str(proposal),
               max_frames=len(rgb), fps=fps, max_retries=args.max_retries,
               init_depth_mode=args.init_depth_mode, est_refine_iter=args.est_refine_iter,
               track_refine_iter=args.track_refine_iter)
    chosen, chosen_quality = None, None
    for iteration in range(2):
        attempt = out / ("attempt_%d" % iteration)
        attempt.mkdir(exist_ok=True)
        cfg.update(output=str(attempt / "poses.npy"), selected_masks=str(attempt / "masks.npz"),
                   report=str(attempt / "quality.json"))
        atomic_json(attempt / "config.json", cfg)
        (attempt / "quality.json").unlink(missing_ok=True)
        try:
            command(args.fp_python, "object_recovery_fp", ["--config", attempt / "config.json"])
        except subprocess.CalledProcessError:
            if not (attempt / "quality.json").exists():
                raise
            failed = json.loads((attempt / "quality.json").read_text())
            if failed.get("state") != "failed":
                raise
        quality = json.loads((attempt / "quality.json").read_text())
        if quality["state"] == "complete":
            from galatea_recon.pipeline.object_quality import aggregate
            if chosen is None or aggregate(quality["final"]) >= aggregate(chosen_quality["final"])-.01:
                chosen, chosen_quality = attempt, quality
        requests = quality["segmentation_requests"]
        if iteration or not requests:
            break
        second = out / "retry_proposals.npz"
        command(sys.executable, "object_proposals", ["--cam-dir", cam, "--masks", args.masks,
                "--object-name", args.object_name, "--frames", ",".join(map(str, requests)),
                "--out", second, "--max-frames", len(rgb), "--max-tracks", 4])
        merged = out / "all_proposals.npz"
        merge_proposals(proposal, second, merged)
        cfg["proposals"] = str(merged)
    if chosen is None:
        raise RuntimeError("Automatic segmentation and pose retries exhausted without an image-supported pose; see attempt quality reports")
    for name in ("poses.npy", "poses_observations.npz", "masks.npz", "quality.json"):
        temporary = out / (name+".tmp")
        shutil.copyfile(chosen / name, temporary)
        os.replace(temporary, out / name)
    outputs = {name: file_hash(out / name) for name in
               ("poses.npy", "poses_observations.npz", "masks.npz", "quality.json")}
    atomic_json(report_path, dict(policy=POLICY, state="complete", inputs=inputs, outputs=outputs,
                                 elapsed_s=time.monotonic()-start, selected_attempt=chosen.name))
    print("[object-recovery] complete: %s" % out, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Coordinate bounded segmentation/FP retries in their existing environments.")
    for name in ("cam-dir", "mesh", "masks", "depth-dir", "baseline-poses", "fp-dir", "fp-python", "object-name", "output-dir"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--max-frames", type=int, default=999999)
    parser.add_argument("--init-depth-mode", choices=("npy", "measured"), default="npy")
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--est-refine-iter", type=int, default=5)
    parser.add_argument("--track-refine-iter", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-color-fallback", action="store_true")
    parser.add_argument("--color-trigger", type=float, default=.75)
    parser.add_argument("--color-min-gain", type=float, default=.05)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
