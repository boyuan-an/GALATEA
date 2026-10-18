import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from galatea_recon.pipeline.fp_runner import _ensure_texture, depth_provider, load_K
from galatea_recon.pipeline.object_quality import (
    POLICY, EvidenceRenderer, accept_candidate, aggregate, atomic_json, mask_box, quality_confidence,
)
from galatea_recon.pipeline.observation_gaps import (
    fill_poses, mask_observed, object_observations, observation_path, pose_observed,
)


def initial_observation_frame(metrics, tracks):
    for f, metric in enumerate(metrics):
        if metric["assessable"]:
            return f
    for f in range(len(metrics)):
        if any(mask_observed(track[f]) for track in tracks):
            return f
    return 0


class Recovery:
    def __init__(self, config):
        self.cfg = config
        self.started = time.monotonic()
        self.cam = Path(config["cam_dir"])
        self.rgb_paths = sorted((self.cam / "rgb").glob("*.png"))[:config["max_frames"]]
        self.n = len(self.rgb_paths)
        self.K, self.width, self.height = load_K(str(self.cam))
        self.rgb_cache, self.depth_cache = {}, {}
        self.base = (np.load(config["baseline_poses"])[:self.n].copy()
                     if Path(config["baseline_poses"]).exists() else np.full((self.n, 4, 4), np.nan))
        if self.base.shape != (self.n, 4, 4):
            raise ValueError("baseline poses do not match the RGB timeline")
        with np.load(config["masks"]) as masks:
            initial = np.stack([masks[f"obj_{f}"] for f in range(self.n)]).astype(bool)
            self.hands = np.stack([(masks[f"fg_{f}"] & ~initial[f])
                                   if f"fg_{f}" in masks else np.zeros_like(initial[f])
                                   for f in range(self.n)])
        self.tracks = [initial]
        self.track_info = [dict(source="original", frame=0)]
        if config.get("proposals") and Path(config["proposals"]).exists():
            with np.load(config["proposals"]) as bank:
                shape = tuple(bank["shape"].astype(int))
                if shape != initial.shape:
                    raise ValueError("segmentation proposal timeline/size mismatch")
                self.track_info += json.loads(str(bank["metadata"].item()))
                for index in range(len(self.track_info)-1):
                    track = np.unpackbits(bank[f"track_{index}"], count=int(np.prod(shape))).reshape(shape)
                    self.tracks.append(track.astype(bool))
        fp_dir = str(Path(config["fp_dir"]).resolve())
        sys.path.insert(0, fp_dir)
        os.chdir(fp_dir)
        import torch
        import nvdiffrast.torch as dr
        from estimater import FoundationPose
        self.torch = torch
        mesh = trimesh.load(config["mesh"], force="mesh", process=False)
        _ensure_texture(mesh, config["mesh"])
        self.mesh = mesh
        self.renderer = EvidenceRenderer(mesh, self.K, self.width, self.height)
        self.est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                                  mesh=mesh, glctx=dr.RasterizeCudaContext(), debug=0,
                                  debug_dir=str(Path(config["output"]).parent / "fp_debug"))
        self.origin = self.est.get_tf_to_centered_mesh().detach().cpu().numpy()
        self.inverse_origin = np.linalg.inv(self.origin)
        self.events, self.requests = [], []
        self.fps = float(config.get("fps", 24))
        self.lookahead = max(3, min(12, round(.4*self.fps)))
        self.confirm = max(2, round(.08*self.fps))
        self.cooldown = max(3, round(.30*self.fps))
        self.max_retries = int(config.get("max_retries", 8))
        self.selected = np.zeros(self.n, np.int32)
        self.motion_seed = None

    def rgb(self, frame):
        if frame not in self.rgb_cache:
            image = cv2.cvtColor(cv2.imread(str(self.rgb_paths[frame])), cv2.COLOR_BGR2RGB)
            self.rgb_cache[frame] = cv2.resize(image, (self.width, self.height))
        return self.rgb_cache[frame]

    def depth(self, frame):
        if frame not in self.depth_cache:
            mode = self.cfg.get("init_depth_mode", "npy") if frame == 0 else "npy"
            self.depth_cache[frame] = depth_provider(mode, str(self.cam), self.cfg["depth_dir"],
                frame, self.rgb_paths[frame].name, self.height, self.width, 1000.0)
        return self.depth_cache[frame]

    def measured(self, frame):
        metadata = self.cam / "prep_metadata.json"
        if not metadata.exists():
            return None
        info = json.loads(metadata.read_text())
        if (info.get("input_mode") == "rgb-only" or info.get("depth_source") == "predicted"
                or not (info.get("input_mode") == "rgbd"
                        or info.get("depth_source") in ("measured", "sensor"))):
            return None
        path = self.cam / "depth" / self.rgb_paths[frame].name
        if not path.exists():
            return None
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            return None
        return cv2.resize(depth.astype(np.float32)/1000, (self.width, self.height),
                          interpolation=cv2.INTER_NEAREST)

    def evaluate(self, frame, poses, track):
        from galatea_recon.pipeline.object_quality import score_masks
        metrics = []
        for pose in poses:
            if pose_observed(pose):
                metrics.extend(self.renderer.evaluate([pose], self.tracks[track][frame], self.rgb(frame),
                                                       self.hands[frame], self.measured(frame)))
            else:
                metrics.append(score_masks(np.zeros_like(self.tracks[track][frame]),
                                            self.tracks[track][frame], self.rgb(frame)))
        return metrics

    def follow(self, pose, frame):
        self.est.pose_last = self.torch.as_tensor(pose @ self.inverse_origin,
                                                  dtype=self.torch.float32, device="cuda")
        result = self.est.track_one(rgb=self.rgb(frame), depth=self.depth(frame), K=self.K,
                                   iteration=int(self.cfg.get("track_refine_iter", 2)))
        return np.asarray(result).reshape(4, 4)

    def register(self, frame, track):
        mask = self.tracks[track][frame]
        if not mask_observed(mask) or np.count_nonzero(mask & (self.depth(frame) > .001)) < 4:
            return []
        self.est.register(K=self.K, rgb=self.rgb(frame), depth=self.depth(frame), ob_mask=mask,
                          iteration=int(self.cfg.get("est_refine_iter", 5)))
        poses = self.est.poses.detach().cpu().numpy() @ self.origin
        chosen = []
        for pose in poses[:40]:
            if not pose_observed(pose):
                continue
            if all(Rotation.from_matrix(pose[:3, :3] @ p[:3, :3].T).magnitude() > np.deg2rad(25)
                   for p in chosen):
                chosen.append(pose)
            if len(chosen) >= 6:
                break
        if self.motion_seed is not None and pose_observed(self.motion_seed):
            seed = self.motion_seed
            projected = self.renderer.render([seed])[0] > 0
            old_box = mask_box(projected)
            target_box = mask_box(self.renderer.resize(mask.astype(np.uint8), True))
            if old_box is not None and target_box is not None:
                old_size = np.linalg.norm(old_box[2:]-old_box[:2])
                new_size = np.linalg.norm(target_box[2:]-target_box[:2])
                ratio = float(np.clip(old_size/max(new_size, 1), .5, 2))
                center_local = np.asarray(self.mesh.bounds).mean(0)
                old_center = seed[:3, :3] @ center_local + seed[:3, 3]
                delta_uv = (target_box[:2]+target_box[2:]-old_box[:2]-old_box[2:])/2
                shifted = old_center.copy()
                shifted[:2] += delta_uv / np.array([self.renderer.K[0, 0], self.renderer.K[1, 1]]) * old_center[2]
                for scale in (1.0, ratio, .9*ratio, 1.1*ratio):
                    candidate = seed.copy()
                    candidate[:3, 3] = shifted*scale - candidate[:3, :3] @ center_local
                    chosen.append(candidate)
        return chosen

    def verify(self, frame, pose, track):
        poses, metrics = [pose], [self.evaluate(frame, [pose], track)[0]]
        for f in range(frame+1, min(self.n, frame+self.lookahead+1)):
            if not mask_observed(self.tracks[track][f]):
                break
            p = self.follow(poses[-1], f)
            if not pose_observed(p):
                break
            poses.append(p)
            metrics.append(self.evaluate(f, [p], track)[0])
        return poses, metrics

    def hypotheses(self, frame, preferred=0, identity_anchor=None):
        hypotheses = []
        order = [preferred] + sorted([t for t in range(len(self.tracks)) if t != preferred],
                                     key=lambda t: abs(self.track_info[t]["frame"]-frame))
        for track in order[:6]:
            if identity_anchor is not None and track != preferred:
                previous = self.tracks[preferred][identity_anchor]
                alternate = self.tracks[track][identity_anchor]
                intersection = np.count_nonzero(previous & alternate)
                support = intersection / max(1, min(previous.sum(), alternate.sum()))
                if support < .5:
                    self.events.append(dict(frame=int(frame), action="different_target_rejected",
                                            track=int(track), anchor=int(identity_anchor)))
                    continue
            try:
                poses = self.register(frame, track)
                if not poses:
                    continue
                metrics = self.evaluate(frame, poses, track)
                for idx in np.argsort([-m["score"] for m in metrics])[:3]:
                    hypotheses.append((metrics[idx]["score"], track, poses[idx]))
            except (ValueError, RuntimeError) as error:
                if "out of memory" in str(error).lower():
                    raise
                self.events.append(dict(frame=int(frame), action="registration_failed",
                                        track=int(track), error=str(error)))
        hypotheses.sort(key=lambda x: x[0], reverse=True)
        verified = []
        for _, track, pose in hypotheses[:4]:
            poses, metrics = self.verify(frame, pose, track)
            verified.append((aggregate(metrics), track, poses, metrics))
        verified.sort(key=lambda x: x[0], reverse=True)
        return verified

    def run(self):
        result = self.base.copy()
        baseline_metrics = [self.evaluate(f, [result[f]], 0)[0] for f in range(self.n)]
        initial = initial_observation_frame(baseline_metrics, self.tracks)
        candidates = self.hypotheses(initial)
        current, retrack = 0, False
        if pose_observed(self.base[initial]):
            self.motion_seed = self.base[initial].copy()
        if candidates:
            _, track, poses, metrics = candidates[0]
            old = baseline_metrics[initial:initial+len(metrics)]
            accepted = accept_candidate(old, metrics)
            self.events.append(dict(frame=initial, action="initial_selection", accepted=accepted,
                old_score=aggregate(old), new_score=aggregate(metrics), candidate=int(track),
                tested=len(candidates), selected_source=self.track_info[track]))
            if accepted:
                current, retrack = track, True
                result[initial] = poses[0]
                next_pose = poses[0]
                for f in range(initial-1, -1, -1):
                    if not mask_observed(self.tracks[current][f]):
                        break
                    proposal = self.follow(next_pose, f)
                    metric = self.evaluate(f, [proposal], current)[0]
                    if not metric["assessable"] or metric["score"] < .43:
                        break
                    result[f], next_pose = proposal, proposal
            elif aggregate(old) < .50:
                self.requests.append(initial)
        else:
            self.requests.append(initial)
        history, bad_run, last_retry, retries = [], 0, -999, 0
        last_good = initial
        for f in range(self.n):
            self.selected[f] = current
            if retrack and f > initial and mask_observed(self.tracks[current][f]):
                try:
                    proposal = self.follow(result[f-1], f)
                    if pose_observed(proposal):
                        result[f] = proposal
                except RuntimeError as error:
                    if "out of memory" in str(error).lower():
                        raise
            metric = self.evaluate(f, [result[f]], current)[0]
            history.append(metric)
            recent = [m["score"] for m in history[max(0, f-12):f] if m["assessable"]]
            relative_bad = len(recent) >= 3 and metric["score"] < np.median(recent)-.18
            bad = metric["assessable"] and (metric["score"] < .43 or (relative_bad and metric["score"] < .62))
            bad_run = bad_run+1 if bad else 0
            if metric["assessable"] and metric["score"] >= .53:
                last_good = f
                self.motion_seed = result[f].copy()
            if (f > initial and bad_run >= self.confirm and f-last_retry >= self.cooldown
                    and retries < self.max_retries and f < self.n-2):
                retries += 1
                last_retry = f
                candidates = self.hypotheses(f, current, identity_anchor=last_good)
                if candidates:
                    _, track, poses, metrics = candidates[0]
                    old_poses, old = self.verify(f, result[f], current)
                    if len(old) < len(metrics):
                        old += [self.evaluate(t, [self.base[t]], current)[0]
                                for t in range(f+len(old), f+len(metrics))]
                    old = old[:len(metrics)]
                    accepted = accept_candidate(old, metrics, min_gain=.06)
                    self.events.append(dict(frame=f, action="reacquire", accepted=accepted,
                        old_score=aggregate(old), new_score=aggregate(metrics), candidate=int(track),
                        confirmation_old=[m["score"] for m in old],
                        confirmation_new=[m["score"] for m in metrics]))
                    if accepted:
                        result[f] = poses[0]
                        current, retrack = track, True
                        self.selected[f] = current
                        history[f] = metrics[0]
                        next_pose = result[f]
                        for t in range(f-1, max(last_good, f-round(self.fps))-1, -1):
                            proposal = self.follow(next_pose, t)
                            if not pose_observed(proposal):
                                break
                            m = self.evaluate(t, [proposal], current)[0]
                            if not m["assessable"] or m["score"] < max(.43, history[t]["score"]+.025):
                                break
                            result[t], self.selected[t], history[t], next_pose = proposal, current, m, proposal
                        bad_run = 0
                    elif len(self.requests) < 3:
                        self.requests.append(f)
                elif len(self.requests) < 3:
                    self.requests.append(f)
            if f % 40 == 0:
                print("[recovery] frame %d/%d score %.3f track %d" % (f, self.n, metric["score"], current), flush=True)
        confidence = np.asarray([quality_confidence(m) for m in history], np.float32)
        mask_valid = object_observations({f"obj_{f}": self.tracks[int(self.selected[f])][f]
                                              for f in range(self.n)}, self.n)
        anchors = confidence >= .35
        estimated = ~anchors
        if anchors.any():
            result = fill_poses(result, anchors)
        if not anchors.any() or not np.isfinite(result).all():
            atomic_json(self.cfg["report"], dict(policy=POLICY, state="failed",
                reason="No image-supported initial pose", events=self.events,
                segmentation_requests=sorted(set(self.requests + [self.n//2]))[:3]))
            raise RuntimeError("No image-supported object pose after automatic recovery")
        confidence[estimated] = 0
        final_metrics = list(history)
        for f in np.flatnonzero(estimated):
            final_metrics[f] = self.evaluate(int(f), [result[f]], int(self.selected[f]))[0]
        output = Path(self.cfg["output"])
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, result)
        np.savez(observation_path(output), object_observed=mask_valid,
                 pose_observed=anchors, pose_estimated=estimated, fp_confidence=confidence,
                 reacquired_frames=np.array([e["frame"] for e in self.events
                    if e["action"] == "reacquire" and e.get("accepted")], np.int64))
        store = {}
        for f in range(self.n):
            mask = self.tracks[int(self.selected[f])][f]
            store[f"obj_{f}"] = mask
            store[f"fg_{f}"] = mask | self.hands[f]
        np.savez_compressed(self.cfg["selected_masks"], **store)
        atomic_json(self.cfg["report"], dict(policy=POLICY, state="complete", frames=self.n,
            elapsed_s=time.monotonic()-self.started, baseline=baseline_metrics, final=final_metrics,
            tracked_before_completion=history,
            events=self.events, segmentation_requests=sorted(set(self.requests))[:3],
            selected_tracks=self.selected.tolist(), fp_confidence=confidence.tolist(),
            estimated_frames=np.flatnonzero(estimated).tolist(),
            candidate_tracks=len(self.tracks), geometry_quality="image_evidence_only"))


def main():
    parser = argparse.ArgumentParser(
        description="Bounded FP hypothesis verification and image-triggered track recovery.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    Recovery(config).run()


if __name__ == "__main__":
    main()
