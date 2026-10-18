#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def transformed_centroid(vertices: np.ndarray, pose: np.ndarray) -> np.ndarray:
    centroid = vertices.mean(axis=0)
    return pose[:3, :3] @ centroid + pose[:3, 3]


def canonicalize_clip(
    clip: dict[str, Any],
    native_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    clip_id = clip["clip_id"]
    frames = int(clip["num_frames"])
    native = native_root / clip_id
    phase = output_root / "_phase3" / clip_id
    pose_path = native / "obj_mesh" / "object_mesh_poses_grasp.npy"
    object_poses = np.full((frames, 4, 4), np.nan, dtype=np.float32)
    object_error = None
    try:
        candidate = np.load(pose_path, allow_pickle=False).astype(np.float32)
        if candidate.shape != (frames, 4, 4):
            raise ValueError(f"pose shape {candidate.shape} != {(frames, 4, 4)}")
        object_poses[:] = candidate
    except Exception as error:
        object_error = f"{type(error).__name__}: {error}"

    hand_joints = np.full((frames, 21, 3), np.nan, dtype=np.float32)
    hand_vertices = np.full((frames, 778, 3), np.nan, dtype=np.float32)
    faces: np.ndarray | None = None
    frame_records: list[dict[str, Any]] = []
    known_mesh = trimesh.load(clip["mesh"], force="mesh", process=False)
    known_vertices = np.asarray(known_mesh.vertices, dtype=np.float64)
    rgb_names = [path.stem for path in sorted((native / "rgb").glob("*.png"))]
    if len(rgb_names) != frames:
        rgb_names = [f"{index:06d}" for index in range(frames)]

    for frame_index, name in enumerate(rgb_names):
        params_path = native / "human_hand" / f"{name}_mano_data.npy"
        mesh_paths = sorted((native / "human_hand").glob(f"{name}_right_3d_*.obj"))
        if not params_path.exists() or not mesh_paths:
            continue
        try:
            payload = np.load(params_path, allow_pickle=True).item()
            meshes = [trimesh.load(path, force="mesh", process=False) for path in mesh_paths]
            if np.isfinite(object_poses[frame_index]).all():
                object_center = transformed_centroid(known_vertices, object_poses[frame_index])
                selected = min(
                    range(len(meshes)),
                    key=lambda index: float(
                        np.linalg.norm(
                            np.asarray(meshes[index].vertices).mean(axis=0) - object_center
                        )
                    ),
                )
            else:
                selected = 0
            keys = list(payload)
            key = f"person_{selected}"
            if key not in payload:
                if selected >= len(keys):
                    raise KeyError(f"no parameter entry for selected hand {selected}: {keys}")
                key = keys[selected]
            joints = np.asarray(payload[key]["mano_retarget_hand_joints"], dtype=np.float32)
            vertices = np.asarray(meshes[selected].vertices, dtype=np.float32)
            candidate_faces = np.asarray(meshes[selected].faces, dtype=np.int32)
            if joints.shape != (21, 3) or not np.isfinite(joints).all():
                raise ValueError(f"invalid hand joints {joints.shape}")
            if vertices.shape != (778, 3) or not np.isfinite(vertices).all():
                raise ValueError(f"invalid MANO vertices {vertices.shape}")
            if candidate_faces.ndim != 2 or candidate_faces.shape[1] != 3:
                raise ValueError(f"invalid MANO faces {candidate_faces.shape}")
            if faces is None:
                faces = candidate_faces
            elif not np.array_equal(faces, candidate_faces):
                raise ValueError("MANO topology changed between frames")
            hand_joints[frame_index] = joints
            hand_vertices[frame_index] = vertices
            frame_records.append(
                {
                    "frame": frame_index,
                    "rgb_name": name,
                    "selected_hand": selected,
                    "parameter_key": key,
                    "candidate_hands": len(meshes),
                }
            )
        except Exception as error:
            frame_records.append(
                {
                    "frame": frame_index,
                    "rgb_name": name,
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    if faces is None:
        faces = np.empty((0, 3), dtype=np.int32)
    atomic_npy(phase / "gr_hsz.npy", object_poses)
    atomic_npy(phase / "gr_hsz_hand.npy", hand_joints)
    atomic_npy(phase / "gr_hsz_handverts.npy", hand_vertices)
    atomic_npy(phase / "mano_right_faces.npy", faces)
    record = {
        "clip_id": clip_id,
        "dataset": clip["dataset"],
        "frames": frames,
        "native_object_pose": str(pose_path.resolve()),
        "object_frames_valid": int(np.isfinite(object_poses).all(axis=(1, 2)).sum()),
        "hand_frames_valid": int(np.isfinite(hand_joints).all(axis=(1, 2)).sum()),
        "mano_mesh_frames_valid": int(np.isfinite(hand_vertices).all(axis=(1, 2)).sum()),
        "mano_faces": int(len(faces)),
        "object_error": object_error,
        "hand_selection": "official closest hand-to-posed-object centroid rule",
        "joint_source": "official mano_retarget_hand_joints",
        "mesh_source": "official camera-space right_3d MANO OBJ",
        "frame_records": frame_records,
    }
    atomic_json(phase / "VIDEOMANIP_CANONICAL_STATUS.json", record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--clip-id")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    clips = json.loads(args.manifest.read_text())["clips"]
    if args.clip_id:
        clips = [clip for clip in clips if clip["clip_id"] == args.clip_id]
        if not clips:
            raise SystemExit(f"clip not in manifest: {args.clip_id}")
    native_root = args.native_root.resolve()
    output_root = args.output_root.resolve()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.workers == 1 or len(clips) == 1:
        records = [canonicalize_clip(clip, native_root, output_root) for clip in clips]
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(clips))) as pool:
            futures = [
                pool.submit(canonicalize_clip, clip, native_root, output_root) for clip in clips
            ]
            records = [future.result() for future in futures]
    report = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "protocol": "official VideoManip known-mesh reconstruction; no GT read by adapter",
        "clips": records,
        "summary": {
            "total": len(records),
            "object_complete": sum(
                record["object_frames_valid"] == record["frames"] for record in records
            ),
            "hand_complete": sum(
                record["hand_frames_valid"] == record["frames"] for record in records
            ),
            "object_any": sum(record["object_frames_valid"] > 0 for record in records),
            "hand_any": sum(record["hand_frames_valid"] > 0 for record in records),
        },
    }
    atomic_json(args.output_root / "canonicalization_report.json", report)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
