#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import itertools
import json
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    os.replace(temporary, path)


def _points(path: Path, count: int = 3000) -> np.ndarray:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = list(loaded.geometry.values())
        if not geometries:
            raise RuntimeError(f"mesh has no geometry: {path}")
        loaded = trimesh.util.concatenate(
            [value for value in geometries if isinstance(value, trimesh.Trimesh)]
        )
    vertices = np.asarray(loaded.vertices, dtype=np.float64)
    if not len(vertices):
        raise RuntimeError(f"mesh has no vertices: {path}")
    if isinstance(loaded, trimesh.Trimesh) and len(loaded.faces):
        return np.asarray(trimesh.sample.sample_surface(loaded, count, seed=0)[0], dtype=np.float64)
    random = np.random.RandomState(0)
    return vertices[random.choice(len(vertices), min(count, len(vertices)), replace=False)]


def _pca_basis(points: np.ndarray) -> np.ndarray:
    covariance = np.cov(points.T)
    values, vectors = np.linalg.eigh(covariance)
    basis = vectors[:, np.argsort(values)[::-1]]
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    return basis


def _proper_axis_maps() -> list[np.ndarray]:
    result: list[np.ndarray] = []
    for permutation in itertools.permutations(range(3)):
        matrix = np.eye(3)[:, permutation]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            candidate = matrix @ np.diag(signs)
            if np.linalg.det(candidate) > 0.5:
                result.append(candidate)
    return result


def _mesh_frame_alignment(known_mesh: Path, reconstructed_mesh: Path) -> dict[str, Any]:
    known = _points(known_mesh)
    reconstructed = _points(reconstructed_mesh)
    known_center = (known.min(axis=0) + known.max(axis=0)) * 0.5
    reconstructed_center = (reconstructed.min(axis=0) + reconstructed.max(axis=0)) * 0.5
    known_centered = known - known_center
    reconstructed_centered = reconstructed - reconstructed_center
    known_radius = float(np.sqrt(np.mean(np.sum(known_centered**2, axis=1))))
    reconstructed_radius = float(np.sqrt(np.mean(np.sum(reconstructed_centered**2, axis=1))))
    if min(known_radius, reconstructed_radius) <= 1e-12:
        raise RuntimeError("degenerate mesh for local-frame alignment")
    known_normalized = known_centered / known_radius
    reconstructed_normalized = reconstructed_centered / reconstructed_radius
    known_basis = _pca_basis(known_normalized)
    reconstructed_basis = _pca_basis(reconstructed_normalized)
    tree_reconstructed = cKDTree(reconstructed_normalized)
    candidates: list[tuple[float, np.ndarray]] = []
    for axis_map in _proper_axis_maps():
        rotation = reconstructed_basis @ axis_map @ known_basis.T
        moved = known_normalized @ rotation.T
        forward = tree_reconstructed.query(moved, k=1)[0]
        backward = cKDTree(moved).query(reconstructed_normalized, k=1)[0]
        score = float(0.5 * (np.mean(forward) + np.mean(backward)))
        candidates.append((score, rotation))
    score, rotation = min(candidates, key=lambda value: value[0])
    return {
        "rotation": rotation,
        "known_center": known_center,
        "reconstructed_center": reconstructed_center,
        "normalized_symmetric_chamfer": score,
        "known_radius_m": known_radius,
        "reconstructed_radius_native": reconstructed_radius,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dai_mesh_alignment(
    clip: dict[str, Any],
    known_mesh: Path,
    reconstructed_mesh: Path,
) -> dict[str, Any]:
    metadata_path = clip.get("mesh_alignment_metadata")
    if not metadata_path:
        return _mesh_frame_alignment(known_mesh, reconstructed_mesh)
    metadata = json.loads(Path(metadata_path).read_text())
    if metadata.get("protocol_id") != "doas-known-mesh-aligned-cache-v1":
        raise RuntimeError(f"unexpected DO AS I DO alignment protocol: {metadata_path}")
    if metadata.get("known_metric_mesh_sha256") != _sha256(known_mesh):
        raise RuntimeError(f"known mesh changed after alignment: {known_mesh}")
    if metadata.get("native_rgb_reconstructed_mesh_sha256") != _sha256(reconstructed_mesh):
        raise RuntimeError(
            f"native reconstructed mesh changed after alignment: {reconstructed_mesh}"
        )
    return {
        "rotation": np.asarray(metadata["rotation"], dtype=np.float64),
        "known_center": np.asarray(metadata["known_center"], dtype=np.float64),
        "reconstructed_center": np.asarray(
            metadata["reconstructed_center"],
            dtype=np.float64,
        ),
        "normalized_symmetric_chamfer": float(metadata["normalized_symmetric_chamfer"]),
        "known_radius_m": float(metadata["known_radius_m"]),
        "reconstructed_radius_native": float(metadata["reconstructed_radius_native"]),
    }


def _orthogonalize(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    result = left @ right
    if np.linalg.det(result) < 0:
        left[:, -1] *= -1
        result = left @ right
    return result


def _compose_known_mesh_poses(
    native_poses: np.ndarray,
    native_scales: np.ndarray,
    alignment: dict[str, Any],
) -> np.ndarray:
    output = np.full((len(native_poses), 4, 4), np.nan, dtype=np.float64)
    align_rotation = alignment["rotation"]
    known_center = alignment["known_center"]
    reconstructed_center = alignment["reconstructed_center"]
    for index, (pose, scale) in enumerate(zip(native_poses, native_scales)):
        if not np.isfinite(pose).all() or not np.isfinite(scale):
            continue
        native_rotation = _orthogonalize(pose[:3, :3])
        output[index] = np.eye(4)
        output[index, :3, :3] = native_rotation @ align_rotation
        local_offset = float(scale) * reconstructed_center - align_rotation @ known_center
        output[index, :3, 3] = pose[:3, 3] + native_rotation @ local_offset
    return output.astype(np.float32)


def _ego_hand(
    data: dict[str, Any],
    frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.full((frames, 21, 3), np.nan, dtype=np.float32)
    vertices = np.full((frames, 778, 3), np.nan, dtype=np.float32)
    for frame_index, frame in enumerate(data.get("frame_data", [])[:frames]):
        joints = frame.get("joints_3d_pred") or []
        frame_vertices = frame.get("vertices_3d") or []
        sides = frame.get("hand_is_right") or []
        for hand_index, is_right in enumerate(sides):
            if is_right and hand_index < len(joints):
                candidate = np.asarray(joints[hand_index], dtype=np.float32)
                if candidate.shape == (21, 3) and np.isfinite(candidate).all():
                    result[frame_index] = candidate
                if hand_index < len(frame_vertices):
                    candidate_vertices = np.asarray(
                        frame_vertices[hand_index],
                        dtype=np.float32,
                    )
                    if (
                        candidate_vertices.shape == (778, 3)
                        and np.isfinite(candidate_vertices).all()
                    ):
                        vertices[frame_index] = candidate_vertices
                break
    faces = np.asarray(data.get("mano_faces"), dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise RuntimeError(f"invalid EgoInfinity MANO faces: {faces.shape}")
    return result, vertices, faces


def _save_hand_geometry(
    canonical_hand_output: Path,
    joints: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    _atomic_npy(canonical_hand_output, joints)
    _atomic_npy(canonical_hand_output.with_name("gr_hsz_handverts.npy"), vertices)
    _atomic_npy(canonical_hand_output.with_name("mano_right_faces.npy"), faces)


def _canonicalize_ego(clip: dict[str, Any]) -> dict[str, Any]:
    native_output = Path(clip["native_output"])
    with gzip.open(native_output, "rb") as stream:
        data = pickle.load(stream)
    hand_output, hand_vertices, hand_faces = _ego_hand(data, int(clip["frames"]))
    _save_hand_geometry(
        Path(clip["canonical_hand_output"]),
        hand_output,
        hand_vertices,
        hand_faces,
    )
    pose_info = data.get("pose_track_info") or {}
    if not pose_info:
        raise RuntimeError("pose_track_info is empty")
    object_id = (
        0
        if 0 in pose_info
        else ("0" if "0" in pose_info else sorted(pose_info, key=lambda value: int(value))[0])
    )
    info = pose_info[object_id]
    native_poses = np.asarray(info.get("T_seq"), dtype=np.float64)
    if native_poses.ndim != 3 or native_poses.shape[1:] != (4, 4):
        raise RuntimeError(f"invalid EgoInfinity T_seq shape: {native_poses.shape}")
    mesh_info = data.get("sam3_mesh_info") or {}
    meta = mesh_info.get(object_id, mesh_info.get(str(object_id), {}))
    reconstructed_mesh = Path(meta.get("ply_path", ""))
    if not reconstructed_mesh.exists():
        reconstructed_mesh = native_output.parent / "sam3_meshes" / f"obj_{object_id}.ply"
    alignment = _mesh_frame_alignment(
        Path(clip["contract_inputs"]["known_metric_mesh"]),
        reconstructed_mesh,
    )
    scale = float(info.get("scale_correction", meta.get("canonical_scale", 1.0)))
    scales = np.full(len(native_poses), scale, dtype=np.float64)
    object_output = _compose_known_mesh_poses(native_poses, scales, alignment)
    _atomic_npy(Path(clip["canonical_object_output"]), object_output)
    return {
        "object_id": int(object_id),
        "native_object": str(native_output),
        "native_reconstructed_mesh": str(reconstructed_mesh),
        "scale_correction": scale,
        "alignment_score": alignment["normalized_symmetric_chamfer"],
        "object_frames_valid": int(np.isfinite(object_output).all(axis=(1, 2)).sum()),
        "hand_frames_valid": int(np.isfinite(hand_output).all(axis=(1, 2)).sum()),
        "hand_mesh_frames_valid": int(np.isfinite(hand_vertices).all(axis=(1, 2)).sum()),
    }


def _layout_poses(path: Path, frames: int) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text())
    poses = np.full((frames, 4, 4), np.nan, dtype=np.float64)
    scales = np.full(frames, np.nan, dtype=np.float64)
    optimized_scale = (payload.get("translation_scale_optimization") or {}).get("mesh_scale")

    def uniform_scale(value: Any) -> float:
        values = np.asarray(value, dtype=np.float64).reshape(-1)
        if not len(values) or not np.isfinite(values).all():
            raise RuntimeError(f"invalid DO AS I DO mesh scale: {value}")
        if len(values) > 1 and not np.allclose(values, values[0], rtol=1e-5, atol=1e-8):
            raise RuntimeError(f"non-uniform DO AS I DO mesh scale is unsupported: {value}")
        return float(values.mean())

    default_scale = uniform_scale(
        optimized_scale if optimized_scale is not None else payload.get("mesh_scale", 1.0)
    )
    if payload.get("frame") != "camera_frame":
        raise RuntimeError("DO AS I DO layout is not in camera_frame")
    for fallback_index, item in enumerate(payload.get("objects", [])):
        index = int(item.get("frame_index", item.get("frame_idx", fallback_index)))
        if not 0 <= index < frames:
            continue
        local = item.get("local_to_scene") or {}
        quaternion = local.get("quat_wxyz_camera_frame")
        translation = local.get("translation_camera_frame")
        if quaternion is None or translation is None:
            continue
        w, x, y, z = (float(value) for value in quaternion)
        poses[index] = np.eye(4)
        poses[index, :3, :3] = Rotation.from_quat([x, y, z, w]).as_matrix()
        poses[index, :3, 3] = np.asarray(translation, dtype=np.float64)
        scales[index] = (
            default_scale
            if optimized_scale is not None
            else uniform_scale(local.get("scale", default_scale) or default_scale)
        )
    scales[~np.isfinite(scales)] = default_scale
    return poses, scales


def _dai_hand(path: Path, frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = np.full((frames, 21, 3), np.nan, dtype=np.float32)
    vertices = np.full((frames, 778, 3), np.nan, dtype=np.float32)
    with np.load(path) as data:
        joints = np.asarray(data["right_joints"], dtype=np.float32)
        native_vertices = np.asarray(data["right_vertices"], dtype=np.float32)
        faces = np.asarray(data["right_faces"], dtype=np.int32)
        count = min(frames, len(joints), len(native_vertices))
        result[:count] = joints[:count]
        vertices[:count] = native_vertices[:count]
        if "right_valid" in data:
            valid = np.asarray(data["right_valid"][:count]).astype(bool).reshape(-1)
            result[:count][~valid] = np.nan
            vertices[:count][~valid] = np.nan
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise RuntimeError(f"invalid DO AS I DO MANO faces: {faces.shape}")
    return result, vertices, faces


def _canonicalize_dai(clip: dict[str, Any]) -> dict[str, Any]:
    native_object = Path(clip["native_object_output"])
    reconstructed_mesh = Path(clip["native_reconstructed_mesh"])
    native_hand = Path(clip["native_hand_output"])
    frames = int(clip["frames"])
    hand_output, hand_vertices, hand_faces = _dai_hand(native_hand, frames)
    _save_hand_geometry(
        Path(clip["canonical_hand_output"]),
        hand_output,
        hand_vertices,
        hand_faces,
    )
    poses, scales = _layout_poses(native_object, frames)
    known_mesh = Path(clip["contract_inputs"]["known_metric_mesh"])
    alignment = _dai_mesh_alignment(clip, known_mesh, reconstructed_mesh)
    object_output = _compose_known_mesh_poses(poses, scales, alignment)
    _atomic_npy(Path(clip["canonical_object_output"]), object_output)
    return {
        "native_object": str(native_object),
        "native_hand": str(native_hand),
        "native_reconstructed_mesh": str(reconstructed_mesh),
        "alignment_score": alignment["normalized_symmetric_chamfer"],
        "object_frames_valid": int(np.isfinite(object_output).all(axis=(1, 2)).sum()),
        "hand_frames_valid": int(np.isfinite(hand_output).all(axis=(1, 2)).sum()),
        "hand_mesh_frames_valid": int(np.isfinite(hand_vertices).all(axis=(1, 2)).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runlist", type=Path, required=True)
    parser.add_argument("--only", default=None)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    runlist = json.loads(args.runlist.read_text())
    baseline = runlist["baseline"]
    selected = set(args.only.split(",")) if args.only else None
    records: list[dict[str, Any]] = []
    failures = 0
    for clip in runlist["clips"]:
        if selected is not None and clip["clip_id"] not in selected:
            continue
        record: dict[str, Any] = {"clip_id": clip["clip_id"]}
        try:
            details = (
                _canonicalize_ego(clip)
                if baseline == "egoinfinity_official"
                else _canonicalize_dai(clip)
            )
            record.update({"status": "ok", **details})
        except Exception as error:
            failures += 1
            record.update(
                {"status": "missing_or_failed", "error": f"{type(error).__name__}: {error}"}
            )
        records.append(record)
    report = (
        args.runlist.parent.parent
        / "canonical_predictions"
        / args.runlist.stem
        / "canonicalization_report.json"
    )
    _atomic_json(
        report,
        {
            "baseline": baseline,
            "source_runlist": str(args.runlist.resolve()),
            "coordinate_convention": runlist["canonical_output_contract"],
            "clips": records,
        },
    )
    print(
        json.dumps(
            {
                "baseline": baseline,
                "clips": len(records),
                "failures": failures,
                "report": str(report),
            },
            indent=2,
        )
    )
    if args.strict and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
