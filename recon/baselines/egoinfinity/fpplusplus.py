#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

FPPLUSPLUS_COMMIT = "58aa7154de23ae86d542366858fc6bcca3b747b0"
PROTOCOL_ID = "egoinfinity_known_mesh_aligned_solid_fpplusplus_v3"
SOLID_MESH_COLOR_RGB = [0, 159, 237]
SURFACE_BACKFILL_PROTOCOL_ID = "egoinfinity_native_surface_backfill_v1"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy", delete=False) as stream:
        temporary = Path(stream.name)
        np.save(stream, value)
    os.replace(temporary, path)


def _load_pkl(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rb") as stream:
        return pickle.load(stream)


def _mapping_get(mapping: dict, key: int) -> Any:
    if key in mapping:
        return mapping[key]
    return mapping.get(str(key))


def _unpack_mask(frame: dict[str, Any], oid: int) -> np.ndarray:
    obj = _mapping_get(frame.get("sam3_obj_data") or {}, oid)
    if not isinstance(obj, dict):
        raise RuntimeError(f"object {oid} has no SAM3 data on the initialization frame")
    packed, shape = obj.get("mask_packed"), obj.get("mask_shape")
    if packed is None or shape is None:
        raise RuntimeError(f"object {oid} has no packed SAM3 mask")
    height, width = int(shape[0]), int(shape[1])
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    if bits.size < height * width:
        raise RuntimeError(f"object {oid} has a truncated packed mask")
    mask = bits[: height * width].reshape(height, width).astype(bool)
    if int(mask.sum()) < 16:
        raise RuntimeError(f"object {oid} initialization mask is empty/degenerate")
    return mask


def _objects(data: dict[str, Any]) -> list[tuple[int, dict, dict]]:
    mesh_info = data.get("sam3_mesh_info") or {}
    pose_info = data.get("pose_track_info") or {}
    result = []
    for raw_oid, meta in mesh_info.items():
        try:
            oid = int(raw_oid)
        except (TypeError, ValueError):
            continue
        info = _mapping_get(pose_info, oid)
        if isinstance(meta, dict) and isinstance(info, dict) and info.get("T_seq") is not None:
            result.append((oid, meta, info))
    return sorted(result, key=lambda item: item[0])


def _clip_dirs(
    root: Path,
    only: str | None,
    worker: int | None = None,
    workers: int | None = None,
) -> list[Path]:
    allow = {item.strip() for item in only.split(",") if item.strip()} if only else None
    clips = sorted(
        path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")
    )
    if allow is not None:
        clips = [path for path in clips if path.name in allow]
        missing = allow - {path.name for path in clips}
        if missing:
            raise RuntimeError(f"unknown clips under {root}: {sorted(missing)}")
    if worker is not None or workers is not None:
        if worker is None or workers is None or workers <= 0 or not 0 <= worker < workers:
            raise RuntimeError("worker sharding requires 0 <= worker < workers")
        clips = [path for index, path in enumerate(clips) if index % workers == worker]
    return clips


def _testcase_dir(fp_root: Path, clip_id: str, oid: int) -> Path:
    return fp_root / (clip_id if oid == 0 else f"{clip_id}__oid{oid}")


def _resolve_surface_mesh(clip_dir: Path, meta: dict, oid: int) -> Path:
    candidates = []
    configured = meta.get("surface_mesh_path")
    if configured:
        configured_path = Path(configured)
        candidates.append(
            configured_path if configured_path.is_absolute() else clip_dir / configured_path
        )
    candidates.append(clip_dir / "sam3_meshes" / f"obj_{oid}_surface.glb")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(
        f"{clip_dir.name} oid={oid}: missing SAM3D triangle surface; run backfill first"
    )


def _resolve_reconstructed_mesh(clip_dir: Path, meta: dict, oid: int) -> Path:
    candidates = []
    configured = meta.get("ply_path")
    if configured:
        configured_path = Path(configured)
        candidates.append(
            configured_path if configured_path.is_absolute() else clip_dir / configured_path
        )
    candidates.append(clip_dir / "sam3_meshes" / f"obj_{oid}.ply")
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise RuntimeError(f"{clip_dir.name} oid={oid}: missing native SAM3D reconstruction")


def _validate_surface_mesh(path: Path) -> dict[str, Any]:
    import trimesh

    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        meshes = [
            geometry
            for geometry in loaded.geometry.values()
            if hasattr(geometry, "faces") and len(geometry.faces)
        ]
        if not meshes:
            raise RuntimeError(f"surface GLB contains no triangle geometry: {path}")
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = loaded
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 100:
        raise RuntimeError(f"surface GLB has too few valid vertices: {path}")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 100:
        raise RuntimeError(f"surface GLB has too few triangles: {path}")
    if not np.isfinite(vertices).all():
        raise RuntimeError(f"surface GLB has non-finite vertices: {path}")
    extents = np.ptp(vertices, axis=0)
    if not 0.05 <= float(extents.max()) <= 10.0:
        raise RuntimeError(f"surface GLB canonical extent is implausible: {extents.tolist()}")
    return {
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "canonical_extents": [float(value) for value in extents],
        "sha256": _sha256_file(path),
    }


def _decode_rgb(frame: dict[str, Any]) -> np.ndarray:
    import cv2

    encoded = frame.get("img_rgb")
    if not encoded:
        raise RuntimeError("frame has no encoded RGB image")
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode RGB image")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _quat_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    first = first / max(float(np.linalg.norm(first)), 1e-12)
    second = second / max(float(np.linalg.norm(second)), 1e-12)
    return float(np.degrees(2.0 * np.arccos(np.clip(abs(float(first @ second)), 0.0, 1.0))))


def _backfill_one(clip_dir: Path, data: dict[str, Any], oid: int, meta: dict) -> dict[str, Any]:
    target = clip_dir / "sam3_meshes" / f"obj_{oid}_surface.glb"
    sidecar = target.with_suffix(".json")
    if target.is_file():
        stats = _validate_surface_mesh(target)
        return {"state": "already_complete", "mesh": str(target), **stats}

    frames = data.get("frame_data") or []
    init_frame = int(meta.get("init_frame", 0))
    if not 0 <= init_frame < len(frames):
        raise RuntimeError(f"{clip_dir.name} oid={oid}: invalid init_frame={init_frame}")
    rgb = _decode_rgb(frames[init_frame])
    mask = _unpack_mask(frames[init_frame], oid)
    target.parent.mkdir(parents=True, exist_ok=True)
    audit_ply = target.with_name(f"obj_{oid}_surface_source_gs.ply")

    ego_repo = Path(__file__).resolve().parents[2] / "3rd" / "EgoInfinity"
    sys.path.insert(0, str(ego_repo))
    from egoinfinity.pipeline.sam3d_client import reconstruct_object

    result = reconstruct_object(
        rgb=rgb,
        mask=mask,
        out_ply_path=str(audit_ply),
        out_mesh_path=str(target),
        seed=42,
        quality="tier1",
        timeout=1800.0,
    )
    if Path(result.mesh_path or "").resolve() != target.resolve():
        raise RuntimeError(
            f"SAM3D worker did not return the requested surface mesh: {result.mesh_path}"
        )

    expected_t = np.asarray(meta.get("canonical_translation"), dtype=np.float64).reshape(-1)
    expected_q = np.asarray(meta.get("canonical_rotation_quat"), dtype=np.float64).reshape(-1)
    expected_s = float(meta.get("canonical_scale"))
    translation_error = float(np.max(np.abs(result.translation - expected_t)))
    rotation_error = _quat_angle_deg(result.rotation_quat.astype(np.float64), expected_q)
    scale_rel_error = abs(float(result.scale) - expected_s) / max(abs(expected_s), 1e-9)
    if translation_error > 1e-3 or rotation_error > 0.25 or scale_rel_error > 1e-3:
        raise RuntimeError(
            f"{clip_dir.name} oid={oid}: deterministic SAM3D backfill disagrees with "
            f"the original inference (t={translation_error:.6g}m, "
            f"R={rotation_error:.4g}deg, scale_rel={scale_rel_error:.6g})"
        )
    stats = _validate_surface_mesh(target)
    record = {
        "protocol_id": SURFACE_BACKFILL_PROTOCOL_ID,
        "created_at": _timestamp(),
        "clip_id": clip_dir.name,
        "oid": oid,
        "init_frame": init_frame,
        "sam3d_seed": 42,
        "same_inference_transform_check": {
            "translation_max_abs_m": translation_error,
            "rotation_deg": rotation_error,
            "scale_relative": scale_rel_error,
        },
        "surface_mesh": str(target.resolve()),
        "audit_gaussian_ply": str(audit_ply.resolve()),
        "surface_stats": stats,
    }
    _atomic_json(sidecar, record)
    return {"state": "backfilled", "mesh": str(target), **stats}


def backfill(args: argparse.Namespace) -> None:
    worker = None
    socket_path = None
    if args.spawn_worker:
        ego_repo = Path(__file__).resolve().parents[2] / "3rd" / "EgoInfinity"
        sys.path.insert(0, str(ego_repo))
        socket_path = f"/tmp/ei_fp_surface_{os.environ.get('USER', 'user')}_{os.getpid()}.sock"
        os.environ["SAM3D_WORKER_SOCKET"] = socket_path
        from egoinfinity.pipeline.sam3d_runner import spawn_sam3d_worker

        worker, _ = spawn_sam3d_worker(socket_path, args.worker_log)
    records = []
    try:
        for clip_dir in _clip_dirs(args.favorites_root, args.only, args.worker, args.workers):
            pkl_path = clip_dir / "pipeline_result.pkl.gz"
            if not pkl_path.is_file():
                continue
            data = _load_pkl(pkl_path)
            for oid, meta, _ in _objects(data):
                records.append(
                    {
                        "clip_id": clip_dir.name,
                        "oid": oid,
                        **_backfill_one(clip_dir, data, oid, meta),
                    }
                )
    finally:
        if worker is not None:
            worker.terminate()
            try:
                worker.wait(timeout=20)
            except subprocess.TimeoutExpired:
                worker.kill()
            for suffix in ("", ".ready"):
                try:
                    Path(str(socket_path) + suffix).unlink()
                except FileNotFoundError:
                    pass
    report = {
        "protocol_id": SURFACE_BACKFILL_PROTOCOL_ID,
        "created_at": _timestamp(),
        "objects": records,
    }
    if args.report:
        _atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _input_digest(
    frames: list[dict[str, Any]],
    start: int,
    mask: np.ndarray,
    mesh_path: Path,
    intrinsics: list[list[float]],
    scale: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(
        json.dumps({"start": start, "K": intrinsics, "scale": scale}, sort_keys=True).encode()
    )
    digest.update(np.ascontiguousarray(mask).tobytes())
    with mesh_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    for frame in frames[start:]:
        digest.update(frame["img_rgb"])
        digest.update(frame["depth_png"])
    return digest.hexdigest()


def _write_if_changed(path: Path, content: bytes) -> None:
    if path.is_file() and path.read_bytes() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def _known_mesh_alignment_matrix(
    known_mesh: Path,
    reconstructed_mesh: Path,
    native_scale_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    from baselines.common.canonical import _mesh_frame_alignment

    alignment = _mesh_frame_alignment(known_mesh, reconstructed_mesh)
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    known_center = np.asarray(alignment["known_center"], dtype=np.float64)
    reconstructed_center = np.asarray(
        alignment["reconstructed_center"],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = native_scale_m * reconstructed_center - rotation @ known_center
    return matrix, alignment


def _bake_known_metric_mesh(
    known_mesh: Path,
    reconstructed_mesh: Path,
    native_scale_m: float,
    output_mesh: Path,
) -> dict[str, Any]:
    import trimesh

    if not known_mesh.is_file():
        raise RuntimeError(f"known metric mesh does not exist: {known_mesh}")
    loaded = trimesh.load(known_mesh, process=False)
    if isinstance(loaded, trimesh.Scene):
        geometries = [
            value
            for value in loaded.dump(concatenate=False)
            if isinstance(value, trimesh.Trimesh) and len(value.faces)
        ]
        if not geometries:
            raise RuntimeError(f"known metric mesh contains no triangles: {known_mesh}")
        loaded = geometries[0] if len(geometries) == 1 else trimesh.util.concatenate(geometries)
    if not isinstance(loaded, trimesh.Trimesh) or not len(loaded.faces):
        raise RuntimeError(f"known metric mesh contains no triangles: {known_mesh}")
    matrix, alignment = _known_mesh_alignment_matrix(
        known_mesh,
        reconstructed_mesh,
        native_scale_m,
    )
    baked = loaded.copy()
    baked.apply_transform(matrix)
    solid_rgba = np.asarray([*SOLID_MESH_COLOR_RGB, 255], dtype=np.uint8)
    baked.visual = trimesh.visual.ColorVisuals(
        mesh=baked,
        vertex_colors=np.tile(solid_rgba, (len(baked.vertices), 1)),
    )
    exported = baked.export(file_type="glb")
    if not isinstance(exported, bytes):
        raise RuntimeError(f"failed to export aligned known mesh: {known_mesh}")
    _write_if_changed(output_mesh, exported)
    output_stats = _validate_surface_mesh(output_mesh)
    return {
        "alignment_policy": (
            "geometry-only 24 proper PCA axis maps plus bidirectional "
            "normalized Chamfer; known metric size preserved"
        ),
        "known_metric_mesh": str(known_mesh.resolve()),
        "known_metric_mesh_sha256": _sha256_file(known_mesh),
        "native_sam3d_reconstruction": str(reconstructed_mesh.resolve()),
        "native_sam3d_reconstruction_sha256": _sha256_file(reconstructed_mesh),
        "known_to_sam3d_metric_matrix": matrix.tolist(),
        "rotation": np.asarray(alignment["rotation"]).tolist(),
        "known_center_m": np.asarray(alignment["known_center"]).tolist(),
        "sam3d_center_native": np.asarray(alignment["reconstructed_center"]).tolist(),
        "native_scale_m_per_unit": native_scale_m,
        "normalized_symmetric_chamfer": float(alignment["normalized_symmetric_chamfer"]),
        "fpplusplus_apply_scale": 1.0,
        "appearance_policy": ("source textures and materials discarded; uniform solid color only"),
        "solid_color_rgb": list(SOLID_MESH_COLOR_RGB),
        "baked_mesh": str(output_mesh.resolve()),
        "baked_mesh_stats": output_stats,
        "uses_gt_pose": False,
        "uses_gt_mask": False,
        "uses_future_depth": False,
    }


def _prepare_one(
    clip_dir: Path,
    data: dict[str, Any],
    oid: int,
    meta: dict,
    info: dict,
    fp_root: Path,
    known_mesh: Path,
) -> dict[str, Any]:
    frames = data.get("frame_data") or []
    init_frame = int(meta.get("init_frame", 0))
    if not 0 <= init_frame < len(frames):
        raise RuntimeError(f"{clip_dir.name} oid={oid}: invalid init_frame={init_frame}")
    mask = _unpack_mask(frames[init_frame], oid)
    reconstructed_mesh = _resolve_reconstructed_mesh(clip_dir, meta, oid)
    focal = float(data.get("dp_focal", 0))
    cx, cy = float(data.get("cx", 0)), float(data.get("cy", 0))
    if focal <= 0 or cx <= 0 or cy <= 0:
        raise RuntimeError(f"{clip_dir.name}: invalid EgoInfinity intrinsics")
    intrinsics = [[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]]
    native_scale = float(info.get("scale_correction") or meta.get("canonical_scale") or 0)
    if not 0.005 <= native_scale <= 5.0:
        raise RuntimeError(
            f"{clip_dir.name} oid={oid}: implausible SAM3D metric scale {native_scale}"
        )

    testcase = _testcase_dir(fp_root, clip_dir.name, oid)
    testcase.mkdir(parents=True, exist_ok=True)
    (testcase / "color").mkdir(exist_ok=True)
    (testcase / "depth").mkdir(exist_ok=True)
    (testcase / "mesh").mkdir(exist_ok=True)
    tracking_mesh = testcase / "mesh" / "object.glb"
    alignment = _bake_known_metric_mesh(
        known_mesh.resolve(),
        reconstructed_mesh,
        native_scale,
        tracking_mesh,
    )
    digest = _input_digest(frames, init_frame, mask, tracking_mesh, intrinsics, 1.0)
    metadata_path = testcase / "egoinfinity_fpplusplus.json"
    old_metadata = None
    if metadata_path.is_file():
        old_metadata = json.loads(metadata_path.read_text())
        if old_metadata.get("input_digest") != digest and (testcase / "pose.npy").exists():
            raise RuntimeError(
                f"{testcase}: inputs changed after pose generation; refusing stale FP++ output"
            )
    for local_index, frame in enumerate(frames[init_frame:]):
        if not frame.get("img_rgb") or not frame.get("depth_png"):
            raise RuntimeError(f"{clip_dir.name}: frame {init_frame + local_index} lacks RGB/depth")
        _write_if_changed(testcase / "color" / f"{local_index}.jpg", frame["img_rgb"])
        _write_if_changed(testcase / "depth" / f"{local_index}.png", frame["depth_png"])
    import cv2

    ok, encoded_mask = cv2.imencode(".png", mask.astype(np.uint8) * 255)
    if not ok:
        raise RuntimeError(f"{clip_dir.name} oid={oid}: failed to encode initialization mask")
    _write_if_changed(testcase / "0_mask.png", bytes(encoded_mask))
    _write_if_changed(testcase / "oid.txt", f"{oid}\n".encode())

    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _timestamp(),
        "fpplusplus_commit": FPPLUSPLUS_COMMIT,
        "clip_id": clip_dir.name,
        "oid": oid,
        "total_frames": len(frames),
        "init_frame": init_frame,
        "tracking_frames": len(frames) - init_frame,
        "cam_K": intrinsics,
        "mesh_apply_scale_m": 1.0,
        "known_mesh_alignment": alignment,
        "input_digest": digest,
        "inputs": {
            "rgb": "EgoInfinity frame_data.img_rgb",
            "depth": "EgoInfinity predicted metric depth_png",
            "mask": "EgoInfinity SAM3 mask at sam3_mesh_info.init_frame",
            "mesh": (
                "benchmark known metric mesh, geometry-only aligned into "
                "the native SAM3D local axes, with source textures/materials "
                "discarded and a uniform solid color"
            ),
            "intrinsics": "EgoInfinity dp_focal/cx/cy",
            "allowed_benchmark_inputs_used": ["known_metric_mesh"],
            "forbidden_benchmark_inputs_used": [],
        },
        "fpplusplus_parameters": {
            "activate_2d_tracker": True,
            "activate_kalman_filter": True,
            "kf_measurement_noise_scale": 0.05,
            "est_refine_iter": 10,
            "track_refine_iter": 5,
            "force_apply_color": True,
            "apply_color_rgb": list(SOLID_MESH_COLOR_RGB),
            "parameter_source": (
                "official FP++ README 'Inference with your own data' command "
                "plus parser defaults at the pinned commit"
            ),
        },
    }
    if (
        old_metadata is not None
        and old_metadata.get("protocol_id") == PROTOCOL_ID
        and old_metadata.get("input_digest") == digest
    ):
        for completion_key in (
            "completed_at",
            "raw_pose_sha256",
            "full_pose_sha256",
            "pose_validation",
        ):
            if completion_key in old_metadata:
                metadata[completion_key] = old_metadata[completion_key]
    _atomic_json(metadata_path, metadata)
    return {"state": "prepared", "testcase": str(testcase), "input_digest": digest}


def prepare(args: argparse.Namespace) -> None:
    runlist = json.loads(args.runlist.read_text())
    if runlist.get("baseline") != "egoinfinity_official":
        raise RuntimeError(f"unexpected EgoInfinity runlist: {args.runlist}")
    known_mesh_by_clip = {}
    for clip in runlist.get("clips", []):
        clip_id = str(clip["clip_id"])
        known_mesh = (clip.get("contract_inputs") or {}).get("known_metric_mesh")
        if not known_mesh:
            raise RuntimeError(f"{clip_id}: runlist has no known_metric_mesh")
        known_mesh_by_clip[clip_id] = Path(known_mesh)
    records = []
    for clip_dir in _clip_dirs(args.favorites_root, args.only, args.worker, args.workers):
        if clip_dir.name not in known_mesh_by_clip:
            raise RuntimeError(f"{clip_dir.name}: missing from benchmark runlist")
        pkl_path = clip_dir / "pipeline_result.pkl.gz"
        if not pkl_path.is_file():
            records.append({"clip_id": clip_dir.name, "state": "no_pipeline_result"})
            continue
        data = _load_pkl(pkl_path)
        for oid, meta, info in _objects(data):
            if bool(info.get("is_static_global", False)):
                records.append(
                    {"clip_id": clip_dir.name, "oid": oid, "state": "static_global_no_fp_required"}
                )
                continue
            records.append(
                {
                    "clip_id": clip_dir.name,
                    "oid": oid,
                    **_prepare_one(
                        clip_dir,
                        data,
                        oid,
                        meta,
                        info,
                        args.fp_root,
                        known_mesh_by_clip[clip_dir.name],
                    ),
                }
            )
    report = {"protocol_id": PROTOCOL_ID, "created_at": _timestamp(), "objects": records}
    if args.report:
        _atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _validate_pose_array(pose: np.ndarray, expected_frames: int, label: str) -> dict[str, float]:
    if pose.shape != (expected_frames, 4, 4):
        raise RuntimeError(f"{label}: expected {(expected_frames, 4, 4)}, got {pose.shape}")
    if not np.isfinite(pose).all():
        raise RuntimeError(f"{label}: pose has NaN/Inf")
    bottom_error = float(np.max(np.abs(pose[:, 3] - np.array([0, 0, 0, 1]))))
    rotations = pose[:, :3, :3].astype(np.float64)
    orth_error = float(np.max(np.abs(rotations @ rotations.transpose(0, 2, 1) - np.eye(3))))
    determinants = np.linalg.det(rotations)
    det_error = float(np.max(np.abs(determinants - 1.0)))
    if bottom_error > 1e-4 or orth_error > 2e-3 or det_error > 2e-3:
        raise RuntimeError(
            f"{label}: invalid SE(3), bottom={bottom_error}, orth={orth_error}, det={det_error}"
        )
    translations = pose[:, :3, 3]
    if float(np.max(np.linalg.norm(translations, axis=1))) > 50.0:
        raise RuntimeError(f"{label}: implausible translation")
    return {
        "bottom_row_max_error": bottom_error,
        "rotation_orthogonality_max_error": orth_error,
        "rotation_determinant_max_error": det_error,
    }


def run_testcase(args: argparse.Namespace) -> None:
    testcase = args.testcase.resolve()
    fp_repo = args.fp_repo.resolve()
    metadata_path = testcase / "egoinfinity_fpplusplus.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("protocol_id") != PROTOCOL_ID:
        raise RuntimeError(f"wrong testcase protocol: {metadata_path}")
    actual_commit = subprocess.check_output(
        ["git", "-C", str(fp_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != FPPLUSPLUS_COMMIT or metadata.get("fpplusplus_commit") != actual_commit:
        raise RuntimeError(f"unfrozen FP++ checkout: {actual_commit}")

    source_dir = fp_repo / "src"
    sys.path.insert(0, str(source_dir))
    import obj_pose_track as official_fp

    parameters = metadata["fpplusplus_parameters"]
    official_fp.args = SimpleNamespace(
        apply_scale=float(metadata["mesh_apply_scale_m"]),
        force_apply_color=bool(parameters["force_apply_color"]),
        apply_color=list(parameters["apply_color_rgb"]),
        kf_measurement_noise_scale=float(parameters["kf_measurement_noise_scale"]),
    )
    raw_path = testcase / "pose_from_init.npy"
    official_fp.pose_track(
        str(testcase / "color"),
        str(testcase / "depth"),
        str(testcase / "mesh" / "object.glb"),
        str(testcase / "0_mask.png"),
        np.asarray(metadata["cam_K"], dtype=np.float64),
        str(raw_path),
        None,
        None,
        None,
        int(parameters["est_refine_iter"]),
        int(parameters["track_refine_iter"]),
        bool(parameters["activate_2d_tracker"]),
        bool(parameters["activate_kalman_filter"]),
    )
    if not raw_path.is_file():
        raise RuntimeError(f"official FP++ did not produce {raw_path}")
    raw_pose = np.load(raw_path)
    stats = _validate_pose_array(raw_pose, int(metadata["tracking_frames"]), str(raw_path))
    full_pose = np.full((int(metadata["total_frames"]), 4, 4), np.nan, dtype=np.float32)
    full_pose[int(metadata["init_frame"]) :] = raw_pose.astype(np.float32)
    _atomic_npy(testcase / "pose.npy", full_pose)
    metadata["completed_at"] = _timestamp()
    metadata["raw_pose_sha256"] = _sha256_file(raw_path)
    metadata["full_pose_sha256"] = _sha256_file(testcase / "pose.npy")
    metadata["pose_validation"] = stats
    _atomic_json(metadata_path, metadata)
    print(json.dumps({"state": "complete", "testcase": str(testcase), **stats}, indent=2))


def validate(args: argparse.Namespace) -> None:
    records = []
    failures = []
    for clip_dir in _clip_dirs(args.favorites_root, args.only, args.worker, args.workers):
        pkl_path = clip_dir / "pipeline_result.pkl.gz"
        if not pkl_path.is_file():
            continue
        data = _load_pkl(pkl_path)
        for oid, _meta, info in _objects(data):
            if bool(info.get("is_static_global", False)):
                records.append({"clip_id": clip_dir.name, "oid": oid, "state": "static_global"})
                continue
            testcase = _testcase_dir(args.fp_root, clip_dir.name, oid)
            try:
                metadata = json.loads((testcase / "egoinfinity_fpplusplus.json").read_text())
                pose = np.load(testcase / "pose.npy")
                init_frame = int(metadata["init_frame"])
                if pose.shape != (int(metadata["total_frames"]), 4, 4):
                    raise RuntimeError(f"wrong full pose shape {pose.shape}")
                if init_frame and not np.isnan(pose[:init_frame]).all():
                    raise RuntimeError("pre-init frames must be NaN")
                stats = _validate_pose_array(
                    pose[init_frame:], int(metadata["tracking_frames"]), str(testcase)
                )
                records.append({"clip_id": clip_dir.name, "oid": oid, "state": "valid", **stats})
            except Exception as error:
                failures.append({"clip_id": clip_dir.name, "oid": oid, "error": str(error)})
    report = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _timestamp(),
        "objects": records,
        "failures": failures,
    }
    if args.report:
        _atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _copy_clip_for_bake(source: Path, bake_cache: Path) -> Path:
    destination = bake_cache / "favorites" / source.name
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / "pipeline_result.pkl.gz", destination / "pipeline_result.pkl.gz")
    source_meshes = source / "sam3_meshes"
    destination_meshes = destination / "sam3_meshes"
    if source_meshes.exists() and not destination_meshes.exists():
        destination_meshes.symlink_to(source_meshes.resolve(), target_is_directory=True)
    return destination


def bake(args: argparse.Namespace) -> None:
    validation_args = argparse.Namespace(
        favorites_root=args.favorites_root,
        fp_root=args.fp_root,
        only=args.only,
        report=None,
        worker=None,
        workers=None,
    )
    validate(validation_args)
    source_dirs = _clip_dirs(args.favorites_root, args.only)
    copied_dirs = [
        _copy_clip_for_bake(source, args.bake_cache)
        for source in source_dirs
        if (source / "pipeline_result.pkl.gz").is_file()
    ]
    if not copied_dirs:
        raise RuntimeError("no EgoInfinity pipeline outputs to bake")
    environment = os.environ.copy()
    environment["ACTION100M_CACHE"] = str(args.bake_cache)
    environment["EGOINFINITY_FP_ROOT"] = str(args.fp_root.resolve())
    ego_command = json.loads(args.ego_command) if args.ego_command else [str(args.ego_python)]
    command = [
        *ego_command,
        "-m",
        "egoinfinity.pipeline.post_tracking.bake_fp_pose",
        "--fp_root",
        str(args.fp_root.resolve()),
        f"--only={','.join(path.name for path in copied_dirs)}",
        "--force",
    ]
    completed = subprocess.run(command, cwd=args.ego_repo, env=environment)
    if completed.returncode:
        raise RuntimeError(f"EgoInfinity FP++ bake failed with exit {completed.returncode}")

    records, clip_records, failures = [], [], []
    for source, copied in zip(
        [path for path in source_dirs if (path / "pipeline_result.pkl.gz").is_file()], copied_dirs
    ):
        before = _load_pkl(source / "pipeline_result.pkl.gz")
        after = _load_pkl(copied / "pipeline_result.pkl.gz")
        for oid, _meta, before_info in _objects(before):
            after_info = _mapping_get(after.get("pose_track_info") or {}, oid)
            old_pose = np.asarray(before_info["T_seq"])
            new_pose = np.asarray(after_info["T_seq"])
            if old_pose.shape != new_pose.shape or not np.isfinite(new_pose).all():
                failures.append(
                    {
                        "clip_id": source.name,
                        "oid": oid,
                        "error": "invalid baked pose shape/finiteness",
                    }
                )
                continue
            translation_error = float(np.max(np.abs(old_pose[:, :3, 3] - new_pose[:, :3, 3])))
            if translation_error > 1e-7:
                failures.append(
                    {
                        "clip_id": source.name,
                        "oid": oid,
                        "error": f"orientation-only bake changed t by {translation_error}m",
                    }
                )
                continue
            records.append(
                {
                    "clip_id": source.name,
                    "oid": oid,
                    "state": "baked",
                    "translation_max_abs_error_m": translation_error,
                }
            )
        expected_objects = len(_objects(before))
        history = (after.get("fp_pose_bake") or {}).get("history") or []
        if expected_objects == 0:
            records.append({"clip_id": source.name, "state": "no_object_output"})
        elif not history:
            failures.append({"clip_id": source.name, "error": "missing fp_pose_bake provenance"})
        else:
            latest = history[-1]
            if (
                int(latest.get("n_oid_baked", -1)) != expected_objects
                or int(latest.get("n_oid_missing", -1)) != 0
            ):
                failures.append(
                    {
                        "clip_id": source.name,
                        "error": (
                            f"incomplete bake provenance: baked={latest.get('n_oid_baked')} "
                            f"missing={latest.get('n_oid_missing')} expected={expected_objects}"
                        ),
                    }
                )
            clip_records.append(
                {
                    "clip_id": source.name,
                    "state": (
                        "baked"
                        if not failures or failures[-1].get("clip_id") != source.name
                        else "failed"
                    ),
                    "elapsed_seconds": latest.get("elapsed_seconds"),
                    "fpplusplus_tracking_objects": sum(
                        not bool(info.get("is_static_global", False))
                        for _oid, _meta, info in _objects(before)
                    ),
                }
            )
    report = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _timestamp(),
        "source_root": str(args.favorites_root.resolve()),
        "baked_favorites_root": str((args.bake_cache / "favorites").resolve()),
        "objects": records,
        "clips": clip_records,
        "failures": failures,
    }
    if args.source_runlist or args.output_runlist:
        if not args.source_runlist or not args.output_runlist:
            raise RuntimeError("--source-runlist and --output-runlist must be supplied together")
        runlist = json.loads(args.source_runlist.read_text())
        if runlist.get("baseline") != "egoinfinity_official":
            raise RuntimeError(f"unexpected source baseline: {runlist.get('baseline')}")
        runlist["baseline_variant"] = "egoinfinity_official_fpplusplus_known_mesh_aligned"
        runlist["fpplusplus_protocol"] = {
            "protocol_id": PROTOCOL_ID,
            "commit": FPPLUSPLUS_COMMIT,
            "testcase_root": str(args.fp_root.resolve()),
            "bake_report": str(args.report.resolve()) if args.report else None,
        }
        for clip in runlist.get("clips", []):
            baked_clip = (args.bake_cache / "favorites" / clip["clip_id"]).resolve()
            clip["clip_dir"] = str(baked_clip)
            clip["native_output"] = str(baked_clip / "pipeline_result.pkl.gz")
            if args.canonical_root:
                phase = args.canonical_root.resolve() / "_phase3" / clip["clip_id"]
                clip["canonical_object_output"] = str(phase / "gr_hsz.npy")
                clip["canonical_hand_output"] = str(phase / "gr_hsz_hand.npy")
        _atomic_json(args.output_runlist, runlist)
        report["derived_runlist"] = str(args.output_runlist.resolve())
    if args.report:
        _atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


def _common_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--favorites-root", type=Path, required=True)
    parser.add_argument("--only", default=None)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--workers", type=int)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    backfill_parser = subparsers.add_parser("backfill")
    _common_selection(backfill_parser)
    backfill_parser.add_argument("--spawn-worker", action="store_true")
    backfill_parser.add_argument(
        "--worker-log", type=Path, default=Path("outputs/baselines/fp_surface_worker.log")
    )
    backfill_parser.set_defaults(function=backfill)

    prepare_parser = subparsers.add_parser("prepare")
    _common_selection(prepare_parser)
    prepare_parser.add_argument("--fp-root", type=Path, required=True)
    prepare_parser.add_argument("--runlist", type=Path, required=True)
    prepare_parser.set_defaults(function=prepare)

    run_parser = subparsers.add_parser("run-testcase")
    run_parser.add_argument("--testcase", type=Path, required=True)
    run_parser.add_argument("--fp-repo", type=Path, required=True)
    run_parser.set_defaults(function=run_testcase)

    validate_parser = subparsers.add_parser("validate")
    _common_selection(validate_parser)
    validate_parser.add_argument("--fp-root", type=Path, required=True)
    validate_parser.set_defaults(function=validate)

    bake_parser = subparsers.add_parser("bake")
    _common_selection(bake_parser)
    bake_parser.add_argument("--fp-root", type=Path, required=True)
    bake_parser.add_argument("--bake-cache", type=Path, required=True)
    default_repo = Path(__file__).resolve().parents[2]
    bake_parser.add_argument(
        "--ego-repo", type=Path, default=default_repo / "third_party" / "EgoInfinity"
    )
    bake_parser.add_argument("--ego-python", type=Path, default=Path(sys.executable))
    bake_parser.add_argument("--source-runlist", type=Path)
    bake_parser.add_argument(
        "--ego-command", help="JSON command prefix for the EgoInfinity environment"
    )
    bake_parser.add_argument("--output-runlist", type=Path)
    bake_parser.add_argument("--canonical-root", type=Path)
    bake_parser.set_defaults(function=bake)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
