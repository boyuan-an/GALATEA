from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from baselines.common.canonical import _mesh_frame_alignment

PROTOCOL_ID = "doas-known-mesh-aligned-cache-v1"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        loaded = loaded.dump(concatenate=True)
    if not isinstance(loaded, trimesh.Trimesh):
        raise RuntimeError(f"not a triangle mesh: {path}")
    if not len(loaded.vertices) or not len(loaded.faces):
        raise RuntimeError(f"empty triangle mesh: {path}")
    return loaded


def _alignment_matrix(alignment: dict[str, Any]) -> np.ndarray:
    rotation = np.asarray(alignment["rotation"], dtype=np.float64)
    known_center = np.asarray(alignment["known_center"], dtype=np.float64)
    reconstructed_center = np.asarray(
        alignment["reconstructed_center"],
        dtype=np.float64,
    )
    scale = float(alignment["reconstructed_radius_native"] / alignment["known_radius_m"])
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = reconstructed_center - scale * rotation @ known_center
    return matrix


def bake_known_mesh(
    known_mesh: Path,
    reconstructed_mesh: Path,
    output_mesh: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    alignment = _mesh_frame_alignment(known_mesh, reconstructed_mesh)
    matrix = _alignment_matrix(alignment)
    source = _load_mesh(known_mesh)
    baked = trimesh.Trimesh(
        vertices=np.asarray(source.vertices, dtype=np.float64),
        faces=np.asarray(source.faces, dtype=np.int64),
        process=False,
        maintain_order=True,
    )
    baked.apply_transform(matrix)
    output_mesh.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_mesh.with_name(output_mesh.stem + ".tmp.obj")
    baked.export(temporary, file_type="obj")
    os.replace(temporary, output_mesh)
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at": _timestamp(),
        "alignment_policy": (
            "deterministic geometry-only 24 proper PCA axis maps plus "
            "bidirectional normalized Chamfer selection"
        ),
        "known_metric_mesh": str(known_mesh.resolve()),
        "known_metric_mesh_sha256": _sha256(known_mesh),
        "native_rgb_reconstructed_mesh": str(reconstructed_mesh.resolve()),
        "native_rgb_reconstructed_mesh_sha256": _sha256(reconstructed_mesh),
        "baked_tracking_mesh": str(output_mesh.resolve()),
        "baked_tracking_mesh_sha256": _sha256(output_mesh),
        "known_to_reconstructed_matrix": matrix.tolist(),
        "rotation": np.asarray(alignment["rotation"]).tolist(),
        "known_center": np.asarray(alignment["known_center"]).tolist(),
        "reconstructed_center": np.asarray(alignment["reconstructed_center"]).tolist(),
        "known_radius_m": float(alignment["known_radius_m"]),
        "reconstructed_radius_native": float(alignment["reconstructed_radius_native"]),
        "native_units_per_meter": float(
            alignment["reconstructed_radius_native"] / alignment["known_radius_m"]
        ),
        "normalized_symmetric_chamfer": float(alignment["normalized_symmetric_chamfer"]),
        "uses_gt_pose": False,
        "uses_future_depth": False,
        "uses_gt_mask": False,
    }
    _atomic_json(metadata_path, metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Known-mesh alignment from the validated GALATEA DO AS I DO adapter.")
    parser.add_argument("--known-mesh", type=Path, required=True)
    parser.add_argument("--reconstructed-mesh", type=Path, required=True)
    parser.add_argument("--output-mesh", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    args = parser.parse_args()
    bake_known_mesh(args.known_mesh, args.reconstructed_mesh, args.output_mesh, args.metadata)


if __name__ == "__main__":
    main()
