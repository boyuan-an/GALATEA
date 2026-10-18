from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import trimesh


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_geometry_only_mesh(source: Path, destination: Path) -> dict[str, object]:
    loaded = trimesh.load(source, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh):
        raise TypeError(f"expected one mesh from {source}, got {type(loaded).__name__}")
    if loaded.vertices.size == 0 or loaded.faces.size == 0:
        raise ValueError(f"empty mesh: {source}")

    geometry = trimesh.Trimesh(
        vertices=np.asarray(loaded.vertices).copy(),
        faces=np.asarray(loaded.faces).copy(),
        process=False,
        validate=False,
    )
    payload = trimesh.exchange.obj.export_obj(
        geometry,
        include_normals=True,
        include_color=False,
        include_texture=False,
    )
    if not isinstance(payload, str):
        raise TypeError(f"unexpected OBJ export payload: {type(payload).__name__}")
    forbidden = ("mtllib ", "usemtl ", "vt ")
    if any(line.startswith(forbidden) for line in payload.splitlines()):
        raise RuntimeError(f"geometry-only export retained appearance data: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return {
        "source_sha256": _sha256(source),
        "geometry_sha256": _sha256(destination),
        "vertices": int(len(geometry.vertices)),
        "faces": int(len(geometry.faces)),
    }
