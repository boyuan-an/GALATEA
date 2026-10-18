from pathlib import Path

import numpy as np
import trimesh

from baselines.common.mesh import _write_geometry_only_mesh


def test_write_geometry_only_mesh_strips_obj_appearance(tmp_path: Path) -> None:
    source = tmp_path / "textured.obj"
    source.write_text(
        "\n".join(
            [
                "mtllib textured.mtl",
                "v 0 0 0",
                "v 1 0 0",
                "v 0 1 0",
                "vt 0 0",
                "vt 1 0",
                "vt 0 1",
                "usemtl printed_texture",
                "f 1/1 2/2 3/3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "textured.mtl").write_text(
        "newmtl printed_texture\nmap_Kd texture.png\n", encoding="utf-8"
    )
    destination = tmp_path / "geometry.obj"

    audit = _write_geometry_only_mesh(source, destination)

    payload = destination.read_text(encoding="utf-8")
    assert not any(line.startswith(("mtllib ", "usemtl ", "vt ")) for line in payload.splitlines())
    result = trimesh.load(destination, force="mesh", process=False)
    assert isinstance(result, trimesh.Trimesh)
    assert np.allclose(result.bounds, [[0, 0, 0], [1, 1, 0]])
    assert audit["vertices"] == 3
    assert audit["faces"] == 1
    assert audit["source_sha256"] != audit["geometry_sha256"]
