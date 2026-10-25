import argparse
import hashlib
import shutil
from pathlib import Path

import coacd
import numpy as np
import trimesh

COLLISION_MESH_FILE = "object_collision.obj"
COLLISION_URDF_FILE = "object_collision.urdf"
COLLISION_URDF = """<?xml version='1.0' encoding='utf-8'?>
<robot name="object">
  <link name="base">
    <visual>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="object.obj" scale="1 1 1" />
      </geometry>
    </visual>
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0" />
      <geometry>
        <mesh filename="object_collision.obj" scale="1 1 1" />
      </geometry>
    </collision>
    <inertial>
      <mass value="0.1" />
      <origin xyz="0 0 0" />
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4" />
    </inertial>
  </link>
</robot>"""


def decompose(mesh_path: Path, output: Path) -> int:
    loaded = trimesh.load(mesh_path, force="mesh", process=False, skip_materials=True)
    parts = coacd.run_coacd(
        coacd.Mesh(loaded.vertices, loaded.faces),
        threshold=0.02,
        max_convex_hull=32,
        preprocess_mode="auto",
        preprocess_resolution=50,
        resolution=2000,
        mcts_nodes=20,
        mcts_iterations=150,
        mcts_max_depth=3,
        merge=True,
        decimate=False,
        max_ch_vertex=256,
        apx_mode="ch",
        seed=0,
    )
    vertices = np.concatenate([part_vertices for part_vertices, _ in parts], axis=0)
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    scale = np.minimum(1.0, (loaded.bounds[1] - loaded.bounds[0]) / np.maximum(bounds[1] - bounds[0], 1e-12))
    translation = loaded.bounds.mean(axis=0) - bounds.mean(axis=0) * scale
    parts = [(part_vertices * scale[None, :] + translation[None, :], faces) for part_vertices, faces in parts]
    vertices = np.concatenate([part_vertices for part_vertices, _ in parts], axis=0)
    bounds = np.stack((vertices.min(axis=0), vertices.max(axis=0)))
    outward = max(float(np.max(loaded.bounds[0] - bounds[0])), float(np.max(bounds[1] - loaded.bounds[1])), 0.0)
    if outward > 0.0001:
        raise ValueError(f"convex decomposition of {mesh_path} grows the mesh bounds by {outward:.6f} m")
    collision_scene = trimesh.Scene()
    for index, (part_vertices, faces) in enumerate(parts):
        collision_scene.add_geometry(
            trimesh.Trimesh(vertices=part_vertices, faces=faces, process=False),
            geom_name=f"part_{index:03d}",
            node_name=f"part_{index:03d}",
        )
    collision_scene.export(output)
    return len(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    args = parser.parse_args()
    coacd.set_log_level("error")
    built = {}
    for mesh_path in sorted((args.bank / "data").glob("*/object.obj")):
        digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        output = mesh_path.with_name(COLLISION_MESH_FILE)
        if digest in built:
            shutil.copy2(built[digest], output)
        else:
            parts = decompose(mesh_path, output)
            built[digest] = output
            print(f"[collision-model] {mesh_path.parent.name}: {parts} convex parts", flush=True)
        mesh_path.with_name(COLLISION_URDF_FILE).write_text(COLLISION_URDF)
    print(f"[collision-model] {len(built)} unique meshes", flush=True)


if __name__ == "__main__":
    main()
