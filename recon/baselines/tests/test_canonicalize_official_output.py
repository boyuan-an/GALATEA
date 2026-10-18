import json

import numpy as np
import trimesh

from baselines.common.canonical import (
    _compose_known_mesh_poses,
    _dai_hand,
    _ego_hand,
    _layout_poses,
)
from baselines.do_as_i_do.mesh import bake_known_mesh


def test_dai_layout_uses_top_level_optimized_scale_and_list_schema(tmp_path):
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "frame": "camera_frame",
                "translation_scale_optimization": {"mesh_scale": 0.04},
                "objects": [
                    {
                        "frame_idx": 0,
                        "local_to_scene": {
                            "translation_camera_frame": [0.1, 0.2, 0.3],
                            "quat_wxyz_camera_frame": [1.0, 0.0, 0.0, 0.0],
                            "scale": [0.1, 0.1, 0.1],
                        },
                    }
                ],
            }
        )
    )
    poses, scales = _layout_poses(layout, 1)
    assert np.allclose(poses[0, :3, 3], [0.1, 0.2, 0.3])
    assert np.allclose(poses[0, :3, :3], np.eye(3))
    assert np.allclose(scales, [0.04])


def test_dai_layout_accepts_uniform_local_scale_list_without_optimization(tmp_path):
    layout = tmp_path / "layout.json"
    layout.write_text(
        json.dumps(
            {
                "frame": "camera_frame",
                "objects": [
                    {
                        "frame_idx": 0,
                        "local_to_scene": {
                            "translation_camera_frame": [0.0, 0.0, 1.0],
                            "quat_wxyz_camera_frame": [1.0, 0.0, 0.0, 0.0],
                            "scale": [0.25, 0.25, 0.25],
                        },
                    }
                ],
            }
        )
    )
    _, scales = _layout_poses(layout, 1)
    assert np.allclose(scales, [0.25])


def test_ego_hand_exports_right_mano_vertices_and_faces():
    left_joints = np.zeros((21, 3), dtype=np.float32)
    right_joints = np.ones((21, 3), dtype=np.float32)
    left_vertices = np.zeros((778, 3), dtype=np.float32)
    right_vertices = np.ones((778, 3), dtype=np.float32) * 2
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    joints, vertices, output_faces = _ego_hand(
        {
            "mano_faces": faces,
            "frame_data": [
                {
                    "hand_is_right": [False, True],
                    "joints_3d_pred": [left_joints, right_joints],
                    "vertices_3d": [left_vertices, right_vertices],
                }
            ],
        },
        1,
    )
    assert np.allclose(joints[0], right_joints)
    assert np.allclose(vertices[0], right_vertices)
    assert np.array_equal(output_faces, faces)


def test_dai_hand_masks_invalid_mano_vertices(tmp_path):
    path = tmp_path / "hands.npz"
    np.savez(
        path,
        right_joints=np.zeros((2, 21, 3), dtype=np.float32),
        right_vertices=np.zeros((2, 778, 3), dtype=np.float32),
        right_faces=np.array([[0, 1, 2]], dtype=np.int32),
        right_valid=np.array([True, False]),
    )
    joints, vertices, _ = _dai_hand(path, 2)
    assert np.isfinite(joints[0]).all() and np.isfinite(vertices[0]).all()
    assert np.isnan(joints[1]).all() and np.isnan(vertices[1]).all()


def test_known_mesh_bake_preserves_scene_geometry_and_exact_inverse(tmp_path):
    known = trimesh.creation.box(extents=[0.08, 0.13, 0.21])
    known.apply_translation([0.02, -0.01, 0.03])
    known_path = tmp_path / "known.obj"
    known.export(known_path)
    angle = np.deg2rad(37.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    reconstructed = known.copy()
    reconstructed.apply_transform(
        np.block(
            [
                [2.7 * rotation, np.array([[0.3], [-0.2], [0.1]])],
                [np.zeros((1, 3)), np.ones((1, 1))],
            ]
        )
    )
    reconstructed_path = tmp_path / "reconstructed.obj"
    reconstructed.export(reconstructed_path)
    baked_path = tmp_path / "baked.obj"
    metadata_path = tmp_path / "alignment.json"
    metadata = bake_known_mesh(
        known_path,
        reconstructed_path,
        baked_path,
        metadata_path,
    )
    baked = trimesh.load(baked_path, force="mesh", process=False)
    expected = trimesh.load(reconstructed_path, force="mesh", process=False)
    assert np.allclose(baked.vertices, expected.vertices, atol=2e-4)
    native_pose = np.eye(4)[None]
    native_pose[0, :3, :3] = rotation
    native_pose[0, :3, 3] = [0.1, 0.2, 0.8]
    scale = metadata["known_radius_m"] / metadata["reconstructed_radius_native"]
    alignment = {
        key: (
            np.asarray(metadata[key])
            if key in {"rotation", "known_center", "reconstructed_center"}
            else metadata[key]
        )
        for key in (
            "rotation",
            "known_center",
            "reconstructed_center",
            "known_radius_m",
            "reconstructed_radius_native",
        )
    }
    known_pose = _compose_known_mesh_poses(
        native_pose,
        np.array([scale]),
        alignment,
    )[0]
    scene_from_baked = (
        scale * np.asarray(baked.vertices) @ native_pose[0, :3, :3].T + native_pose[0, :3, 3]
    )
    known_vertices = np.asarray(trimesh.load(known_path, force="mesh", process=False).vertices)
    scene_from_known = known_vertices @ known_pose[:3, :3].T + known_pose[:3, 3]
    assert np.allclose(scene_from_baked, scene_from_known, atol=1e-5)
