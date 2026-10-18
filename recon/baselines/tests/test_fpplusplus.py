from __future__ import annotations

import gzip
import importlib.util
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import trimesh

from baselines.egoinfinity.fpplusplus import (
    FPPLUSPLUS_COMMIT,
    PROTOCOL_ID,
    _prepare_one,
    _validate_pose_array,
)


def _encoded_frame(value: int = 64) -> dict:
    rgb = np.full((24, 32, 3), value, dtype=np.uint8)
    ok_rgb, rgb_bytes = cv2.imencode(".jpg", rgb)
    depth = np.full((24, 32), 1000, dtype=np.uint16)
    ok_depth, depth_bytes = cv2.imencode(".png", depth)
    assert ok_rgb and ok_depth
    mask = np.zeros((24, 32), dtype=bool)
    mask[5:20, 7:25] = True
    return {
        "img_rgb": bytes(rgb_bytes),
        "depth_png": bytes(depth_bytes),
        "sam3_obj_data": {
            0: {
                "mask_packed": np.packbits(mask.reshape(-1)).tobytes(),
                "mask_shape": mask.shape,
            }
        },
    }


def test_prepare_uses_aligned_known_metric_mesh(tmp_path: Path) -> None:
    clip = tmp_path / "clip"
    mesh_dir = clip / "sam3_meshes"
    mesh_dir.mkdir(parents=True)
    reconstructed = mesh_dir / "obj_0.ply"
    trimesh.creation.icosphere(subdivisions=2, radius=0.5).export(reconstructed)
    known = tmp_path / "known.obj"
    known_geometry = trimesh.creation.icosphere(subdivisions=2, radius=0.5)
    known_geometry.apply_scale([0.1, 0.2, 0.3])
    known_geometry.export(known)
    frames = [_encoded_frame(50), _encoded_frame(60), _encoded_frame(70)]
    data = {"frame_data": frames, "dp_focal": 400.0, "cx": 16.0, "cy": 12.0}
    meta = {"init_frame": 1, "ply_path": str(reconstructed), "canonical_scale": 0.2}
    info = {"T_seq": np.tile(np.eye(4, dtype=np.float32), (3, 1, 1)), "scale_correction": 0.25}

    result = _prepare_one(
        clip,
        data,
        0,
        meta,
        info,
        tmp_path / "testcase",
        known,
    )
    testcase = Path(result["testcase"])
    metadata = json.loads((testcase / "egoinfinity_fpplusplus.json").read_text())
    assert metadata["protocol_id"] == PROTOCOL_ID
    assert metadata["fpplusplus_commit"] == FPPLUSPLUS_COMMIT
    assert metadata["init_frame"] == 1
    assert metadata["tracking_frames"] == 2
    assert metadata["mesh_apply_scale_m"] == 1.0
    assert metadata["known_mesh_alignment"]["known_metric_mesh"] == str(known.resolve())
    assert metadata["known_mesh_alignment"]["native_scale_m_per_unit"] == 0.25
    assert metadata["inputs"]["allowed_benchmark_inputs_used"] == ["known_metric_mesh"]
    assert metadata["inputs"]["forbidden_benchmark_inputs_used"] == []
    assert metadata["fpplusplus_parameters"]["activate_2d_tracker"] is True
    assert metadata["fpplusplus_parameters"]["activate_kalman_filter"] is True
    assert metadata["fpplusplus_parameters"]["track_refine_iter"] == 5
    assert metadata["fpplusplus_parameters"]["force_apply_color"] is True
    assert metadata["known_mesh_alignment"]["appearance_policy"].startswith(
        "source textures and materials discarded"
    )
    assert sorted(path.name for path in (testcase / "color").iterdir()) == ["0.jpg", "1.jpg"]
    assert sorted(path.name for path in (testcase / "depth").iterdir()) == ["0.png", "1.png"]
    assert (testcase / "oid.txt").read_text() == "0\n"
    aligned = trimesh.load(testcase / "mesh" / "object.glb", process=False)
    np.testing.assert_allclose(aligned.volume, known_geometry.volume, rtol=1e-6)
    aligned_mesh = next(iter(aligned.geometry.values()))
    assert not isinstance(aligned_mesh.visual, trimesh.visual.texture.TextureVisuals)


def test_pose_validator_rejects_non_rotation() -> None:
    pose = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    _validate_pose_array(pose, 2, "valid")
    pose[1, 0, 0] = 2.0
    try:
        _validate_pose_array(pose, 2, "invalid")
    except RuntimeError as error:
        assert "invalid SE(3)" in str(error)
    else:
        raise AssertionError("non-rotation pose was accepted")
