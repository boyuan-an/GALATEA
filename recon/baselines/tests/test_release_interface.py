import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import trimesh
from PIL import Image

from baselines.common.execution import REPOSITORY, Execution
from baselines.common.inputs import load_clips, prepare


@pytest.fixture
def inputs(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for i in range(4):
        Image.new("RGB", (64, 48), color=(i * 60, 30, 20)).save(source / f"{i:06d}.png")
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "30000/1001",
            "-i",
            str(source / "%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(source / "video.mp4"),
        ],
        check=True,
    )
    trimesh.creation.box(extents=[0.1, 0.2, 0.3]).export(source / "object.obj")
    np.savetxt(source / "K.txt", [[100, 0, 32], [0, 100, 24], [0, 0, 1]])
    (source / "pose_gt.txt").write_text("must never be copied")
    record = {
        "clip_id": "clip1",
        "video": "source/video.mp4",
        "mesh": "source/object.obj",
        "object_name": "power drill",
        "intrinsics": "source/K.txt",
        "point": [32, 24],
    }
    manifest = tmp_path / "clips.json"
    manifest.write_text(json.dumps({"clips": [record]}))
    return manifest, record


def config_for(method, tmp_path):
    return {"method": method, "repo": str(tmp_path / "upstream"), "fp_repo": str(tmp_path / "fp")}


@pytest.mark.parametrize(
    "method,stages",
    [
        ("videomanip", ["depth", "hand", "segmentation", "object", "canonicalize"]),
        (
            "egoinfinity",
            [
                "frontend",
                "prepare_fpplusplus",
                "track_fpplusplus",
                "bake_fpplusplus",
                "canonicalize",
            ],
        ),
        ("do_as_i_do", ["frontend", "align_mesh", "tracking", "canonicalize"]),
    ],
)
def test_dry_run_stages_and_input_isolation(inputs, tmp_path, method, stages):
    manifest, _ = inputs
    config = config_for(method, tmp_path)
    clip = load_clips(manifest, method)[0]
    root = tmp_path / method
    staged = prepare(clip, method, config, root, False)
    assert staged["frames"] == 4
    assert staged["fps"] == pytest.approx(30000 / 1001)
    assert not list(root.rglob("*gt*"))
    assert not Path(staged["video"]).is_symlink()
    execution = Execution(root, config, True)
    importlib.import_module(f"baselines.{method}.run").run(staged, config, execution)
    records = json.loads((root / "stages.json").read_text())
    assert [record["stage"] for record in records] == stages
    assert all(record["state"] == "planned" for record in records)
    if method == "videomanip":
        command = records[-2]["command"]
        assert command[command.index("--est_refine_iter") + 1] == "5"
        assert command[command.index("--track_refine_iter") + 1] == "2"
    elif method == "egoinfinity":
        metadata = json.loads((Path(staged["native"]) / "manifest.json").read_text())
        assert metadata["preserve_source_frames"] is True
        assert metadata["objects"] == ["power drill", "drill"]
    else:
        assert "--known-mesh" in records[1]["command"]
        assert not any("intrinsics" in value for r in records[:3] for value in r["command"])


def test_resume_validates_original_and_staged_content(inputs, tmp_path):
    manifest, _ = inputs
    clip = load_clips(manifest, "videomanip")[0]
    config = config_for("videomanip", tmp_path)
    root = tmp_path / "run"
    staged = prepare(clip, "videomanip", config, root, False)
    assert prepare(clip, "videomanip", config, root, True) == staged
    changed = {**config, "track_refine_iter": 8}
    with pytest.raises(ValueError, match="inputs or configuration changed"):
        prepare(clip, "videomanip", changed, root, True)
    Path(staged["mesh"]).write_text("corrupt")
    with pytest.raises(ValueError, match="prepared input changed"):
        prepare(clip, "videomanip", config, root, True)


@pytest.mark.parametrize(
    "change",
    [
        {"gt_pose": "hidden.npy"},
        {"hand_side": "left"},
        {"clip_id": "../escape"},
        {"point": [-1, 2]},
    ],
)
def test_reject_invalid_or_disallowed_inputs(inputs, change):
    manifest, record = inputs
    manifest.write_text(json.dumps({"clips": [{**record, **change}]}))
    with pytest.raises(ValueError):
        load_clips(manifest, "videomanip")


def test_stage_failure_is_reported(tmp_path):
    execution = Execution(tmp_path, {}, False)
    with pytest.raises(RuntimeError, match="failed"):
        execution.run("broken", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path)
    record = json.loads((tmp_path / "stages.json").read_text())[0]
    assert record["returncode"] == 7
    assert record["state"] == "failed"
    assert Path(record["log"]).is_file()


def test_missing_environment_is_reported(tmp_path):
    execution = Execution(tmp_path, {}, False)
    with pytest.raises(RuntimeError, match="could not start"):
        execution.run("missing", [str(tmp_path / "no-python")], tmp_path)
    assert json.loads((tmp_path / "stages.json").read_text())[0]["state"] == "failed"


def test_cli_relative_paths_and_no_models_in_dry_run(inputs, tmp_path):
    manifest, _ = inputs
    config = tmp_path / "runtime.json"
    config.write_text(json.dumps(config_for("do_as_i_do", tmp_path)))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "baselines.run",
            "--method",
            "do_as_i_do",
            "--manifest",
            str(manifest),
            "--config",
            str(config),
            "--output",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
        cwd=REPOSITORY,
        check=True,
    )
    status = json.loads((tmp_path / "outputs/do_as_i_do/clip1/status.json").read_text())
    assert status["state"] == "dry_run"


def test_doas_frontend_cannot_inherit_mesh_or_calibration(inputs, tmp_path):
    from baselines.do_as_i_do.run import run

    manifest, _ = inputs
    config = config_for("do_as_i_do", tmp_path)
    clip = prepare(
        load_clips(manifest, "do_as_i_do")[0], "do_as_i_do", config, tmp_path / "run", False
    )
    calls = []

    class Recorder:
        def run(self, name, command, cwd, env=None):
            calls.append((name, env))

    run(clip, config, Recorder())
    first = calls[0][1]
    assert first["DOAS_INPUT_MESH"] == first["DOAS_INPUT_INTRINSICS"] == ""
    assert first["DOAS_STOP_AFTER_STAGE"] == "2"
    tracking = calls[2][1]
    assert tracking["DOAS_INPUT_MESH"].endswith("known_mesh_in_sam3d_frame.obj")
    assert tracking["DOAS_INPUT_INTRINSICS"] == ""
