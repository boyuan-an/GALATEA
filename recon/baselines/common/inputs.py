from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

from .mesh import _sha256, _write_geometry_only_mesh


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_clips(manifest: Path, method: str) -> list[dict]:
    payload = json.loads(manifest.read_text())
    if set(payload) != {"clips"} or not isinstance(payload["clips"], list):
        raise ValueError('manifest must contain a "clips" list only')
    if not payload["clips"]:
        raise ValueError("manifest has no clips")
    allowed = {
        "clip_id",
        "video",
        "mesh",
        "object_name",
        "intrinsics",
        "point",
        "hand_side",
        "frame0_mask",
    }
    clips, seen = [], set()
    for item in payload["clips"]:
        if set(item) - allowed:
            raise ValueError(f"unsupported input fields: {sorted(set(item) - allowed)}")
        clip = dict(item)
        for field in ("clip_id", "video", "mesh", "object_name"):
            if not isinstance(clip.get(field), str) or not clip[field].strip():
                raise ValueError(f"missing {field}")
        clip_id = clip["clip_id"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", clip_id) or clip_id in seen:
            raise ValueError(f"invalid or duplicate clip_id: {clip_id}")
        seen.add(clip_id)
        if clip.get("hand_side", "right") != "right":
            raise ValueError("these published adapters support one interacting right hand")
        clip["hand_side"] = "right"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 _-]*", clip["object_name"]):
            raise ValueError("object_name must be a plain noun phrase (letters, numbers, spaces)")
        if method in {"videomanip", "do_as_i_do"}:
            point = clip.get("point")
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(isinstance(x, (int, float)) and np.isfinite(x) and x >= 0 for x in point)
            ):
                raise ValueError(f"{method} requires an RGB-only frame-0 point [x, y]")
        if method == "videomanip" and not clip.get("intrinsics"):
            raise ValueError("VideoManip requires supplied intrinsics")
        for field in ("video", "mesh", "intrinsics", "frame0_mask"):
            if clip.get(field):
                path = resolve_path(clip[field], manifest.parent)
                if not path.is_file():
                    raise FileNotFoundError(path)
                clip[field] = str(path)
        clips.append(clip)
    return clips


def fingerprint(clip: dict, method: str, config: dict) -> str:
    adapter_root = Path(__file__).resolve().parents[1]
    code = {
        str(path.relative_to(adapter_root)): _sha256(path)
        for path in sorted(adapter_root.rglob("*.py"))
        if "tests" not in path.parts
    }
    code["sources.json"] = _sha256(adapter_root / "sources.json")
    for path in sorted(adapter_root.rglob("*.patch")):
        code[str(path.relative_to(adapter_root))] = _sha256(path)
    values = {"clip": clip, "method": method, "config": config, "adapter_code": code, "inputs": {}}
    for field in ("video", "mesh", "intrinsics", "frame0_mask"):
        if clip.get(field):
            values["inputs"][field] = _sha256(Path(clip[field]))
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def prepare(clip: dict, method: str, config: dict, root: Path, resume: bool) -> dict:
    identity = fingerprint(clip, method, config)
    receipt = root / "inputs.json"
    if root.exists():
        if not resume or not receipt.is_file():
            raise FileExistsError(f"output already exists: {root}; use a new output directory")
        saved = json.loads(receipt.read_text())
        if saved["input_digest"] != identity:
            raise ValueError("inputs or configuration changed; choose a new output directory")
        for name, digest in saved["prepared_hashes"].items():
            path = root / name
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(f"prepared input changed: {path}; choose a new output directory")
        return saved
    root.mkdir(parents=True)
    native = root / "native" / clip["clip_id"]
    native.mkdir(parents=True)
    video = native / (clip["clip_id"] + Path(clip["video"]).suffix)
    shutil.copy2(clip["video"], video)
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate",
                "-of",
                "json",
                str(video),
            ],
            text=True,
        )
    )["streams"][0]
    rate = probe.get("avg_frame_rate", "0/0")
    if rate in {"0/0", "0/1", "0"}:
        rate = probe["r_frame_rate"]
    fps = float(Fraction(rate))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("video has no valid frame rate")
    rgb = native / "rgb"
    rgb.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video),
            "-vsync",
            "0",
            "-start_number",
            "0",
            str(rgb / "%06d.png"),
        ],
        check=True,
    )
    frames = sorted(rgb.glob("*.png"))
    if not frames:
        raise ValueError("video has no decodable frames")
    with Image.open(frames[0]) as image:
        width, height = image.size
    if "point" in clip and not (0 <= clip["point"][0] < width and 0 <= clip["point"][1] < height):
        raise ValueError("frame-0 point is outside the video")
    mesh = root / "inputs" / "object.obj"
    mesh_audit = _write_geometry_only_mesh(Path(clip["mesh"]), mesh)
    intrinsics = None
    if clip.get("intrinsics"):
        values = np.loadtxt(clip["intrinsics"]).reshape(-1)
        if len(values) == 6:
            fx, fy, cx, cy, w, h = values
            if (w, h) != (width, height):
                raise ValueError("intrinsics image size does not match the input video")
            values = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        elif len(values) == 9:
            values = values.reshape(3, 3)
        else:
            raise ValueError("intrinsics must be 3x3 or fx fy cx cy width height")
        if (
            not np.isfinite(values).all()
            or values[0, 0] <= 0
            or values[1, 1] <= 0
            or not np.allclose(values[2], [0, 0, 1])
        ):
            raise ValueError("invalid camera intrinsics")
        intrinsics = root / "inputs" / "intrinsics.txt"
        np.savetxt(intrinsics, values)
    mask = None
    if clip.get("frame0_mask"):
        image = np.asarray(Image.open(clip["frame0_mask"]).convert("L"))
        if image.shape != (height, width) or not image.any():
            raise ValueError("frame0_mask must be nonempty and match the video resolution")
        mask = root / "inputs" / "frame0_mask.png"
        Image.fromarray((image > 0).astype(np.uint8) * 255).save(mask)
    result = {
        **clip,
        "video": str(video),
        "mesh": str(mesh),
        "intrinsics": str(intrinsics) if intrinsics else None,
        "frame0_mask": str(mask) if mask else None,
        "native": str(native),
        "root": str(root),
        "frames": len(frames),
        "width": width,
        "height": height,
        "dataset": "user",
        "input_digest": identity,
        "mesh_audit": mesh_audit,
        "fps": fps,
        "source_frame_rate": rate,
    }
    prepared_files = [
        video,
        mesh,
        *frames,
        *([intrinsics] if intrinsics else []),
        *([mask] if mask else []),
    ]
    result["prepared_hashes"] = {
        str(path.relative_to(root)): _sha256(path) for path in prepared_files
    }
    write_json(receipt, result)
    return result
