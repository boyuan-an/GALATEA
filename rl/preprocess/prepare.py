import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as R, Slerp

import quality
import scene
from reference import mano_layer


def convert(payload, mesh_path: Path, out_dir: Path, target_fps: int, manolayer, bottom_z: float = -0.018) -> float:
    fps = float(payload["framerate"]) if "framerate" in payload else 120.0
    rh, ob = (payload["rhand"], payload["object"])
    go_src = np.asarray(rh["params"]["global_orient"], dtype=np.float32)
    fp_src = np.asarray(rh["params"]["fullpose"], dtype=np.float32)
    tr_src = np.asarray(rh["params"]["transl"], dtype=np.float32)
    obj_go_src = np.asarray(ob["params"]["global_orient"], dtype=np.float32)
    obj_tr_src = np.asarray(ob["params"]["transl"], dtype=np.float32)

    def resample_times(frame_count: int):
        src_t = np.arange(frame_count, dtype=np.float64) / fps
        dst_t = np.arange(0.0, src_t[-1] + 1e-09, 1.0 / target_fps)
        return (src_t, dst_t)

    def resample_linear(x: np.ndarray) -> np.ndarray:
        if fps == target_fps:
            return x.astype(np.float32)
        src_t, dst_t = resample_times(x.shape[0])
        flat = x.reshape(x.shape[0], -1)
        out = np.stack([np.interp(dst_t, src_t, flat[:, i]) for i in range(flat.shape[1])], axis=-1)
        return out.reshape((len(dst_t),) + x.shape[1:]).astype(np.float32)

    def resample_rotvec(x: np.ndarray) -> np.ndarray:
        if fps == target_fps:
            return x.astype(np.float32)
        src_t, dst_t = resample_times(x.shape[0])
        flat = x.reshape(x.shape[0], -1, 3)
        out = np.empty((len(dst_t), flat.shape[1], 3), dtype=np.float32)
        for j in range(flat.shape[1]):
            out[:, j] = Slerp(src_t, R.from_rotvec(flat[:, j]))(dst_t).as_rotvec().astype(np.float32)
        return out.reshape((len(dst_t),) + x.shape[1:]).astype(np.float32)

    ratio = fps / float(target_fps)
    if fps >= target_fps and np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-09):
        sl = slice(0, None, max(1, round(ratio)))
        go, fp, tr, obj_go, obj_tr = (go_src[sl], fp_src[sl], tr_src[sl], obj_go_src[sl], obj_tr_src[sl])
    else:
        go = resample_rotvec(go_src)
        fp = resample_rotvec(fp_src.reshape(fp_src.shape[0], -1, 3)).reshape(-1, fp_src.shape[1])
        tr = resample_linear(tr_src)
        obj_go = resample_rotvec(obj_go_src)
        obj_tr = resample_linear(obj_tr_src)
    frames = go.shape[0]
    pose = torch.tensor(np.concatenate([go, fp], axis=-1), dtype=torch.float32)
    betas = None
    if rh.get("betas") is not None:
        betas = torch.tensor(np.asarray(rh["betas"], dtype=np.float32), dtype=torch.float32)
        if betas.ndim == 1:
            betas = betas.unsqueeze(0).repeat(frames, 1)
    with torch.no_grad():
        verts = manolayer(pose, betas=betas).verts.cpu().numpy()
    verts = verts + tr[:, None]
    obj_verts = np.asarray(trimesh.load(mesh_path, process=False).vertices, dtype=np.float32)
    obj_rot0 = R.from_rotvec(obj_go[0]).as_matrix().astype(np.float32).T
    init_obj_verts = (obj_rot0 @ obj_verts.T).T + obj_tr[0]
    origin = obj_tr[0].copy()
    origin[2] = init_obj_verts[:, 2].min() - float(bottom_z)
    verts = verts - origin[None, None, :]
    tr = tr - origin[None, :]
    obj_tr = obj_tr - origin[None, :]
    demo = {
        "object_global_orient": obj_go.astype(np.float32),
        "object_transl": obj_tr.astype(np.float32),
        "rhand_global_orient_gt": go.astype(np.float32),
        "rhand_transl": tr.astype(np.float32),
        "rhand_verts": verts.astype(np.float32),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / scene.DEMO_FILE, demo, allow_pickle=True)
    shutil.copyfile(mesh_path, out_dir / scene.MESH_FILE)
    return float(((obj_rot0 @ obj_verts.T).T + obj_tr[0])[:, 2].min())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=int, default=30)
    args = parser.parse_args()
    entries = json.loads((args.source / "entries.json").read_text())["entries"]
    plan = json.loads((args.source / "quality_plan.json").read_text())["clips"]
    data_dir = args.output / "data"
    manolayer = mano_layer("cpu")
    for entry in entries:
        payload = quality.apply_plan(quality.load_raw(entry["motion"]), plan[entry["data_index"]])
        bottom = convert(payload, Path(entry["mesh"]), data_dir / entry["dir"], args.target_fps, manolayer)
        if abs(bottom + 0.018) > 1e-05:
            raise RuntimeError(f"{entry['data_index']}: bad frame-0 grounding {bottom}")
        print(f"[prepare] {entry['data_index']} {entry['source']}", flush=True)
    rows = [{key: entry[key] for key in ("data_index", "dir", "source")} for entry in entries]
    scene.atomic_json(data_dir / "manifest.json", {"entries": rows})


if __name__ == "__main__":
    main()
