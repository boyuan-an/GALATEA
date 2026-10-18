import argparse
import json
import os
from glob import glob

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene

from galatea_recon.pipeline.hand_obj_optimizer import FINGER_NAMES, build_mano, seg_lbs_idx
from galatea_recon.pipeline.masks import project_points, render_object_mask
from galatea_recon.pipeline.render_compare_videos import (
    HAND_COLOR_BGR,
    HAND_COLOR_RGB,
    OBJ_COLOR_BGR,
    OBJ_COLOR_RGB,
    build_3d_renderer,
    label,
    load_K,
    make_overlay,
)

CONTACT_RGB = (1.0, 0.12, 0.12)
CONTACT_BGR = (40, 40, 235)


def per_frame_contact_vids(seg_labels_path, seq_id, lbs_seg, n, side="right"):
    spec = json.load(open(seg_labels_path)).get(str(seq_id), [])
    seg2vid = {}
    frames = [set() for _ in range(n)]
    for iv in spec:
        if iv["hand"] != side:
            continue
        for fr in range(max(0, iv["start"]), min(n - 1, iv["end"]) + 1):
            frames[fr].add((iv["finger"], iv["seg"]))
    out = []
    for s in frames:
        if not s:
            out.append(np.zeros(0, np.int64))
            continue
        vids = []
        for (f, k) in s:
            key = (f, k)
            if key not in seg2vid:
                seg2vid[key] = np.where(lbs_seg == seg_lbs_idx(f, k))[0]
            vids.append(seg2vid[key])
        out.append(np.concatenate(vids))
    return out, frames


def legend(img, frames_contact):
    H, W = img.shape[:2]
    rows = [("object", OBJ_COLOR_BGR), ("hand", HAND_COLOR_BGR),
            ("contact", CONTACT_BGR)]
    x0 = W - 168
    cv2.rectangle(img, (x0 - 8, 4), (W - 4, 4 + 22 * len(rows) + 6),
                  (0, 0, 0), -1)
    for i, (t, c) in enumerate(rows):
        y = 22 + i * 22
        cv2.circle(img, (x0 + 6, y - 4), 6, c, -1, cv2.LINE_AA)
        cv2.putText(img, t, (x0 + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 1, cv2.LINE_AA)
    if frames_contact:
        seg_n = {1: "prox", 2: "mid", 3: "dist"}
        txt = ", ".join(sorted(f"{f[:3]}.{seg_n[k]}" for (f, k) in frames_contact))
        cv2.putText(img, f"touch: {txt}", (10, H - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CONTACT_BGR, 2, cv2.LINE_AA)
    return img


def run(args):
    dev = args.device
    cam_dir = os.path.abspath(os.path.expanduser(args.cam_dir))
    K, W, H = load_K(cam_dir)

    omesh = trimesh.load(os.path.expanduser(args.mesh), force="mesh")
    Vobj = np.asarray(omesh.vertices, np.float64)
    Fobj = np.asarray(omesh.faces, np.int64)
    Vobj_h = np.hstack([Vobj, np.ones((len(Vobj), 1))])

    obj_poses = np.load(os.path.expanduser(args.obj_poses))
    hverts = np.load(os.path.expanduser(args.hand_verts)).astype(np.float32)
    hfaces = np.load(os.path.join(os.path.dirname(os.path.expanduser(args.hand_verts)),
                                  "mano_right_faces.npy")).astype(np.int64)
    rgb_files = sorted(glob(os.path.join(cam_dir, "rgb", "*.png")))
    n = min(len(obj_poses), len(rgb_files), len(hverts), args.max_frames)

    _, lbs_seg = build_mano("cpu")
    lbs_seg = lbs_seg.numpy()
    contact_vids, contact_segs = per_frame_contact_vids(
        args.seg_labels, args.seq_id, lbs_seg, n, side=args.side)
    ncon = sum(len(c) > 0 for c in contact_vids)
    print(f"[{args.label}] {n} frames | {ncon} frames have contact | "
          f"obj {len(Vobj)}v hand {hverts.shape[1]}v")

    for p in (args.out_overlay, args.out_3d):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_ov = cv2.VideoWriter(args.out_overlay, fourcc, args.fps, (W, H))
    vw_3d = cv2.VideoWriter(args.out_3d, fourcc, args.fps, (W, H))
    raster_set, lights, blend = build_3d_renderer((H, W), dev)

    allpts = [hverts[::6].reshape(-1, 3)]
    obj_cent = np.zeros((n, 3))
    for i in range(n):
        Pc = (obj_poses[i] @ Vobj_h.T).T[:, :3]
        obj_cent[i] = Pc.mean(0)
        if i % 6 == 0:
            allpts.append(Pc[::20])
    allpts = np.concatenate(allpts, 0)
    scene_center = allpts.mean(0)
    scene_radius = float(np.linalg.norm(allpts - scene_center, axis=1).max())
    cam_dist = max(0.35, scene_radius * 2.6)
    trail = []

    base_hcol = np.tile(np.array(HAND_COLOR_RGB, np.float32), (hverts.shape[1], 1))
    for i in range(n):
        pose = obj_poses[i]
        cvids = contact_vids[i]

        bgr = cv2.imread(rgb_files[i], cv2.IMREAD_COLOR)
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))
        obj_mask = render_object_mask(Vobj, Fobj, K, pose, (W, H), device=dev, dilate_px=0)
        hand_mask = render_object_mask(hverts[i], hfaces, K, np.eye(4), (W, H),
                                       device=dev, dilate_px=0)
        ov = make_overlay(bgr, obj_mask, hand_mask, None)
        if len(cvids):
            uv, _ = project_points(hverts[i][cvids], K)
            for u, v in uv.astype(int):
                if 0 <= u < W and 0 <= v < H:
                    cv2.circle(ov, (u, v), 3, CONTACT_BGR, -1, cv2.LINE_AA)
        ov = label(ov, f"{args.label}  [overlay]")
        ov = legend(ov, contact_segs[i])
        cv2.putText(ov, f"frame {i:03d}/{n-1}", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw_ov.write(ov)

        Pobj = (pose @ Vobj_h.T).T[:, :3].astype(np.float32)
        ov_col = torch.tensor(OBJ_COLOR_RGB, device=dev).repeat(len(Pobj), 1)
        objm = Meshes([torch.tensor(Pobj, device=dev)],
                      [torch.tensor(Fobj, device=dev)],
                      textures=TexturesVertex([ov_col]))
        hcol = base_hcol.copy()
        if len(cvids):
            hcol[cvids] = CONTACT_RGB
        handm = Meshes([torch.tensor(hverts[i], device=dev)],
                       [torch.tensor(hfaces, device=dev)],
                       textures=TexturesVertex([torch.tensor(hcol, device=dev)]))
        scene = join_meshes_as_scene([objm, handm])

        azim = -40.0 + 80.0 * (i / max(1, n - 1))
        Rv, Tv = look_at_view_transform(
            dist=cam_dist, elev=12.0, azim=azim,
            at=(tuple(scene_center.tolist()),), up=((0, -1, 0),), device=dev)
        cam3d = FoVPerspectiveCameras(device=dev, R=Rv, T=Tv, fov=45)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cam3d, raster_settings=raster_set),
            shader=SoftPhongShader(device=dev, cameras=cam3d, lights=lights,
                                   blend_params=blend))
        img = renderer(scene)[0, ..., :3].clamp(0, 1).cpu().numpy()
        frame3d = (img[..., ::-1] * 255).astype(np.uint8).copy()

        trail.append(obj_cent[i])
        cc = torch.tensor(np.array(trail), dtype=torch.float32, device=dev)
        proj = cam3d.transform_points_screen(cc[None], image_size=((H, W),))[0]
        proj = proj.cpu().numpy()[:, :2]
        for k in range(1, len(proj)):
            a = max(0.0, 1.0 - (len(proj) - k) / 25.0)
            if a <= 0:
                continue
            cv2.line(frame3d, tuple(np.round(proj[k - 1]).astype(int)),
                     tuple(np.round(proj[k]).astype(int)),
                     tuple(int(c * a) for c in (60, 200, 255)), 2, cv2.LINE_AA)
        frame3d = label(frame3d, f"{args.label}  [3D orbit]")
        frame3d = legend(frame3d, contact_segs[i])
        cv2.putText(frame3d, f"frame {i:03d}/{n-1}  azim {azim:+.0f}",
                    (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw_3d.write(frame3d)
        if i % 10 == 0:
            print(f"  [{args.label}] frame {i}/{n}")

    vw_ov.release()
    vw_3d.release()
    print(f"[{args.label}] wrote {args.out_overlay}\n[{args.label}] wrote {args.out_3d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--obj_poses", required=True)
    ap.add_argument("--hand_verts", required=True)
    ap.add_argument("--seg_labels", required=True)
    ap.add_argument("--seq_id", required=True)
    ap.add_argument("--side", default="right")
    ap.add_argument("--label", default="grasp")
    ap.add_argument("--out_overlay", required=True)
    ap.add_argument("--out_3d", required=True)
    ap.add_argument("--fps", type=float, default=13.0)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    run(ap.parse_args())
