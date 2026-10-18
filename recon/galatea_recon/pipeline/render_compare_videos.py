import argparse
import os
from glob import glob

import cv2
import numpy as np
import torch
import trimesh
from pytorch3d.renderer import (
    AmbientLights,
    BlendParams,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    TexturesVertex,
    look_at_view_transform,
    PerspectiveCameras,
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.utils import cameras_from_opencv_projection

from galatea_recon.pipeline.masks import render_object_mask, project_points

HAND_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

OBJ_COLOR_BGR = (40, 230, 40)
HAND_COLOR_BGR = (230, 230, 40)
OBJ_COLOR_RGB = (0.20, 0.85, 0.25)
HAND_COLOR_RGB = (0.20, 0.80, 0.95)


def load_K(cam_dir):
    fx, fy, cx, cy, W, H = np.loadtxt(
        os.path.join(cam_dir, "cam_intrinsics.txt")
    ).tolist()
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)
    return K, int(W), int(H)


def skeleton_to_mesh(joints, joint_r=0.006, bone_r=0.004):
    parts = []
    for j in joints:
        s = trimesh.creation.icosphere(subdivisions=1, radius=joint_r)
        s.apply_translation(j)
        parts.append(s)
    for a, b in HAND_BONES:
        pa, pb = joints[a], joints[b]
        v = pb - pa
        L = np.linalg.norm(v)
        if L < 1e-6:
            continue
        cyl = trimesh.creation.cylinder(radius=bone_r, height=L, sections=8)
        zaxis = np.array([0, 0, 1.0])
        d = v / L
        axis = np.cross(zaxis, d)
        s = np.linalg.norm(axis)
        if s < 1e-8:
            R = np.eye(3) if d[2] > 0 else trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])[:3, :3]
        else:
            angle = np.arccos(np.clip(np.dot(zaxis, d), -1, 1))
            R = trimesh.transformations.rotation_matrix(angle, axis / s)[:3, :3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = (pa + pb) / 2.0
        cyl.apply_transform(T)
        parts.append(cyl)
    m = trimesh.util.concatenate(parts)
    return np.asarray(m.vertices, np.float32), np.asarray(m.faces, np.int64)


def make_overlay(rgb, obj_mask, hand_mask, hand_joints_uv=None):
    out = rgb.copy()
    for mask, color in ((obj_mask, OBJ_COLOR_BGR), (hand_mask, HAND_COLOR_BGR)):
        if mask is None:
            continue
        m = mask > 0
        if not m.any():
            continue
        fill = np.zeros_like(out)
        fill[m] = color
        out = cv2.addWeighted(out, 1.0, fill, 0.40, 0.0)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2, cv2.LINE_AA)
    if hand_joints_uv is not None:
        uv = hand_joints_uv.astype(int)
        for a, b in HAND_BONES:
            cv2.line(out, tuple(uv[a]), tuple(uv[b]), HAND_COLOR_BGR, 2, cv2.LINE_AA)
        for p in uv:
            cv2.circle(out, tuple(p), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, tuple(p), 4, HAND_COLOR_BGR, 1, cv2.LINE_AA)
    return out


def label(img, text, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(img, (0, 0), (tw + 18, th + 18), (0, 0, 0), -1)
    cv2.putText(img, text, (9, th + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                cv2.LINE_AA)
    return img


def build_3d_renderer(image_size, device):
    raster = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                   faces_per_pixel=1)
    lights = PointLights(device=device, location=[[0.0, 0.0, -2.0]],
                         ambient_color=[[0.45, 0.45, 0.45]],
                         diffuse_color=[[0.7, 0.7, 0.7]],
                         specular_color=[[0.3, 0.3, 0.3]])
    blend = BlendParams(background_color=(0.0, 0.0, 0.0))
    return raster, lights, blend


def run(source, cam_dir, mesh_path, obj_poses_path, hand_path, hand_kind,
        out_overlay, out_3d, fps, max_frames, device):
    cam_dir = os.path.abspath(os.path.expanduser(cam_dir))
    K, W, H = load_K(cam_dir)

    omesh = trimesh.load(os.path.expanduser(mesh_path), force="mesh")
    Vobj = np.asarray(omesh.vertices, np.float64)
    Fobj = np.asarray(omesh.faces, np.int64)

    obj_poses = np.load(os.path.expanduser(obj_poses_path))
    rgb_files = sorted(glob(os.path.join(cam_dir, "rgb", "*.png")))
    n = min(len(obj_poses), len(rgb_files))
    if max_frames:
        n = min(n, max_frames)

    if hand_kind == "mesh":
        hverts = np.load(os.path.expanduser(hand_path)).astype(np.float32)
        hfaces = np.load(os.path.join(os.path.dirname(os.path.expanduser(hand_path)),
                                      "mano_right_faces.npy")).astype(np.int64)
        n = min(n, len(hverts))
    else:
        hjoints = np.load(os.path.expanduser(hand_path)).astype(np.float32)
        n = min(n, len(hjoints))

    print(f"[{source}] {n} frames | image {W}x{H} | hand={hand_kind} | "
          f"obj mesh {len(Vobj)}v")

    for p in (out_overlay, out_3d):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw_ov = cv2.VideoWriter(out_overlay, fourcc, fps, (W, H))
    vw_3d = cv2.VideoWriter(out_3d, fourcc, fps, (W, H))
    if not vw_ov.isOpened() or not vw_3d.isOpened():
        raise RuntimeError("VideoWriter failed to open")

    raster_set, lights, blend = build_3d_renderer((H, W), device)

    obj_cent = np.zeros((n, 3))
    Vobj_h = np.hstack([Vobj, np.ones((len(Vobj), 1))])
    allpts = []
    for i in range(n):
        Pc = (obj_poses[i] @ Vobj_h.T).T[:, :3]
        obj_cent[i] = Pc.mean(0)
        if i % 6 == 0:
            allpts.append(Pc[::20])
    if hand_kind == "mesh":
        allpts.append(hverts[::6].reshape(-1, 3))
    else:
        allpts.append(hjoints[::6].reshape(-1, 3))
    allpts = np.concatenate(allpts, 0)
    scene_center = allpts.mean(0)
    scene_radius = float(np.linalg.norm(allpts - scene_center, axis=1).max())
    cam_dist = max(0.35, scene_radius * 2.6)
    trail = []

    label_txt = {"ours": "OURS (sil-z + WiLoR)",
                 "phase2": "PHASE-2 (FP + Dyn-HaMR)",
                 "gt": "GT (mocap)"}.get(source, source)

    for i in range(n):
        pose = obj_poses[i]
        bgr = cv2.imread(rgb_files[i], cv2.IMREAD_COLOR)
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))
        obj_mask = render_object_mask(Vobj, Fobj, K, pose, (W, H), device=device,
                                      dilate_px=0)
        hand_mask = None
        hand_uv = None
        if hand_kind == "mesh":
            hand_mask = render_object_mask(hverts[i], hfaces, K, np.eye(4),
                                           (W, H), device=device, dilate_px=0)
        else:
            uv, z = project_points(hjoints[i], K)
            hand_uv = uv
        ov = make_overlay(bgr, obj_mask, hand_mask, hand_uv)
        ov = label(ov, f"{label_txt}  [overlay]")
        cv2.putText(ov, f"frame {i:03d}/{n-1}", (10, H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        vw_ov.write(ov)

        Pobj = (pose @ Vobj_h.T).T[:, :3].astype(np.float32)
        meshes = []
        ov_verts = torch.tensor(Pobj, device=device)
        ov_faces = torch.tensor(Fobj, device=device)
        ov_col = torch.tensor(OBJ_COLOR_RGB, device=device).repeat(len(Pobj), 1)
        meshes.append(Meshes([ov_verts], [ov_faces],
                             textures=TexturesVertex([ov_col])))
        if hand_kind == "mesh":
            hv = torch.tensor(hverts[i], device=device)
            hf = torch.tensor(hfaces, device=device)
        else:
            hv_np, hf_np = skeleton_to_mesh(hjoints[i])
            hv = torch.tensor(hv_np, device=device)
            hf = torch.tensor(hf_np, device=device)
        hcol = torch.tensor(HAND_COLOR_RGB, device=device).repeat(len(hv), 1)
        meshes.append(Meshes([hv], [hf], textures=TexturesVertex([hcol])))
        scene = join_meshes_as_scene(meshes)

        azim = -40.0 + 80.0 * (i / max(1, n - 1))
        elev = 12.0
        Rv, Tv = look_at_view_transform(
            dist=cam_dist, elev=elev, azim=azim,
            at=(tuple(scene_center.tolist()),), up=((0, -1, 0),), device=device)
        cam3d = FoVPerspectiveCameras(device=device, R=Rv, T=Tv, fov=45)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cam3d, raster_settings=raster_set),
            shader=SoftPhongShader(device=device, cameras=cam3d, lights=lights,
                                   blend_params=blend))
        img = renderer(scene)[0, ..., :3].clamp(0, 1).cpu().numpy()
        frame3d = (img[..., ::-1] * 255).astype(np.uint8).copy()

        trail.append(scene_center if False else obj_cent[i])
        cc = torch.tensor(np.array(trail), dtype=torch.float32, device=device)
        proj = cam3d.transform_points_screen(cc[None], image_size=((H, W),))[0]
        proj = proj.cpu().numpy()[:, :2]
        for k in range(1, len(proj)):
            a = max(0.0, 1.0 - (len(proj) - k) / 25.0)
            if a <= 0:
                continue
            p0 = tuple(np.round(proj[k - 1]).astype(int))
            p1 = tuple(np.round(proj[k]).astype(int))
            col = tuple(int(c * a) for c in (60, 200, 255))
            cv2.line(frame3d, p0, p1, col, 2, cv2.LINE_AA)

        frame3d = label(frame3d, f"{label_txt}  [3D orbit]")
        cv2.putText(frame3d, f"frame {i:03d}/{n-1}  azim {azim:+.0f}",
                    (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        vw_3d.write(frame3d)

        if i % 10 == 0:
            print(f"  [{source}] frame {i}/{n}")

    vw_ov.release()
    vw_3d.release()
    print(f"[{source}] wrote {out_overlay}\n[{source}] wrote {out_3d}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=["ours", "phase2", "gt"])
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--obj_poses", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--hand_kind", required=True, choices=["mesh", "skeleton"])
    ap.add_argument("--out_overlay", required=True)
    ap.add_argument("--out_3d", required=True)
    ap.add_argument("--fps", type=float, default=13.0)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    run(args.source, args.cam_dir, args.mesh, args.obj_poses, args.hand,
        args.hand_kind, args.out_overlay, args.out_3d, args.fps, args.max_frames,
        args.device)
