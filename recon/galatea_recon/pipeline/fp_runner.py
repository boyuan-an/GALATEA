import argparse
import os
import sys
from glob import glob

import cv2
import numpy as np
import trimesh
from galatea_recon.pipeline.observation_gaps import (
    fill_poses, object_observations, observation_path, pose_observed,
)


def load_K(cam_dir):
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(cam_dir, "cam_intrinsics.txt")).tolist()
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    return K, int(W), int(H)


def load_obj_pose0(cam_dir):
    f = sorted(glob(os.path.join(cam_dir, "obj_pose_rt", "*.txt")))[0]
    v = np.loadtxt(f).astype(np.float64)
    return v[1:17].reshape(4, 4)


def depth_provider(mode, cam_dir, depth_dir, i, frame_name, H, W, depth_scale):
    if mode == "zero":
        return np.zeros((H, W), np.float32)
    if mode == "measured":
        p = os.path.join(cam_dir, "depth", frame_name)
        d = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        return (d.astype(np.float32) / depth_scale)
    if mode == "npy":
        p = os.path.join(depth_dir, f"{i:06d}.npy")
        return np.load(p).astype(np.float32)
    raise ValueError(mode)


def track_one_hypos(est, rgb, depth_np, K, rel_tfs, iteration):
    import numpy as np
    import torch
    from Utils import bilateral_filter_depth, depth2xyzmap_batch, erode_depth

    depth = torch.as_tensor(depth_np, device="cuda", dtype=torch.float)
    depth = erode_depth(depth, radius=2, device="cuda")
    depth = bilateral_filter_depth(depth, radius=2, device="cuda")
    K_t = torch.as_tensor(K, dtype=torch.float, device="cuda")
    xyz_map = depth2xyzmap_batch(depth[None], K_t[None], zfar=np.inf)[0]

    prev = est.pose_last.reshape(1, 4, 4).data.cpu().numpy()[0]
    hypos = np.asarray([prev] + [tf @ prev for tf in rel_tfs], dtype=np.float32)

    refined, _ = est.refiner.predict(
        mesh=est.mesh, mesh_tensors=est.mesh_tensors, rgb=rgb, depth=depth, K=K,
        ob_in_cams=hypos, normal_map=None, xyz_map=xyz_map,
        mesh_diameter=est.diameter, glctx=est.glctx, iteration=iteration, get_vis=False)
    all_to_score = torch.cat(
        [torch.as_tensor(hypos, device="cuda", dtype=torch.float), refined], dim=0)
    scores, _ = est.scorer.predict(
        mesh=est.mesh, rgb=rgb, depth=depth, K=K,
        ob_in_cams=all_to_score.data.cpu().numpy(), normal_map=None,
        mesh_tensors=est.mesh_tensors, glctx=est.glctx,
        mesh_diameter=est.diameter, get_vis=False)
    best = int(torch.as_tensor(scores).argmax())
    est.pose_last = all_to_score[best]
    return (all_to_score[best] @ est.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4, 4)


def _bake_solid_color(mesh, rgb_path, sam2_masks_path, key="obj_0"):
    sm = np.load(sam2_masks_path)
    if key not in sm.files:
        raise KeyError(f"{key} not in {sam2_masks_path} (keys e.g. {sm.files[:4]})")
    mask = np.asarray(sm[key]).astype(bool)
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    if rgb.shape[:2] != mask.shape:
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]))
    pix = rgb[mask]
    if len(pix) == 0:
        raise ValueError(f"empty {key} mask, cannot sample object color")
    color = np.median(pix, axis=0).astype(np.uint8)
    rgba = np.concatenate([color, [255]]).astype(np.uint8)
    vc = np.tile(rgba, (len(mesh.vertices), 1))
    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vc)
    return color


def _ensure_texture(mesh, mesh_path):
    from PIL import Image
    mat = getattr(getattr(mesh, "visual", None), "material", None)
    if mat is not None and getattr(mat, "image", None) is None:
        d = os.path.dirname(mesh_path)
        imgs = [f for f in sorted(os.listdir(d)) if f.lower().endswith((".png", ".jpg"))]
        if imgs:
            mat.image = Image.open(os.path.join(d, imgs[0])).convert("RGB")
            print(f"[fp] attached texture {imgs[0]} to mesh material")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fp_dir",
                    default=os.environ.get("GALATEA_FP_DIR") or os.path.normpath(os.path.join(
                        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir,
                        "imports", "FoundationPose")))
    ap.add_argument("--depth_mode", choices=["measured", "zero", "npy"], default="measured")
    ap.add_argument("--depth_dir", default=None, help="dir of <frame>.npy for --depth_mode npy")
    ap.add_argument("--rel_transform_path", default=None,
                    help="(variant 4) SpaTracker relative-transform candidates "
                         "(T-1,K,4,4); injected as FP init hypotheses, scorer selects")
    ap.add_argument("--depth_scale", type=float, default=1000.0)
    ap.add_argument("--mesh_scale", type=float, default=1.0)
    ap.add_argument("--track_refine_iter", type=int, default=2)
    ap.add_argument("--init_mode", choices=["register", "gt"], default="register",
                    help="frame-0 pose: register=FP global registration (no GT, "
                         "default); gt=GT first-frame pose (comparison only)")
    ap.add_argument("--sam2_masks", default=None,
                    help="sam2_masks.npz (key obj_0) for register-mode frame-0 mask")
    ap.add_argument("--init_depth_mode", choices=["measured", "npy"], default="measured",
                    help="frame-0 depth for register: measured=known first-frame "
                         "sensor depth (default, an allowed assumption); npy=--depth_dir")
    ap.add_argument("--est_refine_iter", type=int, default=5,
                    help="refine iterations for register-mode frame-0 pose")
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--bake_color", choices=["none", "auto"], default="none",
                    help="auto=paint the mesh with the object's true solid color "
                         "sampled from the frame-0 RGB inside the SAM2 obj_0 mask "
                         "(needs --sam2_masks); none=leave mesh gray (default)")
    ap.add_argument("--fill-missing", action="store_true",
                    help="skip missing/clipped masks, re-register on return, interpolate gaps")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_path = os.path.abspath(os.path.expanduser(args.out))
    cam_dir = os.path.abspath(os.path.expanduser(args.cam_dir))
    mesh_path = os.path.abspath(os.path.expanduser(args.mesh))
    depth_dir = os.path.abspath(os.path.expanduser(args.depth_dir)) if args.depth_dir else None
    if args.sam2_masks:
        args.sam2_masks = os.path.abspath(os.path.expanduser(args.sam2_masks))

    sys.path.insert(0, args.fp_dir)
    os.chdir(args.fp_dir)
    import nvdiffrast.torch as dr
    import torch
    from estimater import FoundationPose, PoseRefinePredictor, ScorePredictor

    K, W, H = load_K(cam_dir)
    rgb_files = sorted(glob(os.path.join(cam_dir, "rgb", "*.png")))
    if args.max_frames:
        rgb_files = rgb_files[: args.max_frames]
    n = len(rgb_files)
    if n == 0:
        raise ValueError("no RGB frames")
    sam_masks = np.load(args.sam2_masks) if args.sam2_masks else None
    first_frame = 0
    visibility = np.ones(n, dtype=bool)
    if args.fill_missing:
        if sam_masks is None or args.init_mode != "register":
            raise ValueError("--fill-missing needs --sam2_masks and --init_mode register")
        visibility = object_observations(sam_masks, n)
        for first_frame in range(n):
            if not visibility[first_frame]:
                continue
            initial_depth = depth_provider(
                args.init_depth_mode if first_frame == 0 else args.depth_mode,
                cam_dir, depth_dir, first_frame, os.path.basename(rgb_files[first_frame]),
                H, W, args.depth_scale)
            mask = np.asarray(sam_masks[f"obj_{first_frame}"], dtype=bool)
            if np.count_nonzero(mask & np.isfinite(initial_depth) & (initial_depth > .001)) < 4:
                visibility[first_frame] = False
                continue
            break
        else:
            raise ValueError("object is unobserved in every frame; cannot initialize or interpolate")
    print(f"[fp] {n} frames | depth_mode={args.depth_mode} | mesh={os.path.basename(args.mesh)}")

    mesh = trimesh.load(mesh_path, force="mesh")
    _ensure_texture(mesh, mesh_path)
    if args.mesh_scale != 1.0:
        mesh.apply_scale(args.mesh_scale)
    if args.bake_color == "auto":
        if not args.sam2_masks:
            raise ValueError("--bake_color auto needs --sam2_masks (obj_0 mask)")
        c = _bake_solid_color(mesh, rgb_files[first_frame], args.sam2_masks, key=f"obj_{first_frame}")
        print(f"[fp] baked solid vertex color RGB={tuple(int(x) for x in c)} "
              f"(sampled from first observed obj mask) — mesh no longer flat gray")
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    dbg = os.path.join(os.path.dirname(os.path.abspath(out_path)), "fp_debug")
    os.makedirs(dbg, exist_ok=True)
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals,
                         mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx,
                         debug=0, debug_dir=dbg)

    if args.init_mode == "gt":
        ob_in_cam = load_obj_pose0(cam_dir)
        adj = ob_in_cam.copy()
        adj[:3, 3] += adj[:3, :3] @ est.model_center
        est.pose_last = torch.as_tensor(adj, dtype=torch.float, device="cuda")
        pose0 = ob_in_cam.copy()
        print("[fp] init_mode=gt (frame-0 pose = GT)")
    else:
        if not args.sam2_masks:
            raise ValueError("--init_mode register needs --sam2_masks (obj_0 mask)")
        ob_mask0 = np.asarray(sam_masks[f"obj_{first_frame}"]).astype(bool)
        rgb0 = cv2.cvtColor(cv2.imread(rgb_files[first_frame]), cv2.COLOR_BGR2RGB)
        if rgb0.shape[:2] != (H, W):
            rgb0 = cv2.resize(rgb0, (W, H))
        depth0 = depth_provider(args.init_depth_mode if first_frame == 0 else args.depth_mode,
                                cam_dir, depth_dir, first_frame,
                                os.path.basename(rgb_files[first_frame]), H, W, args.depth_scale)
        pose0 = est.register(K=K, rgb=rgb0, depth=depth0, ob_mask=ob_mask0,
                             iteration=args.est_refine_iter)
        pose0 = np.asarray(pose0).reshape(4, 4)
        if args.fill_missing and not pose_observed(pose0):
            raise RuntimeError("FoundationPose returned an invalid initial pose")
        print(f"[fp] registered first observed frame={first_frame}, mask={int(ob_mask0.sum())}px")

    rel_cands = None
    if args.rel_transform_path:
        rel_cands = np.load(os.path.abspath(os.path.expanduser(args.rel_transform_path)))
        print(f"[fp] variant-4: rel-transform candidates {rel_cands.shape} (FP hypotheses)")

    poses = []
    observed = np.zeros(n, dtype=bool)
    reacquired = []
    for i, rgb_path in enumerate(rgb_files):
        if args.fill_missing and (i < first_frame or not visibility[i]):
            poses.append(np.full((4, 4), np.nan))
            continue
        color = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        if color.shape[:2] != (H, W):
            color = cv2.resize(color, (W, H))
        depth = depth_provider(args.depth_mode, cam_dir, depth_dir, i,
                               os.path.basename(rgb_path), H, W, args.depth_scale)
        if i == first_frame:
            pose = pose0.copy()
        elif args.fill_missing and not observed[i - 1]:
            mask = np.asarray(sam_masks[f"obj_{i}"], dtype=bool)
            if np.count_nonzero(mask & np.isfinite(depth) & (depth > .001)) < 4:
                poses.append(np.full((4, 4), np.nan))
                continue
            pose = est.register(K=K, rgb=color, depth=depth, ob_mask=mask,
                                iteration=args.est_refine_iter)
            reacquired.append(i)
        elif rel_cands is not None:
            rel_tfs = rel_cands[i - 1] if (i - 1) < len(rel_cands) else []
            pose = track_one_hypos(est, color, depth, K, rel_tfs, args.track_refine_iter)
        else:
            pose = est.track_one(rgb=color, depth=depth, K=K, iteration=args.track_refine_iter)
        if args.fill_missing and not pose_observed(pose):
            poses.append(np.full((4, 4), np.nan))
            continue
        observed[i] = True
        poses.append(np.asarray(pose).reshape(4, 4))
        if i % 50 == 0:
            print(f"  frame {i}/{n}")

    poses = np.stack(poses, 0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if args.fill_missing:
        poses = fill_poses(poses, observed)
        np.savez(observation_path(out_path), object_observed=observed,
                 reacquired_frames=np.asarray(reacquired, dtype=np.int64))
    np.save(out_path, poses)
    print(f"[fp] saved {poses.shape} -> {out_path}")


if __name__ == "__main__":
    main()
