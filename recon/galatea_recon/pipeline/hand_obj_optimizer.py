import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "16")
import argparse
import json
import math
from glob import glob

import cv2
import numpy as np
import torch
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix

from galatea_recon.pipeline import contact_source as cs
from galatea_recon.pipeline.contact_boundary import boundary_inputs, protect_contact_boundaries
from galatea_recon.pipeline.fp_constraints import (
    add_constraint_arguments, center_depth, constraint_config, depth_prior,
    fp_frame_validity, frame_sum_l1, soft_mask_iou,
)
from galatea_recon.pipeline.optimization_safety import (
    REPORT_VERSION, UnsafeOptimizationError, optimize_stage, write_report,
)
from galatea_recon.pipeline.observation_gaps import (
    complete_hands, complete_output_hands, fill_poses, load_object_observed,
    hand_output_anchors, mask_observed, observation_path,
)
from galatea_recon.optimization.loss_terms import (
    bidirectional_chamfer_loss, contact_depth_loss, contact_distribution_smoothness_loss,
    contact_loss, contact_smoothness_loss, keypoint_loss, l1_loss, smoothness_loss)

CONTACT_VIDS = [744, 320, 443, 554, 671, 73, 96, 119, 157, 290]
FINGERTIP_KP = [4, 8, 12, 16, 20]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


def _observed_hand_silhouette(foreground, object_mask, keep, min_pixels=20):
    hand = (np.asarray(foreground, dtype=bool)
            & ~np.asarray(object_mask, dtype=bool)
            & np.asarray(keep, dtype=bool))
    if int(hand.sum()) < min_pixels:
        return None
    return hand.astype(np.float32)


def build_silhouette_renderer(K, H, W, scale, n, device):
    import torch
    from pytorch3d.renderer import (MeshRasterizer, RasterizationSettings,
                                     SoftSilhouetteShader, MeshRenderer)
    from pytorch3d.utils import cameras_from_opencv_projection
    Hs, Ws = int(H * scale), int(W * scale)
    Ks = K.clone(); Ks[:2] *= scale
    cam = cameras_from_opencv_projection(
        R=torch.eye(3, device=device)[None].repeat(n, 1, 1),
        tvec=torch.zeros(n, 3, device=device), camera_matrix=Ks[None].repeat(n, 1, 1),
        image_size=torch.tensor([[Hs, Ws]], dtype=torch.float32, device=device).repeat(n, 1))
    raster = RasterizationSettings(image_size=(Hs, Ws), blur_radius=1e-4, faces_per_pixel=12)
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cam, raster_settings=raster),
                            shader=SoftSilhouetteShader())
    return renderer, (Hs, Ws)


def build_mano(device):
    from galatea_recon.adapters.wilor_assets import mano_right_path
    pkl = mano_right_path()
    from wilor_mini.models.mano_wrapper import MANO
    mano = MANO(model_path=os.path.dirname(pkl), is_rhand=True, use_pca=False,
                flat_hand_mean=False).to(device)
    lbs_seg = mano.lbs_weights.argmax(1)
    return mano, lbs_seg


def seg_lbs_idx(finger, k):
    return 1 + 3 * FINGER_NAMES.index(finger) + (k - 1)


def finger_vertex_groups(verts, kp3d):
    import torch
    anchors = kp3d[[0] + FINGERTIP_KP]
    d = torch.cdist(verts, anchors)
    grp = d.argmin(1)
    return {FINGER_NAMES[k - 1]: torch.where(grp == k)[0] for k in range(1, 6)}


def load_K(cam_dir):
    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(cam_dir, "cam_intrinsics.txt"))
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64), int(W), int(H)


def residual_pose(r6, t):
    R = rotation_6d_to_matrix(r6)
    T = torch.eye(4, device=r6.device).repeat(R.shape[0], 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t
    return T


def project(K, pts):
    uv = pts @ K.T
    return uv[..., :2] / uv[..., 2:3].clamp(min=1e-6)


def backproject(depth, K, mask):
    ys, xs = np.where(mask & (depth > 0.05) & (depth < 8.0))
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32)
    z = depth[ys, xs]
    x = (xs - K[0, 2]) * z / K[0, 0]
    y = (ys - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1).astype(np.float32)


def effective_depth_weight(args):
    weight = 100.0 if args.manip_weights else args.w_depth
    if args.w_depth_override >= 0:
        weight = args.w_depth_override
    return float(weight)


class AdaptiveEarlyStopper:
    def __init__(self, min_iterations, check_every, patience, rel_thresholds):
        if min_iterations < 1:
            raise ValueError("min_iterations must be >= 1")
        if check_every < 1:
            raise ValueError("check_every must be >= 1")
        if patience < 1:
            raise ValueError("patience must be >= 1")
        if not rel_thresholds or any(value <= 0 for value in rel_thresholds.values()):
            raise ValueError("relative thresholds must be non-empty and > 0")
        self.min_iterations = min_iterations
        self.check_every = check_every
        self.patience = patience
        self.rel_thresholds = dict(rel_thresholds)
        self.best = {}
        self.stale_checks = 0
        self.armed = False

    def observe(self, completed_iterations, metrics):
        if completed_iterations % self.check_every != 0:
            return None
        active = {
            name: float(metrics[name])
            for name in self.rel_thresholds
            if name in metrics and math.isfinite(float(metrics[name]))
        }
        if not active:
            return None

        if completed_iterations < self.min_iterations:
            return {
                "iteration": completed_iterations,
                "armed": False,
                "stop": False,
                "stale_checks": 0,
                "metrics": active,
            }

        if not self.armed:
            self.best = dict(active)
            self.stale_checks = 0
            self.armed = True
            return {
                "iteration": completed_iterations,
                "armed": True,
                "stop": False,
                "stale_checks": 0,
                "metrics": active,
                "best": dict(self.best),
                "relative_improvement": {name: 0.0 for name in active},
            }

        meaningful = False
        relative_improvement = {}
        for name, current in active.items():
            if name not in self.best:
                self.best[name] = current
                relative_improvement[name] = 0.0
                continue
            previous_best = self.best[name]
            improvement = (previous_best - current) / max(abs(previous_best), 1e-12)
            relative_improvement[name] = improvement
            if improvement >= self.rel_thresholds[name]:
                self.best[name] = current
                meaningful = True

        self.stale_checks = 0 if meaningful else self.stale_checks + 1
        should_stop = self.stale_checks >= self.patience
        return {
            "iteration": completed_iterations,
            "armed": True,
            "stop": should_stop,
            "stale_checks": self.stale_checks,
            "metrics": active,
            "best": dict(self.best),
            "relative_improvement": relative_improvement,
        }


from galatea_recon.pipeline.object_recovery import load_fp_confidence, perception_identity
from galatea_recon.pipeline.trajectory_cache import (
    normalize_optimization_args, optimization_cache_complete, optimization_inputs,
    optimization_outputs, output_hashes, validate_optimization_outputs,
)


def run(args):
    normalize_optimization_args(args)
    report_path = os.path.splitext(args.out)[0] + "_optimization.json"
    report = {"schema_version": REPORT_VERSION, "state": "running", "stages": [],
              "numerical_status": "unchecked", "geometry_quality": "not_assessed"}
    write_report(report_path, report)
    try:
        report["cache_inputs"] = optimization_inputs(args)
        _run(args, report)
        validate_optimization_outputs(args, report["cache_inputs"])
        report["outputs"] = output_hashes(optimization_outputs(args))
    except Exception as error:
        report["state"] = "failed"
        report["error"] = "%s: %s" % (type(error).__name__, error)
        write_report(report_path, report)
        raise
    report["state"] = "complete"
    report["numerical_status"] = "finite"
    report["optimization_status"] = (
        "recovered_nonfinite" if any(s["recovered_nonfinite"] for s in report["stages"])
        else "restored_best" if any(s["restored_best"] for s in report["stages"])
        else "best_at_final_step"
    )
    write_report(report_path, report)
    print("[phase3] optimization status=%s; audit -> %s"
          % (report["optimization_status"], report_path))


def _run(args, optimization_report):
    dev = "cuda"
    fp_config = constraint_config(args.w_fp_depth, args.fp_depth_free_m,
                                  args.fp_depth_free_fraction, args.fp_depth_evidence_gain)
    boundary_enabled = not getattr(args, "no_contact_boundary_guard", False)
    boundary_identity = boundary_inputs(args.cam_dir, args.seg_labels, args.manual_json,
                                        args.max_frames, boundary_enabled)
    optimization_report["contact_boundary_inputs"] = boundary_identity
    optimization_report["fp_constraints"] = fp_config
    optimization_report["fp_tracking_enabled"] = args.verts_tracking
    optimization_report["perception_inputs"] = perception_identity(args.fp_poses, args.sam2_masks)
    K_np, W, H = load_K(args.cam_dir)
    K = torch.tensor(K_np, dtype=torch.float32, device=dev)
    rgb_files = sorted(glob(os.path.join(args.cam_dir, "rgb", "*.png")))[:args.max_frames]
    n = len(rgb_files)
    fill_missing = getattr(args, "fill_missing", False)
    obj_masks = (np.load(args.sam2_masks)
                 if args.sam2_masks and os.path.exists(args.sam2_masks) else None)
    object_observed = (load_object_observed(args.fp_poses, n, obj_masks)
                       if fill_missing else np.ones(n, dtype=bool))
    object_valid = torch.tensor(object_observed, device=dev)
    fp_confidence = torch.tensor(load_fp_confidence(args.fp_poses, n), device=dev)

    fp = np.load(args.fp_poses)[:n]
    if fill_missing:
        fp = fill_poses(fp, object_observed)
    fp = torch.tensor(fp, dtype=torch.float32, device=dev)
    import trimesh
    m = trimesh.load(args.mesh, force="mesh")
    objV = torch.tensor(np.asarray(m.vertices), dtype=torch.float32, device=dev)

    wd = np.load(args.wilor_hands, allow_pickle=True)
    hands = wd["hands"]
    if fill_missing:
        if len(hands) < n:
            raise ValueError("hand archive is shorter than the RGB sequence")
        hands, hand_observed = complete_hands(
            hands[:n], wd["hand_observed"][:n] if "hand_observed" in wd else None,
            (W, H), require_mano=args.use_mano)
    else:
        hand_observed = np.array([
            fr < len(hands) and bool(len(hands[fr])) for fr in range(n)
        ], dtype=bool)
    hand_valid = torch.tensor(hand_observed, device=dev)
    hand_v0, hand_kp3d0, hand_kp2d, hand_side, has_hand = [], [], [], [], []
    for fr in range(n):
        fh = hands[fr] if fr < len(hands) else []
        if fh:
            h = fh[0]
            hand_v0.append(h["verts"]); hand_kp3d0.append(h["kp3d"])
            hand_kp2d.append(h["kp2d"]); hand_side.append("right" if h["is_right"] > 0.5 else "left")
            has_hand.append(True)
        else:
            hand_v0.append(np.zeros((778, 3), np.float32)); hand_kp3d0.append(np.zeros((21, 3), np.float32))
            hand_kp2d.append(np.zeros((21, 2), np.float32)); hand_side.append("right"); has_hand.append(False)
    hand_v0 = torch.tensor(np.stack(hand_v0), dtype=torch.float32, device=dev)
    hand_kp3d0 = torch.tensor(np.stack(hand_kp3d0), dtype=torch.float32, device=dev)
    hand_kp2d = torch.tensor(np.stack(hand_kp2d), dtype=torch.float32, device=dev)
    has_hand = torch.tensor(has_hand, device=dev)

    if args.finer_contact:
        contact = cs.from_manual_fingers(args.manual_json, args.seq_id, n)
        finger_vids = finger_vertex_groups(hand_v0[0], hand_kp3d0[0])
        from pytorch3d.utils import cameras_from_opencv_projection
        cam_p3d = cameras_from_opencv_projection(
            R=torch.eye(3, device=dev)[None], tvec=torch.zeros(1, 3, device=dev),
            camera_matrix=K[None], image_size=torch.tensor([[H, W]], dtype=torch.float32, device=dev))
        fingertip_vids = torch.tensor(sum([finger_vids[f].tolist() for f in FINGER_NAMES], []),
                                      dtype=torch.long, device=dev)
    else:
        contact = cs.load_contact(args.contact_mode, n, seq_id=args.seq_id,
                                  manual_json=args.manual_json)

    mano = lbs_seg = dpose = finger_mask = seg_contact = None
    go0 = hp0 = betas0 = place_R = place_t = None
    objF = None
    if args.use_mano:
        mano, lbs_seg = build_mano(dev)
        objF = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=dev)
        gl, hl, bl, rl, tl = [], [], [], [], []
        for fr in range(n):
            fh = hands[fr] if fr < len(hands) else []
            h = fh[0] if (fh and "global_orient" in fh[0]) else None
            if h is not None:
                gl.append(h["global_orient"].reshape(3)); hl.append(h["hand_pose"].reshape(15, 3))
                bl.append(h["betas"].reshape(10)); rl.append(h["place_R"]); tl.append(h["place_t"])
            else:
                gl.append(np.zeros(3, np.float32)); hl.append(np.zeros((15, 3), np.float32))
                bl.append(np.zeros(10, np.float32)); rl.append(np.eye(3, dtype=np.float32)); tl.append(np.zeros(3, np.float32))
        go0 = torch.tensor(np.stack(gl), device=dev); hp0 = torch.tensor(np.stack(hl), device=dev)
        betas0 = torch.tensor(np.stack(bl), device=dev)
        place_R = torch.tensor(np.stack(rl), device=dev); place_t = torch.tensor(np.stack(tl), device=dev)
        spec = json.load(open(args.seg_labels)).get(str(args.seq_id), []) if args.seg_labels else []
        seg_contact = [set() for _ in range(n)]
        active = set()
        for iv in spec:
            if iv["hand"] != "right":
                continue
            for fr in range(max(0, iv["start"]), min(n - 1, iv["end"]) + 1):
                seg_contact[fr].add((iv["finger"], iv["seg"]))
            active.add(iv["finger"])
        dpose = torch.zeros(n, 15, 3, device=dev, requires_grad=args.unfreeze)
        finger_mask = torch.zeros(15, device=dev)
        for f in active:
            finger_mask[3 * FINGER_NAMES.index(f):3 * FINGER_NAMES.index(f) + 3] = 1.0

    depth_weight = effective_depth_weight(args)
    depth_clouds, hand_clouds = {}, {}
    if depth_weight != 0.0:
        for fr in range(n):
            dp = os.path.join(args.depth_dir, f"{fr:06d}.npy")
            if obj_masks is not None and os.path.exists(dp):
                depth = np.load(dp)
                okey = f"obj_{fr}"
                if object_observed[fr] and okey in obj_masks.files:
                    pc = backproject(depth, K_np, np.asarray(obj_masks[okey]) > 0)
                    if len(pc) > 30:
                        depth_clouds[fr] = torch.tensor(pc, device=dev)
                if args.hand_depth and hand_valid[fr]:
                    uv = (K_np @ hand_v0[fr].cpu().numpy().T).T
                    uv = (uv[:, :2] / uv[:, 2:3]).astype(int)
                    x0, y0 = uv.min(0); x1, y1 = uv.max(0)
                    mg = max(8, int(0.15 * max(x1 - x0, y1 - y0)))
                    box = np.zeros(depth.shape, bool)
                    box[max(0, y0 - mg):y1 + mg, max(0, x0 - mg):x1 + mg] = True
                    objm = ((np.asarray(obj_masks[okey]) > 0)
                            if okey in obj_masks.files else False)
                    pc = backproject(depth, K_np, box & ~objm)
                    if len(pc) > 30:
                        hand_clouds[fr] = torch.tensor(pc, device=dev)
    else:
        print("[phase3] depth weight is zero; skipping depth-cloud and Chamfer work")

    sil_renderer = sil_tgt = Fc = None
    sil_tgt_hand = hand_faces = None
    if (args.silhouette_z or args.hand_silhouette_z) and obj_masks is not None:
        sil_renderer, (Hs, Ws) = build_silhouette_renderer(K, H, W, args.sil_scale, n, dev)
    if args.silhouette_z and obj_masks is not None:
        Fc = torch.tensor(np.asarray(m.faces), dtype=torch.int64, device=dev)
        sil_obj = torch.zeros(n, Hs, Ws, device=dev)
        sil_valid = torch.ones(n, Hs, Ws, device=dev)
        for fr in range(n):
            okey, fkey = f"obj_{fr}", f"fg_{fr}"
            if (okey not in obj_masks.files or not object_observed[fr]
                    or not mask_observed(obj_masks[okey])):
                sil_valid[fr] = 0; continue
            om = (np.asarray(obj_masks[okey]) > 0).astype(np.float32)
            hand = np.zeros_like(om)
            if fkey in obj_masks.files:
                hand = ((np.asarray(obj_masks[fkey]) > 0) & (om == 0)).astype(np.float32)
            sil_obj[fr] = torch.tensor(cv2.resize(om, (Ws, Hs), interpolation=cv2.INTER_AREA), device=dev)
            sil_valid[fr] = 1.0 - torch.tensor(cv2.resize(hand, (Ws, Hs), interpolation=cv2.INTER_AREA), device=dev)
        sil_tgt = (sil_obj, sil_valid)
    if args.hand_silhouette_z and obj_masks is not None:
        hf_np = np.asarray(mano.faces if args.use_mano else build_mano(dev)[0].faces)
        hand_faces = torch.tensor(hf_np.astype(np.int64), device=dev)
        sil_h = torch.zeros(n, Hs, Ws, device=dev)
        sil_hvalid = torch.zeros(n, Hs, Ws, device=dev)
        for fr in range(n):
            fkey, okey = f"fg_{fr}", f"obj_{fr}"
            if fkey not in obj_masks.files or not bool(hand_valid[fr]):
                continue
            fg = np.asarray(obj_masks[fkey]) > 0
            om = (np.asarray(obj_masks[okey]) > 0) if okey in obj_masks.files else np.zeros_like(fg)
            kp = hand_kp2d[fr].detach().cpu().numpy()
            wrist = kp[0]; fwd = kp[1:].mean(0) - wrist
            fwd = fwd / (np.linalg.norm(fwd) + 1e-6)
            yy, xx = np.mgrid[0:fg.shape[0], 0:fg.shape[1]]
            anchor = wrist - args.hand_hull_dilate * fwd
            keep = ((xx - anchor[0]) * fwd[0] + (yy - anchor[1]) * fwd[1]) > 0
            hnd = _observed_hand_silhouette(fg, om, keep)
            if hnd is None:
                continue
            sil_h[fr] = torch.tensor(cv2.resize(hnd, (Ws, Hs), interpolation=cv2.INTER_AREA), device=dev)
            sil_hvalid[fr] = 1.0 - torch.tensor(
                cv2.resize(om.astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_AREA), device=dev)
        sil_tgt_hand = (sil_h, sil_hvalid)

    obj_r6 = torch.zeros(n, 6, device=dev, requires_grad=True)
    obj_t = torch.zeros(n, 3, device=dev, requires_grad=True)
    hand_r6 = torch.zeros(n, 6, device=dev, requires_grad=True)
    hand_t = torch.zeros(n, 3, device=dev, requires_grad=True)
    with torch.no_grad():
        obj_r6[:, 0] = 1.0; obj_r6[:, 4] = 1.0
        hand_r6[:, 0] = 1.0; hand_r6[:, 4] = 1.0
    objV_h = torch.cat([objV, torch.ones(len(objV), 1, device=dev)], 1)
    S = min(1500, len(objV))
    sub = torch.linspace(0, len(objV) - 1, S).long()
    subV_h = objV_h[sub]
    subN = None
    if depth_weight != 0.0:
        objN = torch.tensor(np.asarray(m.vertex_normals), dtype=torch.float32, device=dev)
        subN = objN[sub]
    with torch.no_grad():
        fp2d = project(K, torch.einsum("nij,vj->nvi", fp, subV_h)[..., :3])

    fp_valid = torch.tensor(fp_frame_validity(
        fp.detach().cpu().numpy(), obj_masks, object_observed), device=dev)
    center_h = torch.cat([(objV.amin(0) + objV.amax(0)) / 2, objV.new_ones(1)])
    fp_z = center_depth(fp, center_h).detach()
    fp_free = max(fp_config["free_m"], fp_config["free_fraction"] *
                  float(torch.linalg.vector_norm(objV.amax(0) - objV.amin(0))))
    fp_iou = None
    fp_depth_valid = fp_valid.clone()
    if args.verts_tracking and fp_config["weight"] > 0 and sil_tgt is not None:
        from pytorch3d.structures import Meshes
        with torch.no_grad():
            initial_vertices = torch.einsum("nij,vj->nvi", fp, objV_h)[..., :3]
            initial_sil = sil_renderer(Meshes(list(initial_vertices), [Fc] * n))[..., 3]
            fp_iou = soft_mask_iou(initial_sil, *sil_tgt).detach()
            fp_depth_valid &= (sil_tgt[0] * sil_tgt[1]).flatten(1).sum(1) > 0
            del initial_sil, initial_vertices
    optimization_report["fp_constraint_diagnostics"] = dict(
        observed_frames=int(fp_valid.sum()), depth_observed_frames=int(fp_depth_valid.sum()),
        confidence_mean=float(fp_confidence.mean()),
        zero_confidence_frames=int((fp_confidence == 0).sum()),
        free_distance_m=fp_free,
        evidence_source="occlusion_aware_silhouette" if fp_iou is not None else "none",
        reference_iou_mean=float(fp_iou[fp_depth_valid].mean())
            if fp_iou is not None and bool(fp_depth_valid.any()) else None,
    )

    if args.manip_weights:
        W = dict(depth=depth_weight, verts_track=0.3, obj_reg=0.0, obj_smooth=50.0,
                 hand_smooth=30.0, hand_reg=3000.0, kp=0.3, contact=30000.0, c_smooth=5.0)
    else:
        W = dict(depth=depth_weight, verts_track=0.3, obj_reg=args.w_reg, obj_smooth=args.w_smooth,
                 hand_smooth=args.w_smooth, hand_reg=args.w_reg, kp=args.w_kp,
                 contact=args.w_contact, c_smooth=args.w_csmooth)
    W["fp_depth"] = fp_config["weight"] if args.verts_tracking else 0.0
    W["sil"] = args.w_silhouette
    W["sil_hand"] = args.w_hand_silhouette
    W["pose_reg"] = args.w_pose_reg; W["pose_smooth"] = args.w_pose_smooth
    if args.w_contact_override >= 0:
        W["contact"] = args.w_contact_override
    if args.w_hand_reg_override >= 0:
        W["hand_reg"] = args.w_hand_reg_override

    def losses(with_contact):
        D_obj, D_hand = residual_pose(obj_r6, obj_t), residual_pose(hand_r6, hand_t)
        obj_pose = D_obj @ fp
        sv = torch.einsum("nij,vj->nvi", obj_pose, subV_h)[..., :3]
        svN = (torch.einsum("nij,vj->nvi", obj_pose[:, :3, :3], subN)
               if subN is not None else None)
        if args.use_mano:
            hp_new = hp0 + dpose * finger_mask[None, :, None]
            out = mano(global_orient=axis_angle_to_matrix(go0.reshape(n, 1, 3)),
                       hand_pose=axis_angle_to_matrix(hp_new), betas=betas0, pose2rot=False)
            rel_v = out.vertices - out.joints[:, 0:1]
            rel_k = out.joints[:, :21] - out.joints[:, 0:1]
            placed_v = torch.einsum("nij,nvj->nvi", place_R, rel_v) + place_t[:, None]
            placed_k = torch.einsum("nij,nvj->nvi", place_R, rel_k) + place_t[:, None]
            hv = torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], placed_v) + D_hand[:, None, :3, 3]
            hk = torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], placed_k) + D_hand[:, None, :3, 3]
        else:
            hv = torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], hand_v0) + D_hand[:, None, :3, 3]
            hk = torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], hand_kp3d0) + D_hand[:, None, :3, 3]
        obj_verts_full = (torch.einsum("nij,vj->nvi", obj_pose, objV_h)[..., :3]
                          if args.use_mano else None)
        L = {k: torch.zeros((), device=dev) for k in W}
        if args.verts_tracking:
            L["verts_track"] = frame_sum_l1(project(K, sv), fp2d, fp_valid, fp_confidence)
        else:
            L["obj_reg"] = obj_t.pow(2).sum()
        L["obj_smooth"] = smoothness_loss(obj_pose[:, :3, 3])
        L["hand_smooth"] = smoothness_loss(hv.mean(1))
        L["hand_reg"] = hand_t.pow(2).sum()
        for fr in range(n):
            if hand_valid[fr]:
                L["kp"] = L["kp"] + keypoint_loss(project(K, hk[fr]), hand_kp2d[fr])
            if fr in depth_clouds:
                ov = sv[fr]
                if args.depth_vis:
                    vis = (svN[fr] * sv[fr]).sum(-1) < 0
                    if int(vis.sum()) > 10:
                        ov = sv[fr][vis]
                L["depth"] = L["depth"] + bidirectional_chamfer_loss(ov, depth_clouds[fr], trim_pct=0.2)
            if args.hand_depth and fr in hand_clouds:
                L["depth"] = L["depth"] + bidirectional_chamfer_loss(hv[fr], hand_clouds[fr], trim_pct=0.3)
            if not hand_valid[fr] or not object_valid[fr]:
                continue
            if with_contact and args.use_mano and seg_contact[fr]:
                from pytorch3d.loss import point_mesh_face_distance
                from pytorch3d.structures import Meshes, Pointclouds
                vids = torch.cat([torch.where(lbs_seg == seg_lbs_idx(f, k))[0]
                                  for (f, k) in seg_contact[fr]])
                L["contact"] = L["contact"] + point_mesh_face_distance(
                    Meshes([obj_verts_full[fr]], [objF]), Pointclouds([hv[fr][vids]]))
            elif with_contact and args.finer_contact:
                for fg in contact.get(fr, {}).get(hand_side[fr], set()):
                    vids = finger_vids[fg]
                    if len(vids) > 0:
                        L["contact"] = L["contact"] + contact_depth_loss(
                            hv[fr][vids], sv[fr], cam_p3d, top_k=30, delta=args.contact_delta)
        if with_contact and args.finer_contact:
            cf = [fr for fr in range(n)
                  if hand_valid[fr] and object_valid[fr]
                  and contact.get(fr, {}).get(hand_side[fr], set())]
            if len(cf) >= 2:
                idx = torch.tensor(cf, device=dev)
                ft, ovv = hv[idx][:, fingertip_vids], sv[idx]
                if ft.shape[1] > 0 and ovv.shape[1] > 0:
                    L["c_smooth"] = (contact_smoothness_loss(ft, ovv, top_k=30)
                                     + contact_distribution_smoothness_loss(ft, ovv))
        current_iou = None
        if args.silhouette_z and sil_renderer is not None:
            from pytorch3d.structures import Meshes
            ov_full = torch.einsum("nij,vj->nvi", obj_pose, objV_h)[..., :3]
            meshes = Meshes(verts=[ov_full[i] for i in range(n)], faces=[Fc] * n)
            sil = sil_renderer(meshes)[..., 3]
            obj_tgt, valid = sil_tgt
            L["sil"] = (((sil - obj_tgt) ** 2) * valid).sum() / (valid.sum() + 1.0)
            if fp_iou is not None:
                current_iou = soft_mask_iou(sil, obj_tgt, valid)
        if W["fp_depth"] > 0:
            L["fp_depth"] = depth_prior(
                center_depth(obj_pose, center_h), fp_z, fp_depth_valid, fp_free,
                current_iou, fp_iou, fp_config["evidence_gain"], fp_confidence)
        if args.hand_silhouette_z and sil_renderer is not None:
            from pytorch3d.structures import Meshes
            mh = Meshes(verts=[hv[i] for i in range(n)], faces=[hand_faces] * n)
            silh = sil_renderer(mh)[..., 3]
            htgt, hvalid = sil_tgt_hand
            L["sil_hand"] = (((silh - htgt) ** 2) * hvalid).sum() / (hvalid.sum() + 1.0)
        if args.use_mano and args.unfreeze:
            L["pose_reg"] = (dpose ** 2).sum()
            L["pose_smooth"] = ((dpose[1:] - dpose[:-1]) ** 2).sum()
        return sum(W[k] * L[k] for k in W), L

    if args.manip_weights:
        groups = [{"params": [obj_r6], "lr": 1e-4}, {"params": [obj_t], "lr": 3e-3},
                  {"params": [hand_r6], "lr": 1e-3}, {"params": [hand_t], "lr": 3e-3}]
    else:
        groups = [{"params": [obj_r6, obj_t, hand_r6, hand_t], "lr": args.lr}]
    if args.use_mano and args.unfreeze:
        groups.append({"params": [dpose], "lr": 5e-3})

    _contact_iters = args.contact_iters
    _print_every = max(1, int(os.environ.get("PRINT_EVERY", "100")))
    stages = [("geom", False, args.iters), ("contact", True, _contact_iters)] if args.two_stage \
        else [("single", args.finer_contact or args.use_mano, args.iters)]
    early_stop_audit = optimization_report["stages"]
    for name, with_contact, niter in stages:
        opt = torch.optim.Adam(groups)
        stopper = None
        if args.early_stop:
            minimum = args.early_stop_min_contact if with_contact else args.early_stop_min_geom
            thresholds = {"total": args.early_stop_rel_total}
            if with_contact:
                thresholds["contact"] = args.early_stop_rel_contact
            stopper = AdaptiveEarlyStopper(
                min_iterations=minimum,
                check_every=args.early_stop_check_every,
                patience=args.early_stop_patience,
                rel_thresholds=thresholds,
            )
        post_step = None
        if args.use_mano and args.unfreeze:
            post_step = lambda: dpose.clamp_(-args.pose_clamp, args.pose_clamp)
        try:
            stage_report = optimize_stage(
                name, opt, lambda: losses(with_contact), niter,
                stopper=stopper, post_step=post_step, print_every=_print_every,
            )
        except UnsafeOptimizationError as error:
            early_stop_audit.append(error.report)
            raise
        early_stop_audit.append(stage_report)
        if stage_report["recovered_nonfinite"]:
            optimization_report["skipped_stages"] = [
                stage[0] for stage in stages[len(early_stop_audit):]
            ]
            print("[phase3] numerical failure recovered; remaining stages skipped")
            break

    with torch.no_grad():
        final_obj_pose = residual_pose(obj_r6, obj_t) @ fp
        dz = (center_depth(final_obj_pose, center_h) - fp_z)[fp_valid].abs()
        optimization_report["fp_constraint_diagnostics"].update(
            mean_abs_depth_correction_m=float(dz.mean()) if dz.numel() else None,
            max_abs_depth_correction_m=float(dz.max()) if dz.numel() else None,
        )
        obj_pose = final_obj_pose.cpu().numpy()
        D_hand = residual_pose(hand_r6, hand_t)
        if args.use_mano:
            hp_new = hp0 + dpose * finger_mask[None, :, None]
            out = mano(global_orient=axis_angle_to_matrix(go0.reshape(n, 1, 3)),
                       hand_pose=axis_angle_to_matrix(hp_new), betas=betas0, pose2rot=False)
            pv = torch.einsum("nij,nvj->nvi", place_R, out.vertices - out.joints[:, 0:1]) + place_t[:, None]
            pk = torch.einsum("nij,nvj->nvi", place_R, out.joints[:, :21] - out.joints[:, 0:1]) + place_t[:, None]
            hv_ref = (torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], pv) + D_hand[:, None, :3, 3]).cpu().numpy()
            hand_kp3d_ref = (torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], pk) + D_hand[:, None, :3, 3]).cpu().numpy()
        else:
            hand_kp3d_ref = (torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], hand_kp3d0)
                             + D_hand[:, None, :3, 3]).cpu().numpy()
            hv_ref = (torch.einsum("nij,nvj->nvi", D_hand[:, :3, :3], hand_v0)
                      + D_hand[:, None, :3, 3]).cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    contact_frames = np.array([
        bool(seg_contact[fr]) if seg_contact is not None else
        (bool(contact.get(fr, {}).get(hand_side[fr], set())) if args.finer_contact
         else hand_side[fr] in contact.get(fr, set()))
        for fr in range(n)
    ], dtype=bool)
    if fill_missing:
        obj_pose = fill_poses(obj_pose, object_observed)
        output_anchors = hand_output_anchors(hand_observed, object_observed, contact_frames)
        hv_ref = complete_output_hands(hv_ref, obj_pose, output_anchors,
                                       object_observed, contact_frames)
        hand_kp3d_ref = complete_output_hands(hand_kp3d_ref, obj_pose, output_anchors,
                                             object_observed, contact_frames)
        np.savez(observation_path(args.out), object_observed=object_observed,
                 fp_reference_confidence=fp_confidence.detach().cpu().numpy(),
                 hand_observed=hand_observed, object_filled=~object_observed,
                 hand_filled=~output_anchors)
    obj_pose, hv_ref, hand_kp3d_ref, boundary_report = protect_contact_boundaries(
        obj_pose, hv_ref, hand_kp3d_ref, contact_frames,
        fps=boundary_identity["fps"], object_observed=object_observed,
        hand_observed=hand_observed, fp_poses=fp.detach().cpu().numpy(),
        enabled=boundary_enabled)
    optimization_report["contact_boundary_guard"] = boundary_report
    print("[phase3] contact boundary guard: %d object / %d hand frames adjusted"
          % (len(boundary_report["changed_object_frames"]),
             len(boundary_report["changed_hand_frames"])))
    for label, array in (("object poses", obj_pose), ("hand joints", hand_kp3d_ref),
                         ("hand vertices", hv_ref)):
        if not np.isfinite(array).all():
            raise RuntimeError("Non-finite %s after optimization/completion; refusing export" % label)
    np.save(args.out, obj_pose)
    np.save(args.out.replace(".npy", "_hand.npy"), hand_kp3d_ref)
    np.save(args.out.replace(".npy", "_handverts.npy"), hv_ref)
    if hand_faces is not None:
        np.save(os.path.join(os.path.dirname(os.path.abspath(args.out)), "mano_right_faces.npy"),
                hand_faces.cpu().numpy().astype(np.int64))
    if args.early_stop:
        audit_path = os.path.splitext(args.out)[0] + "_early_stop.json"
        with open(audit_path, "w") as output_file:
            json.dump({
                "enabled": True,
                "config": {
                    "check_every": args.early_stop_check_every,
                    "patience": args.early_stop_patience,
                    "min_geom": args.early_stop_min_geom,
                    "min_contact": args.early_stop_min_contact,
                    "relative_total": args.early_stop_rel_total,
                    "relative_contact": args.early_stop_rel_contact,
                },
                "stages": early_stop_audit,
            }, output_file, indent=2)
        print(f"[phase3] early-stop audit -> {audit_path}")
    print(f"[phase3] refined object + hand (verts+kp){' [MANO,unfreeze]' if args.use_mano and args.unfreeze else ''} -> {args.out}")


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--fp_poses", required=True, help="FoundationPose init poses.npy")
    ap.add_argument("--wilor_hands", required=True, help="wilor_hands.npz")
    ap.add_argument("--fill-missing", action="store_true",
                    help="complete missing geometry and exclude synthetic image/contact targets")
    ap.add_argument("--depth_dir", required=True, help="MoGe per-frame npy dir")
    ap.add_argument("--sam2_masks", default=None, help="sam2_masks.npz (obj masks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-contact-boundary-guard", action="store_true",
                    help="disable automatic local contact-boundary continuity protection")
    ap.add_argument("--contact_mode", choices=["manual", "always"], default="always")
    ap.add_argument("--finer_contact", action="store_true",
                    help="V2: per-finger contact + contact_depth_loss + contact smoothness")
    ap.add_argument("--w_csmooth", type=float, default=5.0)
    ap.add_argument("--contact_delta", type=float, default=0.05,
                    help="huber delta for contact (m); must be ~cm scale, not 1mm")
    ap.add_argument("--hand_depth", action="store_true",
                    help="V3: hand verts <-> hand-region MoGe depth cloud")
    ap.add_argument("--w_hand_depth", type=float, default=1.0)
    ap.add_argument("--full_objective", action="store_true",
                    help="turn on ALL: finer_contact+hand_depth+verts_tracking+two_stage+depth_vis+manip_weights")
    ap.add_argument("--verts_tracking", action="store_true", help="frame-summed object 2D reproj L1 to FP")
    add_constraint_arguments(ap)
    ap.add_argument("--two_stage", action="store_true", help="geom_opt then contact_opt")
    ap.add_argument("--depth_vis", action="store_true", help="depth loss on front-facing verts only")
    ap.add_argument("--manip_weights", action="store_true", help="manipulation loss weights and per-parameter learning rates")
    ap.add_argument("--silhouette_z", action="store_true",
                    help="occlusion-aware silhouette-z: rendered obj silhouette <-> SAM2 mask (hand excluded)")
    ap.add_argument("--w_silhouette", type=float, default=300.0)
    ap.add_argument("--hand_silhouette_z", action="store_true",
                    help="HAND absolute-z anchor: MANO-hand silhouette <-> kp-hull hand mask (forearm/obj excluded)")
    ap.add_argument("--w_hand_silhouette", type=float, default=50000.0)
    ap.add_argument("--hand_hull_dilate", type=int, default=25,
                    help="px setback (behind wrist, toward elbow) of the forearm cut-line; keeps palm heel")
    ap.add_argument("--sil_scale", type=float, default=0.25, help="silhouette render res scale")
    ap.add_argument("--use_mano", action="store_true",
                    help="MANO hand (needs wilor_hands with mano params) + per-segment face-patch contact")
    ap.add_argument("--unfreeze", action="store_true", help="optimise finger Δpose (limited)")
    ap.add_argument("--seg_labels", default=None, help="per-segment contact labels json")
    ap.add_argument("--w_pose_reg", type=float, default=2000.0, help="strong: keep Δpose ~0")
    ap.add_argument("--w_pose_smooth", type=float, default=500.0)
    ap.add_argument("--pose_clamp", type=float, default=0.15, help="±rad hard clamp on Δpose")
    ap.add_argument("--w_contact_override", type=float, default=-1.0,
                    help=">=0 overrides contact weight (point_mesh_face_distance needs a smaller weight)")
    ap.add_argument("--w_depth_override", type=float, default=-1.0,
                    help=">=0 overrides depth weight (e.g. 0 to isolate silhouette-z)")
    ap.add_argument("--w_hand_reg_override", type=float, default=-1.0,
                    help=">=0 overrides hand_reg (wrist-position reg); 0 frees the wrist for contact")
    ap.add_argument("--object_name", default="object")
    ap.add_argument("--seq_id", default=None)
    ap.add_argument("--manual_json", default=None)
    ap.add_argument("--max_frames", type=int, default=60)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--contact-iters", type=int, default=max(1, int(os.environ.get("CONTACT_ITERS", "200"))))
    ap.add_argument("--check-cache", action="store_true", help="validate cached optimization without GPU inference")
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--w_contact", type=float, default=1.0)
    ap.add_argument("--w_depth", type=float, default=1.0)
    ap.add_argument("--w_kp", type=float, default=0.01)
    ap.add_argument("--w_smooth", type=float, default=10.0)
    ap.add_argument("--w_reg", type=float, default=5.0)
    ap.add_argument("--early-stop", action="store_true",
                    help="adaptively stop converged optimization stages")
    ap.add_argument("--early-stop-check-every", type=int, default=10)
    ap.add_argument("--early-stop-patience", type=int, default=3)
    ap.add_argument("--early-stop-min-geom", type=int, default=80)
    ap.add_argument("--early-stop-min-contact", type=int, default=80)
    ap.add_argument("--early-stop-rel-total", type=float, default=0.002,
                    help="meaningful total-loss improvement ratio (default 0.2%%)")
    ap.add_argument("--early-stop-rel-contact", type=float, default=0.005,
                    help="meaningful raw-contact improvement ratio (default 0.5%%)")
    return ap


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    if args.check_cache:
        raise SystemExit(0 if optimization_cache_complete(args) else 1)
    run(args)
