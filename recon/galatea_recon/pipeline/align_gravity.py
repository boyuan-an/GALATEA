import argparse
import json
import os

import numpy as np


def load_intrinsics(path):
    fx, fy, cx, cy, W, H = np.loadtxt(path)
    return float(fx), float(fy), float(cx), float(cy), int(W), int(H)


def backproject(depth, fx, fy, cx, cy):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H, 0:W]
    z = depth
    x = (xs - cx) / fx * z
    y = (ys - cy) / fy * z
    return np.stack([x, y, z], -1)


def fit_support_plane(pts, iters=1500, thr=0.012, max_pts=60000, seed=0,
                      min_inlier_frac=0.05):
    rng = np.random.default_rng(seed)
    if len(pts) < 3:
        raise ValueError(f"support-plane fit needs >=3 valid depth points, got {len(pts)}")
    if len(pts) > max_pts:
        pts = pts[rng.choice(len(pts), max_pts, replace=False)]
    N = len(pts)
    best_inl, best = -1, None
    for _ in range(iters):
        i = rng.choice(N, 3, replace=False)
        p = pts[i]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n = n / nn
        d = -n @ p[0]
        inl = int((np.abs(pts @ n + d) < thr).sum())
        if inl > best_inl:
            best_inl, best = inl, (n, d)
    if best is None:
        raise ValueError("RANSAC found no valid plane (all sampled triplets degenerate)")
    n, d = best
    m = np.abs(pts @ n + d) < thr
    frac = float(m.mean())
    if frac < min_inlier_frac:
        raise ValueError(f"support-plane inlier fraction {frac:.3f} < {min_inlier_frac}; "
                         "no dominant plane in frame-0 depth (bad/sparse depth?)")
    P = pts[m]
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c)
    n = vt[-1]
    n = n / np.linalg.norm(n)
    d = -n @ c
    if n[2] > 0:
        n, d = -n, -d
    return n, d, frac


def build_cam_to_world(up, d):
    ez = up / np.linalg.norm(up)
    fwd = np.array([0.0, 0.0, 1.0])
    ey = fwd - (fwd @ ez) * ez
    if np.linalg.norm(ey) < 1e-6:
        ey = np.array([0.0, 1.0, 0.0]) - (np.array([0.0, 1.0, 0.0]) @ ez) * ez
    ey = ey / np.linalg.norm(ey)
    ex = np.cross(ey, ez)
    ex = ex / np.linalg.norm(ex)
    R_cw = np.stack([ex, ey, ez], 0)
    o = -d * ez
    t = -R_cw @ o
    M = np.eye(4)
    M[:3, :3] = R_cw
    M[:3, 3] = t
    return M


def make_viz(out_png, obj_pos_w, hand_pos_w, tag, up_cam, tilt_deg, inlier_frac):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    T = len(obj_pos_w)
    t = np.arange(T)
    fig = plt.figure(figsize=(13, 5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    xs = obj_pos_w[:, 0]
    ys = obj_pos_w[:, 1]
    pad = 0.15
    gx = np.linspace(xs.min() - pad, xs.max() + pad, 2)
    gy = np.linspace(ys.min() - pad, ys.max() + pad, 2)
    GX, GY = np.meshgrid(gx, gy)
    ax.plot_surface(GX, GY, np.zeros_like(GX), alpha=0.15, color="gray")
    p = ax.scatter(obj_pos_w[:, 0], obj_pos_w[:, 1], obj_pos_w[:, 2],
                   c=t, cmap="viridis", s=12, label="object")
    if hand_pos_w is not None:
        ax.plot(hand_pos_w[:, 0], hand_pos_w[:, 1], hand_pos_w[:, 2],
                color="crimson", lw=1.0, alpha=0.7, label="hand")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y fwd (m)")
    ax.set_zlabel("Z up (m)")
    ax.set_zlim(0, max(0.3, obj_pos_w[:, 2].max() * 1.1))
    ax.set_title(f"{tag}\nZ-up world (plane=z=0)")
    fig.colorbar(p, ax=ax, shrink=0.6, label="frame")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, obj_pos_w[:, 2] * 100, color="seagreen", label="object height")
    if hand_pos_w is not None:
        ax2.plot(t, hand_pos_w[:, 2] * 100, color="crimson", alpha=0.7, label="hand height")
    ax2.axhline(0, color="gray", ls="--", lw=0.8)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("height above support (cm)")
    ax2.set_title(f"gravity from frame-0 depth\nup_cam={np.round(up_cam,3)}  tilt={tilt_deg:.1f}deg  plane_inliers={inlier_frac*100:.0f}%")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam_dir", required=True, help="cam0 dir with depth/000000.png + cam_intrinsics.txt")
    ap.add_argument("--poses", required=True, help="gr_hsz.npy (T,4,4) obj->cam")
    ap.add_argument("--handverts", default=None, help="gr_hsz_handverts.npy (T,V,3)")
    ap.add_argument("--out_poses", required=True)
    ap.add_argument("--out_handverts", default=None)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_viz", default=None)
    ap.add_argument("--ransac_iters", type=int, default=1500)
    ap.add_argument("--ransac_thr", type=float, default=0.012)
    ap.add_argument("--min_inlier_frac", type=float, default=0.05,
                    help="abort if the support plane has fewer inliers than this fraction")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    import cv2

    fx, fy, cx, cy, W, H = load_intrinsics(os.path.join(args.cam_dir, "cam_intrinsics.txt"))
    depth = cv2.imread(os.path.join(args.cam_dir, "depth", "000000.png"), cv2.IMREAD_UNCHANGED)
    depth = depth.astype(np.float32) / 1000.0
    pts = backproject(depth, fx, fy, cx, cy).reshape(-1, 3)
    pts = pts[(pts[:, 2] > 0.1) & (pts[:, 2] < 5.0)]
    if len(pts) < 500:
        raise ValueError(f"only {len(pts)} valid depth points in frame-0 "
                         "(>=500 expected); first-frame depth is missing/empty")

    up, d, inlier_frac = fit_support_plane(pts, args.ransac_iters, args.ransac_thr,
                                           min_inlier_frac=args.min_inlier_frac)
    if up[2] > -0.05:
        print(f"[WARN] support-plane normal barely faces the camera (up_cam={np.round(up,3)}, "
              "up_z >= -0.05): the fit may be a wall/side surface, not the table -- verify gravity.")
    grav = -up
    tilt = float(np.degrees(np.arccos(np.clip(grav @ np.array([0, 1, 0.0]), -1, 1))))
    M = build_cam_to_world(up, d)

    poses = np.load(args.poses)
    poses_w = M[None] @ poses
    np.save(args.out_poses, poses_w.astype(np.float32))

    obj_pos_w = poses_w[:, :3, 3]

    hand_pos_w = None
    if args.handverts and os.path.exists(args.handverts):
        hv = np.load(args.handverts)
        Rcw, t = M[:3, :3], M[:3, 3]
        hv_w = hv @ Rcw.T + t
        if args.out_handverts:
            np.save(args.out_handverts, hv_w.astype(np.float32))
        hand_pos_w = hv_w.mean(1)

    meta = {
        "tag": args.tag,
        "up_cam": up.tolist(),
        "gravity_cam": grav.tolist(),
        "plane_offset_d": float(d),
        "plane_inlier_frac": inlier_frac,
        "tilt_vs_cam_ydown_deg": tilt,
        "world_convention": "Z-up; support plane z=0; +Y = camera-forward projected to ground",
        "cam_to_world": M.tolist(),
        "obj_height_cm": {
            "min": float(obj_pos_w[:, 2].min() * 100),
            "max": float(obj_pos_w[:, 2].max() * 100),
            "mean": float(obj_pos_w[:, 2].mean() * 100),
        },
    }
    with open(args.out_json, "w") as f:
        json.dump(meta, f, indent=2)

    if args.out_viz:
        make_viz(args.out_viz, obj_pos_w, hand_pos_w, args.tag or os.path.basename(args.poses),
                 up, tilt, inlier_frac)

    print(f"[{args.tag}] up_cam={np.round(up,3)} tilt={tilt:.1f}deg inliers={inlier_frac*100:.0f}% "
          f"obj_h={meta['obj_height_cm']['min']:.1f}..{meta['obj_height_cm']['max']:.1f}cm -> {args.out_poses}")


if __name__ == "__main__":
    main()
