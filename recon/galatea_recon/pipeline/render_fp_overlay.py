import argparse
import os

import cv2
import numpy as np
import trimesh

from galatea_recon.pipeline.h2o_loader import H2OSequence
from galatea_recon.pipeline.masks import render_object_mask


def render(cam_dir, object_models_root, poses_path, out_path, mesh_path=None,
           n_tiles=20, cols=5, tile_w=384, max_frames=None, device="cuda"):
    seq = H2OSequence(cam_dir, object_models_root)
    mesh = trimesh.load(os.path.expanduser(mesh_path or seq.mesh_path), force="mesh")
    V = np.asarray(mesh.vertices, np.float64)
    F = np.asarray(mesh.faces)
    poses = np.load(os.path.expanduser(poses_path))

    n = min(len(poses), seq.num_frames)
    if max_frames:
        n = min(n, max_frames)
    if n == 0:
        raise ValueError(f"no frames to render (poses={len(poses)}, seq={seq.num_frames})")
    W, H = seq.width, seq.height
    tile_h = int(round(H * tile_w / W))
    n_tiles = min(n_tiles, n)
    idxs = [int(round(x)) for x in np.linspace(0, n - 1, n_tiles)]

    tiles = []
    for i in idxs:
        bgr = cv2.cvtColor(seq.load_rgb(i), cv2.COLOR_RGB2BGR)
        if bgr.shape[:2] != (H, W):
            bgr = cv2.resize(bgr, (W, H))
        m = render_object_mask(V, F, seq.K, poses[i], (W, H), device=device, dilate_px=0) > 0
        ov = bgr.copy()
        ov[m] = (ov[m] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, (0, 255, 0), 2)
        ov = cv2.resize(ov, (tile_w, tile_h))
        cv2.putText(ov, f"f{i} z={poses[i, 2, 3]:.2f}", (6, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        tiles.append(ov)

    rows = int(np.ceil(n_tiles / cols))
    blank = np.zeros((tile_h, tile_w, 3), np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])

    bar = np.zeros((34, grid.shape[1], 3), np.uint8)
    z = poses[:n, 2, 3]
    tag = os.path.basename(os.path.dirname(cam_dir.rstrip("/"))) or "seq"
    cv2.putText(bar, f"{tag}  FP pose overlay (red=mesh, green=contour)  "
                f"n={n}  z[{z.min():.2f}-{z.max():.2f}]m",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    grid = np.vstack([bar, grid])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, grid)
    print(f"[fp-viz] {n_tiles} tiles ({rows}x{cols}) z[{z.min():.2f}-{z.max():.2f}]m -> {out_path}")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FP-pose object-mesh overlay montage")
    ap.add_argument("--cam_dir", required=True)
    ap.add_argument("--object_models_root", required=True)
    ap.add_argument("--poses", required=True, help="FP object->camera poses .npy (e.g. bg_cal.npy)")
    ap.add_argument("--out", required=True, help="output montage PNG")
    ap.add_argument("--mesh", default=None, help="object mesh (default: loader's mesh_path)")
    ap.add_argument("--n_tiles", type=int, default=20)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    render(args.cam_dir, args.object_models_root, args.poses, args.out,
           mesh_path=args.mesh, n_tiles=args.n_tiles, cols=args.cols,
           max_frames=args.max_frames, device=args.device)
