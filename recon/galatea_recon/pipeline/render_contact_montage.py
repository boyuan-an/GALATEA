import argparse
import glob
import os

import cv2
import numpy as np


def _parse_windows(spec):
    out = []
    for w in (spec or "").split(","):
        w = w.strip()
        if not w:
            continue
        s, e = w.split("-")
        out.append((int(s), int(e)))
    return out


def _in_window(i, wins):
    return any(s <= i <= e for (s, e) in wins)


def render(
    cam_dir,
    out_path,
    stride=3,
    cols=8,
    tile_w=240,
    fps=24.0,
    max_frames=None,
    window="",
    object_name="",
    tag="",
):
    rgb_files = sorted(glob.glob(os.path.join(cam_dir, "rgb", "*.png")))
    if not rgb_files:
        raise FileNotFoundError(f"no rgb/*.png under {cam_dir}")
    n = len(rgb_files)
    if max_frames:
        n = min(n, int(max_frames))
    wins = _parse_windows(window)

    idxs = list(range(0, n, max(1, stride)))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    h0, w0 = cv2.imread(rgb_files[0]).shape[:2]
    tile_h = int(round(h0 * tile_w / w0))

    tiles = []
    for i in idxs:
        img = cv2.imread(rgb_files[i])
        img = cv2.resize(img, (tile_w, tile_h))
        label = f"f{i}  {i / fps:.2f}s"
        cv2.rectangle(img, (0, 0), (len(label) * 11 + 8, 24), (0, 0, 0), -1)
        cv2.putText(
            img, label, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
        )
        if wins:
            color = (0, 220, 0) if _in_window(i, wins) else (0, 0, 220)
            cv2.rectangle(img, (0, 0), (tile_w - 1, tile_h - 1), color, 4)
        tiles.append(img)

    rows = int(np.ceil(len(tiles) / cols))
    blank = np.zeros((tile_h, tile_w, 3), np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank)
    grid = np.vstack([np.hstack(tiles[r * cols : (r + 1) * cols]) for r in range(rows)])

    bar = np.zeros((34, grid.shape[1], 3), np.uint8)
    hdr = f"{tag or os.path.basename(cam_dir)}"
    if object_name:
        hdr += f"  obj='{object_name}'"
    hdr += f"  frames={n} @ {fps:g}fps  every {stride}f"
    hdr += f"  window=[{window}] (green=contact)" if wins else "  -> fill window in contact.yaml"
    cv2.putText(bar, hdr, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    grid = np.vstack([bar, grid])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, grid)
    print(
        f"[contact-montage] {len(idxs)} tiles ({rows}x{cols}) frames=0..{n-1} "
        f"stride={stride} {'VERIFY '+window if wins else 'annotate'} -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Dense RGB montage for contact-window verification")
    ap.add_argument("--cam_dir", required=True, help="cam0 dir containing rgb/*.png")
    ap.add_argument("--out", required=True, help="output montage PNG")
    ap.add_argument("--stride", type=int, default=3, help="sample every N frames (default 3)")
    ap.add_argument("--cols", type=int, default=8, help="grid columns (default 8)")
    ap.add_argument("--tile_w", type=int, default=240, help="per-tile width px (default 240)")
    ap.add_argument(
        "--fps", type=float, default=24.0, help="capture fps for the seconds label (default 24)"
    )
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument(
        "--window", default="", help='verify mode: "s-e[,s-e...]" highlights the contact window'
    )
    ap.add_argument("--object-name", dest="object_name", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    render(
        a.cam_dir,
        a.out,
        stride=a.stride,
        cols=a.cols,
        tile_w=a.tile_w,
        fps=a.fps,
        max_frames=a.max_frames,
        window=a.window,
        object_name=a.object_name,
        tag=a.tag,
    )
