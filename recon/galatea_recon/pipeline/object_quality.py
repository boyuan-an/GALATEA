import hashlib
import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


POLICY = "object-recovery-v1"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_hash(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def mask_box(mask):
    y, x = np.where(mask)
    return None if not len(x) else np.array([x.min(), y.min(), x.max()+1, y.max()+1])


def outline(mask):
    mask = np.asarray(mask, np.uint8)
    return (mask - cv2.erode(mask, np.ones((3, 3), np.uint8))).astype(bool)


def score_masks(projected, target, rgb, occluder=None, rendered_depth=None,
                measured_depth=None, diameter=0.2):
    projected, target = np.asarray(projected, bool), np.asarray(target, bool)
    hidden = np.zeros_like(target) if occluder is None else np.asarray(occluder, bool)
    hidden &= ~target
    visible = ~hidden
    p, t = projected & visible, target & visible
    area_p, area_t = int(p.sum()), int(t.sum())
    intersection = int((p & t).sum())
    union = int((p | t).sum())
    iou = intersection / max(union, 1)
    precision = intersection / max(area_p, 1)
    recall = intersection / max(area_t, 1)
    boundary_visible = ~cv2.dilate(hidden.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    pb, tb = outline(projected) & boundary_visible, outline(target) & boundary_visible
    tolerance = max(1.5, .025 * np.sqrt(max(area_t, 1)))
    if pb.any() and tb.any():
        pd = cv2.distanceTransform((~pb).astype(np.uint8), cv2.DIST_L2, 3)
        td = cv2.distanceTransform((~tb).astype(np.uint8), cv2.DIST_L2, 3)
        bp = float(np.mean(td[pb] <= tolerance))
        br = float(np.mean(pd[tb] <= tolerance))
        boundary_f1 = 2 * bp * br / max(bp + br, 1e-8)
    else:
        boundary_f1 = 0.0
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    edges = cv2.Canny(gray.astype(np.uint8), 40, 120).astype(bool)
    if edges.any() and pb.any():
        distance = cv2.distanceTransform((~edges).astype(np.uint8), cv2.DIST_L2, 3)
        rgb_edge = float(np.exp(-distance[pb] / tolerance).mean())
    else:
        rgb_edge = 0.0
    depth_score, depth_error = None, None
    if measured_depth is not None and rendered_depth is not None:
        d, r = np.asarray(measured_depth), np.asarray(rendered_depth)
        keep = p & t & np.isfinite(d) & (d > .01) & (r > .01)
        if keep.sum() >= max(10, area_t * .1):
            depth_error = float(np.median(np.abs(d[keep] - r[keep])))
            depth_score = float(np.exp(-depth_error / max(.015, .1 * diameter)))
    score = .60 * iou + .25 * boundary_f1 + .15 * rgb_edge
    if depth_score is not None:
        score = .90 * score + .10 * depth_score
    visible_fraction = area_p / max(int(projected.sum()), 1)
    assessable = area_t >= 12
    clipped = bool(target[0].any() or target[-1].any() or target[:, 0].any() or target[:, -1].any())
    return dict(score=float(score), iou=float(iou), precision=float(precision),
                recall=float(recall), boundary_f1=float(boundary_f1), rgb_edge=rgb_edge,
                depth_error_m=depth_error, visible_fraction=float(visible_fraction),
                target_pixels=area_t, assessable=bool(assessable and not clipped))


def quality_confidence(metric):
    if not metric["assessable"]:
        return 0.0
    return float(np.clip((metric["score"] - .25) / .40, 0, 1))


def aggregate(metrics):
    valid = [m for m in metrics if m["assessable"]]
    if not valid:
        return 0.0
    scores = np.array([m["score"] for m in valid])
    return float(.65 * np.median(scores) + .35 * np.quantile(scores, .2))


def accept_candidate(old_metrics, new_metrics, min_gain=.045, minimum=.48):
    if len(old_metrics) != len(new_metrics) or len(new_metrics) < 2:
        return False
    valid = [i for i, m in enumerate(new_metrics) if m["assessable"]]
    if len(valid) < 2:
        return False
    old, new = aggregate(old_metrics), aggregate(new_metrics)
    improvements = [new_metrics[i]["score"] - old_metrics[i]["score"] for i in valid]
    return bool(new >= minimum and new - old >= min_gain
                and np.mean(np.asarray(improvements) >= .015) >= .6
                and np.quantile([new_metrics[i]["score"] for i in valid], .2) >= .30)


class EvidenceRenderer:
    def __init__(self, mesh, K, width, height, max_size=320):
        import torch
        import nvdiffrast.torch as dr
        self.torch, self.dr = torch, dr
        scale = min(1.0, max_size / max(width, height))
        self.width, self.height = max(8, round(width * scale)), max(8, round(height * scale))
        self.K = np.asarray(K, float).copy()
        self.K[0] *= self.width / width
        self.K[1] *= self.height / height
        self.vertices = torch.as_tensor(np.asarray(mesh.vertices).copy(), dtype=torch.float32, device="cuda")
        self.faces = torch.as_tensor(np.asarray(mesh.faces).copy(), dtype=torch.int32, device="cuda")
        self.context = dr.RasterizeCudaContext()
        self.diameter = float(np.linalg.norm(np.ptp(np.asarray(mesh.vertices), axis=0)))

    def resize(self, image, nearest=False):
        return cv2.resize(image, (self.width, self.height),
                          interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)

    def render(self, poses):
        torch = self.torch
        with torch.no_grad():
            tf = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device="cuda").reshape(-1, 4, 4)
            points = torch.einsum("bij,vj->bvi", tf[:, :3, :3], self.vertices) + tf[:, None, :3, 3]
            x, y, z = points.unbind(-1)
            fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
            near, far = .005, 100.0
            clip = torch.stack((2*(fx*x+cx*z)/self.width-z,
                                z-2*(fy*y+cy*z)/self.height,
                                (far+near)/(far-near)*z-2*far*near/(far-near), z), -1)
            rast, _ = self.dr.rasterize(self.context, clip.contiguous(), self.faces,
                                       resolution=[self.height, self.width])
            depth, _ = self.dr.interpolate(points[..., 2:3].contiguous(), rast, self.faces)
            depth = (depth[..., 0] * (rast[..., 3] > 0)).flip(1)
            return depth.cpu().numpy()

    def evaluate(self, poses, target, rgb, occluder=None, measured_depth=None):
        t = self.resize(np.asarray(target, np.uint8), True).astype(bool)
        image = self.resize(rgb)
        hand = None if occluder is None else self.resize(np.asarray(occluder, np.uint8), True).astype(bool)
        measured = None if measured_depth is None else self.resize(measured_depth, True)
        return [score_masks(d > 0, t, image, hand, d, measured, self.diameter)
                for d in self.render(poses)]
