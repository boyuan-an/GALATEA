import cv2
import numpy as np
import torch
from pytorch3d.renderer import (
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
)
from pytorch3d.renderer.cameras import PerspectiveCameras as _PC
from pytorch3d.structures import Meshes
from pytorch3d.utils import cameras_from_opencv_projection


def project_points(pts_cam, K):
    z = pts_cam[:, 2]
    safe = np.where(z == 0, 1e-9, z)
    u = K[0, 0] * pts_cam[:, 0] / safe + K[0, 2]
    v = K[1, 1] * pts_cam[:, 1] / safe + K[1, 2]
    return np.stack([u, v], 1), z


def _opencv_camera(K, pose_obj2cam, image_size, device):
    W, H = image_size
    R = torch.tensor(pose_obj2cam[:3, :3], dtype=torch.float32, device=device)[None]
    t = torch.tensor(pose_obj2cam[:3, 3], dtype=torch.float32, device=device)[None]
    Kt = torch.tensor(K, dtype=torch.float32, device=device)[None]
    img_size = torch.tensor([[H, W]], dtype=torch.float32, device=device)
    return cameras_from_opencv_projection(R=R, tvec=t, camera_matrix=Kt, image_size=img_size)


def render_object_mask(verts_obj, faces, K, pose_obj2cam, image_size, device="cuda",
                       dilate_px=7):
    W, H = image_size
    cams = _opencv_camera(K, pose_obj2cam, (W, H), device)
    verts = torch.tensor(verts_obj, dtype=torch.float32, device=device)
    faces_t = torch.tensor(np.asarray(faces), dtype=torch.int64, device=device)
    mesh = Meshes(verts=[verts], faces=[faces_t])

    raster = MeshRasterizer(
        cameras=cams,
        raster_settings=RasterizationSettings(
            image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False,
            max_faces_per_bin=max(10000, len(faces)),
        ),
    )
    frags = raster(mesh)
    mask = (frags.pix_to_face[0, ..., 0] >= 0).cpu().numpy().astype(np.uint8)

    if dilate_px and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def render_hand_mask(hand_joints_cam, K, image_size, device=None, dilate_px=25):
    W, H = image_size
    mask = np.zeros((H, W), np.uint8)
    for J in hand_joints_cam:
        if J is None or len(J) == 0:
            continue
        uv, z = project_points(J, K)
        uv = uv[z > 0]
        if len(uv) < 3:
            continue
        pts = uv.astype(np.int32)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)
    if dilate_px and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
        mask = cv2.dilate(mask, k, iterations=1)
    return mask


def background_mask(fg_anchor, fg_cur, depth_cur, depth_min=0.1, depth_max=8.0):
    fg = (fg_anchor > 0) | (fg_cur > 0)
    valid = (depth_cur > depth_min) & (depth_cur < depth_max)
    return (~fg) & valid
