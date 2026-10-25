import json
import os
from functools import lru_cache

import numpy as np
import torch
from manotorch.manolayer import ManoLayer

from armhand.data.rotations import aa_to_rotmat, rotmat_to_aa

KEYPOINTS = {
    "index_proximal": ("joint", 1),
    "index_intermediate": ("joint", 2),
    "index_distal": ("joint", 3),
    "index_tip": ("vertex", 353),
    "middle_proximal": ("joint", 4),
    "middle_intermediate": ("joint", 5),
    "middle_distal": ("joint", 6),
    "middle_tip": ("vertex", 467),
    "pinky_proximal": ("joint", 7),
    "pinky_intermediate": ("joint", 8),
    "pinky_distal": ("joint", 9),
    "pinky_tip": ("vertex", 695),
    "ring_proximal": ("joint", 10),
    "ring_intermediate": ("joint", 11),
    "ring_distal": ("joint", 12),
    "ring_tip": ("vertex", 576),
    "thumb_proximal": ("joint", 13),
    "thumb_intermediate": ("joint", 14),
    "thumb_distal": ("joint", 15),
    "thumb_tip": ("vertex", 766),
}
PALM_JOINTS = (1, 4, 10, 7)


def _normalize(vector, eps, name):
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    if torch.any(norm <= eps):
        raise ValueError(f"degenerate palm geometry for {name}")
    return vector / norm


def palm_rotation(wrist, palm, eps=1e-08):
    z_axis = _normalize(palm.mean(dim=-2) - wrist, eps, "wrist-to-knuckle axis")
    y_seed = palm[..., 0, :] - palm[..., 3, :]
    y_seed = y_seed - (y_seed * z_axis).sum(dim=-1, keepdim=True) * z_axis
    y_axis = _normalize(y_seed, eps, "pinky-to-index axis")
    x_axis = _normalize(torch.linalg.cross(y_axis, z_axis, dim=-1), eps, "palm normal")
    y_axis = _normalize(torch.linalg.cross(z_axis, x_axis, dim=-1), eps, "lateral axis")
    z_axis = _normalize(torch.linalg.cross(x_axis, y_axis, dim=-1), eps, "longitudinal axis")
    rotation = torch.stack((x_axis, y_axis, z_axis), dim=-1)
    if not torch.isfinite(rotation).all() or torch.any(torch.linalg.det(rotation) <= 0.0):
        raise ValueError("palm frame is not a proper rotation")
    return rotation


def wrist_from_joints(joints):
    wrist = joints[..., 0, :]
    return (wrist, palm_rotation(wrist, joints[..., list(PALM_JOINTS), :]))


def tail_jump_frames(
    wrist_rotation,
    finger_points,
    tail_fraction=0.85,
    rotation_step_deg=20.0,
    point_step_m=0.08,
    combined_point_step_m=0.05,
):
    frames = wrist_rotation.shape[0]
    if frames < 2:
        return torch.empty(0, dtype=torch.long, device=wrist_rotation.device)
    tail_start = int(torch.floor(torch.tensor(frames * tail_fraction)).item())
    relative = wrist_rotation[1:] @ wrist_rotation[:-1].transpose(-1, -2)
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    angle_deg = torch.rad2deg(torch.acos(((trace - 1.0) * 0.5).clamp(-1.0, 1.0)))
    point_step = torch.linalg.vector_norm(finger_points[1:] - finger_points[:-1], dim=-1).amax(dim=-1)
    bad = (point_step > point_step_m) | (angle_deg > rotation_step_deg) & (point_step > combined_point_step_m)
    destination = torch.arange(1, frames, device=wrist_rotation.device, dtype=torch.long)
    return torch.unique(destination[bad & (destination >= tail_start)], sorted=True)


def mano_layer(device):
    return ManoLayer(
        rot_mode="axisang",
        side="right",
        center_idx=None,
        mano_assets_root="data/mano_v1_2",
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)


class ReferenceLoader:
    def __init__(self, data_dir, device, world_transform, reject_jumps=True):
        self.device = device
        self.reject_jumps = bool(reject_jumps)
        self.manolayer = mano_layer(device)
        with open(os.path.join(data_dir, "manifest.json")) as f:
            entries = json.load(f)["entries"]
        self.data_indices = [str(entry["data_index"]) for entry in entries]
        self.data_paths = {
            str(entry["data_index"]): os.path.join(data_dir, entry["dir"], "motion.npy") for entry in entries
        }
        offset = np.eye(4)
        offset[:3, :3] = aa_to_rotmat(np.array([-np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        offset[:3, 3] = np.array([0.0, 0.018, 0.0])
        self.transform = world_transform @ torch.tensor(offset, dtype=torch.float32, device=world_transform.device)

    @lru_cache(maxsize=None)
    def __getitem__(self, data_index):
        raw = np.load(self.data_paths[data_index], allow_pickle=True).item()
        verts = torch.tensor(raw["rhand_verts"], device=self.device)
        joints = torch.matmul(self.manolayer.th_J_regressor, verts).detach()
        wrist_pos, wrist_rot = wrist_from_joints(joints)
        keypoints = {
            name: (joints[:, index] if source == "joint" else verts[:, index].detach())
            for name, (source, index) in KEYPOINTS.items()
        }
        rotation, translation = self.transform[:3, :3], self.transform[:3, 3]
        wrist_pos = (rotation @ wrist_pos.T).T + translation
        wrist_rot = rotmat_to_aa(rotation @ wrist_rot)
        keypoints = {name: (rotation @ point.T).T + translation for name, point in keypoints.items()}
        if self.reject_jumps:
            bad = tail_jump_frames(aa_to_rotmat(wrist_rot), torch.stack(list(keypoints.values()), dim=1))
            if bad.numel():
                raise ValueError(f"{self.data_paths[data_index]}: reference jump at frame {int(bad[0])}")
        return {"wrist_pos": wrist_pos, "wrist_rot": wrist_rot, "keypoints": keypoints}
