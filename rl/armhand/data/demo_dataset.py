import json
import os
from functools import lru_cache

import chamfer_distance
import numpy as np
import torch
import trimesh
from manotorch.manolayer import ManoLayer
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from scipy.ndimage import gaussian_filter1d

from .rotations import aa_to_rotmat, rotmat_to_aa

KEYPOINT_JOINTS = {
    "index_proximal": 1,
    "index_intermediate": 2,
    "index_distal": 3,
    "middle_proximal": 4,
    "middle_intermediate": 5,
    "middle_distal": 6,
    "pinky_proximal": 7,
    "pinky_intermediate": 8,
    "pinky_distal": 9,
    "ring_proximal": 10,
    "ring_intermediate": 11,
    "ring_distal": 12,
    "thumb_proximal": 13,
    "thumb_intermediate": 14,
    "thumb_distal": 15,
}
KEYPOINT_VERTICES = {"index_tip": 353, "middle_tip": 467, "pinky_tip": 695, "ring_tip": 576, "thumb_tip": 766}
TIP_KEYPOINTS = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]


def _normalize(vector):
    return vector / torch.linalg.vector_norm(vector, dim=-1, keepdim=True)


def wrist_frame_from_joints(joints):
    wrist = joints[..., 0, :]
    mcp = joints[..., [1, 4, 10, 7], :]
    z_axis = _normalize(mcp.mean(dim=-2) - wrist)
    y_seed = mcp[..., 0, :] - mcp[..., 3, :]
    y_seed = y_seed - (y_seed * z_axis).sum(dim=-1, keepdim=True) * z_axis
    y_axis = _normalize(y_seed)
    x_axis = _normalize(torch.linalg.cross(y_axis, z_axis, dim=-1))
    y_axis = _normalize(torch.linalg.cross(z_axis, x_axis, dim=-1))
    z_axis = _normalize(torch.linalg.cross(x_axis, y_axis, dim=-1))
    return wrist, torch.stack((x_axis, y_axis, z_axis), dim=-1)


def linear_velocity(p, dt):
    velocity = np.gradient(p.cpu().numpy(), axis=0) / dt
    velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
    return torch.from_numpy(velocity).to(p)


def angular_velocity(r, dt):
    diff_aa = rotmat_to_aa(r[1:] @ r[:-1].transpose(-1, -2)).cpu().numpy()
    diff_angle = np.linalg.norm(diff_aa, axis=-1)
    diff_axis = diff_aa / (diff_angle[:, :, None] + 1e-08)
    velocity = diff_axis * diff_angle[:, :, None] / dt
    velocity = np.concatenate([velocity, velocity[-1:]], axis=0)
    velocity = gaussian_filter1d(velocity, 2, axis=0, mode="nearest")
    return torch.from_numpy(velocity).to(r)


class DemoDataset:
    def __init__(self, *, data_dir, ik_dir, device, world_transform, max_seq_len, fps):
        self.data_dir = data_dir
        self.ik_dir = ik_dir
        self.device = device
        self.max_seq_len = max_seq_len
        self.dt = 1 / fps
        self.nearest = chamfer_distance.ChamferDistance()
        self.hand_model = ManoLayer(
            rot_mode="axisang",
            side="right",
            center_idx=None,
            mano_assets_root="data/mano_v1_2",
            use_pca=False,
            flat_hand_mean=True,
        ).to(device)
        with open(os.path.join(data_dir, "manifest.json")) as f:
            entries = json.load(f)["entries"]
        self.sequence_dirs = {
            str(e["data_index"]): os.path.join(data_dir, e.get("dir", e["data_index"])) for e in entries
        }
        offset = np.eye(4)
        offset[:3, :3] = aa_to_rotmat(np.array([-np.pi / 2, 0, 0])) @ aa_to_rotmat(np.array([0, 0, np.pi / 2]))
        offset[:3, 3] = np.array([0.0, 0.018, 0.0])
        self.table_surface_z = float(world_transform[2, 3].detach().cpu().item())
        self.transform = world_transform @ torch.tensor(offset, dtype=torch.float32, device=world_transform.device)

    def _surface_points(self, mesh):
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state()
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        points = sample_points_from_meshes(mesh, 1000).to(self.device).squeeze(0)
        torch.random.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        return points

    def _hand_vertices(self, raw):
        if "rhand_verts" in raw:
            return torch.tensor(raw["rhand_verts"], device=self.device)
        pose = np.concatenate([raw["rhand_global_orient"], raw["rhand_hand_pose"]], axis=1)
        with torch.no_grad():
            verts = self.hand_model(
                torch.tensor(pose, device=self.device, dtype=torch.float32),
                torch.tensor(raw["rhand_betas"], device=self.device, dtype=torch.float32),
            ).verts
        return verts + torch.tensor(raw["rhand_transl"], device=self.device, dtype=torch.float32)[:, None]

    @lru_cache(maxsize=None)
    def __getitem__(self, data_index):
        seq_dir = self.sequence_dirs[data_index]
        raw = np.load(os.path.join(seq_dir, "motion.npy"), allow_pickle=True).item()
        mesh = trimesh.load(os.path.join(seq_dir, "object.obj"), process=False)
        length = len(raw["object_global_orient"])
        object_pose = np.eye(4)[None].repeat(length, axis=0)
        object_pose[:, :3, :3] = aa_to_rotmat(raw["object_global_orient"]).transpose(0, 2, 1)
        object_pose[:, :3, 3] = raw["object_transl"]
        object_pose = torch.tensor(object_pose, device=self.device)
        hand_verts = self._hand_vertices(raw)
        hand_joints = torch.matmul(self.hand_model.th_J_regressor, hand_verts).detach()
        wrist_pos, wrist_rot = wrist_frame_from_joints(hand_joints)
        keypoints = {}
        for name in ("index", "middle", "pinky", "ring", "thumb"):
            for part in ("proximal", "intermediate", "distal"):
                keypoints[f"{name}_{part}"] = hand_joints[:, KEYPOINT_JOINTS[f"{name}_{part}"]]
            keypoints[f"{name}_tip"] = hand_verts[:, KEYPOINT_VERTICES[f"{name}_tip"]].detach()
        surface = self._surface_points(
            Meshes(
                verts=torch.from_numpy(mesh.vertices[None, ...]).float(),
                faces=torch.from_numpy(mesh.faces[None, ...]).float(),
            )
        )
        data = {
            "object_points": surface,
            "object_urdf": os.path.join(seq_dir, "object_collision.urdf"),
            "object_pose": torch.tensor(np.stack(object_pose.cpu()), device=self.device, dtype=torch.float),
            "wrist_pos": wrist_pos,
            "wrist_rot": wrist_rot,
            "keypoints": keypoints,
        }
        self._to_world(data, surface)
        self._ground_on_table(data, mesh)
        self._align_with_arm(data, data_index)
        return data

    def _to_world(self, data, surface):
        rotation, translation = self.transform[:3, :3], self.transform[:3, 3]
        data["object_pose"] = self.transform @ data["object_pose"]
        data["wrist_pos"] = (rotation @ data["wrist_pos"].T).T + translation
        data["wrist_rot"] = rotmat_to_aa(rotation @ data["wrist_rot"])
        for k in data["keypoints"]:
            data["keypoints"][k] = (rotation @ data["keypoints"][k].T).T + translation
        object_points = (data["object_pose"][:, :3, :3] @ surface.T[None]).transpose(-1, -2) + data["object_pose"][
            :, :3, 3
        ][:, None]
        tips = torch.cat([data["keypoints"][k][:, None] for k in TIP_KEYPOINTS], dim=1)
        tips_sq_distance, _, _, _ = self.nearest(tips, object_points)
        data["tips_distance"] = torch.sqrt(tips_sq_distance)
        data["object_vel"] = linear_velocity(data["object_pose"][:, None, :3, 3], self.dt).squeeze(1)
        data["object_ang_vel"] = angular_velocity(data["object_pose"][:, None, :3, :3], self.dt).squeeze(1)
        data["wrist_vel"] = linear_velocity(data["wrist_pos"][:, None], self.dt).squeeze(1)
        data["wrist_ang_vel"] = angular_velocity(aa_to_rotmat(data["wrist_rot"][:, None]), self.dt).squeeze(1)
        data["keypoints_vel"] = {k: linear_velocity(v, self.dt) for k, v in data["keypoints"].items()}
        for key in (
            "object_pose",
            "object_vel",
            "object_ang_vel",
            "wrist_pos",
            "wrist_rot",
            "wrist_vel",
            "wrist_ang_vel",
            "tips_distance",
        ):
            data[key] = data[key][: self.max_seq_len]
        for group in ("keypoints", "keypoints_vel"):
            data[group] = {k: v[: self.max_seq_len] for k, v in data[group].items()}

    def _ground_on_table(self, data, mesh):
        verts = torch.tensor(np.asarray(mesh.vertices), device=self.device, dtype=torch.float32)
        first_frame = (data["object_pose"][0, :3, :3] @ verts.T).T + data["object_pose"][0, :3, 3]
        shift = self.table_surface_z - first_frame[:, 2].min()
        if torch.abs(shift) <= 0.0001:
            return
        data["object_pose"][:, 2, 3] += shift
        data["wrist_pos"][:, 2] += shift
        for k in data["keypoints"]:
            data["keypoints"][k][:, 2] += shift

    def _rotate_about_z(self, data, theta_deg, pivot):
        theta = np.radians(float(theta_deg))
        c, s = (np.cos(theta), np.sin(theta))
        rz = torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], device=self.device, dtype=torch.float32)
        center = torch.tensor([float(pivot[0]), float(pivot[1]), 0.0], device=self.device, dtype=torch.float32)

        def rotate_point(p):
            return (p - center) @ rz.T + center

        def rotate_vector(v):
            return v @ rz.T

        data["object_pose"][:, :3, 3] = rotate_point(data["object_pose"][:, :3, 3])
        data["object_pose"][:, :3, :3] = torch.einsum("ij,tjk->tik", rz, data["object_pose"][:, :3, :3])
        data["wrist_pos"] = rotate_point(data["wrist_pos"])
        data["wrist_rot"] = rotmat_to_aa(torch.einsum("ij,tjk->tik", rz, aa_to_rotmat(data["wrist_rot"])))
        for k in data["keypoints"]:
            data["keypoints"][k] = rotate_point(data["keypoints"][k])
        for key in ("object_vel", "object_ang_vel", "wrist_vel", "wrist_ang_vel"):
            data[key] = rotate_vector(data[key])
        for k in data["keypoints_vel"]:
            data["keypoints_vel"][k] = rotate_vector(data["keypoints_vel"][k])

    def _align_with_arm(self, data, data_index):
        ik = np.load(os.path.join(self.ik_dir, f"{data_index}_fr3_retarget.npz"))
        length = min(int(ik["arm_qpos"].shape[0]), int(self.max_seq_len))
        theta = float(ik["corot_theta"]) if "corot_theta" in ik.files else 0.0
        if abs(theta) > 1e-09:
            self._rotate_about_z(data, theta, np.asarray(ik["corot_pivot"], dtype=np.float64).reshape(-1))
        shift = torch.tensor(np.asarray(ik["shift_xy"], dtype=np.float32), device=self.device, dtype=torch.float32)
        data["object_pose"][:, :2, 3] += shift
        data["wrist_pos"][:, :2] += shift
        for k in data["keypoints"]:
            data["keypoints"][k][:, :2] += shift
        for key in (
            "object_pose",
            "object_vel",
            "object_ang_vel",
            "wrist_pos",
            "wrist_rot",
            "wrist_vel",
            "wrist_ang_vel",
            "tips_distance",
        ):
            data[key] = data[key][:length].contiguous()
        for group in ("keypoints", "keypoints_vel"):
            data[group] = {k: v[:length].contiguous() for k, v in data[group].items()}
        data["arm_reset_qpos"] = torch.tensor(ik["arm_qpos_reset"][:length], device=self.device, dtype=torch.float32)
        data["object_height_offset"] = torch.tensor(
            ik["deficit"][:length].reshape(-1), device=self.device, dtype=torch.float32
        )
        data["object_pos_tolerance"] = torch.tensor(
            ik["pos_err"][:length].reshape(-1), device=self.device, dtype=torch.float32
        )
        data["object_bottom_z"] = torch.tensor(
            np.asarray(ik["reference_object_bottom_z"]).reshape(-1)[:length], device=self.device, dtype=torch.float32
        )
        data["reset_valid"] = torch.tensor(
            np.asarray(ik["reset_collision_free"]).reshape(-1)[:length].astype(np.bool_),
            device=self.device,
            dtype=torch.bool,
        )
