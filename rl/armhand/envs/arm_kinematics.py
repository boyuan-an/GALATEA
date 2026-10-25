import xml.etree.ElementTree as ET

import numpy as np
import torch


def _rpy_to_matrix(rpy):
    cx, cy, cz = np.cos(rpy)
    sx, sy, sz = np.sin(rpy)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(xyz, rpy):
    t = np.eye(4)
    t[:3, :3] = _rpy_to_matrix(np.asarray(rpy, float))
    t[:3, 3] = np.asarray(xyz, float)
    return t


class ArmKinematics:
    def __init__(self, urdf_path, base_pos, base_yaw, arm_joint_names, target_link, device):
        self.device = device
        arm_joint_names = list(arm_joint_names)
        root = ET.parse(urdf_path).getroot()
        parents = {}
        for joint in root.findall("joint"):
            origin = joint.find("origin")
            xyz = [0, 0, 0]
            rpy = [0, 0, 0]
            if origin is not None:
                if origin.get("xyz"):
                    xyz = [float(x) for x in origin.get("xyz").split()]
                if origin.get("rpy"):
                    rpy = [float(x) for x in origin.get("rpy").split()]
            axis = [0, 0, 1]
            axis_node = joint.find("axis")
            if axis_node is not None and axis_node.get("xyz"):
                axis = [float(x) for x in axis_node.get("xyz").split()]
            parents[joint.find("child").get("link")] = (
                joint.find("parent").get("link"),
                _transform(xyz, rpy),
                joint.get("type"),
                np.array(axis, float),
                joint.get("name"),
            )
        chain = []
        link = target_link
        while link in parents:
            chain.append(parents[link])
            link = parents[link][0]
        chain = chain[::-1]
        origins = []
        axes = []
        self._joint_slots = []
        for _, origin, joint_type, axis, name in chain:
            origins.append(origin)
            self._joint_slots.append(
                arm_joint_names.index(name) if joint_type == "revolute" and name in arm_joint_names else -1
            )
            axes.append(axis / (np.linalg.norm(axis) + 1e-12))
        self._origins = torch.tensor(np.stack(origins), device=device, dtype=torch.float32)
        self._axes = torch.tensor(np.stack(axes), device=device, dtype=torch.float32)
        base = np.eye(4)
        base[:3, :3] = _rpy_to_matrix(np.array([0.0, 0.0, float(base_yaw)]))
        base[:3, 3] = np.asarray(base_pos, float)
        self._base = torch.tensor(base, device=device, dtype=torch.float32)

    def _axis_rotation(self, axis, angle):
        n = angle.shape[0]
        axis = axis / (axis.norm() + 1e-12)
        k = torch.zeros(3, 3, device=self.device, dtype=torch.float32)
        k[0, 1], k[0, 2] = (-axis[2], axis[1])
        k[1, 0], k[1, 2] = (axis[2], -axis[0])
        k[2, 0], k[2, 1] = (-axis[1], axis[0])
        eye = torch.eye(3, device=self.device, dtype=torch.float32)
        s = torch.sin(angle).view(n, 1, 1)
        c = torch.cos(angle).view(n, 1, 1)
        return eye.unsqueeze(0) + s * k.unsqueeze(0) + (1.0 - c) * (k @ k).unsqueeze(0)

    def forward(self, q):
        n = q.shape[0]
        t = self._base.unsqueeze(0).expand(n, 4, 4).contiguous()
        for link, slot in enumerate(self._joint_slots):
            t = t @ self._origins[link].unsqueeze(0)
            if slot >= 0:
                rotation = (
                    torch.eye(4, device=self.device, dtype=torch.float32).unsqueeze(0).expand(n, 4, 4).contiguous()
                )
                rotation[:, :3, :3] = self._axis_rotation(self._axes[link], q[:, slot])
                t = t @ rotation
        return t
