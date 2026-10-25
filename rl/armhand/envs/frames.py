import torch


def _conjugate(quat):
    return torch.cat((-quat[..., :3], quat[..., 3:4]), dim=-1)


def _multiply(left, right):
    left_xyz, left_w = (left[..., :3], left[..., 3:4])
    right_xyz, right_w = (right[..., :3], right[..., 3:4])
    xyz = left_w * right_xyz + right_w * left_xyz + torch.cross(left_xyz, right_xyz, dim=-1)
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat((xyz, w), dim=-1)


def _rotate(quat, vector):
    quat_xyz = quat[..., :3]
    uv = torch.cross(quat_xyz, vector, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vector + 2.0 * (quat[..., 3:4] * uv + uuv)


def _broadcast(value, target):
    while value.ndim < target.ndim:
        value = value.unsqueeze(-2)
    return value.expand(*target.shape[:-1], value.shape[-1])


def vector_to_base(vector, base_quat):
    return _rotate(_conjugate(_broadcast(base_quat, vector)), vector)


def position_to_base(position, base_pos, base_quat):
    return vector_to_base(position - _broadcast(base_pos, position), base_quat)


def quaternion_to_base(quat, base_quat):
    return _multiply(_conjugate(_broadcast(base_quat, quat)), quat)
