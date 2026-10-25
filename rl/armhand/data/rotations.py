import numpy as np
import torch
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    axis_angle_to_quaternion,
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    quaternion_to_matrix,
)


def _apply(functions, value):
    is_numpy = isinstance(value, np.ndarray)
    if is_numpy:
        value = torch.FloatTensor(value)
    for function in functions:
        value = function(value)
    return value.detach().cpu().numpy() if is_numpy else value


def aa_to_rotmat(axis_angle):
    return _apply((axis_angle_to_matrix,), axis_angle)


def rotmat_to_aa(matrix):
    return _apply((matrix_to_quaternion, quaternion_to_axis_angle), matrix)


def aa_to_quat(axis_angle):
    return _apply((axis_angle_to_quaternion,), axis_angle)


def rotmat_to_quat(matrix):
    return _apply((matrix_to_quaternion,), matrix)


def quat_to_rotmat(quaternion):
    return _apply((quaternion_to_matrix,), quaternion)
