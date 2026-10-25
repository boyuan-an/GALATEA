from typing import Dict, List, Tuple

import numpy as np
import torch
from isaacgym.torch_utils import normalize_angle, quat_conjugate, quat_mul
from torch import Tensor


@torch.jit.script
def quat_to_angle_axis(q: Tensor) -> Tuple[Tensor, Tensor]:
    sin_theta = torch.sqrt(1 - q[..., 3] * q[..., 3])
    angle = 2 * torch.acos(q[..., 3])
    angle = normalize_angle(angle)
    sin_theta_expand = sin_theta.unsqueeze(-1)
    axis = q[..., 0:3] / sin_theta_expand
    mask = torch.abs(sin_theta) > 1e-05
    default_axis = torch.zeros_like(axis)
    default_axis[..., -1] = 1
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    mask_expand = mask.unsqueeze(-1)
    axis = torch.where(mask_expand, axis, default_axis)
    return (angle, axis)


@torch.jit.script
def compute_tracking_reward(
    reset_buf: Tensor,
    progress_buf: Tensor,
    running_progress_buf: Tensor,
    states: Dict[str, Tensor],
    targets: Dict[str, Tensor],
    max_length: Tensor,
    scale_factor: float,
    fail_tol: float,
    keypoint_groups: Dict[str, List[int]],
    obj_rest_z: Tensor,
    lift_reward_scale: float,
    lift_min_height: float,
    wrist_pos_weight: float,
    level_1_weight: float,
    level_2_weight: float,
    wrist_rot_weight: float,
    wrist_rot_k: float,
    free_space_finger_scale: float,
    contact_finger_scale: float,
    object_contact_distance: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Tensor], Tensor]:
    wrist_pos = states["wrist_state"][:, :3]
    wrist_quat = states["wrist_state"][:, 3:7]
    wrist_vel = states["wrist_state"][:, 7:10]
    wrist_ang_vel = states["wrist_state"][:, 10:13]
    wrist_pos_error = torch.norm(targets["wrist_pos"] - wrist_pos, dim=-1)
    joints_pos = states["body_state"][:, 1:, :3]
    joints_vel = states["body_state"][:, 1:, 7:10]
    joint_errors = torch.norm(targets["joints_pos"] - joints_pos, dim=-1)
    thumb_error = joint_errors[:, [k - 1 for k in keypoint_groups["thumb_tip"]]].mean(dim=-1)
    index_error = joint_errors[:, [k - 1 for k in keypoint_groups["index_tip"]]].mean(dim=-1)
    middle_error = joint_errors[:, [k - 1 for k in keypoint_groups["middle_tip"]]].mean(dim=-1)
    ring_error = joint_errors[:, [k - 1 for k in keypoint_groups["ring_tip"]]].mean(dim=-1)
    pinky_error = joint_errors[:, [k - 1 for k in keypoint_groups["pinky_tip"]]].mean(dim=-1)
    level_1_error = joint_errors[:, [k - 1 for k in keypoint_groups["level_1_joints"]]].mean(dim=-1)
    level_2_error = joint_errors[:, [k - 1 for k in keypoint_groups["level_2_joints"]]].mean(dim=-1)

    reward_wrist_pos = torch.exp(-40 * wrist_pos_error)
    reward_thumb = torch.exp(-100 * thumb_error)
    reward_index = torch.exp(-90 * index_error)
    reward_middle = torch.exp(-80 * middle_error)
    reward_pinky = torch.exp(-60 * pinky_error)
    reward_ring = torch.exp(-60 * ring_error)
    reward_level_1 = torch.exp(-50 * level_1_error)
    reward_level_2 = torch.exp(-40 * level_2_error)
    reward_wrist_vel = torch.exp(-1 * (targets["wrist_vel"] - wrist_vel).abs().mean(dim=-1))
    reward_wrist_ang_vel = torch.exp(-1 * (targets["wrist_ang_vel"] - wrist_ang_vel).abs().mean(dim=-1))
    reward_joints_vel = torch.exp(-1 * (targets["joints_vel"] - joints_vel).abs().mean(dim=-1).mean(-1))
    wrist_rot_error = quat_to_angle_axis(quat_mul(targets["wrist_quat"], quat_conjugate(wrist_quat)))[0]
    reward_wrist_rot = torch.exp(-wrist_rot_k * wrist_rot_error.abs())

    obj_pos = states["object_pos"]
    obj_quat = states["object_quat"]
    obj_vel = states["object_vel"]
    obj_ang_vel = states["object_ang_vel"]
    target_obj_pos = targets["object_pos"]
    obj_pos_error = torch.norm(target_obj_pos - obj_pos, dim=-1)
    reward_obj_pos = torch.exp(-80 * obj_pos_error)
    obj_rot_error = quat_to_angle_axis(quat_mul(targets["object_quat"], quat_conjugate(obj_quat)))[0].abs()
    reward_obj_rot = torch.exp(-3 * obj_rot_error)
    reward_obj_vel = torch.exp(-1 * (targets["object_vel"] - obj_vel).abs().mean(dim=-1))
    reward_obj_ang_vel = torch.exp(-1 * (targets["object_ang_vel"] - obj_ang_vel).abs().mean(dim=-1))
    reward_power = torch.exp(-10 * targets["power"])

    error_buf = (
        (torch.norm(wrist_vel, dim=-1) > 100)
        | (torch.norm(wrist_ang_vel, dim=-1) > 200)
        | (torch.norm(joints_vel, dim=-1).mean(-1) > 100)
        | (torch.abs(states["dq"]).mean(-1) > 200)
        | (torch.norm(obj_vel, dim=-1) > 100)
        | (torch.norm(obj_ang_vel, dim=-1) > 200)
    )
    failed_hand = (
        (thumb_error > 0.04 / 0.7 * scale_factor * fail_tol)
        | (index_error > 0.045 / 0.7 * scale_factor * fail_tol)
        | (middle_error > 0.05 / 0.7 * scale_factor * fail_tol)
        | (pinky_error > 0.06 / 0.7 * scale_factor * fail_tol)
        | (ring_error > 0.06 / 0.7 * scale_factor * fail_tol)
        | (level_1_error > 0.07 / 0.7 * scale_factor * fail_tol)
        | (level_2_error > 0.08 / 0.7 * scale_factor * fail_tol)
    )
    failed_obj_rot = obj_rot_error / np.pi * 180 > 30 / 0.343 * scale_factor ** 3
    failed_obj_pos = obj_pos_error > 0.02 / 0.343 * scale_factor ** 3 * fail_tol + targets["object_pos_tolerance"]
    failed = (failed_obj_pos | failed_hand | failed_obj_rot) & (running_progress_buf >= 8) | error_buf

    contact_history = targets["tip_contact_history"]
    reference_contact = torch.any(targets["tips_distance"] < object_contact_distance, dim=-1)
    policy_contact = torch.any(contact_history.any(1), dim=-1)
    object_contact_gate = (reference_contact | policy_contact).to(reward_obj_pos.dtype)
    reward_object = (
        5.0 * reward_obj_pos + reward_obj_rot + (0.1 * reward_obj_vel + 0.1 * reward_obj_ang_vel)
    ) * object_contact_gate

    ref_lift = target_obj_pos[:, 2] - obj_rest_z
    cur_lift = obj_pos[:, 2] - obj_rest_z
    reward_lift = lift_reward_scale * ((ref_lift > lift_min_height) & (cur_lift > lift_min_height)).to(
        reward_obj_pos.dtype
    )
    lift_gate = torch.where(
        (ref_lift > lift_min_height) & (cur_lift <= lift_min_height),
        torch.zeros_like(reward_wrist_pos),
        torch.ones_like(reward_wrist_pos),
    )

    finger_scale = torch.where(
        targets["reference_hand_free"],
        torch.ones_like(reward_wrist_pos) * free_space_finger_scale,
        torch.ones_like(reward_wrist_pos) * contact_finger_scale,
    )
    reward_wrist = wrist_pos_weight * reward_wrist_pos + wrist_rot_weight * reward_wrist_rot
    reward_fingers = (
        0.9 * reward_thumb
        + 0.8 * reward_index
        + 0.75 * reward_middle
        + 0.6 * reward_pinky
        + 0.6 * reward_ring
        + level_1_weight * reward_level_1
        + level_2_weight * reward_level_2
    ) * finger_scale
    reward_velocity = 0.1 * reward_wrist_vel + 0.05 * reward_wrist_ang_vel + 0.1 * reward_joints_vel
    premature_contact = torch.any(targets["reference_fingers_free"] & targets["pad_object_contact"], dim=-1)
    contact_gate = torch.where(premature_contact, torch.zeros_like(reward_wrist_pos), torch.ones_like(reward_wrist_pos))
    reward_hand = (reward_wrist + reward_fingers + reward_velocity) * contact_gate

    reward = reward_hand * lift_gate + reward_object + reward_lift + 0.5 * reward_power + 0.5
    succeeded = (progress_buf + 4 >= max_length) & ~failed
    reset_buf = torch.where(succeeded | failed, torch.ones_like(reset_buf), reset_buf)
    reward_terms = {
        "reward_wrist": reward_wrist,
        "reward_fingers": reward_fingers,
        "reward_velocity": reward_velocity,
        "reward_hand": reward_hand,
        "reward_object": reward_object,
        "reward_lift": reward_lift,
        "reward_power": reward_power,
        "object_position_error": obj_pos_error,
    }
    return (reward, reset_buf, succeeded, failed, reward_terms, error_buf)


@torch.jit.script
def nearest_surface_distance(query_points: Tensor, surface_points: Tensor) -> Tensor:
    delta = query_points.unsqueeze(2) - surface_points.unsqueeze(1)
    min_sq_distance, _ = torch.min(torch.sum(delta * delta, dim=-1), dim=-1)
    return torch.sqrt(torch.clamp(min_sq_distance, min=0.0))


@torch.jit.script
def pad_distance_score(surface_distance: Tensor, active: Tensor, tolerance: float, sigma: float) -> Tensor:
    excess = torch.clamp(surface_distance - tolerance, min=0.0)
    score = torch.exp(-torch.square(excess / sigma)).mean(dim=-1)
    return score * active.to(surface_distance.dtype)


@torch.jit.script
def torque_penalty(
    dof_force: Tensor, effort_limit: Tensor, deadzone: float, scale: float, max_penalty: float
) -> Tensor:
    fraction = torch.abs(dof_force) / torch.clamp(effort_limit, min=1e-06)
    excess = torch.clamp((fraction - deadzone) / (1.0 - deadzone), min=0.0)
    return -torch.clamp(scale * torch.square(excess).mean(dim=-1), max=max_penalty)


def velocity_fraction(joint_velocity, limits):
    return torch.abs(joint_velocity) / limits.to(dtype=joint_velocity.dtype).unsqueeze(0)


def action_rate(command, previous, valid):
    rate = torch.square(command - previous).sum(dim=-1)
    return torch.where(valid, rate, torch.zeros_like(rate))


def farthest_point_sample(points, count):
    count = min(int(count), int(points.shape[0]))
    center = points.mean(dim=0)
    first = torch.argmin(torch.sum((points - center) ** 2, dim=-1))
    chosen = torch.empty(count, dtype=torch.long, device=points.device)
    chosen[0] = first
    min_sq_distance = torch.sum((points - points[first]) ** 2, dim=-1)
    for index in range(1, count):
        farthest = torch.argmax(min_sq_distance)
        chosen[index] = farthest
        min_sq_distance = torch.minimum(min_sq_distance, torch.sum((points - points[farthest]) ** 2, dim=-1))
    return chosen
