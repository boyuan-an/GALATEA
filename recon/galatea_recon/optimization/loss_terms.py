import torch
from pytorch3d.loss import point_mesh_distance


def l1_loss(A, B):
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A.shape={A.shape}, B.shape={B.shape}")
    return torch.nn.functional.l1_loss(A, B)


def l2_loss(A, B):
    if A.shape != B.shape:
        raise ValueError(f"Shape mismatch: A.shape={A.shape}, B.shape={B.shape}")
    return torch.nn.functional.mse_loss(A, B)


def huber_loss(a, delta=1e-2):
    abs_a = torch.abs(a)
    quadratic = 0.5 * abs_a**2
    linear = delta * (abs_a - 0.5 * delta)
    loss = torch.where(abs_a < delta, quadratic, linear)
    return loss.mean()


def bidirectional_chamfer_loss(pred_verts, gt_points, trim_pct=0.2):
    from pytorch3d.ops import knn_points

    device = pred_verts.device
    if pred_verts.shape[0] == 0 or gt_points.shape[0] == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pred = pred_verts.float().unsqueeze(0)
    gt = gt_points.float().unsqueeze(0)

    pred2gt_dists = knn_points(pred, gt, K=1).dists.squeeze(0).squeeze(-1)

    gt2pred_dists = knn_points(gt, pred, K=1).dists.squeeze(0).squeeze(-1)

    k_p = max(1, int(len(pred2gt_dists) * (1.0 - trim_pct)))
    k_g = max(1, int(len(gt2pred_dists) * (1.0 - trim_pct)))
    trimmed_p, _ = pred2gt_dists.topk(k_p, largest=False)
    trimmed_g, _ = gt2pred_dists.topk(k_g, largest=False)

    return trimmed_p.mean() + trimmed_g.mean()


def contact_loss(verts_A, verts_B, num_vertices=2000, top_k=200, delta=0.001):
    if verts_A.shape[0] > num_vertices:
        verts_A = verts_A[torch.randperm(verts_A.shape[0])[:num_vertices]]
    if verts_B.shape[0] > num_vertices:
        verts_B = verts_B[torch.randperm(verts_B.shape[0])[:num_vertices]]

    distances = torch.cdist(verts_A.float(), verts_B.float())

    min_distances_A_to_B = torch.min(distances, dim=1).values

    min_distances_B_to_A = torch.min(distances, dim=0).values

    all_min_distances = torch.cat([min_distances_A_to_B, min_distances_B_to_A], dim=0)

    sorted_min_distances, _ = torch.sort(all_min_distances)

    top_k = min(top_k, len(sorted_min_distances))
    clipped_distances = torch.clamp(sorted_min_distances[:top_k], min=0.0)

    contact_loss_value = huber_loss(clipped_distances, delta=delta)

    return contact_loss_value


def contact_depth_loss(
    verts_A, verts_B, cameras, num_vertices=2000, top_k=200, delta=0.001, screen_dist_thresh=20.0
):
    if verts_A.shape[0] == 0 or verts_B.shape[0] == 0:
        return torch.tensor(0.0, device=verts_A.device, requires_grad=True)

    if verts_A.shape[0] > num_vertices:
        verts_A = verts_A[torch.randperm(verts_A.shape[0], device=verts_A.device)[:num_vertices]]
    if verts_B.shape[0] > num_vertices:
        verts_B = verts_B[torch.randperm(verts_B.shape[0], device=verts_B.device)[:num_vertices]]

    screen_A = cameras.transform_points_screen(verts_A.unsqueeze(0)).squeeze(0)[:, :2]
    screen_B = cameras.transform_points_screen(verts_B.unsqueeze(0)).squeeze(0)[:, :2]

    dists_2d = torch.cdist(screen_B.float(), screen_A.float())
    min_dists_2d = dists_2d.min(dim=1).values

    mask = min_dists_2d < screen_dist_thresh
    if mask.sum() == 0:
        return torch.tensor(0.0, device=verts_A.device, requires_grad=True)
    verts_B = verts_B[mask]

    distances_3d = torch.cdist(verts_A.float(), verts_B.float())
    nn_idx_A = distances_3d.argmin(dim=1)
    nn_idx_B = distances_3d.argmin(dim=0)

    view_transform = cameras.get_world_to_view_transform()
    z_A = view_transform.transform_points(verts_A.unsqueeze(0)).squeeze(0)[:, 2]
    z_B = view_transform.transform_points(verts_B.unsqueeze(0)).squeeze(0)[:, 2]

    depth_diff_A = (z_A - z_B[nn_idx_A]).abs()
    depth_diff_B = (z_B - z_A[nn_idx_B]).abs()

    all_depth_diffs = torch.cat([depth_diff_A, depth_diff_B], dim=0)
    sorted_diffs, _ = all_depth_diffs.sort()

    top_k = min(top_k, len(sorted_diffs))
    return huber_loss(sorted_diffs[:top_k], delta=delta)


def contact_smoothness_loss(verts_A_seq, verts_B_seq, top_k=200):
    window_size = len(verts_A_seq)

    if window_size < 2:
        device = verts_A_seq[0].device if window_size > 0 else torch.device("cpu")
        return torch.tensor(0.0, device=device, requires_grad=True)

    verts_A_middle = verts_A_seq[window_size // 2]
    verts_B_middle = verts_B_seq[window_size // 2]

    distances_middle = torch.cdist(verts_A_middle.float(), verts_B_middle.float())

    distances_flat = distances_middle.flatten()
    top_k = min(top_k, len(distances_flat))
    top_k_values, top_k_indices = torch.topk(distances_flat, top_k, largest=False)

    num_B = verts_B_middle.shape[0]
    idx_A = top_k_indices // num_B
    idx_B = top_k_indices % num_B

    distance_sequences = []
    for t in range(window_size):
        verts_A_t = verts_A_seq[t]
        verts_B_t = verts_B_seq[t]

        selected_A = verts_A_t[idx_A]
        selected_B = verts_B_t[idx_B]

        distances_t = torch.norm(selected_A - selected_B, dim=-1)
        distance_sequences.append(distances_t)

    distance_sequences = torch.stack(distance_sequences, dim=0)

    avg_distance_per_pair = torch.mean(distance_sequences, dim=0)

    diff_from_mean = torch.abs(
        distance_sequences - avg_distance_per_pair.unsqueeze(0)
    )

    return torch.mean(diff_from_mean)


def contact_distribution_smoothness_loss(
    human_contact_verts_seq, obj_verts_seq, temperature=100.0, num_obj_verts=2000
):
    T = len(human_contact_verts_seq)
    if T < 2:
        dev = human_contact_verts_seq[0].device if T > 0 else torch.device("cpu")
        return torch.tensor(0.0, device=dev, requires_grad=True)

    obj_verts = obj_verts_seq
    if obj_verts.shape[1] > num_obj_verts:
        subsample_idx = torch.linspace(0, obj_verts.shape[1] - 1, num_obj_verts).long()
        obj_verts = obj_verts[:, subsample_idx]

    dists = torch.cdist(obj_verts.float(), human_contact_verts_seq.float())
    min_dists = torch.min(dists, dim=2).values
    distributions = torch.softmax(-min_dists * temperature, dim=1)
    diff = distributions[1:] - distributions[:-1]
    return diff.pow(2).sum(dim=-1).mean()


def contact_center_loss(verts_A, verts_B):
    center_A = torch.mean(verts_A, dim=0)
    center_B = torch.mean(verts_B, dim=0)
    return torch.nn.functional.l1_loss(center_A, center_B)


def ground_loss(verts_seq, gravity_axis="z", height=0.14):
    if gravity_axis == "y":
        min_verts_seq = torch.min(verts_seq[:, :, 1], dim=1).values
    elif gravity_axis == "z":
        min_verts_seq = torch.min(verts_seq[:, :, 2], dim=1).values
    else:
        raise ValueError(f"Invalid gravity axis: {gravity_axis}")

    return torch.mean(torch.abs(min_verts_seq - height))


def penetration_loss(verts_A, sdf_B, threshold=0.0):
    sdf_values = sdf_B.query(verts_A, method="grid")

    penetration = threshold - sdf_values
    penetration = torch.clamp(penetration, min=0.0)

    return penetration.sum()


def keypoint_loss(pred_keypoints, gt_keypoints, gt_conf=None, conf_thres=0.6):
    if gt_conf is not None:
        gt_keypoints = gt_keypoints[gt_conf > conf_thres]
        pred_keypoints = pred_keypoints[gt_conf > conf_thres]

    if pred_keypoints.shape[0] == 0:
        return torch.tensor(0.0)

    return torch.nn.functional.l1_loss(pred_keypoints, gt_keypoints)


def smoothness_loss(seq, beta=1.0):
    first_order_loss = torch.nn.functional.l1_loss(seq[1:] - seq[:-1], torch.zeros_like(seq[1:]))
    second_order_loss = torch.nn.functional.l1_loss(
        seq[2:] - 2 * seq[1:-1] + seq[:-2],
        torch.zeros_like(seq[2:]),
    )
    return first_order_loss + beta * second_order_loss


def pose_smoothness_loss(pose_seq, beta=1.0, joint_weights=None):
    from pytorch3d.transforms import axis_angle_to_matrix, so3_log_map

    if pose_seq.dim() != 3:
        raise ValueError(
            f"Expected pose_seq to have 3 dimensions (frame_num, N, 3), got {pose_seq.shape}"
        )

    frame_num, num_joints, _ = pose_seq.shape
    device = pose_seq.device

    if frame_num < 2:
        return torch.tensor(0.0, device=device, requires_grad=True)

    pose_flat = pose_seq.reshape(-1, 3)
    rot_mats = axis_angle_to_matrix(pose_flat)
    rot_mats = rot_mats.reshape(frame_num, num_joints, 3, 3)

    if joint_weights is None:
        joint_weights = torch.ones(num_joints, device=device)
    else:
        joint_weights = joint_weights.to(device)

    R_t = rot_mats[:-1]
    R_t_plus_1 = rot_mats[1:]

    R_t_inv = R_t.transpose(-2, -1)
    R_rel = torch.matmul(R_t_inv, R_t_plus_1)

    R_rel_flat = R_rel.reshape(-1, 3, 3)
    log_R_rel_flat = so3_log_map(R_rel_flat)

    log_velocities = log_R_rel_flat.reshape(frame_num - 1, num_joints, 3)

    vel_norms = torch.norm(log_velocities, dim=-1)
    weighted_vel_norms = vel_norms**2 * joint_weights.unsqueeze(0)
    vel_loss = weighted_vel_norms.sum() / (frame_num - 1)

    acc_loss = torch.tensor(0.0, device=device)

    if frame_num >= 3:
        accelerations = log_velocities[1:] - log_velocities[:-1]

        acc_norms = torch.norm(accelerations, dim=-1)
        weighted_acc_norms = acc_norms**2 * joint_weights.unsqueeze(0)
        acc_loss = weighted_acc_norms.sum() / (frame_num - 2)

    return vel_loss + beta * acc_loss


def reg_loss(new_seq, old_seq, use_l2=False):
    if new_seq.shape != old_seq.shape:
        raise ValueError(
            f"Shape mismatch: new_seq.shape={new_seq.shape}, old_seq.shape={old_seq.shape}"
        )
    if use_l2:
        return torch.nn.functional.mse_loss(new_seq, old_seq)
    else:
        return torch.nn.functional.l1_loss(new_seq, old_seq)
