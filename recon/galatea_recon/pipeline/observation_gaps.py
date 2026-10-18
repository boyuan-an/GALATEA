from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


MIN_MASK_PIXELS = 20


def mask_observed(mask, min_pixels=MIN_MASK_PIXELS):
    return mask is not None and np.count_nonzero(mask) >= min_pixels


def object_observations(masks, n, candidate=None):
    valid = np.zeros(n, dtype=bool)
    previous_area = None
    lost = False
    for i in range(n):
        key = f"obj_candidate_{candidate}_{i}"
        if candidate is None or key not in masks:
            key = f"obj_{i}"
        mask = np.asarray(masks[key], dtype=bool) if key in masks else None
        if not mask_observed(mask):
            lost = True
            continue
        area = int(mask.sum())
        clipped = mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
        if clipped or (lost and previous_area is not None and area < 0.25 * previous_area):
            lost = True
            continue
        valid[i] = True
        previous_area = area
        lost = False
    return valid


def pose_observed(pose):
    pose = np.asarray(pose)
    return bool(
        pose.shape == (4, 4)
        and np.isfinite(pose).all()
        and pose[2, 3] > 0
        and np.allclose(pose[3], [0, 0, 0, 1], atol=1e-4)
        and np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-3)
        and np.linalg.det(pose[:3, :3]) > 0.99
    )


def validate_observed(observed, n):
    observed = np.asarray(observed)
    if observed.dtype != np.bool_ or observed.shape != (n,):
        raise ValueError("observation validity must be a boolean array of shape (T,)")
    if not observed.any():
        raise ValueError("no valid observations in the entire sequence; cannot interpolate")
    return observed


def gap_spans(observed):
    observed = validate_observed(observed, len(observed))
    anchors = np.flatnonzero(observed)
    edges = np.flatnonzero(np.diff(np.r_[True, observed, True]))
    for start, end in edges.reshape(-1, 2):
        yield int(start), int(end), max(int(anchors[0]), int(start - 1)), min(
            int(anchors[-1]), int(end)
        )


def fill_linear(values, observed):
    values = np.asarray(values)
    observed = validate_observed(observed, len(values))
    if not np.isfinite(values[observed]).all():
        raise ValueError("non-finite values on an observed frame")
    result = values.copy()
    for start, end, left, right in gap_spans(observed):
        if left == right:
            result[start:end] = values[left]
        else:
            alpha = (np.arange(start, end) - left) / (right - left)
            alpha = alpha.reshape((-1,) + (1,) * (values.ndim - 1))
            result[start:end] = (1 - alpha) * values[left] + alpha * values[right]
    return result


def fill_poses(poses, observed):
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError("poses must have shape (T,4,4)")
    observed = validate_observed(observed, len(poses))
    if not all(pose_observed(p) for p in poses[observed]):
        raise ValueError("invalid rigid pose on an observed frame")
    result = poses.copy()
    result[:, :3, 3] = fill_linear(poses[:, :3, 3], observed)
    for start, end, left, right in gap_spans(observed):
        if left == right:
            result[start:end] = poses[left]
        else:
            result[start:end, :3, :3] = Slerp(
                [left, right], Rotation.from_matrix(poses[[left, right], :3, :3])
            )(np.arange(start, end)).as_matrix()
            result[start:end, 3] = [0, 0, 0, 1]
    return result


def observation_path(poses_path):
    path = Path(poses_path)
    return path.with_name(path.stem + "_observations.npz")


def load_object_observed(poses_path, n, masks=None):
    path = observation_path(poses_path)
    if path.exists():
        with np.load(path) as data:
            return validate_observed(data["object_observed"][:n], n)
    if masks is not None:
        return validate_observed(object_observations(masks, n), n)
    return np.ones(n, dtype=bool)


def hand_observed(hand, image_size=None, require_mano=False):
    keys = ["verts", "kp3d", "kp2d"]
    if require_mano:
        keys += ["global_orient", "hand_pose", "betas", "place_R", "place_t"]
    if any(k not in hand or not np.isfinite(hand[k]).all() for k in keys):
        return False
    if np.median(np.asarray(hand["kp3d"])[:, 2]) <= 0:
        return False
    if image_size is not None:
        width, height = image_size
        points = np.asarray(hand["kp2d"])
        inside = (points[:, 0] >= 0) & (points[:, 0] < width)
        inside &= (points[:, 1] >= 0) & (points[:, 1] < height)
        if not inside.all():
            return False
    return True


def complete_hands(hands, observed=None, image_size=None, require_mano=False):
    import copy

    from galatea_recon.pipeline.clean_wilor_hands import interpolate_hand

    usable = np.array(
        [bool(len(frame)) and hand_observed(frame[0], image_size, require_mano) for frame in hands],
        dtype=bool,
    )
    if observed is not None:
        observed = np.asarray(observed)
        if observed.dtype != np.bool_ or observed.shape != usable.shape:
            raise ValueError("hand observation validity must have shape (T,) and boolean dtype")
        usable &= observed
    validate_observed(usable, len(hands))
    result = np.empty(len(hands), dtype=object)
    for i, frame in enumerate(hands):
        result[i] = copy.deepcopy(list(frame))
    for start, end, left, right in gap_spans(usable):
        for i in range(start, end):
            hand = (
                copy.deepcopy(hands[left][0])
                if left == right
                else interpolate_hand(hands[left][0], hands[right][0], (i - left) / (right - left))
            )
            result[i] = [hand]
    return result, usable


def complete_output_hands(points, poses, hand_valid, object_valid, contact):
    result = fill_linear(points, hand_valid)
    for start, end, left, right in gap_spans(hand_valid):
        if not (np.all(contact[left : right + 1]) and object_valid[left] and object_valid[right]):
            continue
        local_left = (points[left] - poses[left, :3, 3]) @ poses[left, :3, :3]
        local_right = (points[right] - poses[right, :3, 3]) @ poses[right, :3, :3]
        for i in range(start, end):
            alpha = 0.0 if left == right else (i - left) / (right - left)
            local = (1 - alpha) * local_left + alpha * local_right
            result[i] = local @ poses[i, :3, :3].T + poses[i, :3, 3]
    return result


def hand_output_anchors(hand_observed, object_observed, contact):
    joint = hand_observed & (~contact | object_observed)
    return joint if joint.any() else hand_observed.copy()
