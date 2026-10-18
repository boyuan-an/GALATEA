import math


def constraint_config(weight=1.0, free_m=0.01, free_fraction=0.05, evidence_gain=0.05):
    values = [float(v) for v in (weight, free_m, free_fraction, evidence_gain)]
    if not all(math.isfinite(v) for v in values):
        raise ValueError("FP constraint settings must be finite")
    weight, free_m, free_fraction, evidence_gain = values
    if weight < 0 or free_m <= 0 or free_fraction < 0 or evidence_gain <= 0:
        raise ValueError("FP weight/fraction must be nonnegative; free_m/gain positive")
    return dict(version=1, tracking_reduction="sum_frame_means", weight=weight,
                free_m=free_m, free_fraction=free_fraction, evidence_gain=evidence_gain)


def add_constraint_arguments(parser):
    config = constraint_config()
    parser.add_argument("--w_fp_depth", type=float, default=config["weight"],
                        help="soft FP center-depth prior weight; 0 disables it")
    parser.add_argument("--fp_depth_free_m", type=float, default=config["free_m"],
                        help="minimum unpenalized depth correction in meters")
    parser.add_argument("--fp_depth_free_fraction", type=float, default=config["free_fraction"],
                        help="unpenalized correction as a fraction of mesh diagonal")
    parser.add_argument("--fp_depth_evidence_gain", type=float, default=config["evidence_gain"],
                        help="silhouette IoU gain that reduces depth penalty by four")


def fp_frame_validity(poses, masks=None, observed=None):
    import numpy as np

    valid = np.isfinite(poses).all(axis=(1, 2)) & (poses[:, 2, 3] > 0)
    if observed is not None:
        valid &= np.asarray(observed, dtype=bool)
    if masks is not None:
        for frame in range(len(poses)):
            key = f"obj_{frame}"
            mask = np.asarray(masks[key], dtype=bool) if key in masks else None
            if (mask is None or mask.ndim != 2 or mask.sum() < 20
                    or mask[0].any() or mask[-1].any()
                    or mask[:, 0].any() or mask[:, -1].any()):
                valid[frame] = False
    return valid


def frame_sum_l1(prediction, reference, valid, confidence=None):
    values = (prediction[valid] - reference[valid]).abs().flatten(1).mean(1)
    if confidence is not None:
        values = values * confidence[valid].detach()
    return values.sum()


def soft_mask_iou(alpha, target, visible):
    intersection = (alpha * target * visible).flatten(1).sum(1)
    union = ((alpha + target - alpha * target) * visible).flatten(1).sum(1)
    return intersection / union.clamp_min(1e-6)


def center_depth(poses, center_h):
    return (poses[:, 2, :] * center_h).sum(-1)


def depth_prior(current_z, reference_z, valid, free_distance,
                current_iou=None, reference_iou=None, evidence_gain=0.05, confidence=None):
    delta = current_z[valid] - reference_z[valid]
    penalty = (delta.abs() / free_distance - 1).clamp_min(0).square()
    if reference_iou is not None:
        trust = reference_iou[valid].detach().clamp(0, 1)
        gain = (current_iou[valid] - reference_iou[valid].detach()).clamp_min(0)
        penalty = penalty * trust / (1 + gain / evidence_gain).square()
    if confidence is not None:
        penalty = penalty * confidence[valid].detach()
    return penalty.sum()
