import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

VERSION = 1


def boundary_config(enabled=True):
    return dict(
        version=VERSION,
        enabled=bool(enabled),
        blend_seconds=0.5,
        velocity_seconds=0.25,
        min_step_m_at_24fps=0.02,
        min_step_deg_at_24fps=20.0,
        outlier_ratio=4.0,
    )


def sequence_fps(cam_dir):
    path = Path(cam_dir) / "prep_metadata.json"
    if not path.exists():
        return 24.0
    data = json.loads(path.read_text())
    fps = float(data.get("output_fps", data.get("source_fps", 24.0)))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("prep_metadata.json must contain a positive finite FPS")
    return fps


def boundary_inputs(cam_dir, seg_labels, manual_json, max_frames, enabled=True):
    def digest(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest() if path else None

    return dict(
        config=boundary_config(enabled),
        fps=sequence_fps(cam_dir),
        seg_labels_sha256=digest(seg_labels),
        manual_sha256=digest(manual_json),
        max_frames=int(max_frames),
    )


def _rot_steps(rotations):
    return Rotation.from_matrix(rotations[1:] @ rotations[:-1].transpose(0, 2, 1)).as_rotvec()


def _runs(mask):
    padded = np.r_[False, mask, False].astype(np.int8)
    return zip(np.flatnonzero(np.diff(padded) == 1), np.flatnonzero(np.diff(padded) == -1))


def _smoothstep(value):
    value = np.clip(value, 0, 1)
    return value * value * (3 - 2 * value)


def _repair_track(centers, rotations, contact, observed, fps, reference=None):
    centers = np.asarray(centers, dtype=np.float64)
    rotations = np.asarray(rotations, dtype=np.float64)
    new_c, new_r = centers.copy(), rotations.copy()
    n = len(centers)
    width = max(2, int(round(0.5 * fps)))
    velocity_frames = max(3, int(round(0.25 * fps)))
    min_distance, min_angle = 0.02 * 24 / fps, np.deg2rad(20) * 24 / fps
    records = []
    if n < 2:
        return new_c, new_r, records
    linear, angular = np.diff(centers, axis=0), _rot_steps(rotations)
    for start, stop in _runs(~contact):
        candidates = []
        for entering, outside, inside in ((False, start, start - 1), (True, stop - 1, stop)):
            if inside < 0 or inside >= n:
                continue
            seam = max(outside, inside)
            record = dict(frame=int(seam), entering=entering, status="unchanged")
            records.append(record)
            if not (observed[outside] and observed[inside]):
                record["status"] = "unobserved_boundary"
                continue
            edge_ids = (
                np.arange(max(start, stop - velocity_frames), stop - 1)
                if entering
                else np.arange(start, min(stop - 1, start + velocity_frames - 1))
            )
            edge_ids = edge_ids[observed[edge_ids] & observed[edge_ids + 1]]
            if len(edge_ids) < 2:
                record["status"] = "insufficient_outside_samples"
                continue
            velocity = np.median(linear[edge_ids], axis=0)
            spin = np.median(angular[edge_ids], axis=0)
            linear_limit = max(
                min_distance, 4 * float(np.median(np.linalg.norm(linear[edge_ids], axis=1)))
            )
            angular_limit = max(
                min_angle, 4 * float(np.median(np.linalg.norm(angular[edge_ids], axis=1)))
            )
            seam_step = linear[seam - 1]
            seam_spin = angular[seam - 1]
            distance = float(np.linalg.norm(seam_step - velocity))
            angle = float(np.linalg.norm(seam_spin - spin))
            move, turn = distance > linear_limit, angle > angular_limit
            record.update(
                before_step_m=float(np.linalg.norm(seam_step)),
                before_rotation_deg=float(np.rad2deg(np.linalg.norm(seam_spin))),
                excess_step_m=distance,
                excess_rotation_deg=float(np.rad2deg(angle)),
                threshold_m=linear_limit,
                threshold_deg=float(np.rad2deg(angular_limit)),
            )
            if reference is not None:
                ref_c, ref_r = reference
                ref_step = ref_c[seam] - ref_c[seam - 1]
                ref_spin = Rotation.from_matrix(ref_r[seam] @ ref_r[seam - 1].T).as_rotvec()
                if np.linalg.norm(seam_step - ref_step) < max(min_distance, 0.25 * distance):
                    move = False
                if np.linalg.norm(seam_spin - ref_spin) < max(min_angle, 0.25 * angle):
                    turn = False
            if not (move or turn):
                if distance > linear_limit or angle > angular_limit:
                    record["status"] = "input_supported_motion"
                continue
            sign = -1 if entering else 1
            delta = centers[inside] + sign * velocity - centers[outside] if move else np.zeros(3)
            target_r = Rotation.from_rotvec(sign * spin).as_matrix() @ rotations[inside]
            correction = (
                Rotation.from_matrix(target_r @ rotations[outside].T).as_rotvec()
                if turn
                else np.zeros(3)
            )
            candidates.append((entering, delta, correction, record))
        if not candidates:
            continue
        ids = np.arange(start, stop)
        offset = np.zeros((len(ids), 3))
        rot_offset = np.zeros_like(offset)
        extent = min(width - 1, len(ids) - 1)
        if extent < 1:
            for *_, record in candidates:
                record["status"] = "insufficient_transition_space"
            continue
        for entering, delta, correction, record in candidates:
            distance_from_seam = stop - 1 - ids if entering else ids - start
            weight = 1 - _smoothstep(distance_from_seam / extent)
            offset += weight[:, None] * delta
            rot_offset += weight[:, None] * correction
        trial_c, trial_r = centers.copy(), rotations.copy()
        trial_c[ids] += offset
        trial_r[ids] = Rotation.from_rotvec(rot_offset).as_matrix() @ rotations[ids]
        edges = np.arange(max(0, start - 1), min(n - 1, stop))
        before_d = np.linalg.norm(linear[edges], axis=1)
        before_a = np.linalg.norm(angular[edges], axis=1)
        after_d = np.linalg.norm(np.diff(trial_c, axis=0)[edges], axis=1)
        after_a = np.linalg.norm(_rot_steps(trial_r)[edges], axis=1)
        accepted = after_d.max() <= before_d.max() + 1e-8 and after_a.max() <= before_a.max() + 1e-8
        for _, delta, correction, record in candidates:
            edge = record["frame"] - 1
            d = float(np.linalg.norm(trial_c[edge + 1] - trial_c[edge]))
            a = float(np.linalg.norm(_rot_steps(trial_r[edge : edge + 2])[0]))
            if np.linalg.norm(delta) > 0:
                accepted &= d <= max(record["threshold_m"], 0.5 * record["before_step_m"])
            if np.linalg.norm(correction) > 0:
                accepted &= np.rad2deg(a) <= max(
                    record["threshold_deg"], 0.5 * record["before_rotation_deg"]
                )
        if accepted:
            new_c[ids], new_r[ids] = trial_c[ids], trial_r[ids]
        for _, _, _, record in candidates:
            record["status"] = "repaired" if accepted else "rejected_new_jump"
            record["transition_frames"] = ids[
                np.any(offset != 0, axis=1) | np.any(rot_offset != 0, axis=1)
            ].tolist()
            record["interval_max_step_before_m"] = float(before_d.max())
            record["interval_max_step_after_m"] = (
                float(after_d.max()) if accepted else float(before_d.max())
            )
    for record in records:
        edge = record["frame"] - 1
        record["after_step_m"] = float(np.linalg.norm(new_c[edge + 1] - new_c[edge]))
        record["after_rotation_deg"] = float(
            np.rad2deg(np.linalg.norm(_rot_steps(new_r[edge : edge + 2])[0]))
        )
    return new_c, new_r, records


def _palm_rotations(joints):
    palm = joints[:, [0, 5, 9, 13, 17]].astype(np.float64)
    palm -= palm.mean(1, keepdims=True)
    reference = palm[np.argmax(np.linalg.norm(palm, axis=(1, 2)))]
    covariance = np.einsum("vi,tvj->tij", reference, palm)
    u, _, vt = np.linalg.svd(covariance)
    correction = np.tile(np.eye(3), (len(joints), 1, 1))
    correction[:, -1, -1] = np.linalg.det(vt.transpose(0, 2, 1) @ u.transpose(0, 2, 1))
    return vt.transpose(0, 2, 1) @ correction @ u.transpose(0, 2, 1)


def _local(points, poses):
    return np.einsum("tji,tvj->tvi", poses[:, :3, :3], points - poses[:, None, :3, 3])


def protect_contact_boundaries(
    poses,
    vertices,
    joints,
    contact,
    *,
    fps=24.0,
    object_observed=None,
    hand_observed=None,
    fp_poses=None,
    enabled=True,
):
    poses, vertices, joints = (np.asarray(a) for a in (poses, vertices, joints))
    n = len(poses)
    contact = np.asarray(contact, dtype=bool)
    if (
        poses.shape != (n, 4, 4)
        or vertices.ndim != 3
        or vertices.shape[0] != n
        or vertices.shape[2] != 3
        or joints.shape != (n, 21, 3)
        or contact.shape != (n,)
    ):
        raise ValueError("inconsistent boundary-guard trajectory shapes")
    if (
        not np.isfinite(fps)
        or fps <= 0
        or any(not np.isfinite(a).all() for a in (poses, vertices, joints))
    ):
        raise ValueError("boundary guard requires finite trajectories and positive FPS")
    observed = [
        np.ones(n, dtype=bool) if value is None else np.asarray(value, dtype=bool)
        for value in (object_observed, hand_observed)
    ]
    if any(a.shape != (n,) for a in observed):
        raise ValueError("observation flags must match trajectory length")
    out_p, out_v, out_j = (a.copy() for a in (poses, vertices, joints))
    report = dict(
        config=boundary_config(enabled),
        fps=float(fps),
        object=[],
        hand=[],
        changed_object_frames=[],
        changed_hand_frames=[],
    )
    if not enabled or n < 2 or not contact.any() or contact.all():
        return out_p, out_v, out_j, report
    reference = None
    if fp_poses is not None:
        fp_poses = np.asarray(fp_poses)
        if fp_poses.shape != poses.shape or not np.isfinite(fp_poses).all():
            raise ValueError("FP reference must be finite and match object poses")
        reference = (fp_poses[:, :3, 3], fp_poses[:, :3, :3])
    centers, rotations, report["object"] = _repair_track(
        poses[:, :3, 3], poses[:, :3, :3], contact, observed[0], fps, reference
    )
    changed = (
        (np.linalg.norm(centers - poses[:, :3, 3], axis=1) > 1e-10)
        | (np.linalg.norm(rotations - poses[:, :3, :3], axis=(1, 2)) > 1e-10)
    ) & ~contact
    out_p[changed, :3, 3], out_p[changed, :3, :3] = centers[changed], rotations[changed]
    report["changed_object_frames"] = np.flatnonzero(changed).tolist()
    local_v, local_j = _local(vertices.astype(float), out_p), _local(joints.astype(float), out_p)
    centers, rotations = local_v.mean(1), _palm_rotations(local_j)
    new_c, new_r, report["hand"] = _repair_track(
        centers, rotations, contact, observed[0] & observed[1], fps
    )
    changed = (
        (np.linalg.norm(new_c - centers, axis=1) > 1e-10)
        | (np.linalg.norm(new_r - rotations, axis=(1, 2)) > 1e-10)
    ) & ~contact
    delta_r = new_r[changed] @ rotations[changed].transpose(0, 2, 1)
    for local, output in ((local_v, out_v), (local_j, out_j)):
        corrected = (
            np.einsum("tij,tvj->tvi", delta_r, local[changed] - centers[changed, None])
            + new_c[changed, None]
        )
        output[changed] = (
            np.einsum("tij,tvj->tvi", out_p[changed, :3, :3], corrected)
            + out_p[changed, None, :3, 3]
        )
    report["changed_hand_frames"] = np.flatnonzero(changed).tolist()
    input_local = _local(vertices.astype(float), poses).mean(1)
    output_local = _local(out_v.astype(float), out_p).mean(1)
    for entry in report["hand"]:
        seam = entry["frame"]
        entry["input_step_m"] = float(np.linalg.norm(input_local[seam] - input_local[seam - 1]))
        entry["output_step_m"] = float(np.linalg.norm(output_local[seam] - output_local[seam - 1]))
    if (
        any(not np.isfinite(a).all() for a in (out_p, out_v, out_j))
        or np.any((poses[:, 2, 3] > 0) & (out_p[:, 2, 3] <= 0))
        or np.any((vertices[..., 2] > 0) & (out_v[..., 2] <= 0))
    ):
        report["rejected"] = "invalid_output_geometry"
        report["changed_object_frames"] = []
        report["changed_hand_frames"] = []
        for entry in report["object"] + report["hand"]:
            if entry["status"] == "repaired":
                entry["status"] = "rejected_invalid_geometry"
                entry["after_step_m"] = entry["before_step_m"]
                entry["after_rotation_deg"] = entry["before_rotation_deg"]
        for entry in report["hand"]:
            entry["output_step_m"] = entry["input_step_m"]
        return poses.copy(), vertices.copy(), joints.copy(), report
    return out_p, out_v, out_j, report
