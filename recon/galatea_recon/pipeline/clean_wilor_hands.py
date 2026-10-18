from __future__ import annotations

import argparse
import copy
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


LINEAR_KEYS = ("verts", "kp3d", "kp2d", "betas", "place_t", "root")


def _rotation_angle_deg(left: np.ndarray, right: np.ndarray) -> float:
    relative = Rotation.from_matrix(left).inv() * Rotation.from_matrix(right)
    return float(np.degrees(relative.magnitude()))


def _slerp_matrix(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    rotations = Rotation.from_matrix(np.stack([left, right]))
    return Slerp([0.0, 1.0], rotations)([alpha]).as_matrix()[0].astype(np.float32)


def _slerp_rotvec(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
    shape = left.shape
    rotations = Rotation.from_rotvec(np.stack([left.reshape(3), right.reshape(3)]))
    return Slerp([0.0, 1.0], rotations)([alpha]).as_rotvec()[0].astype(np.float32).reshape(shape)


def interpolate_hand(left: dict[str, Any], right: dict[str, Any], alpha: float) -> dict[str, Any]:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha}")
    output = copy.deepcopy(left)
    output["is_right"] = 1.0
    for key in LINEAR_KEYS:
        if key in left and key in right:
            a = np.asarray(left[key])
            b = np.asarray(right[key])
            if a.shape != b.shape:
                raise ValueError(f"cannot interpolate {key}: {a.shape} != {b.shape}")
            output[key] = ((1.0 - alpha) * a + alpha * b).astype(np.float32)

    if "global_orient" in left and "global_orient" in right:
        output["global_orient"] = _slerp_rotvec(
            np.asarray(left["global_orient"]), np.asarray(right["global_orient"]), alpha
        )
    if "place_R" in left and "place_R" in right:
        output["place_R"] = _slerp_matrix(
            np.asarray(left["place_R"]), np.asarray(right["place_R"]), alpha
        )
    if "hand_pose" in left and "hand_pose" in right:
        left_pose = np.asarray(left["hand_pose"])
        right_pose = np.asarray(right["hand_pose"])
        if left_pose.shape != right_pose.shape or left_pose.shape[-1] != 3:
            raise ValueError(
                f"cannot interpolate hand_pose: {left_pose.shape} != {right_pose.shape}"
            )
        output["hand_pose"] = np.stack(
            [
                _slerp_rotvec(left_pose[joint], right_pose[joint], alpha)
                for joint in range(len(left_pose))
            ]
        ).astype(np.float32)
    return output


def _position_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    distances = []
    if "place_t" in left and "place_t" in right:
        distances.append(np.linalg.norm(np.asarray(left["place_t"]) - np.asarray(right["place_t"])))
    if "kp3d" in left and "kp3d" in right:
        distances.append(np.linalg.norm(np.asarray(left["kp3d"])[0] - np.asarray(right["kp3d"])[0]))
    if "verts" in left and "verts" in right:
        left_center = np.asarray(left["verts"]).mean(axis=0)
        right_center = np.asarray(right["verts"]).mean(axis=0)
        distances.append(np.linalg.norm(left_center - right_center))
    if not distances:
        raise ValueError("hand detections have no usable 3D position fields")
    return float(max(distances))


def _rotation_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    distances = []
    if "place_R" in left and "place_R" in right:
        distances.append(
            _rotation_angle_deg(np.asarray(left["place_R"]), np.asarray(right["place_R"]))
        )
    if "global_orient" in left and "global_orient" in right:
        left_global = Rotation.from_rotvec(np.asarray(left["global_orient"]).reshape(3)).as_matrix()
        right_global = Rotation.from_rotvec(
            np.asarray(right["global_orient"]).reshape(3)
        ).as_matrix()
        distances.append(_rotation_angle_deg(left_global, right_global))
    if "hand_pose" in left and "hand_pose" in right:
        left_pose = np.asarray(left["hand_pose"]).reshape(-1, 3)
        right_pose = np.asarray(right["hand_pose"]).reshape(-1, 3)
        if left_pose.shape == right_pose.shape:
            for left_joint, right_joint in zip(left_pose, right_pose):
                left_rotation = Rotation.from_rotvec(left_joint).as_matrix()
                right_rotation = Rotation.from_rotvec(right_joint).as_matrix()
                distances.append(_rotation_angle_deg(left_rotation, right_rotation))
    return float(max(distances, default=0.0))


def _choose_primary_right_hands(
    hands_per_frame: np.ndarray,
) -> tuple[list[dict[str, Any] | None], dict[str, Any]]:
    selected: list[dict[str, Any] | None] = []
    previous = None
    input_detections = 0
    discarded_left = 0
    multi_right_frames = []
    missing_right_frames = []

    for frame_index, frame_hands in enumerate(hands_per_frame):
        candidates = list(frame_hands)
        input_detections += len(candidates)
        right_hands = [hand for hand in candidates if float(hand.get("is_right", 0.0)) > 0.5]
        discarded_left += len(candidates) - len(right_hands)
        if not right_hands:
            selected.append(None)
            missing_right_frames.append(frame_index)
            continue
        if len(right_hands) > 1:
            multi_right_frames.append(frame_index)
        if previous is None or len(right_hands) == 1:
            chosen = right_hands[0]
        else:
            chosen = min(
                right_hands,
                key=lambda hand: _position_distance(previous, hand)
                + np.radians(_rotation_distance(previous, hand)) * 0.05,
            )
        chosen = copy.deepcopy(chosen)
        chosen["is_right"] = 1.0
        selected.append(chosen)
        previous = chosen

    return selected, {
        "input_detections": input_detections,
        "discarded_left_detections": discarded_left,
        "multi_right_frames": multi_right_frames,
        "initial_missing_right_frames": missing_right_frames,
    }


def _robust_limit(values: list[float], floor: float, mad_scale: float) -> float:
    if not values:
        return floor
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return max(floor, median + mad_scale * 1.4826 * mad)


def _interpolation_residuals(
    hands: list[dict[str, Any] | None], start: int, end: int
) -> list[tuple[float, float]]:
    left = hands[start - 1]
    right = hands[end + 1]
    if left is None or right is None:
        return []
    denominator = end - start + 2
    residuals = []
    for frame in range(start, end + 1):
        current = hands[frame]
        if current is None:
            return []
        alpha = (frame - start + 1) / denominator
        expected = interpolate_hand(left, right, alpha)
        residuals.append(
            (_position_distance(current, expected), _rotation_distance(current, expected))
        )
    return residuals


def _fill_span(hands: list[dict[str, Any] | None], start: int, end: int) -> None:
    left = hands[start - 1]
    right = hands[end + 1]
    if left is None or right is None:
        raise ValueError("interpolation span must have valid detections on both sides")
    denominator = end - start + 2
    for frame in range(start, end + 1):
        alpha = (frame - start + 1) / denominator
        hands[frame] = interpolate_hand(left, right, alpha)


def clean_wilor_hands(
    hands_per_frame: np.ndarray,
    *,
    max_repair_span: int = 3,
    position_floor_mm: float = 80.0,
    rotation_floor_deg: float = 75.0,
    mad_scale: float = 10.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    if max_repair_span < 1:
        raise ValueError("max_repair_span must be at least 1")
    if position_floor_mm <= 0 or rotation_floor_deg <= 0:
        raise ValueError("jump thresholds must be positive")

    hands, report = _choose_primary_right_hands(hands_per_frame)
    missing_repairs = []
    frame = 0
    while frame < len(hands):
        if hands[frame] is not None:
            frame += 1
            continue
        start = frame
        while frame < len(hands) and hands[frame] is None:
            frame += 1
        end = frame - 1
        if (
            end - start + 1 <= max_repair_span
            and start > 0
            and frame < len(hands)
            and hands[start - 1] is not None
            and hands[frame] is not None
        ):
            _fill_span(hands, start, end)
            missing_repairs.extend(range(start, end + 1))

    position_residuals = []
    rotation_residuals = []
    for frame in range(1, len(hands) - 1):
        residuals = _interpolation_residuals(hands, frame, frame)
        if residuals:
            position_residuals.append(residuals[0][0])
            rotation_residuals.append(residuals[0][1])
    position_limit = _robust_limit(position_residuals, position_floor_mm / 1000.0, mad_scale)
    rotation_limit = _robust_limit(rotation_residuals, rotation_floor_deg, mad_scale)

    jump_repairs: list[dict[str, Any]] = []
    already_repaired = set(missing_repairs)
    for span_length in range(1, max_repair_span + 1):
        for start in range(1, len(hands) - span_length):
            end = start + span_length - 1
            if any(frame in already_repaired for frame in range(start, end + 1)):
                continue
            residuals = _interpolation_residuals(hands, start, end)
            if not residuals:
                continue
            left = hands[start - 1]
            right = hands[end + 1]
            assert left is not None and right is not None
            bridge_steps = span_length + 1
            bridge_is_smooth = (
                _position_distance(left, right) / bridge_steps <= position_limit
                and _rotation_distance(left, right) / bridge_steps <= rotation_limit
            )
            frame_is_bad = [
                position > position_limit or rotation > rotation_limit
                for position, rotation in residuals
            ]
            if not bridge_is_smooth or not all(frame_is_bad):
                continue
            frames = list(range(start, end + 1))
            jump_repairs.append(
                {
                    "frames": frames,
                    "position_residual_mm": [position * 1000.0 for position, _ in residuals],
                    "rotation_residual_deg": [rotation for _, rotation in residuals],
                }
            )
            _fill_span(hands, start, end)
            already_repaired.update(frames)

    unresolved = [frame for frame, hand in enumerate(hands) if hand is None]
    cleaned = np.empty(len(hands), dtype=object)
    for frame, hand in enumerate(hands):
        cleaned[frame] = [hand] if hand is not None else []
    report.update(
        {
            "method": "right-hand selection plus bounded robust interpolation",
            "frames": len(hands),
            "max_repair_span": max_repair_span,
            "position_floor_mm": position_floor_mm,
            "rotation_floor_deg": rotation_floor_deg,
            "mad_scale": mad_scale,
            "effective_position_threshold_mm": position_limit * 1000.0,
            "effective_rotation_threshold_deg": rotation_limit,
            "observed_position_residual_mm_median": (
                float(np.median(position_residuals) * 1000.0) if position_residuals else None
            ),
            "observed_position_residual_mm_max": (
                float(np.max(position_residuals) * 1000.0) if position_residuals else None
            ),
            "observed_rotation_residual_deg_median": (
                float(np.median(rotation_residuals)) if rotation_residuals else None
            ),
            "observed_rotation_residual_deg_max": (
                float(np.max(rotation_residuals)) if rotation_residuals else None
            ),
            "interpolated_missing_frames": missing_repairs,
            "interpolated_jump_spans": jump_repairs,
            "unresolved_missing_frames": unresolved,
        }
    )
    return cleaned, report


def save_wilor_npz(path: Path | str, *, hands: np.ndarray, camera: np.ndarray,
                   observed: np.ndarray | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        extra = {} if observed is None else {"hand_observed": observed}
        np.savez(stream, hands=hands, K=camera, **extra)
    os.replace(temporary, path)


def write_report(path: Path | str, report: dict[str, Any]) -> None:
    path = Path(path)
    payload = dict(report)
    payload["updated_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one right hand per frame and repair short WiLoR trajectory outliers.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-repair-span", type=int, default=3)
    parser.add_argument("--position-floor-mm", type=float, default=80.0)
    parser.add_argument("--rotation-floor-deg", type=float, default=75.0)
    args = parser.parse_args()

    data = np.load(args.input, allow_pickle=True)
    cleaned, report = clean_wilor_hands(
        data["hands"],
        max_repair_span=args.max_repair_span,
        position_floor_mm=args.position_floor_mm,
        rotation_floor_deg=args.rotation_floor_deg,
    )
    save_wilor_npz(args.output, hands=cleaned, camera=data["K"])
    write_report(args.report, report)
    repaired = report["interpolated_missing_frames"]
    jump_spans = report["interpolated_jump_spans"]
    print(
        f"cleaned {len(cleaned)} frames; filled missing={repaired}; "
        f"jump spans={[span['frames'] for span in jump_spans]}"
    )


if __name__ == "__main__":
    main()
