import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import scene

REQUIRED_PATHS = (
    ("rhand", "params", "global_orient"),
    ("rhand", "params", "fullpose"),
    ("rhand", "params", "transl"),
    ("object", "params", "global_orient"),
    ("object", "params", "transl"),
)


def _materialize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == () and value.dtype == object:
            return _materialize(value.item())
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): _materialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_materialize(item) for item in value]
    if isinstance(value, tuple):
        return tuple((_materialize(item) for item in value))
    return copy.deepcopy(value)


def load_raw(path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {str(key): _materialize(archive[key]) for key in archive.files}


def _tree_get(payload: Mapping[str, Any], path) -> Any:
    value: Any = payload
    for key in path:
        if isinstance(value, np.ndarray) and value.shape == () and (value.dtype == object):
            value = value.item()
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(".".join(path))
        value = value[key]
    return value


def frame_count(payload: Mapping[str, Any]) -> int:
    lengths = {".".join(path): int(np.asarray(_tree_get(payload, path)).shape[0]) for path in REQUIRED_PATHS}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"raw modalities have different lengths: {lengths}")
    frames = next(iter(set(lengths.values())))
    if frames < 3:
        raise ValueError(f"sequence is too short: {frames} frames")
    return frames


def _fps(payload: Mapping[str, Any]) -> float:
    fps = float(np.asarray(payload.get("framerate", 120.0)).reshape(()))
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"invalid framerate: {fps}")
    return fps


def so3_geodesic_steps(rotvec: np.ndarray) -> np.ndarray:
    values = np.asarray(rotvec, dtype=np.float64)
    frames = values.shape[0]
    flat = values.reshape(frames, -1, 3)
    matrices = Rotation.from_rotvec(flat.reshape(-1, 3)).as_matrix()
    matrices = matrices.reshape(frames, flat.shape[1], 3, 3)
    relative = np.einsum("tkji,tkjl->tkil", matrices[:-1], matrices[1:])
    cosine = np.clip((np.trace(relative, axis1=-2, axis2=-1) - 1.0) * 0.5, -1.0, 1.0)
    result = np.arccos(cosine)
    return result[:, 0] if values.ndim == 2 else result.reshape((frames - 1,) + values.shape[1:-1])


def so3_geodesic(rotvec_a: np.ndarray, rotvec_b: np.ndarray) -> np.ndarray:
    a = Rotation.from_rotvec(np.asarray(rotvec_a, dtype=np.float64).reshape(-1, 3))
    b = Rotation.from_rotvec(np.asarray(rotvec_b, dtype=np.float64).reshape(-1, 3))
    return (a.inv() * b).magnitude().reshape(np.asarray(rotvec_a).shape[:-1])


def _robust_threshold(values, multiplier: float) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        raise ValueError("cannot derive a threshold from an empty metric")
    median = float(np.median(array))
    mad = float(np.median(np.abs(array - median)))
    return float(median + float(multiplier) * float(1.4826 * mad))


def _rotation_path_excess(rotvec: np.ndarray, left: int, right: int) -> float:
    path = float(np.sum(so3_geodesic_steps(np.asarray(rotvec)[left : right + 1])))
    endpoint = float(so3_geodesic(np.asarray(rotvec)[left], np.asarray(rotvec)[right]))
    return max(0.0, path - endpoint)


def _max_interior_path_excess(rotvec: np.ndarray, max_steps: int, tail_fraction: float) -> float:
    values = np.asarray(rotvec)
    end = max(3, min(len(values), int(math.floor(tail_fraction * len(values)))))
    best = 0.0
    for left in range(end - 2):
        for right in range(left + 2, min(end, left + max_steps + 1)):
            best = max(best, _rotation_path_excess(values, left, right))
    return best


def _candidate_groups(indices: np.ndarray, max_gap: int) -> list:
    groups = []
    for raw_index in np.asarray(indices, dtype=np.int64):
        index = int(raw_index)
        if groups and index - groups[-1][-1] <= max_gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def _has_clean_run(mask: np.ndarray, start: int, run_length: int) -> bool:
    run = 0
    for bad in np.asarray(mask, dtype=bool)[start:]:
        run = 0 if bad else run + 1
        if run >= run_length:
            return True
    return False


def _contact_in_span(payload: Mapping[str, Any], left: int, right: int) -> bool:
    try:
        contact = payload["contact"]
        if isinstance(contact, np.ndarray) and contact.shape == () and (contact.dtype == object):
            contact = contact.item()
        values = np.asarray(contact["object"])
        if values.shape[0] != frame_count(payload):
            return False
        return bool(np.any(values[left : right + 1]))
    except (KeyError, TypeError, ValueError):
        return False


def _clip_metrics(payload: Mapping[str, Any]) -> dict:
    frames = frame_count(payload)
    rhand = _tree_get(payload, ("rhand", "params"))
    obj = _tree_get(payload, ("object", "params"))
    fullpose = np.asarray(rhand["fullpose"], dtype=np.float64)
    return {
        "hand_translation": np.linalg.norm(np.diff(np.asarray(rhand["transl"], dtype=np.float64), axis=0), axis=-1),
        "object_translation": np.linalg.norm(np.diff(np.asarray(obj["transl"], dtype=np.float64), axis=0), axis=-1),
        "wrist": so3_geodesic_steps(np.asarray(rhand["global_orient"], dtype=np.float64)),
        "articulation": so3_geodesic_steps(fullpose.reshape(frames, -1, 3)),
    }


def _repair(left, right, root, joints, frame, channels):
    return {
        "start": left + 1,
        "stop": right,
        "left_anchor": left,
        "right_anchor": right,
        "root": root,
        "fullpose_joint_indices": joints,
        "translation_frame": frame,
        "channels": channels,
    }


def _analyze_one(payload, label, thresholds, tail_fraction, max_window_seconds, min_anchor_seconds, persistent_count):
    frames = frame_count(payload)
    fps = _fps(payload)
    metrics = _clip_metrics(payload)
    max_gap = max(2, int(math.ceil(max_window_seconds * fps)))
    min_anchor_steps = max(2, int(math.ceil(min_anchor_seconds * fps)))
    tail_edge = int(math.floor(tail_fraction * (frames - 1)))
    object_bad = metrics["object_translation"] > thresholds["object_translation"]
    object_cut = None
    rejected = False
    for edge in np.flatnonzero(object_bad):
        edge = int(edge)
        stop = min(len(object_bad), edge + max_gap)
        if int(np.count_nonzero(object_bad[edge:stop])) < persistent_count:
            continue
        if _has_clean_run(object_bad, stop, max_gap):
            rejected = True
        else:
            object_cut = edge + 1
            break
    wrist_bad = metrics["wrist"] > thresholds["wrist"]
    articulation_bad = metrics["articulation"] > thresholds["articulation"]
    hand_translation_bad = metrics["hand_translation"] > thresholds["hand_translation"]
    any_hand_bad = wrist_bad | np.any(articulation_bad, axis=1) | hand_translation_bad
    analysis_stop = object_cut if object_cut is not None else frames
    hand_cut = None
    for group in _candidate_groups(np.flatnonzero(any_hand_bad[: max(0, analysis_stop - 1)]), max_gap):
        first, last = (group[0], group[-1])
        if first >= tail_edge or analysis_stop - 1 - last < min_anchor_steps:
            hand_cut = first + 1 if hand_cut is None else min(hand_cut, first + 1)
            break
    keep_stop = min((value for value in (frames, object_cut or frames, hand_cut or frames)))
    cancellation_gate = thresholds["cancellation"]
    repairs = []
    for group in _candidate_groups(np.flatnonzero(wrist_bad[: max(0, keep_stop - 1)]), max_gap):
        if len(group) < 2:
            continue
        left, right = (group[0], group[-1] + 1)
        if left < min_anchor_steps or right >= keep_stop or keep_stop - 1 - group[-1] < min_anchor_steps:
            continue
        excess = _rotation_path_excess(
            np.asarray(_tree_get(payload, ("rhand", "params", "global_orient"))), left, right
        )
        if excess <= cancellation_gate:
            continue
        object_clean = not bool(np.any(object_bad[left:right]))
        frame = "object_local" if object_clean and _contact_in_span(payload, left, right) else "world"
        repairs.append(_repair(left, right, True, [], frame, ["rhand.params.transl", "rhand.params.global_orient"]))
    fullpose = np.asarray(_tree_get(payload, ("rhand", "params", "fullpose"))).reshape(frames, -1, 3)
    for joint in range(metrics["articulation"].shape[1]):
        for group in _candidate_groups(np.flatnonzero(articulation_bad[: max(0, keep_stop - 1), joint]), max_gap):
            if len(group) < 2:
                continue
            left, right = (group[0], group[-1] + 1)
            if left < min_anchor_steps or right >= keep_stop or keep_stop - 1 - group[-1] < min_anchor_steps:
                continue
            if _rotation_path_excess(fullpose[:, joint], left, right) <= cancellation_gate:
                continue
            repairs.append(_repair(left, right, False, [joint], None, ["rhand.params.fullpose"]))
    repairs = [repair for repair in _merge_repairs(repairs) if repair["right_anchor"] < keep_stop]
    return {
        "label": label,
        "original_frames": frames,
        "keep_stop": keep_stop,
        "rejected": rejected,
        "repairs": repairs,
    }


def _merge_repairs(repairs: list) -> list:
    merged = {}
    for repair in repairs:
        key = (int(repair["left_anchor"]), int(repair["right_anchor"]))
        if key not in merged:
            merged[key] = copy.deepcopy(repair)
            continue
        dst = merged[key]
        dst["root"] = bool(dst["root"] or repair["root"])
        dst["fullpose_joint_indices"] = sorted(
            set(dst["fullpose_joint_indices"]) | set(repair["fullpose_joint_indices"])
        )
        dst["channels"] = sorted(set(dst["channels"]) | set(repair["channels"]))
        if repair["translation_frame"] is not None:
            dst["translation_frame"] = repair["translation_frame"]
    return [merged[key] for key in sorted(merged)]


def analyze(rows, tail_fraction=0.9, max_window_seconds=0.5, min_anchor_seconds=0.25, persistent_count=4) -> dict:
    loaded = []
    pooled = {"object_translation": [], "hand_translation": [], "wrist": [], "articulation": []}
    path_excess = []
    for label, path in rows:
        payload = load_raw(path)
        metrics = _clip_metrics(payload)
        fps = _fps(payload)
        interior_edges = max(1, int(math.floor(tail_fraction * len(metrics["hand_translation"]))))
        pooled["object_translation"].extend(metrics["object_translation"].tolist())
        pooled["hand_translation"].extend(metrics["hand_translation"][:interior_edges].tolist())
        pooled["wrist"].extend(metrics["wrist"].tolist())
        pooled["articulation"].extend(np.max(metrics["articulation"], axis=1).tolist())
        path_excess.append(
            _max_interior_path_excess(
                np.asarray(_tree_get(payload, ("rhand", "params", "global_orient"))),
                max(2, int(math.ceil(max_window_seconds * fps))),
                tail_fraction,
            )
        )
        loaded.append((label, payload))
    thresholds = {
        "object_translation": _robust_threshold(pooled["object_translation"], 12.0),
        "hand_translation": _robust_threshold(pooled["hand_translation"], 7.0),
        "wrist": _robust_threshold(pooled["wrist"], 12.0),
        "articulation": _robust_threshold(pooled["articulation"], 12.0),
        "cancellation": _robust_threshold(path_excess, 8.0),
    }
    return {
        "thresholds": thresholds,
        "clips": {
            label: _analyze_one(
                payload, label, thresholds, tail_fraction, max_window_seconds, min_anchor_seconds, persistent_count
            )
            for label, payload in loaded
        },
    }


def _slerp(rotvec: np.ndarray, left: int, right: int, indices: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)[[left, right]])
    return Slerp([float(left), float(right)], rotations)(indices.astype(np.float64)).as_rotvec()


def _apply_repair(payload: dict, repair: Mapping[str, Any]) -> None:
    rhand = payload["rhand"]["params"]
    obj = payload["object"]["params"]
    left, right = (int(repair["left_anchor"]), int(repair["right_anchor"]))
    indices = np.arange(int(repair["start"]), int(repair["stop"]), dtype=np.int64)
    if not len(indices):
        return
    if bool(repair["root"]):
        hand_position = np.asarray(rhand["transl"], dtype=np.float64)
        hand_rotation = np.asarray(rhand["global_orient"], dtype=np.float64)
        alpha = ((indices - left) / float(right - left))[:, None]
        if repair.get("translation_frame") == "object_local":
            object_position = np.asarray(obj["transl"], dtype=np.float64)
            object_rotation = np.swapaxes(
                Rotation.from_rotvec(np.asarray(obj["global_orient"], dtype=np.float64)).as_matrix(), -1, -2
            )
            local_position_anchors = np.stack(
                [object_rotation[t].T @ (hand_position[t] - object_position[t]) for t in (left, right)]
            )
            local_position = (1.0 - alpha) * local_position_anchors[0] + alpha * local_position_anchors[1]
            hand_position[indices] = (
                np.einsum("tij,tj->ti", object_rotation[indices], local_position) + object_position[indices]
            )
            hand_matrices = Rotation.from_rotvec(hand_rotation).as_matrix()
            local_anchor_matrices = np.stack([object_rotation[t].T @ hand_matrices[t] for t in (left, right)])
            local_interp = Slerp([float(left), float(right)], Rotation.from_matrix(local_anchor_matrices))(
                indices.astype(np.float64)
            ).as_matrix()
            hand_rotation[indices] = Rotation.from_matrix(
                np.einsum("tij,tjk->tik", object_rotation[indices], local_interp)
            ).as_rotvec()
        else:
            hand_position[indices] = (1.0 - alpha) * hand_position[left] + alpha * hand_position[right]
            hand_rotation[indices] = _slerp(hand_rotation, left, right, indices)
        rhand["transl"] = hand_position.astype(np.asarray(rhand["transl"]).dtype, copy=False)
        rhand["global_orient"] = hand_rotation.astype(np.asarray(rhand["global_orient"]).dtype, copy=False)
    fullpose_original = np.asarray(rhand["fullpose"])
    fullpose = fullpose_original.astype(np.float64).reshape(len(fullpose_original), -1, 3)
    for joint in repair.get("fullpose_joint_indices", []):
        joint = int(joint)
        fullpose[indices, joint] = _slerp(fullpose[:, joint], left, right, indices)
    rhand["fullpose"] = fullpose.reshape(fullpose_original.shape).astype(fullpose_original.dtype, copy=False)


def _trim_time_tree(value: Any, original_frames: int, keep_stop: int) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == () and value.dtype == object:
            return _trim_time_tree(value.item(), original_frames, keep_stop)
        if value.ndim and value.shape[0] == original_frames:
            return value[:keep_stop].copy()
        return value.copy()
    if isinstance(value, Mapping):
        return {str(key): _trim_time_tree(item, original_frames, keep_stop) for key, item in value.items()}
    if isinstance(value, list):
        return [_trim_time_tree(item, original_frames, keep_stop) for item in value]
    if isinstance(value, tuple):
        return tuple((_trim_time_tree(item, original_frames, keep_stop) for item in value))
    return copy.deepcopy(value)


def apply_plan(payload: Mapping[str, Any], clip_plan: Mapping[str, Any]) -> dict:
    result = _materialize(payload)
    original_frames = frame_count(result)
    if original_frames != int(clip_plan["original_frames"]):
        raise ValueError(f"payload frames {original_frames} != plan frames {clip_plan['original_frames']}")
    for repair in clip_plan["repairs"]:
        _apply_repair(result, repair)
    keep_stop = int(clip_plan["keep_stop"])
    if not 1 < keep_stop <= original_frames:
        raise ValueError(f"invalid keep_stop {keep_stop}/{original_frames}")
    result = _trim_time_tree(result, original_frames, keep_stop)
    if "n_frames" in result:
        old = np.asarray(result["n_frames"])
        result["n_frames"] = np.asarray(keep_stop, dtype=old.dtype) if old.shape == () else keep_stop
    if frame_count(result) != keep_stop:
        raise AssertionError("quality plan did not produce its declared final length")
    return result


def _iter_array_paths(value: Any, prefix=()):
    if isinstance(value, np.ndarray):
        yield (prefix, value)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _iter_array_paths(item, prefix + (str(key),))


def _value_at_path(value: Mapping[str, Any], path) -> Any:
    current: Any = value
    for key in path:
        current = current[key]
    return current


def _verify_repairs(before: Mapping[str, Any], after: Mapping[str, Any], clip: Mapping[str, Any]) -> None:
    final = int(clip["keep_stop"])
    before_params = before["rhand"]["params"]
    after_params = after["rhand"]["params"]
    root_target = np.zeros(final, dtype=bool)
    fullpose_before = np.asarray(before_params["fullpose"]).reshape(int(clip["original_frames"]), -1, 3)
    fullpose_after = np.asarray(after_params["fullpose"]).reshape(final, -1, 3)
    joint_target = np.zeros((final, fullpose_before.shape[1]), dtype=bool)
    for repair in clip["repairs"]:
        left, right = (int(repair["left_anchor"]), int(repair["right_anchor"]))
        indices = np.arange(int(repair["start"]), int(repair["stop"]), dtype=np.int64)
        if bool(repair["root"]):
            root_target[indices] = True
            for key in ("transl", "global_orient"):
                if not np.array_equal(
                    np.asarray(before_params[key])[[left, right]], np.asarray(after_params[key])[[left, right]]
                ):
                    raise ValueError(f"repair changed a {key} anchor at {left}/{right}")
            before_rot = np.asarray(before_params["global_orient"])
            after_rot = np.asarray(after_params["global_orient"])
            before_excess = _rotation_path_excess(before_rot, left, right)
            after_excess = _rotation_path_excess(after_rot, left, right)
            before_max = float(np.max(so3_geodesic_steps(before_rot[left : right + 1])))
            after_max = float(np.max(so3_geodesic_steps(after_rot[left : right + 1])))
            if after_excess > max(math.radians(1.0), 0.5 * before_excess) and after_max > 0.75 * before_max:
                raise ValueError(f"root repair {left}:{right} did not reduce its rotation anomaly")
        for joint in repair["fullpose_joint_indices"]:
            joint = int(joint)
            joint_target[indices, joint] = True
            if not np.array_equal(fullpose_before[[left, right], joint], fullpose_after[[left, right], joint]):
                raise ValueError(f"repair changed joint {joint} anchor at {left}/{right}")
            before_excess = _rotation_path_excess(fullpose_before[:, joint], left, right)
            after_excess = _rotation_path_excess(fullpose_after[:, joint], left, right)
            if after_excess > max(math.radians(0.25), 0.5 * before_excess):
                raise ValueError(f"joint {joint} repair {left}:{right} did not reduce its path excess")
    for key in ("transl", "global_orient"):
        original = np.asarray(before_params[key])[:final]
        repaired = np.asarray(after_params[key])
        if not np.array_equal(original[~root_target], repaired[~root_target]):
            raise ValueError(f"repair modified non-target rhand.params.{key} samples")
    untouched = ~joint_target
    if not np.array_equal(fullpose_before[:final][untouched], fullpose_after[untouched]):
        raise ValueError("repair modified non-target rhand.params.fullpose samples")
    for key, original_value in before_params.items():
        if key in {"transl", "global_orient", "fullpose"}:
            continue
        original = np.asarray(original_value)
        if original.ndim and original.shape[0] == int(clip["original_frames"]):
            if not np.array_equal(original[:final], np.asarray(after_params[key])):
                raise ValueError(f"repair modified non-target rhand.params.{key}")


def verify(plan: dict, rows) -> set:
    failures = set()
    for label, path in rows:
        clip = plan["clips"][label]
        try:
            payload = load_raw(path)
            result = apply_plan(payload, clip)
            original = frame_count(payload)
            final = frame_count(result)
            for required in REQUIRED_PATHS:
                if not np.isfinite(np.asarray(_tree_get(result, required))).all():
                    raise ValueError(f"non-finite output in {'.'.join(required)}")
            for key in ("transl", "global_orient"):
                if not np.array_equal(
                    np.asarray(payload["object"]["params"][key])[:final], np.asarray(result["object"]["params"][key])
                ):
                    raise ValueError(f"object.{key} was modified")
            for array_path, original_array in _iter_array_paths(payload):
                if not original_array.ndim or original_array.shape[0] != original:
                    continue
                repaired_array = np.asarray(_value_at_path(result, array_path))
                if not repaired_array.ndim or repaired_array.shape[0] != final:
                    raise ValueError(f"temporal array {'.'.join(array_path)} was not trimmed")
            _verify_repairs(payload, result, clip)
        except Exception as error:
            print(f"[quality] {label}: repair check failed: {error}", flush=True)
            failures.add(label)
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--maximum-passes", type=int, default=8)
    args = parser.parse_args()
    raw = args.raw.resolve(strict=True)
    sequences = json.loads((raw / "manifest.json").read_text())
    entries = []
    for index, (name, item) in enumerate(sequences.items()):
        entries.append(
            {
                "data_index": f"g{index}",
                "dir": f"g{index}_{name}",
                "source": name,
                "motion": str((raw / item["motion"]).resolve(strict=True)),
                "mesh": str((raw / item["mesh"]).resolve(strict=True)),
            }
        )
    for _ in range(args.maximum_passes):
        rows = [(entry["data_index"], entry["motion"]) for entry in entries]
        plan = analyze(rows)
        failures = verify(plan, rows)
        excluded = [e for e in entries if plan["clips"][e["data_index"]]["rejected"] or e["data_index"] in failures]
        if not excluded:
            break
        for entry in excluded:
            print(f"[quality] exclude {entry['source']}", flush=True)
        entries = [
            dict(entry, data_index=f"g{index}") for index, entry in enumerate(e for e in entries if e not in excluded)
        ]
        if not entries:
            raise RuntimeError("quality filtering rejected every sequence")
    else:
        raise RuntimeError("quality filtering did not converge")
    scene.atomic_json(args.output / "entries.json", {"entries": entries})
    scene.atomic_json(args.output / "quality_plan.json", plan)
    print(f"[quality] accepted {len(entries)}/{len(sequences)} sequences", flush=True)


if __name__ == "__main__":
    main()
