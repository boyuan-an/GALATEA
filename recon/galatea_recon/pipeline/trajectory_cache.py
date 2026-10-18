import os
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from galatea_recon.pipeline.object_quality import atomic_json, file_hash
from galatea_recon.pipeline.observation_gaps import observation_path
from galatea_recon.pipeline.perception_cache import read_report


INVALID_CACHE = (OSError, ValueError, EOFError, BadZipFile, KeyError, TypeError, AttributeError)


def camera_inputs(cam_dir, max_frames):
    cam = Path(cam_dir)
    frames = sorted((cam / "rgb").glob("*.png"))
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
    if not frames:
        raise ValueError("no RGB frames")
    return {"rgb": [[p.name, file_hash(p)] for p in frames],
            "intrinsics": file_hash(cam / "cam_intrinsics.txt")}


def optional_hash(path):
    return file_hash(path) if path and Path(path).is_file() else None


def output_hashes(paths):
    return {name: file_hash(path) for name, path in paths.items()}


def outputs_match(report, paths):
    try:
        return report.get("outputs") == output_hashes(paths)
    except OSError:
        return False


def hand_inputs(args):
    parameters = {name: getattr(args, name) for name in (
        "max_frames", "hand_side", "hand_conf", "clean_jumps", "fill_missing",
        "max_repair_span", "jump_position_mm", "jump_rotation_deg")}
    return {"version": 1, "camera": camera_inputs(args.cam_dir, args.max_frames),
            "parameters": parameters, "frame0_predictions": optional_hash(args.frame0_predictions)}


def hand_outputs(args):
    root = Path(args.out)
    names = ["wilor_hands.npz", "wilor_hands_raw.npz"]
    if args.clean_jumps or args.fill_missing:
        names.append("wilor_cleanup.json")
    return {name: root / name for name in names}


def validate_hand_outputs(args, inputs):
    n = len(inputs["camera"]["rgb"])
    for name in ("wilor_hands_raw.npz", "wilor_hands.npz"):
        with np.load(Path(args.out) / name, allow_pickle=True) as data:
            hands, camera = data["hands"], data["K"]
            if hands.shape != (n,) or camera.shape != (3, 3) or not np.isfinite(camera).all():
                raise ValueError("WiLoR output does not match the requested timeline/camera")
            if name.endswith("_raw.npz"):
                continue
            if args.fill_missing:
                observed = data["hand_observed"]
                if observed.shape != (n,) or observed.dtype != np.bool_ or not observed.any():
                    raise ValueError("invalid hand observation flags")
            for frame in hands:
                if (args.fill_missing and len(frame) != 1) or (args.clean_jumps and len(frame) > 1):
                    raise ValueError("invalid number of hands in completed right-hand output")
                for hand in frame:
                    if args.hand_side != "all" and (float(hand["is_right"]) > .5) != (args.hand_side == "right"):
                        raise ValueError("cached hand side does not match request")
                    for key, shape in (("verts", (778, 3)), ("kp3d", (21, 3)), ("kp2d", (21, 2))):
                        value = np.asarray(hand[key])
                        if value.shape != shape or not np.isfinite(value).all():
                            raise ValueError("invalid hand geometry: " + key)
                    for key in ("global_orient", "hand_pose", "betas", "place_R", "place_t", "root"):
                        if not np.isfinite(hand[key]).all():
                            raise ValueError("non-finite MANO parameter: " + key)


def hand_cache_complete(args):
    report = read_report(Path(args.out) / "wilor_complete.json")
    try:
        inputs = hand_inputs(args)
        if report.get("state") != "complete" or report.get("inputs") != inputs:
            return False
        if not outputs_match(report, hand_outputs(args)):
            return False
        validate_hand_outputs(args, inputs)
        return True
    except INVALID_CACHE:
        return False


def publish_hand_cache(args, inputs):
    validate_hand_outputs(args, inputs)
    atomic_json(Path(args.out) / "wilor_complete.json", {
        "state": "complete", "inputs": inputs, "outputs": output_hashes(hand_outputs(args)),
    })


def normalize_optimization_args(args):
    if args.full_objective:
        for name in ("finer_contact", "hand_depth", "verts_tracking", "two_stage", "depth_vis", "manip_weights"):
            setattr(args, name, True)
    if not hasattr(args, "contact_iters"):
        args.contact_iters = max(1, int(os.environ.get("CONTACT_ITERS", "200")))
    args.contact_iters = max(1, args.contact_iters)
    return args


def optimization_inputs(args):
    from galatea_recon.pipeline.contact_boundary import boundary_inputs
    from galatea_recon.pipeline.fp_constraints import constraint_config

    normalize_optimization_args(args)
    camera = camera_inputs(args.cam_dir, args.max_frames)
    files = {name: optional_hash(getattr(args, name)) for name in (
        "mesh", "fp_poses", "wilor_hands", "sam2_masks", "seg_labels", "manual_json")}
    files["fp_observations"] = optional_hash(observation_path(args.fp_poses))
    files["prep_metadata"] = optional_hash(Path(args.cam_dir) / "prep_metadata.json")
    n = len(camera["rgb"])
    files["depth"] = [optional_hash(Path(args.depth_dir) / f"{f:06d}.npy") for f in range(n)]
    excluded = {"cam_dir", "depth_dir", "out", "check_cache", *(
        "mesh", "fp_poses", "wilor_hands", "sam2_masks", "seg_labels", "manual_json")}
    return {"version": 1, "camera": camera, "files": files,
            "fp_constraints": constraint_config(args.w_fp_depth, args.fp_depth_free_m,
                                                args.fp_depth_free_fraction, args.fp_depth_evidence_gain),
            "contact_boundary": boundary_inputs(args.cam_dir, args.seg_labels, args.manual_json,
                                                 args.max_frames, not args.no_contact_boundary_guard),
            "parameters": {key: value for key, value in vars(args).items() if key not in excluded}}


def optimization_outputs(args):
    out = Path(args.out)
    paths = {suffix: out.with_name(out.stem + suffix) for suffix in (
        ".npy", "_hand.npy", "_handverts.npy")}
    if args.fill_missing:
        paths["observations"] = observation_path(out)
    if args.hand_silhouette_z and args.sam2_masks:
        paths["faces"] = out.parent / "mano_right_faces.npy"
    if args.early_stop:
        paths["early_stop"] = out.with_name(out.stem + "_early_stop.json")
    return paths


def validate_optimization_outputs(args, inputs):
    n = len(inputs["camera"]["rgb"])
    paths = optimization_outputs(args)
    for suffix, shape in ((".npy", (n, 4, 4)), ("_hand.npy", (n, 21, 3)),
                          ("_handverts.npy", (n, 778, 3))):
        value = np.load(paths[suffix], allow_pickle=False)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError("invalid joint output: " + suffix)
    if args.fill_missing:
        with np.load(paths["observations"], allow_pickle=False) as data:
            for key in ("object_observed", "hand_observed", "object_filled", "hand_filled"):
                if data[key].shape != (n,) or data[key].dtype != np.bool_:
                    raise ValueError("invalid joint observation flags: " + key)


def optimization_cache_complete(args):
    report = read_report(Path(args.out).with_name(Path(args.out).stem + "_optimization.json"))
    try:
        inputs = optimization_inputs(args)
        if report.get("state") != "complete" or report.get("cache_inputs") != inputs:
            return False
        if not outputs_match(report, optimization_outputs(args)):
            return False
        validate_optimization_outputs(args, inputs)
        return True
    except INVALID_CACHE:
        return False
