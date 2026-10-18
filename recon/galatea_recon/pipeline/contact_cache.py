import argparse
import hashlib
import json
from pathlib import Path

from galatea_recon.pipeline.object_quality import atomic_json, file_hash
from galatea_recon.pipeline.perception_cache import read_report


PARAMETERS = (
    "score_threshold", "nms_iou", "interaction_threshold", "median_width",
    "min_length", "max_gap", "start_trim", "end_trim", "object_match_iou",
    "hand_match_iou", "match_margin",
)


def frame_paths(args):
    frames = sorted(p for p in (Path(args.cam_dir) / "rgb").iterdir()
                    if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("max_frames must be positive")
        frames = frames[:args.max_frames]
    if not frames:
        raise ValueError("no RGB frames found")
    return frames


def contact_inputs(args):
    root = Path(args.hoidetr_root)
    sources = sorted({p for directory in ("projects", "demo")
                      for p in (root / directory).rglob("*.py")})
    return {
        "version": 1,
        "frames": [[p.name, file_hash(p)] for p in frame_paths(args)],
        "max_frames": args.max_frames,
        "target_masks": file_hash(args.target_masks),
        "target_hands": file_hash(args.target_hands),
        "parameters": {name: getattr(args, name) for name in PARAMETERS},
        "model_config": file_hash(root / args.model_config),
        "checkpoint": file_hash(root / args.checkpoint),
        "model_sources": {str(p.relative_to(root)): file_hash(p) for p in sources},
        "modules": {name: file_hash(Path(__file__).with_name(name + ".py"))
                    for name in ("contact_cache", "hoidetr_contact")},
    }


def report_hash(report):
    content = {k: v for k, v in report.items() if k != "report_sha256"}
    return hashlib.sha256(json.dumps(content, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def cached_contact(path, inputs):
    report = read_report(path)
    try:
        frames = report["frames"]
        n = len(inputs["frames"])
        if (report.get("state") != "complete" or report.get("source") != "hoidetr-auto"
                or report.get("inputs") != inputs or report.get("error_frames") != []
                or report.get("num_frames") != n or len(frames) != n
                or report.get("report_sha256") != report_hash(report)):
            return None
        if any(f["frame_idx"] != i or f["file_name"] != inputs["frames"][i][0]
               for i, f in enumerate(frames)):
            return None
        validate_windows(report["window"], n)
        return report
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def publish_contact(path, report):
    report["report_sha256"] = report_hash(report)
    atomic_json(path, report)


def validate_windows(window, num_frames):
    import re

    if not isinstance(window, str) or not re.fullmatch(r"\d+-\d+(,\d+-\d+)*", window):
        raise ValueError("invalid contact window format: " + str(window))
    for interval in window.split(","):
        start, end = map(int, interval.split("-"))
        if not 0 <= start <= end < num_frames:
            raise ValueError("contact window {} is outside valid frames 0-{}".format(
                interval, num_frames - 1))


def manual_window(sidecar):
    import yaml

    path = Path(sidecar)
    if not path.is_file():
        return ""
    value = yaml.safe_load(path.read_text()) or {}
    if not isinstance(value, dict):
        raise ValueError("contact sidecar must be a mapping: " + str(path))
    window = str(value.get("window") or "").strip()
    if not window or value.get("source") != "hoidetr-auto":
        return window
    original = value.get("auto_window")
    return window if original is not None and window != str(original) else ""


def main():
    parser = argparse.ArgumentParser(
        description="Contact cache identity and explicit/manual sidecar resolution.")
    parser.add_argument("sidecar")
    args = parser.parse_args()
    print(manual_window(args.sidecar))


if __name__ == "__main__":
    main()
