from __future__ import annotations

import argparse
import fcntl
import importlib
import json
from pathlib import Path

from baselines.common.execution import Execution
from baselines.common.inputs import load_clips, prepare, resolve_path, write_json
from baselines.setup import verify_runtime

METHODS = ("videomanip", "egoinfinity", "do_as_i_do")


def main():
    parser = argparse.ArgumentParser(
        description="Run a released GALATEA baseline on explicit video/mesh inputs.")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated clip IDs")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare inputs and record commands without model execution",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only inputs with matching content/config fingerprints",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config.get("method") != args.method:
        parser.error("configuration method does not match --method")
    for key in ("repo", "fp_repo"):
        if key in config:
            config[key] = str(resolve_path(config[key], config_path.parent))
    if "repo" not in config:
        parser.error("configuration must specify repo")
    if args.method != "do_as_i_do" and "fp_repo" not in config:
        parser.error("configuration must specify fp_repo")
    if not args.dry_run and not args.prepare_only:
        verify_runtime(args.method, config)
    clips = load_clips(args.manifest.resolve(), args.method)
    if args.only:
        selected = set(args.only.split(","))
        unknown = selected - {clip["clip_id"] for clip in clips}
        if unknown:
            parser.error(f"unknown clip IDs: {sorted(unknown)}")
        clips = [clip for clip in clips if clip["clip_id"] in selected]
    module = importlib.import_module(f"baselines.{args.method}.run")
    failures = []
    for clip in clips:
        root = args.output.resolve() / args.method / clip["clip_id"]
        try:
            prepared = prepare(clip, args.method, config, root, args.resume)
        except Exception as error:
            print(f"{clip['clip_id']}: {type(error).__name__}: {error}", flush=True)
            failures.append(clip["clip_id"])
            continue
        execution = Execution(root, config, args.dry_run)
        lock = (root / ".run.lock").open("a")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            failures.append(clip["clip_id"])
            print(f"{clip['clip_id']}: another process is already running this output", flush=True)
            continue
        try:
            if not args.prepare_only:
                module.run(prepared, config, execution)
            state = "prepared" if args.prepare_only else ("dry_run" if args.dry_run else "complete")
            write_json(
                root / "status.json",
                {
                    "method": args.method,
                    "state": state,
                    "input_digest": prepared["input_digest"],
                    "clip_id": clip["clip_id"],
                },
            )
        except Exception as error:
            failures.append(clip["clip_id"])
            write_json(
                root / "status.json",
                {
                    "method": args.method,
                    "state": "failed",
                    "clip_id": clip["clip_id"],
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            print(f"{clip['clip_id']}: {error}", flush=True)
        finally:
            lock.close()
    if failures:
        raise SystemExit(f"Failed clips: {', '.join(failures)}")


if __name__ == "__main__":
    main()
