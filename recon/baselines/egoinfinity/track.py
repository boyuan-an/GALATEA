from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Execute FP++ for every dynamic object prepared by the EgoInfinity adapter.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--fp-repo", type=Path, required=True)
    parser.add_argument("--python-command", required=True)
    args = parser.parse_args()
    command = json.loads(args.python_command)
    if not isinstance(command, list) or not command:
        parser.error("--python-command must encode a nonempty argument list")
    for metadata in sorted(args.root.glob("*/egoinfinity_fpplusplus.json")):
        subprocess.run(
            command
            + [
                "-m",
                "baselines.egoinfinity.fpplusplus",
                "run-testcase",
                "--testcase",
                str(metadata.parent),
                "--fp-repo",
                str(args.fp_repo),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
