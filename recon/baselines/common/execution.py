from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

from .inputs import write_json

REPOSITORY = Path(__file__).resolve().parents[2]


def python_command(config: dict, name: str) -> list[str]:
    value = config.get(name, [sys.executable])
    if not isinstance(value, list) or not value or not all(isinstance(x, str) for x in value):
        raise ValueError(f"{name} must be a nonempty command argument list")
    return value


class Execution:
    def __init__(self, root: Path, config: dict, dry_run: bool):
        self.root, self.config, self.dry_run = root, config, dry_run
        self.records = []

    def run(self, name: str, command: list, cwd: Path, env: dict | None = None):
        command = [str(x) for x in command]
        environment = os.environ.copy()
        environment.update(self.config.get("env", {}))
        environment.update(env or {})
        environment["PYTHONPATH"] = str(REPOSITORY) + os.pathsep + environment.get("PYTHONPATH", "")
        environment["PYTHONUNBUFFERED"] = "1"
        log = self.root / "logs" / (name + ".log")
        record = {
            "stage": name,
            "command": command,
            "cwd": str(cwd),
            "log": str(log),
            "state": "planned",
        }
        self.records.append(record)
        write_json(self.root / "stages.json", self.records)
        print(f"[{name}] {shlex.join(command)}", flush=True)
        if self.dry_run:
            return
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with log.open("a") as stream:
            stream.write(f"\n$ {shlex.join(command)}\n")
            stream.flush()
            try:
                completed = subprocess.run(
                    command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT
                )
            except OSError as error:
                record.update(
                    state="failed", error=str(error), elapsed_seconds=time.monotonic() - started
                )
                write_json(self.root / "stages.json", self.records)
                raise RuntimeError(f"{name} could not start: {error}; see {log}") from error
        record.update(
            returncode=completed.returncode,
            elapsed_seconds=time.monotonic() - started,
            state="complete" if completed.returncode == 0 else "failed",
        )
        write_json(self.root / "stages.json", self.records)
        if completed.returncode:
            raise RuntimeError(f"{name} failed ({completed.returncode}); see {log}")
