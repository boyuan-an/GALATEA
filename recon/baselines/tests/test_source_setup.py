import json
import subprocess
from pathlib import Path

import pytest

from baselines.setup import prepare_source


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def test_source_pin_and_edit_detection(tmp_path):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    git(upstream, "init", "-q")
    git(upstream, "config", "user.email", "test@example.invalid")
    git(upstream, "config", "user.name", "Test")
    (upstream / "module.py").write_text("value = 1\n")
    git(upstream, "add", "module.py")
    git(upstream, "commit", "-qm", "fixture")
    revision = git(upstream, "rev-parse", "HEAD")
    patch = tmp_path / "change.patch"
    patch.write_text("--- a/module.py\n+++ b/module.py\n@@ -1 +1 @@\n-value = 1\n+value = 2\n")
    spec = {
        "key": "fixture",
        "path": "fixture",
        "url": str(upstream),
        "revision": revision,
        "patches": [str(patch)],
    }
    destination = tmp_path / "installed"
    prepare_source(spec, {"fixture": spec}, destination, "recipe")
    installed = destination / "fixture"
    assert git(installed, "rev-parse", "HEAD") == revision
    assert (installed / "module.py").read_text() == "value = 2\n"
    prepare_source(spec, {"fixture": spec}, destination, "recipe", check_only=True)
    assert json.loads((installed / ".galatea-baseline-source.json").read_text())["files"]
    (installed / "module.py").write_text("value = 3\n")
    with pytest.raises(RuntimeError, match="prepared source changed"):
        prepare_source(spec, {"fixture": spec}, destination, "recipe")
    assert (installed / "module.py").read_text() == "value = 3\n"


def test_check_only_never_downloads(tmp_path):
    spec = {
        "key": "missing",
        "path": "missing",
        "url": "https://invalid.example/repo",
        "revision": "0" * 40,
    }
    with pytest.raises(RuntimeError, match="not prepared"):
        prepare_source(spec, {"missing": spec}, tmp_path, "recipe", check_only=True)
    assert not (tmp_path / "missing").exists()
