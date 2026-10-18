from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("sources.json")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def recipe_digest(manifest: dict) -> str:
    digest = hashlib.sha256(MANIFEST.read_bytes())
    for spec in manifest["repositories"]:
        for patch in spec.get("patches", []):
            digest.update(patch.encode())
            digest.update((REPOSITORY / patch).read_bytes())
    return digest.hexdigest()


def call(command, cwd=None, check=True):
    environment = os.environ.copy()
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    return subprocess.run(
        [str(x) for x in command],
        cwd=cwd,
        env=environment,
        check=check,
        text=True,
        capture_output=True,
    )


def apply_patch(repo: Path, patch: Path):
    if not patch.read_text().strip():
        return
    command = [
        "git",
        "-C",
        repo,
        "apply",
        "--whitespace=nowarn",
        "--exclude=*.jpg",
        "--exclude=*.png",
        "--exclude=*.webp",
        "--exclude=*.gif",
    ]
    if call(command + ["--reverse", "--check", patch], check=False).returncode == 0:
        return
    call(command + ["--check", patch])
    call(command + [patch])


def prepare_source(
    spec: dict, specs: dict, destination: Path, manifest_digest: str, check_only=False
):
    repo = destination / spec["path"]
    receipt = repo / ".galatea-baseline-source.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["manifest_sha256"] != manifest_digest:
            raise RuntimeError(f"source recipe changed; use a fresh destination: {repo}")
        head = call(["git", "-C", repo, "rev-parse", "HEAD"]).stdout.strip()
        if head != spec["revision"]:
            raise RuntimeError(f"source revision changed: {repo}")
        for name, digest in saved["files"].items():
            path = repo / name
            if not path.is_file() or sha256(path) != digest:
                raise RuntimeError(f"prepared source changed: {path}")
        print(f"verified {spec['key']} @ {head[:12]}")
        return
    if check_only:
        raise RuntimeError(f"source is not prepared; run baselines.setup for {spec['key']}: {repo}")
    if not (repo / ".git").exists():
        if repo.exists() and any(repo.iterdir()):
            raise RuntimeError(f"refusing nonempty non-Git destination: {repo}")
        repo.parent.mkdir(parents=True, exist_ok=True)
        call(["git", "clone", "--no-checkout", spec["url"], repo])
        call(["git", "-C", repo, "checkout", "--detach", spec["revision"]])
    else:
        head = call(["git", "-C", repo, "rev-parse", "HEAD"]).stdout.strip()
        if head != spec["revision"]:
            raise RuntimeError(
                f"existing checkout has revision {head}, expected {spec['revision']}: {repo}"
            )
        if call(
            ["git", "-C", repo, "diff", "HEAD", "--quiet", "--ignore-submodules"], check=False
        ).returncode:
            raise RuntimeError(
                f"existing checkout has local edits; use a fresh destination: {repo}"
            )
    if spec.get("submodules"):
        call(["git", "-C", repo, "submodule", "update", "--init", "--recursive"])
    for item in spec.get("upstream_patches", []):
        patch = destination / specs[item["repo"]]["path"] / item["path"]
        apply_patch(repo, patch)
    copied = []
    for item in spec.get("copies", []):
        source = destination / specs[item["repo"]]["path"] / item["source"]
        target = repo / item["target"]
        pairs = (
            [
                (path, target / path.relative_to(source))
                for path in source.rglob("*")
                if path.is_file()
            ]
            if source.is_dir()
            else [(source, target)]
        )
        for source_file, target_file in pairs:
            if target_file.exists() and source_file.read_bytes() != target_file.read_bytes():
                raise RuntimeError(f"refusing to overwrite existing source file: {target_file}")
            target_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, target_file)
            copied.append(str(target_file.relative_to(repo)))
    for name in spec.get("patches", []):
        apply_patch(repo, REPOSITORY / name)
    changed = call(
        ["git", "-C", repo, "diff", "HEAD", "--name-only", "--ignore-submodules"]
    ).stdout.splitlines()
    names = sorted(set(changed + copied))
    files = {name: sha256(repo / name) for name in names if (repo / name).is_file()}
    receipt.write_text(
        json.dumps(
            {"manifest_sha256": manifest_digest, "revision": spec["revision"], "files": files},
            indent=2,
        )
        + "\n"
    )
    print(f"prepared {spec['key']} @ {spec['revision'][:12]}")


def verify_runtime(method: str, config: dict):
    manifest = json.loads(MANIFEST.read_text())
    specs = {spec["key"]: spec for spec in manifest["repositories"]}
    primary_path = Path(specs[method]["path"])
    for key in manifest["methods"][method]:
        spec = specs[key]
        if key in {"videomanip_fp", "fpplusplus"}:
            repo = Path(config["fp_repo"])
        else:
            repo = Path(config["repo"]) / Path(spec["path"]).relative_to(primary_path)
        prepare_source({**spec, "path": "."}, specs, repo, recipe_digest(manifest), check_only=True)


def main():
    manifest = json.loads(MANIFEST.read_text())
    parser = argparse.ArgumentParser(
        description=(
            "Fetch pinned upstream sources and apply the recorded GALATEA adaptations. This "
            "prepares source only; install each method's environments and weights using the "
            "upstream documentation linked in the method README."))
    parser.add_argument("--method", choices=[*manifest["methods"], "all"], required=True)
    parser.add_argument("--destination", type=Path, default=REPOSITORY / "third_party")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify source pins and prepared-file hashes, without downloading",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    selected = (
        set(spec["key"] for spec in manifest["repositories"])
        if args.method == "all"
        else set(manifest["methods"][args.method])
    )
    specs = {spec["key"]: spec for spec in manifest["repositories"]}
    for spec in manifest["repositories"]:
        if spec["key"] not in selected:
            continue
        if args.dry_run:
            print(f"{spec['url']} @ {spec['revision']} -> {args.destination / spec['path']}")
        else:
            try:
                prepare_source(
                    spec, specs, args.destination.resolve(), recipe_digest(manifest), args.check
                )
            except subprocess.CalledProcessError as error:
                raise SystemExit(error.stderr.strip() or str(error)) from error


if __name__ == "__main__":
    main()
