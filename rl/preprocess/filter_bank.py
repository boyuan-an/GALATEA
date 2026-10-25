import argparse
import os
import shutil
from pathlib import Path

import scene


def filter_bank(source: Path, output: Path, exclude=()) -> list:
    if output.exists():
        raise FileExistsError(output)
    building = output.with_name(output.name + ".building")
    shutil.rmtree(building, ignore_errors=True)
    (building / "data").mkdir(parents=True)
    (building / "ik").mkdir()
    accepted = []
    for row in scene.load_entries(source):
        arm = scene.ik_path(source, row["data_index"])
        if row["data_index"] in exclude or not arm.is_file():
            continue
        new = f"g{len(accepted)}"
        accepted.append(dict(row, data_index=new))
        shutil.copytree(source / "data" / row["dir"], building / "data" / row["dir"], copy_function=os.link)
        os.link(arm, scene.ik_path(building, new))
    if not accepted:
        raise RuntimeError("no trajectories left after filtering")
    scene.atomic_json(building / "data" / "manifest.json", {"entries": accepted})
    os.replace(building, output)
    print(f"[filter] kept {len(accepted)} trajectories -> {output}", flush=True)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    filter_bank(args.source.resolve(strict=True), args.output.resolve())


if __name__ == "__main__":
    main()
