import os

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

import arm_continuity
import arm_ik
import scene
from hand import HAND_DOF_NAMES

RMS_MAX_DEG = 70.0
IK_RMS_MAX_DEG = 60.0
LIMIT_MARGIN_MIN_DEG = 5.0
MIDDLE_DIRECTION_TOLERANCE_DEG = 20.1
RNG_BASE = 4700
LOCAL_SEED_COUNT = 12
LOCAL_SEED_SIGMA_RAD = 0.55
UNIFORM_SEED_COUNT = 12
COARSE_STEP_DEG = 5.0
FINE_STEP_DEG = 0.25
FINAL_STEP_DEG = 0.05


def _wrap_deg(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def _origin_transform(joint) -> np.ndarray:
    origin = joint.find("origin")
    T = np.eye(4)
    if origin is None:
        return T
    T[:3, 3] = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ")
    T[:3, :3] = R.from_euler("xyz", np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return T


def middle_finger_vector(urdf: Path) -> np.ndarray:
    root = ET.parse(urdf).getroot()
    by_child = {j.find("child").attrib["link"]: j for j in root.findall("joint")}
    qmap = dict(zip(HAND_DOF_NAMES, np.zeros(len(HAND_DOF_NAMES))))

    def link_transform(link: str) -> np.ndarray:
        chain = []
        while link != "right_hand_C_MC":
            joint = by_child[link]
            chain.append(joint)
            link = joint.find("parent").attrib["link"]
        T = np.eye(4)
        for joint in reversed(chain):
            T = T @ _origin_transform(joint)
            if joint.attrib["type"] in ("revolute", "continuous"):
                axis = np.fromstring(joint.find("axis").attrib.get("xyz", "1 0 0"), sep=" ")
                X = np.eye(4)
                X[:3, :3] = R.from_rotvec(axis * qmap[joint.attrib["name"]]).as_matrix()
                T = T @ X
        return T

    return link_transform("right_middle_fingertip")[:3, 3] - link_transform("right_middle_MCP_VL")[:3, 3]


def joint_stats(q):
    q = np.asarray(q, float)
    delta = q - scene.DEFAULT_Q
    margin = np.minimum(q - arm_ik.LO, arm_ik.HI - q)
    return {
        "joint_l2_rad": float(np.linalg.norm(delta)),
        "joint_rms_deg": float(np.degrees(np.sqrt(np.mean(delta * delta)))),
        "minimum_limit_margin_deg": float(np.degrees(np.min(margin))),
    }


def _perpendicular_error(rot, middle_local) -> float:
    axis = rot @ middle_local
    heading = float(np.degrees(np.arctan2(axis[1], axis[0])))
    return float(abs(abs(_wrap_deg(heading)) - 90.0))


def _solve_candidate(T, bT, seeds, max_it, qref):
    quality_valid = []
    for seed_index, seed in enumerate(seeds):
        q, ep, er = arm_ik.solve(T, bT, np.asarray(seed, float), qref, max_it, damp=arm_ik.DAMP * 0.5)
        if ep > scene.POS_GATE or er > scene.ROT_GATE:
            continue
        stats = joint_stats(q)
        if stats["joint_rms_deg"] > RMS_MAX_DEG or stats["minimum_limit_margin_deg"] < LIMIT_MARGIN_MIN_DEG:
            continue
        quality_valid.append(
            {"score": stats["joint_l2_rad"], "q": np.asarray(q, dtype=np.float64), "seed_index": seed_index}
        )
    quality_valid.sort(key=lambda row: (row["score"], row["seed_index"]))
    return quality_valid[0] if quality_valid else None


def search_first_frame(clip: str, inputs: dict, target_xy: np.ndarray, middle_local: np.ndarray):
    bT = arm_ik.base_transform(scene.BASE_POS, scene.BASE_YAW)
    wrist0, wrist_rot0 = (inputs["wrist"][0], inputs["wrist_rot"][0])
    pivot_xy = inputs["obj"][:, :2].mean(axis=0)
    rng = np.random.default_rng(RNG_BASE + int(clip[1:]))
    seeds = [scene.DEFAULT_Q.copy()]
    seeds.extend(
        (
            np.clip(scene.DEFAULT_Q + rng.normal(0.0, LOCAL_SEED_SIGMA_RAD, 7), arm_ik.LO, arm_ik.HI)
            for _ in range(LOCAL_SEED_COUNT)
        )
    )
    seeds.extend((rng.uniform(arm_ik.LO, arm_ik.HI) for _ in range(UNIFORM_SEED_COUNT)))

    def target_at(yaw_deg):
        rz = R.from_euler("z", yaw_deg, degrees=True).as_matrix()
        pivot = np.array([pivot_xy[0], pivot_xy[1], 0.0])
        pos = rz @ (wrist0 - pivot) + pivot
        pos[:2] += target_xy - pivot_xy
        rot = rz @ R.from_rotvec(wrist_rot0).as_matrix()
        return (arm_ik.target_pose(pos, R.from_matrix(rot).as_rotvec()), rot)

    def solve(target, candidate_seeds, max_it):
        for qref in (scene.DEFAULT_Q, None):
            answer = _solve_candidate(target, bT, candidate_seeds, max_it, qref)
            if answer is not None:
                return answer
        return None

    def search(yaws, candidate_seeds, max_it, constrained):
        found = []
        for yaw in yaws:
            target, rot = target_at(yaw)
            if constrained and _perpendicular_error(rot, middle_local) > MIDDLE_DIRECTION_TOLERANCE_DEG + 1e-08:
                continue
            answer = solve(target, candidate_seeds, max_it)
            if answer is not None:
                found.append((answer["score"], yaw, answer, rot))
        found.sort(key=lambda x: (x[0], x[1]))
        return found

    coarse_grid = [float(yaw) for yaw in np.arange(-180.0, 180.0, COARSE_STEP_DEG)]
    coarse = search(coarse_grid, [scene.DEFAULT_Q], 800, False)
    if not coarse:
        coarse = search(coarse_grid, seeds, 1000, False)
    if not coarse:
        raise RuntimeError(f"{clip}: no valid frame-0 yaw on the full-circle grid")
    fine = search(
        sorted({_wrap_deg(coarse[0][1] + d) for d in np.arange(-7.0, 7.0001, FINE_STEP_DEG)}),
        [scene.DEFAULT_Q],
        800,
        False,
    )
    if not fine:
        raise RuntimeError(f"{clip}: coarse yaw had no valid fine neighbour")
    verified = search([row[1] for row in fine[:9]], seeds, 1000, False)
    if not verified:
        raise RuntimeError(f"{clip}: no multi-start verified frame-0 solution")
    seed_yaw, seed_q = (verified[0][1], verified[0][2]["q"])
    final = search(
        [_wrap_deg(seed_yaw + d) for d in np.arange(-0.35, 0.3501, FINAL_STEP_DEG)],
        [scene.DEFAULT_Q, seed_q],
        1200,
        False,
    )
    if not final:
        raise RuntimeError(f"{clip}: final refinement found no valid solution")
    _, unconstrained_yaw, unconstrained, unconstrained_rot = final[0]
    _, rot_zero = target_at(0.0)
    axis_zero = rot_zero @ middle_local
    heading_zero = float(np.degrees(np.arctan2(axis_zero[1], axis_zero[0])))
    perpendicular_yaws = [_wrap_deg(90.0 - heading_zero), _wrap_deg(-90.0 - heading_zero)]
    semantic_yaws = set(perpendicular_yaws)
    for exact_yaw in perpendicular_yaws:
        semantic_yaws.add(_wrap_deg(exact_yaw - MIDDLE_DIRECTION_TOLERANCE_DEG))
        semantic_yaws.add(_wrap_deg(exact_yaw + MIDDLE_DIRECTION_TOLERANCE_DEG))
    semantic_yaws.update(
        (
            yaw
            for _, yaw, _, rot in coarse
            if _perpendicular_error(rot, middle_local) <= MIDDLE_DIRECTION_TOLERANCE_DEG + 1e-08
        )
    )
    if _perpendicular_error(unconstrained_rot, middle_local) <= MIDDLE_DIRECTION_TOLERANCE_DEG + 1e-08:
        semantic_yaws.add(float(unconstrained_yaw))
    screened = search(sorted(semantic_yaws), [scene.DEFAULT_Q, unconstrained["q"]], 1000, True)
    if not screened:
        raise RuntimeError(f"{clip}: no reachable yaw near either table short-edge direction")
    fine_semantic_yaws = set(semantic_yaws)
    for _, center_yaw, _, _ in screened[:4]:
        fine_semantic_yaws.update((_wrap_deg(center_yaw + delta) for delta in np.arange(-2.5, 2.5001, FINE_STEP_DEG)))
    warm_q = [row[2]["q"] for row in screened[:4]]
    refined = search(sorted(fine_semantic_yaws), [scene.DEFAULT_Q, unconstrained["q"], *warm_q], 1200, True)
    if not refined:
        raise RuntimeError(f"{clip}: short-edge refinement found no valid solution")
    verified_semantic = []
    for _, yaw, candidate, _ in refined[:9]:
        target, rot = target_at(yaw)
        answer = solve(target, seeds + [unconstrained["q"], candidate["q"]], 1200)
        if answer is not None:
            verified_semantic.append((answer["score"], yaw, answer, rot))
    verified_semantic.sort(key=lambda x: (x[0], x[1]))
    if not verified_semantic:
        raise RuntimeError(f"{clip}: short-edge multi-start verification failed")
    _, yaw, winner, _ = verified_semantic[0]
    final_semantic = search(
        sorted({_wrap_deg(yaw + delta) for delta in np.arange(-0.35, 0.3501, FINAL_STEP_DEG)} | {yaw}),
        [scene.DEFAULT_Q, unconstrained["q"], winner["q"]],
        1200,
        True,
    )
    final_semantic.extend(verified_semantic)
    final_semantic.sort(key=lambda x: (x[0], x[1]))
    _, yaw, winner, rot = final_semantic[0]
    if _perpendicular_error(rot, middle_local) > MIDDLE_DIRECTION_TOLERANCE_DEG + 1e-08:
        raise RuntimeError(f"{clip}: middle-finger direction misses the table short edge")
    return (float(yaw), winner["q"])


def solve_trajectory(clip: str, inputs: dict, target_xy: np.ndarray, yaw_deg: float, q0: np.ndarray):
    obj, objrot, wrist, wrist_rot = (inputs["obj"], inputs["objrot"], inputs["wrist"], inputs["wrist_rot"])
    n = len(obj)
    theta = float(np.radians(float(yaw_deg)))
    pivot = obj[:, :2].mean(0)
    obj, objrot, wrist, wrist_rot = scene.rotate_scene(obj, objrot, wrist, wrist_rot, theta, pivot)
    shift = target_xy - obj[:, :2].mean(0)
    obj[:, :2] += shift
    wrist[:, :2] += shift
    bT = arm_ik.base_transform(scene.BASE_POS, scene.BASE_YAW)
    targets = [arm_ik.target_pose(wrist[t], wrist_rot[t]) for t in range(n)]
    q0 = np.asarray(q0, dtype=np.float64)
    pos_error, rot_error = arm_ik.pose_error(q0, bT, targets[0])
    stats = joint_stats(q0)
    if (
        pos_error > scene.POS_GATE
        or rot_error > scene.ROT_GATE
        or stats["joint_rms_deg"] > IK_RMS_MAX_DEG
        or stats["minimum_limit_margin_deg"] < LIMIT_MARGIN_MIN_DEG
    ):
        raise RuntimeError(f"{clip}: frame-0 configuration fails the quality gates")
    armq, _, _, n_unreach = arm_continuity.solve_continuous(targets, bT, np.random.default_rng(100 + int(clip[1:])), q0)
    pe, re = (np.zeros(n), np.zeros(n))
    for t in range(n):
        pe[t], re[t] = arm_ik.pose_error(armq[t], bT, targets[t])
    if (
        not (np.isfinite(armq).all() and np.isfinite(pe).all() and np.isfinite(re).all())
        or pe.max() >= scene.POS_GATE
        or re.max() >= scene.ROT_GATE
    ):
        raise RuntimeError(
            f"{clip}: trajectory IK failed: pos_max={pe.max() * 1000:.3f}mm rot_max={np.degrees(re.max()):.3f}deg unreachable={n_unreach}"
        )
    import trimesh

    verts = np.asarray(trimesh.load(inputs["demo_dir"] / scene.MESH_FILE, force="mesh", process=False).vertices, float)
    world_verts = np.einsum("tij,nj->tni", objrot, verts) + obj[:, None, :]
    deficit = np.clip(scene.TABLE_Z + scene.RESET_FLOOR_MARGIN - world_verts[..., 2].min(1), 0.0, None)
    armq_reset = np.zeros_like(armq)
    reset_pe = np.zeros(n)
    reset_re = np.zeros(n)
    for t in range(n):
        target = arm_ik.target_pose(wrist[t] + [0.0, 0.0, deficit[t]], wrist_rot[t])
        q, ep, er = arm_ik.solve_ball(target, bT, armq[t], arm_ik.SPEED_CAP)
        if ep >= scene.POS_GATE or er >= scene.ROT_GATE:
            q, ep, er = arm_ik.solve_restarts(target, bT, armq[t], np.random.default_rng(300 + t))
        armq_reset[t], reset_pe[t], reset_re[t] = (q, ep, er)
    if (
        not (np.isfinite(armq_reset).all() and np.isfinite(reset_pe).all() and np.isfinite(reset_re).all())
        or reset_pe.max() >= scene.POS_GATE
        or reset_re.max() >= scene.ROT_GATE
    ):
        raise RuntimeError(
            f"{clip}: reset IK failed: pos_max={reset_pe.max() * 1000:.3f}mm rot_max={np.degrees(reset_re.max()):.3f}deg"
        )
    if float(np.linalg.norm(armq[0] - q0)) > 1e-10:
        raise RuntimeError(f"{clip}: trajectory solver changed the frame-0 configuration")
    return {
        "arm_qpos": armq,
        "arm_qpos_reset": armq_reset,
        "deficit": deficit,
        "pos_err": pe,
        "rot_err": re,
        "reset_pos_err": reset_pe,
        "reset_rot_err": reset_re,
        "ref_wrist_pos": wrist,
        "ref_wrist_rot": wrist_rot,
        "shift_xy": shift,
        "corot_theta": np.degrees(theta),
        "corot_pivot": np.array([pivot[0], pivot[1], 0.0]),
        "first_frame_q0": q0,
    }


def solve_clip(bank: Path, clip: str, target_xy, urdf: Path, output: Path) -> None:
    arm_ik.configure(str(urdf))
    entry = next(row for row in scene.load_entries(bank) if row["data_index"] == clip)
    target_xy = np.asarray(target_xy, dtype=np.float64)
    yaw_deg, q0 = search_first_frame(clip, scene.load_ik_inputs(bank, entry), target_xy, middle_finger_vector(urdf))
    arm = solve_trajectory(clip, scene.load_ik_inputs(bank, entry), target_xy, yaw_deg, q0)
    scene.atomic_npz(output, arm)
    print(
        f"[ik] {clip} yaw={yaw_deg:+.2f}deg pos_max={arm['pos_err'].max() * 1000:.2f}mm "
        f"rot_max={np.degrees(arm['rot_err'].max()):.2f}deg",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("--clip")
    parser.add_argument("--target-xy", type=float, nargs=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    bank = args.bank.resolve(strict=True)
    urdf = args.urdf.resolve(strict=True)
    if args.clip is not None:
        solve_clip(bank, args.clip, args.target_xy, urdf, args.output)
        return
    plan = json.loads(args.plan.read_text())["clips"]
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(bank / "ik", ignore_errors=True)
    (bank / "ik").mkdir()

    def run(clip: str) -> dict:
        clip_work = work / clip
        if clip_work.exists():
            shutil.rmtree(clip_work)
        clip_work.mkdir(parents=True)
        output = clip_work / f"{clip}_fr3_retarget.npz"
        xy = plan[clip]["target_object_xy"]
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--bank",
            str(bank),
            "--plan",
            str(args.plan),
            "--urdf",
            str(urdf),
            "--work",
            str(work),
            "--clip",
            clip,
            "--target-xy",
            repr(xy[0]),
            repr(xy[1]),
            "--output",
            str(output),
        ]
        started = time.time()
        with (clip_work / "run.log").open("wb") as log:
            try:
                returncode = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout or None, check=False
                ).returncode
            except subprocess.TimeoutExpired:
                returncode = None
        row = {
            "status": "passed" if returncode == 0 and output.is_file() else "failed",
            "wall_time_s": time.time() - started,
        }
        if returncode is None:
            row["status"] = "timeout"
        if row["status"] == "passed":
            shutil.copy2(output, scene.ik_path(bank, clip))
        else:
            row["log_tail"] = (clip_work / "run.log").read_text(errors="replace").splitlines()[-5:]
        return row

    clips = list(plan)
    rows = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, len(clips))) as pool:
        futures = {pool.submit(run, clip): clip for clip in clips}
        for future in as_completed(futures):
            clip = futures[future]
            rows[clip] = future.result()
            print(f"[ik] {clip} {rows[clip]['status']} wall={rows[clip]['wall_time_s']:.1f}s", flush=True)
    rows = {clip: rows[clip] for clip in clips}
    scene.atomic_json(bank / "ik_report.json", rows)
    passed = sum(row["status"] == "passed" for row in rows.values())
    print(f"[ik] passed {passed}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
