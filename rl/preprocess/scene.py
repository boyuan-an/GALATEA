import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as R

import arm_ik

TABLE_Z = 0.415
BASE_POS = np.array([0.0, 0.95, 0.0], dtype=np.float64)
BASE_YAW = -np.pi / 2
DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
POS_GATE = 0.01
ROT_GATE = np.radians(5.0)
RESET_FLOOR_MARGIN = 0.006
DQ_GATE = 0.295
DEMO_FILE = "motion.npy"
MESH_FILE = "object.obj"


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def atomic_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **payload)
    os.replace(temporary, path)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def load_entries(bank: Path) -> list:
    return json.loads((bank / "data" / "manifest.json").read_text())["entries"]


def ik_path(bank: Path, clip: str) -> Path:
    return bank / "ik" / f"{clip}_fr3_retarget.npz"


def scene_transform(table_z: float = TABLE_Z):
    base_r = R.from_rotvec([0, 0, -np.pi / 2]).as_matrix() @ R.from_rotvec([np.pi / 2, 0, 0]).as_matrix()
    off_r = R.from_rotvec([-np.pi / 2, 0, 0]).as_matrix() @ R.from_rotvec([0, 0, np.pi / 2]).as_matrix()
    return (base_r @ off_r, np.array([0.0, 0.0, table_z]) + base_r @ np.array([0.0, 0.018, 0.0]))


def object_world(demo_dir: Path, table_z: float = TABLE_Z, as_float64: bool = False):
    demo = np.load(demo_dir / DEMO_FILE, allow_pickle=True).item()
    mr, mt = scene_transform(table_z)
    pos = (mr @ np.asarray(demo["object_transl"], dtype=np.float64 if as_float64 else None).T).T + mt
    rot = np.array([mr @ R.from_rotvec(o).as_matrix().T for o in demo["object_global_orient"]])
    return (pos.astype(np.float64), rot.astype(np.float64))


def rotate_scene(pos, rot, wrist, wrist_rot, theta, pivot_xy):
    rz = R.from_euler("z", theta).as_matrix()
    piv = np.array([pivot_xy[0], pivot_xy[1], 0.0])
    pos2 = (rz @ (pos - piv).T).T + piv
    rot2 = np.einsum("ij,tjk->tik", rz, rot)
    wrist2 = (rz @ (wrist - piv).T).T + piv
    wrist_rot2 = R.from_matrix(np.einsum("ij,tjk->tik", rz, R.from_rotvec(wrist_rot).as_matrix())).as_rotvec()
    return (pos2, rot2, wrist2, wrist_rot2)


def apply_arm_scene(arm, pos, rot):
    theta = float(arm["corot_theta"])
    if abs(theta) > 1e-09:
        piv = np.asarray(arm["corot_pivot"], float).reshape(-1)
        rz = R.from_euler("z", np.radians(theta)).as_matrix()
        piv3 = np.array([piv[0], piv[1], 0.0])
        pos = (rz @ (pos - piv3).T).T + piv3
        rot = np.einsum("ij,tjk->tik", rz, rot)
    pos[:, :2] += np.asarray(arm["shift_xy"], float)
    return (pos, rot)


def object_bottoms(bank: Path, entry: dict, arm: dict, table_z: float = TABLE_Z, as_float64: bool = False):
    import trimesh

    demo_dir = bank / "data" / entry["dir"]
    pos, rot = apply_arm_scene(arm, *object_world(demo_dir, table_z, as_float64))
    n = len(np.asarray(arm["arm_qpos_reset"]))
    pos, rot = (pos[:n], rot[:n])
    vertices = np.asarray(trimesh.load(demo_dir / MESH_FILE, force="mesh", process=False).vertices, dtype=np.float64)
    world = np.einsum("tij,nj->tni", rot, vertices) + pos[:, None]
    return (world[..., 2].min(axis=1), world, pos, rot, vertices)


def load_ik_inputs(bank: Path, entry: dict) -> dict:
    demo_dir = bank / "data" / entry["dir"]
    obj, objrot = object_world(demo_dir)
    with np.load(bank / "wrist_targets" / f"{entry['data_index']}.npz") as targets:
        wrist = np.asarray(targets["position"], dtype=np.float64)
        wrist_rot = np.asarray(targets["rotation"], dtype=np.float64)
    if not len(obj) == len(wrist) == len(wrist_rot):
        raise ValueError(f"{entry['data_index']}: object and wrist lengths differ")
    return {"demo_dir": demo_dir, "obj": obj, "objrot": objrot, "wrist": wrist, "wrist_rot": wrist_rot}


def default_wrist_pose(urdf: Path):
    arm_ik.configure(str(urdf))
    return arm_ik.fk(DEFAULT_Q, arm_ik.base_transform(BASE_POS, BASE_YAW))[0]


def default_wrist_yaw(wrist_rot: np.ndarray, default_rot: np.ndarray):
    rw = R.from_rotvec(wrist_rot).as_matrix()

    def errors(theta):
        rz = R.from_euler("z", theta).as_matrix()
        aligned = np.einsum("ij,tjk->tik", rz, rw)
        return R.from_matrix(np.einsum("ij,tjk->tik", default_rot.T, aligned)).magnitude()

    def objective(theta):
        a = errors(theta)
        return float(np.mean(a * a))

    grid = np.linspace(-np.pi, np.pi, 1441)
    values = np.array([objective(theta) for theta in grid])
    k = int(values.argmin())
    step = float(grid[1] - grid[0])
    lo, hi = (max(-np.pi, grid[k] - 2 * step), min(np.pi, grid[k] + 2 * step))
    return float(minimize_scalar(objective, bounds=(lo, hi), method="bounded").x)
