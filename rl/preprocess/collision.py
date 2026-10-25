import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import hppfcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R

import scene

TABLE_X = (-0.6, 0.4)
TABLE_Y = (-0.8, 0.8)
TABLE_THICKNESS = 0.03
PENETRATION_TOLERANCE = 0.0002
GROUNDING_TOLERANCE = 0.0005
CONTACT_OFFSET = 0.005
AUDIT_KEYS = (
    "reset_collision_free",
    "reference_object_bottom_z",
    "reset_table_collision_link_names",
    "reset_table_signed_distance_m",
)
_ROBOT_CACHE = {}


def strip_audit(arm: dict) -> dict:
    return {key: value for key, value in arm.items() if key not in AUDIT_KEYS}


def _rpy(rpy):
    return R.from_euler("xyz", rpy).as_matrix()


def _T(Rm, t):
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = t
    return T


def _aa(axis, ang):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-09 or abs(ang) < 1e-12:
        return np.eye(3)
    return R.from_rotvec(axis / n * ang).as_matrix()


def parse_joints(urdf_path):
    joints = []
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        o = j.find("origin")
        xyz, rpy = (np.zeros(3), np.zeros(3))
        if o is not None:
            if o.get("xyz"):
                xyz = np.array([float(x) for x in o.get("xyz").split()])
            if o.get("rpy"):
                rpy = np.array([float(x) for x in o.get("rpy").split()])
        ax = j.find("axis")
        axis = np.array([float(x) for x in ax.get("xyz").split()]) if ax is not None else np.array([0, 0, 1.0])
        joints.append(
            dict(
                name=j.get("name"),
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                xyz=xyz,
                rpy=rpy,
                axis=axis,
            )
        )
    return joints


def fk_all(root_link, T0, joints, angles):
    kids = {}
    for j in joints:
        kids.setdefault(j["parent"], []).append(j)
    world = {root_link: T0}
    stack = [root_link]
    while stack:
        cur = stack.pop()
        for j in kids.get(cur, []):
            Tj = _T(_rpy(j["rpy"]), j["xyz"])
            if j["type"] == "revolute":
                Tj = Tj @ _T(_aa(j["axis"], angles.get(j["name"], 0.0)), np.zeros(3))
            world[j["child"]] = world[cur] @ Tj
            stack.append(j["child"])
    return world


def _origin(collision) -> np.ndarray:
    transform = np.eye(4)
    node = collision.find("origin")
    if node is not None:
        transform[:3, 3] = np.fromstring(node.attrib.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = R.from_euler("xyz", np.fromstring(node.attrib.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


def _convex(vertices: np.ndarray):
    points = hppfcl.StdVec_Vec3f()
    points.extend([np.asarray(vertex, dtype=np.float64) for vertex in vertices])
    return hppfcl.Convex.convexHull(points, True, "Qt")


class RobotTable:
    def __init__(self, urdf: Path):
        self.joints = parse_joints(str(urdf))
        children = {j["child"] for j in self.joints}
        self.root = sorted({j["parent"] for j in self.joints} - children)[0]
        self.arm_joint_names = [f"fr3_joint{i}" for i in range(1, 8)]
        self.base = np.eye(4)
        self.base[:3, :3] = R.from_euler("z", float(scene.BASE_YAW)).as_matrix()
        self.base[:3, 3] = scene.BASE_POS
        xml = ET.parse(urdf).getroot()
        joint_nodes = {joint.attrib["name"]: joint for joint in xml.findall("joint")}
        limits = [joint_nodes[name].find("limit").attrib for name in self.arm_joint_names]
        self.arm_lower = np.asarray([float(limit["lower"]) for limit in limits], dtype=np.float64)
        self.arm_upper = np.asarray([float(limit["upper"]) for limit in limits], dtype=np.float64)
        self.link_pieces = {}
        for link in xml.findall("link"):
            pieces = []
            for collision in link.findall("collision"):
                mesh_node = collision.find("geometry/mesh")
                mesh = trimesh.load(urdf.parent / mesh_node.attrib["filename"], force="mesh", process=False).convex_hull
                points = np.asarray(mesh.vertices, dtype=np.float64)
                scale = np.fromstring(mesh_node.attrib.get("scale", "1 1 1"), sep=" ")
                if scale.size == 1:
                    scale = np.repeat(scale, 3)
                points *= scale[None]
                transform = _origin(collision)
                points = (transform[:3, :3] @ points.T).T + transform[:3, 3]
                pieces.append(_convex(points))
            if pieces:
                self.link_pieces[link.attrib["name"]] = pieces
        self.link_names = list(self.link_pieces)
        self.arm_links = np.array([name.startswith("fr3_link") or name == "base" for name in self.link_names])

    def fk(self, arm_qpos):
        return fk_all(
            self.root, self.base, self.joints, {name: float(arm_qpos[i]) for i, name in enumerate(self.arm_joint_names)}
        )

    def table_distances(self, arm_qpos, table_top):
        n = len(arm_qpos)
        collision = np.zeros((n, len(self.link_names)), dtype=np.bool_)
        signed_distance = np.full((n, len(self.link_names)), np.inf, dtype=np.float32)
        table = hppfcl.Box(np.array([TABLE_X[1] - TABLE_X[0], TABLE_Y[1] - TABLE_Y[0], TABLE_THICKNESS]))
        table_transform = hppfcl.Transform3f(
            np.array(
                [
                    (TABLE_X[0] + TABLE_X[1]) / 2.0,
                    (TABLE_Y[0] + TABLE_Y[1]) / 2.0,
                    float(table_top) - TABLE_THICKNESS / 2.0,
                ]
            )
        )
        request = hppfcl.DistanceRequest()
        request.enable_nearest_points = True
        for frame, q in enumerate(np.asarray(arm_qpos, dtype=np.float64)):
            world = self.fk(q)
            for column, link in enumerate(self.link_names):
                link_transform = hppfcl.Transform3f(world[link][:3, :3], world[link][:3, 3])
                best = np.inf
                for piece in self.link_pieces[link]:
                    result = hppfcl.DistanceResult()
                    best = min(
                        best, float(hppfcl.distance(piece, link_transform, table, table_transform, request, result))
                    )
                signed_distance[frame, column] = best
                collision[frame, column] = best < -PENETRATION_TOLERANCE
        return (collision, signed_distance)


def audit_one(bank: Path, entry: dict, urdf: Path, write: bool, table_top: float = scene.TABLE_Z) -> dict:
    path = scene.ik_path(bank, entry["data_index"])
    arm = scene.load_npz(path)
    arm_reset = np.asarray(arm["arm_qpos_reset"], dtype=np.float64)
    arm_reference = np.asarray(arm["arm_qpos"], dtype=np.float64)
    wrist_pos = np.asarray(arm["ref_wrist_pos"], dtype=np.float64)
    deficit = np.asarray(arm["deficit"], dtype=np.float64).reshape(-1)
    reference_bottom, world_vertices, object_position, object_rotation, object_vertices = scene.object_bottoms(
        bank, entry, arm, table_top, as_float64=True
    )
    reset_bottom = reference_bottom + deficit
    robot = _ROBOT_CACHE.get(str(urdf))
    if robot is None:
        robot = _ROBOT_CACHE[str(urdf)] = RobotTable(urdf)
    collision, signed_distance = robot.table_distances(arm_reset, table_top)
    n = len(arm_reset)
    reference_position_error = np.zeros(n, dtype=np.float64)
    reference_rotation_error = np.zeros(n, dtype=np.float64)
    reset_position_error = np.zeros(n, dtype=np.float64)
    reset_rotation_error = np.zeros(n, dtype=np.float64)
    object_distance = np.full(n, np.inf, dtype=np.float64)
    object_shape = _convex(object_vertices)
    object_reset_position = object_position.copy()
    object_reset_position[:, 2] += deficit
    request = hppfcl.DistanceRequest()
    request.enable_nearest_points = True
    target_rotations = R.from_rotvec(np.asarray(arm["ref_wrist_rot"], dtype=np.float64)).as_matrix()
    for frame in range(n):
        reference_wrist = robot.fk(arm_reference[frame])["right_hand_C_MC"]
        reference_position_error[frame] = np.linalg.norm(reference_wrist[:3, 3] - wrist_pos[frame])
        reference_rotation_error[frame] = R.from_matrix(target_rotations[frame] @ reference_wrist[:3, :3].T).magnitude()
        reset_world = robot.fk(arm_reset[frame])
        wrist = reset_world["right_hand_C_MC"]
        target_position = wrist_pos[frame].copy()
        target_position[2] += deficit[frame]
        reset_position_error[frame] = np.linalg.norm(wrist[:3, 3] - target_position)
        reset_rotation_error[frame] = R.from_matrix(target_rotations[frame] @ wrist[:3, :3].T).magnitude()
        object_transform = hppfcl.Transform3f(object_rotation[frame], object_reset_position[frame])
        for link in robot.link_names:
            link_transform = hppfcl.Transform3f(reset_world[link][:3, :3], reset_world[link][:3, 3])
            for piece in robot.link_pieces[link]:
                result = hppfcl.DistanceResult()
                distance = float(
                    hppfcl.distance(piece, link_transform, object_shape, object_transform, request, result)
                )
                object_distance[frame] = min(object_distance[frame], distance)
    hand_columns = ~robot.arm_links
    reset_object_ok = reset_bottom >= table_top - GROUNDING_TOLERANCE
    geometry_free = ~collision[:, hand_columns].any(axis=1) & reset_object_ok
    contact_band_clear = (signed_distance[:, hand_columns] >= CONTACT_OFFSET).all(axis=1) & (
        reset_bottom - table_top >= CONTACT_OFFSET
    )
    object_clear = object_distance >= CONTACT_OFFSET
    reset_ik_valid = (reset_position_error < scene.POS_GATE) & (reset_rotation_error < scene.ROT_GATE)
    reset_in_limits = ((arm_reset >= robot.arm_lower[None] - 1e-07) & (arm_reset <= robot.arm_upper[None] + 1e-07)).all(
        axis=1
    )
    reference_in_limits = (
        (arm_reference >= robot.arm_lower[None] - 1e-07) & (arm_reference <= robot.arm_upper[None] + 1e-07)
    ).all(axis=1)
    support = world_vertices[0][world_vertices[0, :, 2] <= reference_bottom[0] + GROUNDING_TOLERANCE]
    support_min, support_max = (support[:, :2].min(axis=0), support[:, :2].max(axis=0))
    reasons = []
    if abs(float(reference_bottom[0]) - table_top) > GROUNDING_TOLERANCE:
        reasons.append("frame-0 object is not on the table")
    if not (
        support_min[0] >= TABLE_X[0]
        and support_max[0] <= TABLE_X[1]
        and support_min[1] >= TABLE_Y[0]
        and support_max[1] <= TABLE_Y[1]
    ):
        reasons.append("frame-0 object support lies outside the table")
    if reset_position_error.max() >= scene.POS_GATE or reset_rotation_error.max() >= scene.ROT_GATE:
        reasons.append("reset IK error above the gate")
    if reference_position_error.max() >= scene.POS_GATE or reference_rotation_error.max() >= scene.ROT_GATE:
        reasons.append("reference IK error above the gate")
    if not reset_in_limits.all():
        reasons.append("reset configuration outside the joint limits")
    if not reference_in_limits.all():
        reasons.append("reference configuration outside the joint limits")
    candidate = geometry_free & contact_band_clear & object_clear & reset_ik_valid & reset_in_limits
    if not candidate.any():
        reasons.append("no collision-free reset frame")
    elif not candidate[0]:
        reasons.append("frame-0 reset is not collision-free")
    accepted = not reasons
    if write:
        scene.atomic_npz(
            path,
            {
                **strip_audit(arm),
                "reset_collision_free": (candidate if accepted else np.zeros_like(candidate)).astype(np.bool_),
                "reference_object_bottom_z": reference_bottom.astype(np.float32),
                "reset_table_collision_link_names": np.asarray(robot.link_names),
                "reset_table_signed_distance_m": signed_distance.astype(np.float32),
            },
        )
    return {"accepted": accepted, "reasons": reasons, "frame0_object_clear": bool(object_clear[0])}


def audit_bank(bank: Path, urdf: Path, workers: int, write: bool = True) -> dict:
    entries = scene.load_entries(bank)
    with ProcessPoolExecutor(max_workers=min(workers, len(entries))) as pool:
        futures = [pool.submit(audit_one, bank, entry, urdf, write) for entry in entries]
        rows = {entry["data_index"]: future.result() for entry, future in zip(entries, futures)}
    for clip, row in rows.items():
        if not row["accepted"]:
            print(f"[collision] {clip} rejected: {'; '.join(row['reasons'])}", flush=True)
    scene.atomic_json(bank / "collision_report.json", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=32)
    args = parser.parse_args()
    rows = audit_bank(args.bank.resolve(strict=True), args.urdf.resolve(strict=True), args.workers)
    print(f"[collision] accepted {sum(row['accepted'] for row in rows.values())}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
