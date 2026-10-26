import hashlib
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from gym import spaces
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_conjugate, quat_mul, quat_rotate, quat_rotate_inverse

from armhand.data.demo_dataset import DemoDataset
from armhand.data.rotations import aa_to_quat, aa_to_rotmat, quat_to_rotmat, rotmat_to_aa, rotmat_to_quat
from armhand.utils.torch_math import scale, tensor_clamp

from .arm_kinematics import ArmKinematics
from .base import SimTask
from .collision import hand_self_collision_masks
from .frames import position_to_base, quaternion_to_base, vector_to_base
from .physics_randomization import PhysicsRandomizer
from .rewards import (
    action_rate,
    compute_tracking_reward,
    farthest_point_sample,
    nearest_surface_distance,
    pad_distance_score,
    quat_to_angle_axis,
    torque_penalty,
    velocity_fraction,
)
from .robot import (
    ARM_DAMPING,
    ARM_EFFORT,
    ARM_JOINTS,
    ARM_STIFFNESS,
    ARM_VELOCITY_LIMITS,
    FINGERTIP_BODIES,
    HAND_BODIES,
    HAND_JOINT_DRIVE,
    HAND_JOINT_LIMITS,
    KEYPOINT_GROUPS,
    NUM_HAND_DOFS,
    PAD_BODIES,
    REFERENCE_KEYPOINTS,
    WRIST_BODY,
)
from .shape_code import ShapeEncoder

DONE_REASONS = (
    "timeline_success",
    "object_position",
    "object_rotation",
    "hand_thumb",
    "hand_index",
    "hand_middle",
    "hand_ring",
    "hand_pinky",
    "hand_level_1",
    "hand_level_2",
    "contact_mismatch",
    "numerical_safety",
    "arm_velocity_limit",
    "hand_velocity_limit",
    "other_failure",
    "eval_max_steps_incomplete",
)
DONE_METRICS = (
    "progress_step",
    "wrist_position_error_m",
    "wrist_rotation_error_deg",
    "object_position_error_m",
    "object_rotation_error_deg",
    "arm_velocity_limit_fraction_max",
    "hand_velocity_limit_fraction_max",
)
NUM_ACTOR_DOFS = len(ARM_JOINTS) + NUM_HAND_DOFS
NUM_BODIES = len(HAND_BODIES)


def _asset_content_key(urdf_path):
    digest = hashlib.sha256(b"urdf\0")
    with open(urdf_path, "rb") as f:
        urdf_bytes = f.read()
    digest.update(urdf_bytes)
    for mesh in ET.fromstring(urdf_bytes).iter("mesh"):
        name = mesh.attrib.get("filename", "")
        if name:
            digest.update(b"\0mesh\0" + name.encode("utf-8"))
            with open(os.path.join(os.path.dirname(urdf_path), name), "rb") as f:
                digest.update(f.read())
    return digest.hexdigest()


class ArmHandTrackEnv(SimTask):
    proprioception_dim = 4 * NUM_ACTOR_DOFS + 13
    privileged_dim = NUM_ACTOR_DOFS + 13 + 4 * len(PAD_BODIES) + 4
    target_dim = 128 + 5 + 23 + (NUM_BODIES - 1) * 9 + 23 + NUM_BODIES
    actor_dim = (
        2 * NUM_ACTOR_DOFS
        + 3
        + 4
        + 3 * len(PAD_BODIES)
        + 3
        + 4
        + 3
        + 4
        + 4
        + (NUM_BODIES - 1) * 3
        + 3
        + 4
        + 4
        + 5
        + 128
    )

    def __init__(self, cfg, rl_device, sim_device):
        self.cfg = cfg
        env_cfg = cfg["env"]
        reward_cfg = env_cfg["reward"]
        self.max_episode_length = env_cfg["episodeLength"]
        self.data_indices = list(env_cfg["dataIndices"])
        self.random_state_init = env_cfg["randomStateInit"]
        self.reset_frame_fraction = env_cfg["resetFrameFraction"]
        self.failure_tolerance = env_cfg["failureTolerance"]
        self.tighten_factor = env_cfg["tightenFactor"]
        self.tighten_steps = env_cfg["tightenSteps"]
        self.tighten_schedule = env_cfg["tightenSchedule"]
        self.arm_action_scale = env_cfg["armActionScale"]
        self.arm_smoothing = env_cfg["armSmoothing"]
        self.arm_joint_margin = np.radians(env_cfg["armJointMarginDeg"])
        self.hand_smoothing = env_cfg["handSmoothing"]
        self.hand_target_deadband = env_cfg["handTargetDeadband"]
        self.object_pos_noise = env_cfg["objectPosNoise"]
        self.object_rot_noise = env_cfg["objectRotNoiseDeg"]
        self.gravity_ramp_steps = env_cfg["gravityRampSteps"]
        self.gravity_update_interval = env_cfg["gravityUpdateInterval"]
        self.gravity_initial_scale = env_cfg["gravityInitialScale"]
        self.floor_lift_fps = env_cfg["floorLiftFps"]
        self.self_collision = env_cfg["handSelfCollision"]
        mask_cfg = env_cfg["handActionMask"]
        self.action_mask_probability = mask_cfg["probability"]
        self.action_mask_joints = mask_cfg["joints"]
        self.action_mask_start_steps = mask_cfg["startMaxSteps"]
        self.action_mask_end_steps = mask_cfg["endMaxSteps"]
        self.action_mask_curriculum_steps = mask_cfg["curriculumSteps"]
        self.action_mask_sigma_min = mask_cfg["sigmaMin"]
        self.reward_cfg = reward_cfg
        self.hand_velocity_gate_threshold = reward_cfg["handVelocity"]["unlockSuccessRate"]
        self.hand_velocity_gate_unlocked = False
        self.hand_velocity_gate_epoch = -1
        self.hand_velocity_gate_last_success = float("nan")
        self.action_mask_step = 0
        self.eval_mode = False
        self.record_done_reasons = False
        self.done_reason_bits = None
        self.done_metrics = None
        self.mass_rng = np.random.default_rng(np.random.SeedSequence([int(cfg["seed"]) & 4294967295, 1297371716]))
        cfg["env"]["numActions"] = NUM_ACTOR_DOFS
        super().__init__(cfg, rl_device, sim_device)

        self.randomizer = PhysicsRandomizer(self, env_cfg["physicsRandomization"])
        self.mask_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.mask_joints = torch.zeros((self.num_envs, NUM_HAND_DOFS), dtype=torch.bool, device=self.device)
        self.mask_previous_command = torch.zeros((self.num_envs, NUM_HAND_DOFS), device=self.device)
        self.obs_dict["target"] = torch.zeros((self.num_envs, self.target_dim), device=self.device)
        self.obs_dict["actor"] = torch.zeros((self.num_envs, self.actor_dim), device=self.device)
        self.obs_space = spaces.Dict(
            {
                "proprioception": spaces.Box(low=-np.inf, high=np.inf, shape=(self.proprioception_dim,)),
                "privileged": spaces.Box(low=-np.inf, high=np.inf, shape=(self.privileged_dim,)),
                "target": spaces.Box(low=-np.inf, high=np.inf, shape=(self.target_dim,)),
                "actor": spaces.Box(low=-np.inf, high=np.inf, shape=(self.actor_dim,)),
            }
        )
        self.demo_shape_code = ShapeEncoder(128, 0.2, self.device)(self.demo_bank["object_points"])
        self.shape_code = self.demo_shape_code[self.demo_ids]
        initial_points = self.demo_bank["object_points"][self.demo_ids].to(device=self.device, dtype=torch.float32)
        sample_ids = farthest_point_sample(initial_points[0].reshape(-1, 3), reward_cfg["padContact"]["surfaceSamples"])
        self.surface_samples = initial_points.index_select(1, sample_ids).contiguous()
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._refresh()

    @property
    def num_demos(self):
        return len(self.data_indices)

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.8
        self.sim = self.gym.create_sim(self.device_id, -1, gymapi.SIM_PHYSX, self.sim_params)
        plane = gymapi.PlaneParams()
        plane.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane)
        self._create_envs()
        self._gravity_frame = None
        self.base_gravity = None
        self._update_gravity()

    def _update_gravity(self):
        frame = self.gym.get_frame_count(self.sim)
        if self._gravity_frame is not None and frame - self._gravity_frame < self.gravity_update_interval:
            return
        self._gravity_frame = frame
        params = self.gym.get_sim_params(self.sim)
        if self.base_gravity is None:
            self.base_gravity = (params.gravity.x, params.gravity.y, params.gravity.z)
        decay = 1 - 1 / self.gravity_ramp_steps * min(frame, self.gravity_ramp_steps)
        ramp = self.gravity_initial_scale * decay + (1 - decay)
        params.gravity.x = self.base_gravity[0] * ramp
        params.gravity.y = self.base_gravity[1] * ramp
        params.gravity.z = self.base_gravity[2] * ramp
        self.gym.set_sim_params(self.sim, params)

    def _create_envs(self):
        env_cfg = self.cfg["env"]
        lower = gymapi.Vec3(-1.0, -1.0, 0.0)
        upper = gymapi.Vec3(1.0, 1.0, 1.0)
        table_options = gymapi.AssetOptions()
        table_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 1.0, 1.6, 0.03, table_options)
        table_pos = gymapi.Vec3(-0.1, 0, env_cfg["tableSurfaceZ"] - 0.015)
        self.table_surface_z = table_pos.z + 0.015
        base_pos = [float(v) for v in env_cfg["robotBasePos"]]
        base_yaw = float(env_cfg["robotBaseYaw"])
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(*base_pos)
        robot_pose.r = gymapi.Quat.from_euler_zyx(0, 0, base_yaw)
        self.base_pos = torch.tensor(base_pos, device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
        self.base_quat = torch.tensor(
            [0.0, 0.0, np.sin(base_yaw / 2.0), np.cos(base_yaw / 2.0)], device=self.device, dtype=torch.float32
        ).expand(self.num_envs, -1)
        world = np.eye(4)
        world[:3, :3] = aa_to_rotmat(np.array([0, 0, -np.pi / 2])) @ aa_to_rotmat(np.array([np.pi / 2, 0, 0]))
        world[:3, 3] = np.array([0, 0, self.table_surface_z])
        world_transform = torch.tensor(world, device=self.device, dtype=torch.float32)
        dataset = DemoDataset(
            data_dir=env_cfg["dataDir"],
            ik_dir=env_cfg["ikDir"],
            device=self.device,
            world_transform=world_transform,
            max_seq_len=self.max_episode_length,
            fps=env_cfg["referenceFps"],
        )

        urdf = os.path.abspath(env_cfg["robotUrdf"])
        options = gymapi.AssetOptions()
        options.thickness = 0.001
        options.angular_damping = 0.01
        options.linear_damping = 0.01
        options.max_linear_velocity = 50
        options.max_angular_velocity = 100
        options.fix_base_link = True
        options.disable_gravity = True
        options.flip_visual_attachments = False
        options.collapse_fixed_joints = False
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        options.use_mesh_materials = True
        options.use_physx_armature = True
        robot_asset = self.gym.load_asset(self.sim, *os.path.split(urdf), options)
        dof_names = list(self.gym.get_asset_dof_names(robot_asset))
        self.arm_dof_indices = [dof_names.index(n) for n in ARM_JOINTS]
        self.hand_dof_indices = [i for i, n in enumerate(dof_names) if n not in ARM_JOINTS]
        self.arm_dofs = torch.tensor(self.arm_dof_indices, device=self.device, dtype=torch.long)
        self.hand_dofs = torch.tensor(self.hand_dof_indices, device=self.device, dtype=torch.long)

        body_names = list(self.gym.get_asset_rigid_body_names(robot_asset))
        shape_ranges = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        body_shapes = {
            name: list(range(shape_ranges[i].start, shape_ranges[i].start + shape_ranges[i].count))
            for i, name in enumerate(body_names)
        }
        hand_body_set = set(HAND_BODIES) | set(PAD_BODIES)
        self.hand_shape_ids = [s for name in body_names if name in hand_body_set for s in body_shapes[name]]
        self.pad_shape_ids = [s for name in body_names if name.endswith("_elastomer") for s in body_shapes[name]]
        shape_props = self.gym.get_asset_rigid_shape_properties(robot_asset)
        for prop in shape_props:
            prop.friction = 4.0
            prop.rolling_friction = 0.01
            prop.torsion_friction = 0.01
        pad_shape_set = set(self.pad_shape_ids)
        for s in self.hand_shape_ids:
            shape_props[s].friction = env_cfg["padFriction"] if s in pad_shape_set else env_cfg["metalFriction"]
        self.gym.set_asset_rigid_shape_properties(robot_asset, shape_props)

        dof_props = self.gym.get_asset_dof_properties(robot_asset)
        for i, name in enumerate(dof_names):
            if name in HAND_JOINT_LIMITS:
                dof_props["lower"][i], dof_props["upper"][i] = HAND_JOINT_LIMITS[name]
        for i, name in enumerate(dof_names):
            dof_props["driveMode"][i] = gymapi.DOF_MODE_POS
            if name in ARM_JOINTS:
                k = ARM_JOINTS.index(name)
                dof_props["stiffness"][i] = float(ARM_STIFFNESS[k])
                dof_props["damping"][i] = float(ARM_DAMPING[k])
                dof_props["effort"][i] = float(ARM_EFFORT[k])
            else:
                (
                    dof_props["stiffness"][i],
                    dof_props["damping"][i],
                    dof_props["armature"][i],
                    dof_props["friction"][i],
                ) = HAND_JOINT_DRIVE[name]
        lower_limits = np.asarray(dof_props["lower"])
        upper_limits = np.asarray(dof_props["upper"])
        self.hand_lower = torch.tensor(lower_limits[self.hand_dof_indices].copy(), device=self.device)
        self.hand_upper = torch.tensor(upper_limits[self.hand_dof_indices].copy(), device=self.device)
        self.hand_velocity_limits = torch.tensor(
            np.asarray(dof_props["velocity"])[self.hand_dof_indices].copy(), device=self.device
        )
        self.effort_limits = torch.tensor(np.asarray(dof_props["effort"], dtype=np.float32).copy(), device=self.device)
        arm_lower = lower_limits[self.arm_dof_indices].astype(np.float64)
        arm_upper = upper_limits[self.arm_dof_indices].astype(np.float64)
        arm_lower = np.minimum(arm_lower + self.arm_joint_margin, arm_upper)
        arm_upper = np.maximum(arm_upper - self.arm_joint_margin, arm_lower)
        self.arm_lower = torch.tensor(arm_lower, device=self.device, dtype=torch.float)
        self.arm_upper = torch.tensor(arm_upper, device=self.device, dtype=torch.float)
        self.arm_velocity_limits = torch.tensor(ARM_VELOCITY_LIMITS, device=self.device, dtype=torch.float)
        self.arm_max_step = torch.tensor(
            ARM_VELOCITY_LIMITS * float(self.cfg["sim"]["dt"]), device=self.device, dtype=torch.float
        )
        self.arm_kinematics = ArmKinematics(urdf, base_pos, base_yaw, ARM_JOINTS, WRIST_BODY, self.device)
        arm_shape_ids = [
            s for name in body_names if name.startswith("fr3") or name == "base" for s in body_shapes[name]
        ]
        self_collision_masks = hand_self_collision_masks(body_names, body_shapes) if self.self_collision else {}

        demos = [dataset[index] for index in self.data_indices]
        self._build_demo_bank(demos)
        self.demo_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long) % len(demos)
        self.env_group_ids = self.demo_group_ids[self.demo_ids].clone()
        self.demo_seq_len = self.demo_bank["seq_len"][self.demo_ids]

        robot_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        robot_shapes = self.gym.get_asset_rigid_shape_count(robot_asset)
        mass_lower, mass_upper = self.cfg["env"]["physicsRandomization"]["objectMass"]
        mass_targets = self.mass_rng.uniform(mass_lower, mass_upper, size=self.num_envs).astype(np.float32)
        object_assets = {}
        asset_keys = {}
        self.envs = []
        self.robot_actors = []
        self.table_actors = []
        self.object_actors = []
        object_nominal_mass = []
        object_com = []
        num_per_row = int(np.sqrt(self.num_envs))
        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)
            demo = demos[i % len(demos)]
            urdf_path = demo["object_urdf"]
            if urdf_path not in asset_keys:
                asset_keys[urdf_path] = _asset_content_key(urdf_path)
            asset_key = asset_keys[urdf_path]
            if asset_key not in object_assets:
                object_options = gymapi.AssetOptions()
                object_options.override_com = True
                object_options.override_inertia = True
                object_options.convex_decomposition_from_submeshes = True
                object_options.thickness = 0.002
                object_options.max_linear_velocity = 1000.0
                object_options.max_angular_velocity = 64.0
                object_options.fix_base_link = False
                object_options.disable_gravity = False
                object_options.enable_gyroscopic_forces = True
                object_options.vhacd_enabled = False
                object_options.density = 1000.0
                object_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
                object_assets[asset_key] = self.gym.load_asset(self.sim, *os.path.split(urdf_path), object_options)
            object_asset = object_assets[asset_key]
            object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
            self.gym.begin_aggregate(env_ptr, robot_bodies + 1 + object_bodies + 5, robot_shapes + 1 + object_shapes + 5, True)
            robot = self.gym.create_actor(env_ptr, robot_asset, robot_pose, "robot", i, 0 if self.self_collision else 1)
            props = self.gym.get_actor_rigid_shape_properties(env_ptr, robot)
            for shape_id, mask in self_collision_masks.items():
                props[shape_id].filter = int(props[shape_id].filter) | int(mask)
            for shape_id in arm_shape_ids:
                props[shape_id].filter = int(props[shape_id].filter) | 2
            self.gym.set_actor_rigid_shape_properties(env_ptr, robot, props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot)
            self.gym.set_actor_dof_properties(env_ptr, robot, dof_props)
            table_pose = gymapi.Transform()
            table_pose.p = gymapi.Vec3(table_pos.x, table_pos.y, table_pos.z)
            table = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, 2)
            table_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table)
            table_props[0].friction = 0.1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table, table_props)
            object_actor = self._create_object(env_ptr, i, object_asset, demo["object_pose"][0])
            body_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_actor)
            inertia = body_props[0].inertia
            inertia = np.array(
                [
                    [inertia.x.x, inertia.x.y, inertia.x.z],
                    [inertia.y.x, inertia.y.y, inertia.y.z],
                    [inertia.z.x, inertia.z.y, inertia.z.z],
                ]
            )
            inertia = (0.5 * (inertia + inertia.T)).astype(np.float32, copy=True)
            body_props[0].mass = min(0.5, body_props[0].mass)
            nominal_mass = float(body_props[0].mass)
            mass = float(mass_targets[i])
            inertia = inertia * (mass / max(nominal_mass, 1e-09))
            body_props[0].mass = mass
            body_props[0].inertia.x.x, body_props[0].inertia.x.y, body_props[0].inertia.x.z = (
                float(v) for v in inertia[0]
            )
            body_props[0].inertia.y.x, body_props[0].inertia.y.y, body_props[0].inertia.y.z = (
                float(v) for v in inertia[1]
            )
            body_props[0].inertia.z.x, body_props[0].inertia.z.y, body_props[0].inertia.z.z = (
                float(v) for v in inertia[2]
            )
            self.gym.set_actor_rigid_body_properties(env_ptr, object_actor, body_props)
            object_nominal_mass.append(nominal_mass)
            object_com.append(torch.tensor([body_props[0].com.x, body_props[0].com.y, body_props[0].com.z]))
            self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.robot_actors.append(robot)
            self.table_actors.append(table)
            self.object_actors.append(object_actor)
        self.object_nominal_mass = torch.tensor(object_nominal_mass, device=self.device)
        self.object_com = torch.stack(object_com, dim=0).to(self.device)
        self._init_buffers()

    def _create_object(self, env_ptr, i, asset, pose_matrix):
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(pose_matrix[0, 3], pose_matrix[1, 3], pose_matrix[2, 3])
        axis_angle = rotmat_to_aa(pose_matrix[:3, :3])
        angle = torch.norm(axis_angle)
        axis = axis_angle / angle
        pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(axis[0], axis[1], axis[2]), angle)
        actor = self.gym.create_actor(env_ptr, asset, pose, "object", i, 0)
        props = self.gym.get_actor_rigid_shape_properties(env_ptr, actor)
        for prop in props:
            prop.friction = 0.5
            prop.restitution = 0.0
            prop.rolling_friction = 0.0
            prop.torsion_friction = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, actor, props)
        placeholder_options = gymapi.AssetOptions()
        placeholder_options.fix_base_link = True
        for k in range(5):
            placeholder = self.gym.create_box(self.sim, 0.02, 0.04, 0.06, placeholder_options)
            self.gym.create_actor(env_ptr, placeholder, gymapi.Transform(), f"placeholder_{k}", self.num_envs + 1, 1)
        return actor

    def _build_demo_bank(self, demos):
        lengths = [len(d["object_pose"]) for d in demos]
        max_len = max(lengths)

        def pad(value):
            if len(value) >= max_len:
                return value
            return torch.cat([value, value[-1:].repeat(max_len - len(value), *[1 for _ in value.shape[1:]])], dim=0)

        def stack(values, padded=True):
            return torch.stack([pad(v) if padded else v for v in values])

        bank = {"seq_len": torch.tensor(lengths, device=self.device)}
        for key in (
            "object_pose",
            "object_vel",
            "object_ang_vel",
            "wrist_pos",
            "wrist_rot",
            "wrist_vel",
            "wrist_ang_vel",
            "tips_distance",
            "arm_reset_qpos",
            "object_height_offset",
            "object_pos_tolerance",
            "reset_valid",
        ):
            bank[key] = stack([d[key] for d in demos])
        bank["joints_pos"] = stack([torch.cat([d["keypoints"][k] for k in REFERENCE_KEYPOINTS], dim=-1) for d in demos])
        bank["joints_vel"] = stack(
            [torch.cat([d["keypoints_vel"][k] for k in REFERENCE_KEYPOINTS], dim=-1) for d in demos]
        )
        bank["object_points"] = stack([d["object_points"] for d in demos], padded=False)
        self.demo_bank = bank
        group_refs = []
        group_members = []
        group_ids = []
        for i, points in enumerate(bank["object_points"]):
            group = next(
                (
                    g
                    for g, ref in enumerate(group_refs)
                    if ref.shape == points.shape and torch.allclose(points, ref, rtol=1e-05, atol=1e-05)
                ),
                None,
            )
            if group is None:
                group = len(group_refs)
                group_refs.append(points)
                group_members.append([])
            group_members[group].append(i)
            group_ids.append(group)
        self.demo_group_ids = torch.tensor(group_ids, device=self.device, dtype=torch.long)
        self.demo_group_members = [torch.tensor(m, device=self.device, dtype=torch.long) for m in group_members]
        bottom = torch.stack([pad(d["object_bottom_z"]) for d in demos])
        self.floor_lift = (self.table_surface_z - bottom).clamp_min(0.0)
        self.floor_lift_vel = torch.zeros_like(self.floor_lift)
        for row, length in enumerate(lengths):
            if length == 1:
                continue
            lift = self.floor_lift[row]
            self.floor_lift_vel[row, 0] = (lift[1] - lift[0]) * self.floor_lift_fps
            self.floor_lift_vel[row, length - 1] = (lift[length - 1] - lift[length - 2]) * self.floor_lift_fps
            if length > 2:
                self.floor_lift_vel[row, 1 : length - 1] = (
                    0.5 * self.floor_lift_fps * (lift[2:length] - lift[: length - 2])
                )

    def _init_buffers(self):
        env_ptr = self.envs[0]
        robot = self.robot_actors[0]
        self.body_handles = {
            name: self.gym.find_actor_rigid_body_handle(env_ptr, robot, name) for name in HAND_BODIES + PAD_BODIES
        }
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self._root_state = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).view(
            self.num_envs, -1, 13
        )
        self._dof_state = gymtorch.wrap_tensor(self.gym.acquire_dof_state_tensor(self.sim)).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(self.gym.acquire_rigid_body_state_tensor(self.sim)).view(
            self.num_envs, -1, 13
        )
        self._net_contact_force = gymtorch.wrap_tensor(self.gym.acquire_net_contact_force_tensor(self.sim)).view(
            self.num_envs, -1, 3
        )
        self._dof_force = gymtorch.wrap_tensor(self.gym.acquire_dof_force_tensor(self.sim)).view(self.num_envs, -1)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._wrist_state = self._rigid_body_state[:, self.body_handles[WRIST_BODY], :]
        self._object_state = self._root_state[:, self.object_actors[0], :]
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.pos_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.last_arm_command = torch.zeros((self.num_envs, len(ARM_JOINTS)), dtype=torch.float, device=self.device)
        self.last_hand_command = torch.zeros((self.num_envs, NUM_HAND_DOFS), dtype=torch.float, device=self.device)
        self.command_history_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.arm_action_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.hand_action_rate = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.action_history = torch.zeros((2, self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.action_history_ptr = 0
        self.action_history_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.just_reset = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.reset_wrist_state = torch.zeros(self.num_envs, 13, device=self.device)
        self.open_hand_joints = None
        self.robot_indices = torch.tensor(
            [self.gym.find_actor_index(env, "robot", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.device,
        )
        self.object_indices = torch.tensor(
            [self.gym.find_actor_index(env, "object", gymapi.DOMAIN_SIM) for env in self.envs],
            dtype=torch.int32,
            device=self.device,
        )
        self.tip_contact_history = torch.ones(self.num_envs, 3, len(PAD_BODIES), device=self.device).bool()
        self.states = {}

    def _demo_value(self, key, env_ids, frame=None):
        demo_ids = self.demo_ids[env_ids]
        if frame is None:
            return self.demo_bank[key][demo_ids]
        while demo_ids.ndim < frame.ndim:
            demo_ids = demo_ids.unsqueeze(-1)
        return self.demo_bank[key][demo_ids, frame]

    def _floor_lift_at(self, env_ids, frame):
        demo_ids = self.demo_ids[env_ids]
        while demo_ids.ndim < frame.ndim:
            demo_ids = demo_ids.unsqueeze(-1)
        return self.floor_lift[demo_ids, frame], self.floor_lift_vel[demo_ids, frame]

    def _body_positions(self, names):
        return torch.stack([self._rigid_body_state[:, self.body_handles[name], :3] for name in names], dim=1)

    def _update_states(self):
        self.states.update(
            {
                "q": self._q,
                "cos_q": torch.cos(self._q),
                "sin_q": torch.sin(self._q),
                "dq": self._qd,
                "wrist_state": self._wrist_state,
            }
        )
        self.states["body_state"] = torch.stack(
            [self._rigid_body_state[:, self.body_handles[name], :][:, :10] for name in HAND_BODIES], dim=1
        )
        if torch.any(self.just_reset):
            self._patch_reset_body_state()
        self.states.update(
            {
                "object_pos": self._object_state[:, :3],
                "object_quat": self._object_state[:, 3:7],
                "object_vel": self._object_state[:, 7:10],
                "object_ang_vel": self._object_state[:, 10:],
            }
        )

    def _patch_reset_body_state(self):
        env_ids = self.just_reset.nonzero(as_tuple=False).flatten()
        body_state = self.states["body_state"]
        reset_wrist = self.reset_wrist_state[env_ids]
        if self.open_hand_joints is None:
            open_joints = torch.stack(
                [self._rigid_body_state[0, self.body_handles[k], :3] for k in HAND_BODIES[1:]], dim=0
            )
            self.open_hand_joints = quat_rotate_inverse(
                self._wrist_state[0, 3:7].unsqueeze(0).expand(len(open_joints), -1),
                open_joints - self._wrist_state[0, :3],
            )
        wrist_state = self.states["wrist_state"].clone()
        wrist_state[env_ids] = reset_wrist
        self.states["wrist_state"] = wrist_state
        body_state[env_ids, 0, :10] = reset_wrist[:, :10]
        local = self.open_hand_joints.unsqueeze(0).expand(len(env_ids), -1, -1)
        wrist_quat = reset_wrist[:, None, 3:7].expand(-1, local.shape[1], -1)
        joints = reset_wrist[:, None, :3] + quat_rotate(wrist_quat.reshape(-1, 4), local.reshape(-1, 3)).reshape_as(
            local
        )
        body_state[env_ids, 1:, :3] = joints
        body_state[env_ids, 1:, 7:10] = torch.zeros_like(joints)

    def _refresh(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self._update_states()

    def _tighten_scale(self):
        if self.tighten_schedule == "const":
            return self.tighten_factor
        frame = self.gym.get_frame_count(self.sim)
        return (np.e * 2) ** (-frame / self.tighten_steps) * (1 - self.tighten_factor) + self.tighten_factor

    def _pad_contacts(self, tip_force):
        probes = torch.stack((self._body_positions(PAD_BODIES), self._body_positions(FINGERTIP_BODIES)), dim=2)
        obj_pos = self.states["object_pos"]
        obj_quat = self.states["object_quat"]
        fingers, per_finger = probes.shape[1], probes.shape[2]
        local = quat_rotate_inverse(
            obj_quat[:, None, None, :].expand(-1, fingers, per_finger, -1).reshape(-1, 4),
            (probes - obj_pos[:, None, None, :]).reshape(-1, 3),
        ).reshape(self.num_envs, fingers * per_finger, 3)
        distance = nearest_surface_distance(local, self.surface_samples).reshape(self.num_envs, fingers, per_finger)
        distance = torch.min(distance, dim=-1)[0]
        pad_cfg = self.reward_cfg["padContact"]
        contact = (torch.norm(tip_force, dim=-1) > pad_cfg["forceThreshold"]) & (distance <= pad_cfg["distance"])
        return contact, distance

    def compute_reward(self):
        reward_cfg = self.reward_cfg
        env_ids = torch.arange(self.num_envs, device=self.device)
        frame = self.progress_buf
        free_distance = reward_cfg["freeSpaceDistance"]
        reference_tips = self._demo_value("tips_distance", env_ids, torch.minimum(frame, self.demo_seq_len.long() - 1))
        fingers_free = reference_tips > free_distance
        lift, lift_vel = self._floor_lift_at(env_ids, frame)
        targets = {"reference_fingers_free": fingers_free, "reference_hand_free": torch.all(fingers_free, dim=-1)}
        wrist_pos = self._demo_value("wrist_pos", env_ids, frame).clone()
        wrist_pos[:, 2] += lift
        targets["wrist_pos"] = wrist_pos
        targets["wrist_quat"] = aa_to_quat(self._demo_value("wrist_rot", env_ids, frame))[:, [1, 2, 3, 0]]
        wrist_vel = self._demo_value("wrist_vel", env_ids, frame).clone()
        wrist_vel[:, 2] += lift_vel
        targets["wrist_vel"] = wrist_vel
        targets["wrist_ang_vel"] = self._demo_value("wrist_ang_vel", env_ids, frame)
        targets["tips_distance"] = self._demo_value("tips_distance", env_ids, frame)
        joints_pos = self._demo_value("joints_pos", env_ids, frame).reshape(self.num_envs, -1, 3).clone()
        joints_pos[..., 2] += lift[:, None]
        joints_vel = self._demo_value("joints_vel", env_ids, frame).reshape(self.num_envs, -1, 3).clone()
        joints_vel[..., 2] += lift_vel[:, None]
        targets["joints_pos"] = joints_pos
        targets["joints_vel"] = joints_vel
        obj_pose = self._demo_value("object_pose", env_ids, frame)
        obj_pos = obj_pose[:, :3, 3].clone()
        obj_pos[..., 2] = obj_pose[:, :3, 3][..., 2] + lift
        targets["object_pos"] = obj_pos
        targets["object_quat"] = rotmat_to_quat(obj_pose[:, :3, :3])[:, [1, 2, 3, 0]]
        obj_vel = self._demo_value("object_vel", env_ids, frame).clone()
        obj_vel[..., 2] = obj_vel[..., 2] + lift_vel
        targets["object_vel"] = obj_vel
        targets["object_ang_vel"] = self._demo_value("object_ang_vel", env_ids, frame)
        targets["object_pos_tolerance"] = self._demo_value("object_pos_tolerance", env_ids, frame)
        tip_force = torch.stack([self._net_contact_force[:, self.body_handles[k], :] for k in PAD_BODIES], axis=1)
        self.tip_contact_history = torch.concat(
            [self.tip_contact_history[:, 1:], (torch.norm(tip_force, dim=-1) > 0)[:, None]], dim=1
        )
        targets["tip_contact_history"] = self.tip_contact_history
        pad_contact, pad_distance = self._pad_contacts(tip_force)
        targets["pad_object_contact"] = pad_contact
        targets["power"] = torch.abs(self._dof_force[:, self.hand_dofs] * self.states["dq"][:, self.hand_dofs]).sum(
            dim=-1
        )
        scale_factor = self._tighten_scale()
        obj_rest_z = self._demo_value(
            "object_pose", env_ids, torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        )[:, 2, 3]
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.success_buf[:],
            self.failure_buf[:],
            self.reward_terms,
            self.error_buf[:],
        ) = compute_tracking_reward(
            self.reset_buf,
            self.progress_buf,
            self.running_progress_buf,
            self.states,
            targets,
            torch.clip(self.demo_seq_len, 0, self.max_episode_length).float(),
            scale_factor,
            self.failure_tolerance,
            KEYPOINT_GROUPS,
            obj_rest_z,
            reward_cfg["lift"],
            reward_cfg["liftHeight"],
            reward_cfg["wristPos"],
            reward_cfg["level1"],
            reward_cfg["level2"],
            reward_cfg["wristRot"],
            reward_cfg["wristRotSharpness"],
            reward_cfg["freeSpaceFingerScale"],
            reward_cfg["contactFingerScale"],
            reward_cfg["objectContactDistance"],
        )

        contact_cfg = reward_cfg["multiFingerContact"]
        contact_gate = (
            (self.running_progress_buf >= contact_cfg["warmupSteps"])
            & torch.any(pad_contact, dim=-1)
            & (~fingers_free).any(dim=-1)
        )
        contact_count = torch.sum(pad_contact.long(), dim=-1)
        contact_reward = torch.zeros_like(contact_count, dtype=torch.float32)
        contact_reward = torch.where(
            contact_count == 4, torch.full_like(contact_reward, contact_cfg["four"]), contact_reward
        )
        contact_reward = torch.where(
            contact_count == 5, torch.full_like(contact_reward, contact_cfg["five"]), contact_reward
        )
        contact_reward = contact_reward * contact_gate.to(dtype=contact_reward.dtype) * contact_cfg["scale"]
        self.rew_buf[:] = self.rew_buf + contact_reward

        hand_cfg = reward_cfg["handVelocity"]
        hand_fraction = velocity_fraction(self.states["dq"][:, self.hand_dofs], self.hand_velocity_limits)
        hand_l2 = torch.square(hand_fraction / hand_cfg["referenceFraction"]).sum(dim=-1)
        hand_l2 = torch.where(self.running_progress_buf >= hand_cfg["warmupSteps"], hand_l2, torch.zeros_like(hand_l2))
        hand_violation = hand_fraction.amax(dim=-1) > hand_cfg["terminateFraction"]
        if self.hand_velocity_gate_unlocked:
            hand_violation = hand_violation & (self.running_progress_buf >= hand_cfg["terminateWarmupSteps"])
        else:
            hand_violation = torch.zeros_like(hand_violation)
        self.rew_buf[:] = self.rew_buf + -torch.clamp(hand_cfg["scale"] * hand_l2, max=hand_cfg["max"])
        self.reset_buf[hand_violation] = 1
        self.failure_buf[hand_violation] = 1
        self.success_buf[hand_violation] = 0

        arm_cfg = reward_cfg["armVelocity"]
        arm_active = self.running_progress_buf >= arm_cfg["warmupSteps"]
        arm_fraction = velocity_fraction(self.states["dq"][:, self.arm_dofs], self.arm_velocity_limits)
        arm_l2 = torch.square(arm_fraction / arm_cfg["referenceFraction"]).sum(dim=-1)
        arm_l2 = torch.where(arm_active, arm_l2, torch.zeros_like(arm_l2))
        arm_violation = (arm_fraction.amax(dim=-1) > arm_cfg["terminateFraction"]) & arm_active
        self.rew_buf[:] = self.rew_buf + -torch.clamp(arm_cfg["scale"] * arm_l2, max=arm_cfg["max"])
        self.reset_buf[arm_violation] = 1
        self.failure_buf[arm_violation] = 1
        self.success_buf[arm_violation] = 0

        self.rew_buf[:] = self.rew_buf + -reward_cfg["armActionRate"] * self.arm_action_rate
        self.rew_buf[:] = self.rew_buf + -reward_cfg["handActionRate"] * self.hand_action_rate
        torque_cfg = reward_cfg["torque"]
        self.rew_buf[:] = self.rew_buf + torque_penalty(
            self._dof_force[:, :NUM_ACTOR_DOFS],
            self.effort_limits,
            torque_cfg["deadzone"],
            torque_cfg["scale"],
            torque_cfg["max"],
        )
        pad_cfg = reward_cfg["padDistance"]
        reference_contacts = (~fingers_free).sum(dim=-1)
        pad_active = (reference_contacts >= pad_cfg["minReferenceContacts"]) & (
            self.running_progress_buf >= pad_cfg["warmupSteps"]
        )
        self.rew_buf[:] = self.rew_buf + pad_cfg["scale"] * pad_distance_score(
            pad_distance, pad_active, pad_cfg["tolerance"], pad_cfg["sigma"]
        )
        if self.record_done_reasons:
            self._record_done_reasons(targets, scale_factor, hand_violation, arm_violation, hand_fraction, arm_fraction)
        self.total_rew_buf += self.rew_buf

    def done_reason_schema(self):
        return DONE_REASONS, DONE_METRICS

    def _record_done_reasons(self, targets, scale_factor, hand_violation, arm_violation, hand_fraction, arm_fraction):
        done = self.reset_buf.bool()
        failure = done & self.failure_buf.bool()
        success = done & self.success_buf.bool()
        obj_pos_error = torch.linalg.vector_norm(targets["object_pos"] - self.states["object_pos"], dim=-1)
        obj_pos_threshold = (
            0.02 / 0.343 * float(scale_factor) ** 3 * self.failure_tolerance + targets["object_pos_tolerance"]
        )
        obj_rot_error = 2.0 * torch.acos(
            torch.sum(self.states["object_quat"] * targets["object_quat"], dim=-1).abs().clamp(0.0, 1.0)
        )
        obj_rot_threshold = 30.0 / 0.343 * float(scale_factor) ** 3 * np.pi / 180.0
        gate = self.running_progress_buf >= 8
        obj_pos_failure = (obj_pos_error > obj_pos_threshold) & gate & failure
        obj_rot_failure = (obj_rot_error > obj_rot_threshold) & gate & failure
        joint_errors = torch.linalg.vector_norm(targets["joints_pos"] - self.states["body_state"][:, 1:, :3], dim=-1)

        def group_error(name):
            return joint_errors[:, [k - 1 for k in KEYPOINT_GROUPS[name]]].mean(dim=-1)

        hand_conditions = (
            group_error("thumb_tip") > 0.04 / 0.7 * float(scale_factor) * self.failure_tolerance,
            group_error("index_tip") > 0.045 / 0.7 * float(scale_factor) * self.failure_tolerance,
            group_error("middle_tip") > 0.05 / 0.7 * float(scale_factor) * self.failure_tolerance,
            group_error("ring_tip") > 0.06 / 0.7 * float(scale_factor) * self.failure_tolerance,
            group_error("pinky_tip") > 0.06 / 0.7 * float(scale_factor) * self.failure_tolerance,
            group_error("level_1_joints") > 0.07 / 0.7 * float(scale_factor) * self.failure_tolerance,
            group_error("level_2_joints") > 0.08 / 0.7 * float(scale_factor) * self.failure_tolerance,
        )
        hand_failures = tuple(condition & gate & failure for condition in hand_conditions)
        none = torch.zeros_like(done)
        numerical = self.error_buf.bool() & failure
        arm_limit = arm_violation & failure
        hand_limit = hand_violation & failure
        known = obj_pos_failure | obj_rot_failure | numerical | arm_limit | hand_limit
        for hand_failure in hand_failures:
            known |= hand_failure
        masks = (
            success,
            obj_pos_failure,
            obj_rot_failure,
            *hand_failures,
            none,
            numerical,
            arm_limit,
            hand_limit,
            failure & ~known,
            none,
        )
        bits = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        for bit, mask in enumerate(masks):
            bits |= mask.to(torch.int64) << bit
        wrist_rot_error = torch.abs(
            quat_to_angle_axis(quat_mul(targets["wrist_quat"], quat_conjugate(self.states["wrist_state"][:, 3:7])))[0]
        )
        wrist_pos_error = torch.linalg.vector_norm(targets["wrist_pos"] - self.states["wrist_state"][:, :3], dim=-1)
        self.done_reason_bits = bits
        self.done_metrics = torch.stack(
            (
                self.progress_buf.to(obj_pos_error.dtype),
                wrist_pos_error,
                wrist_rot_error * (180.0 / np.pi),
                obj_pos_error,
                obj_rot_error * (180.0 / np.pi),
                arm_fraction.amax(dim=-1),
                hand_fraction.amax(dim=-1),
            ),
            dim=-1,
        )

    def update_hand_velocity_gate(self, success_rate, epoch, successful_episodes, total_episodes):
        self.hand_velocity_gate_last_success = float(success_rate)
        unlocked = (
            self.hand_velocity_gate_unlocked
            or float(successful_episodes) / float(total_episodes) > self.hand_velocity_gate_threshold
        )
        newly_unlocked = unlocked and not self.hand_velocity_gate_unlocked
        self.hand_velocity_gate_unlocked = unlocked
        if newly_unlocked:
            self.hand_velocity_gate_epoch = int(epoch)
        return newly_unlocked

    def get_env_state(self):
        return {
            "action_mask_control_step": int(self.action_mask_step),
            "hand_velocity_terminate_unlocked": bool(self.hand_velocity_gate_unlocked),
            "hand_velocity_terminate_unlock_epoch": int(self.hand_velocity_gate_epoch),
            "hand_velocity_terminate_last_eval_success": float(self.hand_velocity_gate_last_success),
        }

    def set_env_state(self, env_state):
        if not isinstance(env_state, dict):
            return
        if "action_mask_control_step" in env_state:
            self.action_mask_step = max(0, int(env_state["action_mask_control_step"]))
        if "hand_velocity_terminate_unlocked" in env_state:
            self.hand_velocity_gate_unlocked = bool(env_state["hand_velocity_terminate_unlocked"])
        if "hand_velocity_terminate_unlock_epoch" in env_state:
            self.hand_velocity_gate_epoch = int(env_state["hand_velocity_terminate_unlock_epoch"])
        if "hand_velocity_terminate_last_eval_success" in env_state:
            value = float(env_state["hand_velocity_terminate_last_eval_success"])
            if math.isfinite(value) and 0.0 <= value <= 1.0:
                self.hand_velocity_gate_last_success = value

    def compute_observations(self):
        self._refresh()
        wrist_state = self.states["wrist_state"]
        self.obs_dict["proprioception"][:] = torch.cat(
            [
                self.states["q"],
                self.states["cos_q"],
                self.states["sin_q"],
                torch.cat([torch.zeros_like(wrist_state[:, :3]), wrist_state[:, 3:]], dim=-1),
                self.last_actions,
            ],
            dim=-1,
        )
        obj_pos = self.states["object_pos"]
        obj_quat = self.states["object_quat"]
        noisy_obj_pos = obj_pos + torch.randn_like(obj_pos) * self.object_pos_noise
        sigma = float(np.radians(self.object_rot_noise)) * 0.5
        noisy_obj_quat = obj_quat + torch.randn_like(obj_quat) * sigma
        noisy_obj_quat = noisy_obj_quat / noisy_obj_quat.norm(dim=-1, keepdim=True).clamp_min(1e-08)
        tip_force = torch.stack([self._net_contact_force[:, self.body_handles[k], :] for k in PAD_BODIES], axis=1)
        tip_force = torch.cat([tip_force, torch.norm(tip_force, dim=-1, keepdim=True)], dim=-1)
        com = (quat_to_rotmat(obj_quat[:, [1, 2, 3, 0]]) @ self.object_com.unsqueeze(-1)).squeeze(-1) + obj_pos
        gravity_z = self.gym.get_sim_params(self.sim).gravity.z
        self.obs_dict["privileged"][:] = torch.cat(
            [
                self.states["dq"],
                obj_pos - wrist_state[:, :3],
                obj_quat,
                self.states["object_vel"],
                self.states["object_ang_vel"],
                tip_force.reshape(self.num_envs, -1),
                com - wrist_state[:, :3],
                (self.object_nominal_mass * -1 * gravity_z).unsqueeze(-1),
            ],
            dim=-1,
        )

        num_envs = self.num_envs
        env_ids = torch.arange(num_envs, device=self.device)
        frame = torch.clamp(self.progress_buf + 1, torch.zeros_like(self.demo_seq_len), self.demo_seq_len - 1)[:, None]
        lift, lift_vel = self._floor_lift_at(env_ids, frame)
        target_wrist_pos = self._demo_value("wrist_pos", env_ids, frame).clone()
        target_wrist_pos[..., 2] += lift
        wrist_pos = wrist_state[:, :3]
        wrist_quat = wrist_state[:, 3:7]
        target_wrist_vel = self._demo_value("wrist_vel", env_ids, frame).clone()
        target_wrist_vel[..., 2] += lift_vel
        target_wrist_quat = aa_to_quat(self._demo_value("wrist_rot", env_ids, frame).reshape(num_envs, -1))[
            :, [1, 2, 3, 0]
        ]
        target_wrist_ang_vel = self._demo_value("wrist_ang_vel", env_ids, frame)
        target_joints_pos = self._demo_value("joints_pos", env_ids, frame).reshape(num_envs, 1, -1, 3).clone()
        target_joints_pos[..., 2] += lift[..., None]
        target_joints_vel = self._demo_value("joints_vel", env_ids, frame).reshape(num_envs, 1, -1, 3).clone()
        target_joints_vel[..., 2] += lift_vel[..., None]
        target_obj_pose = self._demo_value("object_pose", env_ids, frame).reshape(num_envs, 4, 4)
        target_obj_pos = target_obj_pose[:, :3, 3].reshape(num_envs, 1, 3).clone()
        target_obj_pos[..., 2] = target_obj_pos[..., 2] + lift
        target_obj_vel = self._demo_value("object_vel", env_ids, frame).clone()
        target_obj_vel[..., 2] = target_obj_vel[..., 2] + lift_vel
        target_obj_quat = rotmat_to_quat(target_obj_pose[:, :3, :3])[:, [1, 2, 3, 0]]
        target_obj_ang_vel = self._demo_value("object_ang_vel", env_ids, frame).reshape(num_envs, 3)
        joints = self.states["body_state"]
        tips_distance = self._demo_value("tips_distance", env_ids, frame).reshape(num_envs, -1)
        self.obs_dict["target"][:] = torch.cat(
            [
                (target_wrist_pos - wrist_pos[:, None]).reshape(num_envs, -1),
                target_wrist_vel.reshape(num_envs, -1),
                (target_wrist_vel - wrist_state[:, None, 7:10]).reshape(num_envs, -1),
                target_wrist_quat,
                quat_mul(wrist_quat, quat_conjugate(target_wrist_quat)),
                target_wrist_ang_vel.reshape(num_envs, -1),
                (target_wrist_ang_vel - wrist_state[:, None, 10:13]).reshape(num_envs, -1),
                (target_joints_pos - joints[:, None, 1:, :3]).reshape(num_envs, -1),
                target_joints_vel.reshape(num_envs, -1),
                (target_joints_vel - joints[:, None, 1:, 7:10]).reshape(num_envs, -1),
                (target_obj_pos - obj_pos[:, None]).reshape(num_envs, -1),
                target_obj_vel.reshape(num_envs, -1),
                (target_obj_vel - self.states["object_vel"][:, None]).reshape(num_envs, -1),
                target_obj_quat,
                quat_mul(obj_quat, quat_conjugate(target_obj_quat)),
                target_obj_ang_vel,
                target_obj_ang_vel - self.states["object_ang_vel"],
                torch.norm(obj_pos[:, None] - joints[:, :, :3], dim=-1).reshape(num_envs, -1),
                tips_distance,
                self.shape_code,
            ],
            dim=-1,
        )

        base_quat = self.base_quat
        pad_pos = self._body_positions(PAD_BODIES)
        pad_pos_wrist = quat_rotate_inverse(
            wrist_quat[:, None, :].expand(-1, pad_pos.shape[1], -1).reshape(-1, 4),
            (pad_pos - wrist_pos[:, None, :]).reshape(-1, 3),
        ).reshape(num_envs, -1)
        wrist_pos_base = position_to_base(wrist_pos, self.base_pos, base_quat)
        wrist_quat_base = quaternion_to_base(wrist_quat, base_quat)
        target_wrist_quat_base = quaternion_to_base(target_wrist_quat.reshape(num_envs, 1, 4), base_quat)
        obj_pos_base = position_to_base(noisy_obj_pos, self.base_pos, base_quat)
        obj_quat_base = quaternion_to_base(noisy_obj_quat, base_quat)
        target_obj_quat_base = quaternion_to_base(target_obj_quat.reshape(num_envs, 1, 4), base_quat)
        self.obs_dict["actor"][:] = torch.cat(
            [
                self.states["q"],
                self.last_actions,
                wrist_pos_base,
                wrist_quat_base,
                pad_pos_wrist,
                quat_rotate_inverse(wrist_quat, noisy_obj_pos - wrist_pos),
                quat_mul(quat_conjugate(wrist_quat), noisy_obj_quat),
                (position_to_base(target_wrist_pos, self.base_pos, base_quat) - wrist_pos_base[:, None]).reshape(
                    num_envs, -1
                ),
                target_wrist_quat_base.reshape(num_envs, -1),
                quat_mul(wrist_quat_base, quat_conjugate(target_wrist_quat_base.reshape(num_envs, 4))),
                vector_to_base(target_joints_pos - joints[:, None, 1:, :3], base_quat).reshape(num_envs, -1),
                (position_to_base(target_obj_pos, self.base_pos, base_quat) - obj_pos_base[:, None, :]).reshape(
                    num_envs, -1
                ),
                target_obj_quat_base.reshape(num_envs, -1),
                quat_mul(obj_quat_base, quat_conjugate(target_obj_quat_base.reshape(num_envs, 4))),
                tips_distance,
                self.shape_code,
            ],
            dim=-1,
        )
        return self.obs_dict

    def _sample_demo_ids(self, env_ids):
        demo_ids = torch.empty(len(env_ids), device=self.device, dtype=torch.long)
        env_groups = self.env_group_ids[env_ids]
        for group in torch.unique(env_groups):
            env_mask = env_groups == group
            members = self.demo_group_members[int(group.item())]
            count = int(env_mask.sum().item())
            demo_ids[env_mask] = members[torch.randint(0, len(members), (count,), device=self.device)]
        return demo_ids

    def _sample_reset_frames(self, env_ids):
        seq_len = self.demo_seq_len[env_ids].long()
        if not self.random_state_init:
            return torch.zeros_like(seq_len)
        valid = self._demo_value("reset_valid", env_ids).bool()
        frames = torch.arange(valid.shape[1], device=self.device).unsqueeze(0)
        upper = torch.floor(seq_len.float() * float(self.reset_frame_fraction)).long().clamp(min=1)
        allowed = valid & (frames < upper[:, None])
        counts = allowed.sum(dim=1)
        rank = torch.floor(torch.rand_like(counts.float()) * counts.float()).long()
        return ((allowed.long().cumsum(dim=1) > rank[:, None]) & allowed).long().argmax(dim=1)

    def reset_idx(self, env_ids):
        self._refresh()
        self._update_gravity()
        self.randomizer.resample(env_ids)
        demo_ids = self._sample_demo_ids(env_ids)
        self.demo_seq_len[env_ids] = self.demo_bank["seq_len"][demo_ids].to(self.demo_seq_len.device)
        self.demo_ids[env_ids] = demo_ids
        self.shape_code[env_ids] = self.demo_shape_code[demo_ids].to(self.shape_code.device)
        frames = self._sample_reset_frames(env_ids)
        dof_pos = torch.zeros((len(env_ids), self.num_dofs), device=self.device)
        arm_q = tensor_clamp(
            self._demo_value("arm_reset_qpos", env_ids, frames),
            self.arm_lower.unsqueeze(0),
            self.arm_upper.unsqueeze(0),
        )
        dof_pos[:, self.arm_dofs] = arm_q
        self._q[env_ids, :] = dof_pos
        self._qd[env_ids, :] = torch.zeros_like(dof_pos)
        self.pos_targets[env_ids, :] = dof_pos
        obj_pose = self._demo_value("object_pose", env_ids, frames)
        self._object_state[env_ids, :3] = obj_pose[:, :3, 3]
        self._object_state[env_ids, 3:7] = rotmat_to_quat(obj_pose[:, :3, :3])[:, [1, 2, 3, 0]]
        self._object_state[env_ids, 7:10] = self._demo_value("object_vel", env_ids, frames)
        self._object_state[env_ids, 10:13] = self._demo_value("object_ang_vel", env_ids, frames)
        self._object_state[env_ids, 2] += self._demo_value("object_height_offset", env_ids, frames)
        wrist_tf = self.arm_kinematics.forward(arm_q)
        self.reset_wrist_state[env_ids] = 0
        self.reset_wrist_state[env_ids, :3] = wrist_tf[:, :3, 3]
        self.reset_wrist_state[env_ids, 3:7] = rotmat_to_quat(wrist_tf[:, :3, :3])[:, [1, 2, 3, 0]]
        robot_indices = self.robot_indices[env_ids].flatten()
        object_indices = self.object_indices[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._dof_state), gymtorch.unwrap_tensor(robot_indices), len(robot_indices)
        )
        root_indices = torch.concat([robot_indices, object_indices])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state), gymtorch.unwrap_tensor(root_indices), len(root_indices)
        )
        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.pos_targets),
            gymtorch.unwrap_tensor(robot_indices),
            len(robot_indices),
        )
        self.progress_buf[env_ids] = frames
        self.mask_remaining[env_ids] = 0
        self.mask_joints[env_ids] = False
        self.mask_previous_command[env_ids] = 0.0
        self.just_reset[env_ids] = True
        self.running_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.success_buf[env_ids] = 0
        self.failure_buf[env_ids] = 0
        self.error_buf[env_ids] = 0
        self.total_rew_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_arm_command[env_ids] = 0.0
        self.last_hand_command[env_ids] = 0.0
        self.command_history_valid[env_ids] = False
        self.arm_action_rate[env_ids] = 0.0
        self.hand_action_rate[env_ids] = 0.0
        self.prev_targets[env_ids] = dof_pos
        self.action_history_reset[env_ids] = True
        self.tip_contact_history[env_ids] = torch.ones_like(self.tip_contact_history[env_ids]).bool()

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.compute_observations()

    def reset_done(self):
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)
            self.compute_observations()
        return self.obs_dict, done_env_ids

    def _mask_max_duration(self):
        fraction = min(max(float(self.action_mask_step) / float(self.action_mask_curriculum_steps), 0.0), 1.0)
        sigma = 1.0 - (1.0 - float(self.action_mask_sigma_min)) * fraction
        progress = (1.0 - sigma ** 3) / (1.0 - float(self.action_mask_sigma_min) ** 3)
        upper = (
            float(self.action_mask_start_steps)
            + (float(self.action_mask_end_steps) - float(self.action_mask_start_steps)) * progress
        )
        return max(1, int(round(upper)))

    def _mask_hand_targets(self, hand_targets):
        if self.eval_mode:
            self.mask_previous_command.copy_(hand_targets)
            return hand_targets
        start = (self.mask_remaining <= 0) & (
            torch.rand(self.num_envs, device=self.device) < self.action_mask_probability
        )
        if torch.any(start):
            start_ids = start.nonzero(as_tuple=False).flatten()
            count = start_ids.numel()
            joints = (
                torch.rand((count, NUM_HAND_DOFS), device=self.device)
                .topk(self.action_mask_joints, dim=-1, largest=False)
                .indices
            )
            masks = torch.zeros((count, NUM_HAND_DOFS), dtype=torch.bool, device=self.device)
            masks.scatter_(1, joints, True)
            self.mask_joints[start_ids] = masks
            self.mask_remaining[start_ids] = torch.randint(
                1, self._mask_max_duration() + 1, (count,), dtype=torch.long, device=self.device
            )
        active = self.mask_remaining > 0
        executed = torch.where(active[:, None] & self.mask_joints, self.mask_previous_command, hand_targets)
        self.mask_remaining.copy_(torch.clamp(self.mask_remaining - active.long(), min=0))
        self.mask_previous_command.copy_(executed)
        return executed

    def _delay_actions(self, actions):
        if torch.any(self.action_history_reset):
            self.action_history[:, self.action_history_reset] = actions[self.action_history_reset].detach().unsqueeze(0)
            self.action_history_reset[:] = False
        self.action_history_ptr = (self.action_history_ptr + 1) % 2
        self.action_history[self.action_history_ptr] = actions.detach()
        delay = torch.randint(0, 2, (actions.shape[0],), device=actions.device)
        return self.action_history[
            (self.action_history_ptr - delay) % 2, torch.arange(actions.shape[0], device=actions.device)
        ]

    def pre_physics_step(self, actions):
        self.just_reset[:] = False
        self.last_actions.copy_(actions.detach())
        arm_command = actions[:, : len(ARM_JOINTS)].detach()
        hand_command = torch.clamp(actions[:, len(ARM_JOINTS) :], -1.0, 1.0).detach()
        self.arm_action_rate.copy_(action_rate(arm_command, self.last_arm_command, self.command_history_valid))
        self.hand_action_rate.copy_(action_rate(hand_command, self.last_hand_command, self.command_history_valid))
        self.last_arm_command.copy_(arm_command)
        self.last_hand_command.copy_(hand_command)
        self.command_history_valid.fill_(True)
        actions = self._delay_actions(actions)
        prev_hand = self.prev_targets[:, self.hand_dofs]
        prev_arm = self.prev_targets[:, self.arm_dofs]
        hand_targets = scale(torch.clamp(actions[:, len(ARM_JOINTS) :], -1.0, 1.0), self.hand_lower, self.hand_upper)
        hand_targets = self.hand_smoothing * hand_targets + (1.0 - self.hand_smoothing) * prev_hand
        delta = hand_targets - prev_hand
        hand_targets = prev_hand + torch.sign(delta) * torch.clamp(
            torch.abs(delta) - self.hand_target_deadband, min=0.0
        )
        hand_targets = self._mask_hand_targets(tensor_clamp(hand_targets, self.hand_lower, self.hand_upper))
        arm_targets = self._q[:, self.arm_dofs] + actions[:, : len(ARM_JOINTS)] * self.arm_action_scale
        arm_targets = self.arm_smoothing * arm_targets + (1.0 - self.arm_smoothing) * prev_arm
        arm_targets = prev_arm + torch.clamp(arm_targets - prev_arm, -self.arm_max_step, self.arm_max_step)
        arm_targets = tensor_clamp(arm_targets, self.arm_lower, self.arm_upper)
        self.pos_targets[:, self.hand_dofs] = hand_targets
        self.pos_targets[:, self.arm_dofs] = arm_targets
        self.prev_targets[:, self.hand_dofs] = hand_targets
        self.prev_targets[:, self.arm_dofs] = arm_targets
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.pos_targets))

    def post_physics_step(self):
        self.compute_observations()
        self.compute_reward()
        if not self.eval_mode:
            self.action_mask_step += 1
        self.progress_buf += 1
        self.running_progress_buf += 1
