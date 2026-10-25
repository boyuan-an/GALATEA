import numpy as np


class PhysicsRandomizer:
    def __init__(self, env, ranges):
        self.env = env
        self.hand_gain_range = tuple(ranges["handPdScale"])
        self.arm_gain_range = tuple(ranges["armPdScale"])
        self.surface_friction_range = tuple(ranges["surfaceFrictionScale"])
        self.object_friction_range = tuple(ranges["objectFriction"])
        self.table_friction_range = tuple(ranges["tableFriction"])
        self.hand_slots = np.asarray(env.hand_dof_indices, dtype=np.int64)
        self.arm_slots = np.asarray(env.arm_dof_indices, dtype=np.int64)
        env_ptr, robot = env.envs[0], env.robot_actors[0]
        dof = env.gym.get_actor_dof_properties(env_ptr, robot)
        self.nominal_stiffness = np.asarray(dof["stiffness"], dtype=np.float32).copy()
        self.nominal_damping = np.asarray(dof["damping"], dtype=np.float32).copy()
        shapes = env.gym.get_actor_rigid_shape_properties(env_ptr, robot)
        pad_shapes = set(env.pad_shape_ids)
        self.metal_shapes = [i for i in env.hand_shape_ids if i not in pad_shapes]
        self.pad_shapes = [i for i in env.hand_shape_ids if i in pad_shapes]
        self.nominal_shape_friction = {i: np.float32(shapes[i].friction) for i in env.hand_shape_ids}

    @staticmethod
    def _uniform(bounds, rows, cols):
        return np.random.uniform(bounds[0], bounds[1], (rows, cols)).astype(np.float32)

    def sample(self, rows):
        return {
            "hand_kp": self._uniform(self.hand_gain_range, rows, self.hand_slots.size),
            "hand_kd": self._uniform(self.hand_gain_range, rows, self.hand_slots.size),
            "arm_kp": self._uniform(self.arm_gain_range, rows, self.arm_slots.size),
            "arm_kd": self._uniform(self.arm_gain_range, rows, self.arm_slots.size),
            "surface": self._uniform(self.surface_friction_range, rows, 2),
            "object_friction": self._uniform(self.object_friction_range, rows, 1)[:, 0],
            "table_friction": self._uniform(self.table_friction_range, rows, 1)[:, 0],
        }

    def resample(self, env_ids):
        ids = [int(i) for i in env_ids.detach().cpu().flatten().tolist()]
        if not ids:
            return
        draw = self.sample(len(ids))
        env = self.env
        for row, env_id in enumerate(ids):
            env_ptr = env.envs[env_id]
            robot = env.robot_actors[env_id]
            dof = env.gym.get_actor_dof_properties(env_ptr, robot)
            dof["stiffness"][self.hand_slots] = self.nominal_stiffness[self.hand_slots] * draw["hand_kp"][row]
            dof["damping"][self.hand_slots] = self.nominal_damping[self.hand_slots] * draw["hand_kd"][row]
            dof["stiffness"][self.arm_slots] = self.nominal_stiffness[self.arm_slots] * draw["arm_kp"][row]
            dof["damping"][self.arm_slots] = self.nominal_damping[self.arm_slots] * draw["arm_kd"][row]
            env.gym.set_actor_dof_properties(env_ptr, robot, dof)
        for row, env_id in enumerate(ids):
            env_ptr = env.envs[env_id]
            robot = env.robot_actors[env_id]
            metal_scale, pad_scale = (float(v) for v in draw["surface"][row])
            shapes = env.gym.get_actor_rigid_shape_properties(env_ptr, robot)
            for i in self.metal_shapes:
                shapes[i].friction = float(self.nominal_shape_friction[i] * metal_scale)
            for i in self.pad_shapes:
                shapes[i].friction = float(self.nominal_shape_friction[i] * pad_scale)
            env.gym.set_actor_rigid_shape_properties(env_ptr, robot, shapes)
            object_shapes = env.gym.get_actor_rigid_shape_properties(env_ptr, env.object_actors[env_id])
            for shape in object_shapes:
                shape.friction = float(draw["object_friction"][row])
            env.gym.set_actor_rigid_shape_properties(env_ptr, env.object_actors[env_id], object_shapes)
            table_shapes = env.gym.get_actor_rigid_shape_properties(env_ptr, env.table_actors[env_id])
            for shape in table_shapes:
                shape.friction = float(draw["table_friction"][row])
            env.gym.set_actor_rigid_shape_properties(env_ptr, env.table_actors[env_id], table_shapes)
