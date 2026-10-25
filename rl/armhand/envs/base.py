# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import torch
from gym import spaces
from isaacgym import gymapi


class SimTask:
    def __init__(self, cfg, rl_device, sim_device):
        self.device_id = int(sim_device.split(":")[1]) if ":" in sim_device else 0
        self.device = sim_device
        self.rl_device = rl_device
        self.num_envs = cfg["env"]["numEnvs"]
        self.num_actions = cfg["env"]["numActions"]
        self.control_freq_inv = cfg["env"]["controlFrequencyInv"]
        self.clip_actions = cfg["env"]["clipActions"]
        self.obs_space = spaces.Dict(
            {
                "proprioception": spaces.Box(low=-np.inf, high=np.inf, shape=(self.proprioception_dim,)),
                "privileged": spaces.Box(low=-np.inf, high=np.inf, shape=(self.privileged_dim,)),
            }
        )
        self.act_space = spaces.Box(np.ones(self.num_actions) * -1.0, np.ones(self.num_actions) * 1.0)
        self.sim_params = self._parse_sim_params(cfg["sim"])
        self.sim_params.physx.max_gpu_contact_pairs = int(2 * self.sim_params.physx.max_gpu_contact_pairs)
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        self.gym = gymapi.acquire_gym()
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.allocate_buffers()

    @property
    def observation_space(self):
        return self.obs_space

    @property
    def action_space(self):
        return self.act_space

    @staticmethod
    def _parse_sim_params(sim_cfg):
        sim_params = gymapi.SimParams()
        sim_params.dt = sim_cfg["dt"]
        sim_params.num_client_threads = 0
        sim_params.use_gpu_pipeline = True
        sim_params.substeps = sim_cfg["substeps"]
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(*sim_cfg["gravity"])
        for name, value in sim_cfg["physx"].items():
            if name == "contact_collection":
                value = gymapi.ContactCollection(value)
            setattr(sim_params.physx, name, value)
        return sim_params

    def allocate_buffers(self):
        self.obs_dict = {
            "proprioception": torch.zeros((self.num_envs, self.proprioception_dim), device=self.device),
            "privileged": torch.zeros((self.num_envs, self.privileged_dim), device=self.device),
        }
        self.rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.total_rew_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.running_progress_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.success_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.failure_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.error_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.reward_terms = {}

    def step(self, actions):
        self.pre_physics_step(torch.clamp(actions, -self.clip_actions, self.clip_actions))
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)
        self.post_physics_step()
        timeout = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)
        self.extras["time_outs"] = timeout.to(self.rl_device)
        self.extras["reward_dict"] = self.reward_terms
        self.extras["total_rewards"] = self.total_rew_buf
        self.extras["total_steps"] = self.progress_buf
        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
