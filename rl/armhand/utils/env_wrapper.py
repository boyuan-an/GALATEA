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
from collections import OrderedDict, deque

import numpy as np
import torch
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import AlgoObserver


def flatten_dict(d, prefix="", separator="."):
    res = dict()
    for key, value in d.items():
        if isinstance(value, (dict, OrderedDict)):
            res.update(flatten_dict(value, prefix + key + separator, separator))
        else:
            res[prefix + key] = value
    return res


class EnvStatsObserver(AlgoObserver):
    def __init__(self):
        super().__init__()
        self.algo = None
        self.writer = None
        self.ep_infos = []
        self.direct_info = {}
        self.episode_cumulative = dict()
        self.episode_cumulative_avg = dict()
        self.new_finished_episodes = False

    def after_init(self, algo):
        self.algo = algo
        self.writer = self.algo.writer

    def process_infos(self, infos, done_indices):
        if "episode" in infos:
            self.ep_infos.append(infos["episode"])
        if "episode_cumulative" in infos:
            for key, value in infos["episode_cumulative"].items():
                if key not in self.episode_cumulative:
                    self.episode_cumulative[key] = torch.zeros_like(value)
                self.episode_cumulative[key] += value
            for done_idx in done_indices:
                self.new_finished_episodes = True
                done_idx = done_idx.item()
                for key, value in infos["episode_cumulative"].items():
                    if key not in self.episode_cumulative_avg:
                        self.episode_cumulative_avg[key] = deque([], maxlen=self.algo.games_to_track)
                    self.episode_cumulative_avg[key].append(self.episode_cumulative[key][done_idx].item())
                    self.episode_cumulative[key][done_idx] = 0
        if len(infos) > 0:
            self.direct_info = {}
            for k, v in flatten_dict(infos, prefix="", separator="/").items():
                if isinstance(v, (float, int)) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.direct_info[k] = v

    def after_print_stats(self, frame, epoch_num, total_time):
        if self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.algo.device)
                for ep_info in self.ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.algo.device)))
                self.writer.add_scalar("Episode/" + key, torch.mean(infotensor), epoch_num)
            self.ep_infos.clear()
        if self.new_finished_episodes:
            for key, values in self.episode_cumulative_avg.items():
                self.writer.add_scalar(f"episode_cumulative/{key}", np.mean(values), frame)
                self.writer.add_scalar(f"episode_cumulative_min/{key}_min", np.min(values), frame)
                self.writer.add_scalar(f"episode_cumulative_max/{key}_max", np.max(values), frame)
            self.new_finished_episodes = False
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f"{k}/frame", v, frame)
            self.writer.add_scalar(f"{k}/iter", v, epoch_num)
            self.writer.add_scalar(f"{k}/time", v, total_time)


class DictObsVecEnv(vecenv.IVecEnv):
    def __init__(self, config_name):
        self.env = env_configurations.configurations[config_name]["env_creator"]()

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        return self.env.reset()

    def get_env_info(self):
        return {"action_space": self.env.action_space, "observation_space": self.env.observation_space}

    def get_env_state(self):
        return self.env.get_env_state()

    def set_env_state(self, env_state):
        self.env.set_env_state(env_state)

    @property
    def success_buf(self):
        return self.env.success_buf

    @property
    def failure_buf(self):
        return self.env.failure_buf


def register_env(name, creator):
    env_configurations.register(name, {"vecenv_type": "dict_obs", "env_creator": creator})
    vecenv.register("dict_obs", lambda config_name, num_actors: DictObsVecEnv(config_name))
