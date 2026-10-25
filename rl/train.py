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
import os
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf


def resolve_experiment_name(requested_name, base_name, now):
    if requested_name:
        requested_name = str(requested_name)
        if requested_name in {".", ".."} or os.path.basename(requested_name) != requested_name:
            raise ValueError(f"full_experiment_name must be a single directory name, got {requested_name!r}")
        return requested_name
    return base_name + "__" + now.strftime("%m-%d-%H-%M-%S")


@hydra.main(version_base="1.1", config_name="config", config_path="cfg")
def main(cfg: DictConfig):
    import isaacgym
    from hydra.utils import to_absolute_path
    from rl_games.algos_torch.model_builder import register_model, register_network

    from armhand.envs import make_env
    from armhand.rl.model import ContinuousPolicyModel
    from armhand.rl.network import AsymmetricMLPBuilder
    from armhand.rl.runner import Runner
    from armhand.utils.config import omegaconf_to_dict, print_dict
    from armhand.utils.env_wrapper import EnvStatsObserver, register_env
    from armhand.utils.seeding import set_np_formatting, set_seed

    register_model("block_policy", ContinuousPolicyModel)
    register_network("asymmetric_mlp", AsymmetricMLPBuilder)
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)
    print_dict(omegaconf_to_dict(cfg))
    set_np_formatting()
    global_rank = int(os.getenv("RANK", "0"))
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    def create_env():
        return make_env(sim_device=cfg.sim_device, rl_device=cfg.rl_device, multi_gpu=cfg.multi_gpu, cfg=cfg.task)

    register_env("isaac_env", create_env)
    train_config = omegaconf_to_dict(cfg.train)
    train_config["params"]["config"]["device"] = cfg.rl_device
    experiment_name = resolve_experiment_name(
        cfg.get("full_experiment_name"), cfg.train.params.config.name, datetime.now()
    )
    train_config["params"]["config"]["full_experiment_name"] = experiment_name
    cfg.train.params.config.full_experiment_name = experiment_name
    runner = Runner(EnvStatsObserver())
    runner.load(train_config)
    experiment_dir = os.path.join("runs", experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    with open(os.path.join(experiment_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    runner.run(
        {
            "train": True,
            "play": False,
            "checkpoint": cfg.checkpoint,
            "from_ckpt_epoch": cfg.from_ckpt_epoch,
            "frame0_eval_only": cfg.frame0_eval_only,
            "checkpoint_model_only": cfg.checkpoint_model_only,
        }
    )


if __name__ == "__main__":
    main()
