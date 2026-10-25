import os

from armhand.utils.config import omegaconf_to_dict

from .arm_hand_track import ArmHandTrackEnv


def make_env(*, sim_device, rl_device, multi_gpu, cfg):
    if multi_gpu:
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        sim_device = f"cuda:{local_rank}"
        rl_device = f"cuda:{local_rank}"
    return ArmHandTrackEnv(omegaconf_to_dict(cfg), rl_device, sim_device)
