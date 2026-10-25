from functools import partial

import torch
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.torch_ext import safe_filesystem_op
from rl_games.common import common_losses, datasets
from torch import optim

from armhand.rl.base import A2CBase
from armhand.rl.blocks import BLOCK_OBS_KEY, clone_tree, concatenate_trees, permute_tree, slice_tree


class BlockPPOAgent(A2CBase):
    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        build_config = {
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "num_seqs": self.num_actors * self.num_agents,
            "value_size": self.env_info.get("value_size", 1),
            "normalize_input_excluded_keys": self.normalize_input_excluded_keys,
            "block_config": dict(self.config["blocks"]),
            **params,
        }
        self.model = self.network.build(build_config)
        self.model.to(self.ppo_device)
        self.is_rnn = False
        self.last_lr = float(self.last_lr)
        self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=0.0)
        self.value_mean_std = self.model.value_mean_std
        self.algo_observer.after_init(self)
        if self.num_agents != 1:
            raise ValueError("one agent per environment is required")
        self.block_batch_size = (
            self.batch_size + self.batch_size // self.block_settings.num_blocks * self.block_settings.off_policy_repeats
        )
        self.dataset = datasets.PPODataset(
            self.block_batch_size, self.minibatch_size, False, self.is_rnn, self.ppo_device, self.seq_length
        )
        self.dataset.length = (self.block_batch_size + self.minibatch_size - 1) // self.minibatch_size
        self.num_minibatches = len(self.dataset)

    def save(self, fn):
        torch_ext.save_checkpoint(fn, self.get_full_state_weights())

    def restore(self, fn, set_epoch=True):
        checkpoint = safe_filesystem_op(partial(torch.load, map_location="cpu"), fn)
        self.set_full_state_weights(checkpoint, set_epoch=set_epoch)

    def init_tensors(self):
        super().init_tensors()
        self.tensor_list.append("rewards")
        self.experience_buffer.tensor_dict["block_time_outs"] = torch.zeros_like(
            self.experience_buffer.tensor_dict["dones"], dtype=torch.float32
        )
        self.tensor_list.append("block_time_outs")

    def train_actor_critic(self, input_dict):
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        obs_batch = input_dict["obs"]
        lr_mul = 1.0
        curr_e_clip = self.e_clip
        batch_dict = {"is_train": True, "prev_actions": actions_batch, "obs": obs_batch}
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict["prev_neglogp"]
            values = res_dict["values"]
            entropy = res_dict["entropy"]
            mu = res_dict["mus"]
            sigma = res_dict["sigmas"]
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, True, curr_e_clip)
            c_loss = common_losses.critic_loss(
                self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
            )
            b_loss = self.bound_loss(mu)
            entropy_bonus = self.entropy_bonus(entropy, obs_batch)
            a_loss, c_loss, entropy_bonus, b_loss, entropy = (
                term.sum() / term.numel()
                for term in (
                    a_loss.unsqueeze(1),
                    c_loss,
                    entropy_bonus.unsqueeze(1),
                    b_loss.unsqueeze(1),
                    entropy.unsqueeze(1),
                )
            )
            loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy_bonus + b_loss * self.bounds_loss_coef
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None
        self.scaler.scale(loss).backward()
        self.step_optimizer()
        with torch.no_grad():
            kl_values = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, False)
            kl_dist = kl_values.sum() / kl_values.new_tensor(float(kl_values.numel()))
        return a_loss, c_loss, entropy, kl_dist, self.last_lr, lr_mul, mu.detach(), sigma.detach(), b_loss

    def entropy_bonus(self, entropy, obs_batch):
        indices = self.block_conditioner.coefficient_indices(obs_batch[BLOCK_OBS_KEY])
        coefficients = self.block_conditioner.entropy_coefficients(entropy.device)
        return entropy * coefficients[indices]

    @staticmethod
    def bound_loss(mu):
        soft_bound = 1.1
        mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
        mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
        return (mu_loss_low + mu_loss_high).sum(axis=-1)

    def _sample_follower_blocks(self):
        repeat_count = self.block_settings.off_policy_repeats
        if repeat_count == 0:
            return []
        candidates = torch.arange(1, self.block_settings.num_blocks, device=self.ppo_device)
        if self.global_rank == 0:
            selected = candidates[torch.randperm(len(candidates), device=self.ppo_device)[:repeat_count]]
        else:
            selected = torch.empty(repeat_count, dtype=torch.long, device=self.ppo_device)
        if self.multi_gpu:
            torch.distributed.broadcast(selected, src=0)
        return [int(value) for value in selected.cpu().tolist()]

    def _model_values(self, obs):
        return self.model({"is_train": False, "prev_actions": None, "obs": obs, "rnn_states": None})["values"]

    def _off_policy_part(self, batch_dict, repeat_index):
        horizon = self.horizon_length
        block_size = self.block_conditioner.block_size
        first_env = (repeat_index - 1) * block_size
        last_env = first_env + block_size
        transition_slice = slice(first_env * horizon, last_env * horizon)
        part = {
            key: slice_tree(value, transition_slice)
            for key, value in batch_dict.items()
            if key not in ("played_frames", "step_time")
        }
        leader_coefficient = self.block_settings.coefficient_min
        part["obses"][BLOCK_OBS_KEY].fill_(leader_coefficient)
        source_dones = part["dones"]
        target_values = self._model_values(part["obses"])
        final_obs = slice_tree(clone_tree(self.obs["obs"]), slice(first_env, last_env))
        final_obs[BLOCK_OBS_KEY].fill_(leader_coefficient)
        final_dones = self.dones[first_env:last_env]
        final_values = self._model_values(final_obs)
        behavior_value_grid = part["values"].reshape(block_size, horizon, -1)
        value_grid = target_values.reshape(block_size, horizon, -1)
        reward_grid = part["rewards"].reshape(block_size, horizon, -1)
        timeout_grid = part["block_time_outs"].reshape(block_size, horizon, 1)
        done_grid = source_dones.reshape(block_size, horizon)
        next_values = torch.cat([value_grid[:, 1:], final_values.unsqueeze(1)], dim=1)
        next_dones = torch.cat([done_grid[:, 1:], final_dones.unsqueeze(1)], dim=1)
        pure_reward_grid = reward_grid - self.gamma * behavior_value_grid * timeout_grid
        timeout_bootstrap = self.gamma * value_grid * timeout_grid
        returns = (
            pure_reward_grid + timeout_bootstrap + self.gamma * next_values * (1.0 - next_dones.float()).unsqueeze(-1)
        )
        part["values"] = target_values
        part["returns"] = returns.reshape(-1, returns.shape[-1])
        return part

    def postprocess_rollout_batch(self, batch_dict):
        parts = [batch_dict]
        for repeat_index in self._sample_follower_blocks():
            parts.append(self._off_policy_part(batch_dict, repeat_index))
        mixed = {}
        for key in batch_dict:
            if key in ("played_frames", "step_time"):
                mixed[key] = batch_dict[key]
            else:
                mixed[key] = concatenate_trees([part[key] for part in parts])
        permutation = torch.randperm(mixed["returns"].shape[0], device=self.ppo_device)
        for key, value in list(mixed.items()):
            if key not in ("played_frames", "step_time"):
                mixed[key] = permute_tree(value, permutation)
        if mixed["returns"].shape[0] != self.block_batch_size:
            raise RuntimeError(f"built {mixed['returns'].shape[0]} transitions, expected {self.block_batch_size}")
        return mixed
