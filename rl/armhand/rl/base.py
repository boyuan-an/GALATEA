import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from rl_games.algos_torch import model_builder, torch_ext
from rl_games.common import common_losses, schedulers, vecenv
from rl_games.common.experience import ExperienceBuffer
from scipy.stats import linregress
from tensorboardX import SummaryWriter
from torch import nn

from armhand.rl.blocks import BLOCK_OBS_KEY, BlockConditioner, BlockSettings
from armhand.rl.distributed import synchronized_normalize


def swap_and_flatten01(arr):
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    return action * d + m


def soft_clamp(x, lower, upper):
    return lower + torch.sigmoid(4 / (upper - lower) * (x - (lower + upper) / 2)) * (upper - lower)


class A2CBase:
    def __init__(self, base_name, params):
        self.config = config = params["config"]
        self.experiment_name = config["full_experiment_name"]
        self.algo_observer = config["features"]["observer"]
        self.algo_observer.before_init(base_name, config, self.experiment_name)
        self.config["network"] = model_builder.ModelBuilder().load(params)
        self.multi_gpu = config.get("multi_gpu", False)
        self.local_rank = 0
        self.global_rank = 0
        self.world_size = 1
        if self.multi_gpu:
            self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
            self.global_rank = int(os.getenv("RANK", "0"))
            self.world_size = int(os.getenv("WORLD_SIZE", "1"))
            dist.init_process_group("nccl", rank=self.global_rank, world_size=self.world_size)
            config["device"] = "cuda:" + str(self.local_rank)
            if self.global_rank != 0:
                config["print_stats"] = False
                config["lr_schedule"] = None
        self.num_actors = config["num_actors"]
        self.vec_env = vecenv.create_vec_env(config["env_name"], self.num_actors)
        self.env_info = self.vec_env.get_env_info()
        self.ppo_device = config.get("device", "cuda:0")
        self.value_size = self.env_info.get("value_size", 1)
        self.block_settings = BlockSettings.from_config(config["blocks"])
        self.block_conditioner = BlockConditioner(self.block_settings, self.num_actors)
        self.evaluation_mode = False
        self.env_info = dict(self.env_info)
        self.observation_space = self.block_conditioner.extend_observation_space(self.env_info["observation_space"])
        self.env_info["observation_space"] = self.observation_space
        excluded = list(config.get("normalize_input_excluded_keys", []))
        if BLOCK_OBS_KEY not in excluded:
            excluded.append(BLOCK_OBS_KEY)
        config["normalize_input_excluded_keys"] = excluded
        self.normalize_input_excluded_keys = excluded
        self.save_freq = config.get("save_frequency", 0)
        self.save_best_after = config.get("save_best_after", 100)
        self.print_stats = config.get("print_stats", True)
        self.rnn_states = None
        self.max_epochs = config.get("max_epochs", -1)
        self.early_stop_epochs = config.get("early_stop_epochs", 500)
        if config["lr_schedule"] == "adaptive":
            self.kl_threshold = config["kl_threshold"]
            self.scheduler = schedulers.AdaptiveScheduler(self.kl_threshold)
            self.scheduler.min_lr = float(config["adaptive_lr_min"])
            self.scheduler.max_lr = float(config["adaptive_lr_max"])
        else:
            self.scheduler = schedulers.IdentityScheduler()
        self.e_clip = config["e_clip"]
        self.clip_value = config["clip_value"]
        self.network = config["network"]
        self.rewards_shaper = config["reward_shaper"]
        self.num_agents = self.env_info.get("agents", 1)
        self.horizon_length = config["horizon_length"]
        self.seq_length = config.get("seq_length", 4)
        self.obs_shape = self.observation_space
        self.critic_coef = config["critic_coef"]
        self.grad_norm = config["grad_norm"]
        self.gamma = config["gamma"]
        self.tau = config["tau"]
        self.games_to_track = config.get("games_to_track", 100)
        self.game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_success = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.game_fails = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        self.game_success_per_demo = None
        self.game_eplen_per_demo = None
        self.demo_seq_len = None
        self.num_demos_tracked = 0
        self.obs = None
        self.batch_size = self.horizon_length * self.num_actors * self.num_agents
        self.batch_size_envs = self.horizon_length * self.num_actors
        self.minibatch_size = min(config["minibatch_size"], self.batch_size)
        self.num_minibatches = self.batch_size // self.minibatch_size
        assert self.batch_size % self.minibatch_size == 0
        self.mini_epochs_num = config["mini_epochs"]
        self.mixed_precision = config.get("mixed_precision", False)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        self.last_lr = config["learning_rate"]
        self.frame = 0
        self.mean_rewards = self.last_mean_rewards = -1000000000
        self._full_state_restored = False
        self.epoch_num = 0
        self.curr_frames = 0
        self.frame0_eval_interval = int(config.get("frame0_eval_interval", 0) or 0)
        self.frame0_eval_max_steps = int(config.get("frame0_eval_max_steps", 0) or 0)
        self.experiment_dir = os.path.join("runs", self.experiment_name)
        self.nn_dir = os.path.join(self.experiment_dir, "nn")
        self.summaries_dir = os.path.join(self.experiment_dir, "summaries")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        self.entropy_coef = config["entropy_coef"]
        if self.global_rank == 0:
            self.writer = SummaryWriter(self.summaries_dir)
            self.reward_history = []
        else:
            self.writer = None
            self.reward_history = None
        self.actor_loss_func = common_losses.actor_loss
        self.is_tensor_obses = False
        self.actions_num = self.env_info["action_space"].shape[0]
        self.bounds_loss_coef = config.get("bounds_loss_coef", None)
        action_space = self.env_info["action_space"]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.ppo_device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.ppo_device)

    @property
    def device(self):
        return self.ppo_device

    @property
    def raw_env(self):
        return self.vec_env.env

    def step_optimizer(self):
        if self.multi_gpu:
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))
            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def write_stats(
        self,
        total_time,
        epoch_num,
        step_time,
        play_time,
        update_time,
        a_losses,
        c_losses,
        entropies,
        kls,
        last_lr,
        lr_mul,
        frame,
        scaled_time,
        scaled_play_time,
        curr_frames,
    ):
        self.writer.add_scalar("performance/step_inference_rl_update_fps", curr_frames / scaled_time, frame)
        self.writer.add_scalar("performance/step_inference_fps", curr_frames / scaled_play_time, frame)
        self.writer.add_scalar("performance/step_fps", curr_frames / step_time, frame)
        self.writer.add_scalar("performance/rl_update_time", update_time, frame)
        self.writer.add_scalar("performance/step_inference_time", play_time, frame)
        self.writer.add_scalar("performance/step_time", step_time, frame)
        self.writer.add_scalar("losses/a_loss", torch_ext.mean_list(a_losses).item(), frame)
        self.writer.add_scalar("losses/c_loss", torch_ext.mean_list(c_losses).item(), frame)
        self.writer.add_scalar("losses/entropy", torch_ext.mean_list(entropies).item(), frame)
        self.writer.add_scalar("info/last_lr", last_lr * lr_mul, frame)
        self.writer.add_scalar("info/lr_mul", lr_mul, frame)
        self.writer.add_scalar("info/e_clip", self.e_clip * lr_mul, frame)
        self.writer.add_scalar("info/kl", torch_ext.mean_list(kls).item(), frame)
        self.writer.add_scalar("info/epochs", epoch_num, frame)
        self.algo_observer.after_print_stats(frame, epoch_num, total_time)

    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_action_values(self, obs):
        self.model.eval()
        input_dict = {"is_train": False, "prev_actions": None, "obs": obs["obs"], "rnn_states": self.rnn_states}
        with torch.no_grad():
            return self.model(input_dict)

    def get_values(self, obs):
        with torch.no_grad():
            self.model.eval()
            input_dict = {"is_train": False, "prev_actions": None, "obs": obs["obs"], "rnn_states": self.rnn_states}
            return self.model(input_dict)["values"]

    def init_tensors(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            "num_actors": self.num_actors,
            "horizon_length": self.horizon_length,
            "has_central_value": False,
            "use_action_masks": False,
        }
        self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)
        current_rewards_shape = (batch_size, self.value_size)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)
        self.game_reward_dict = {}
        self.current_reward_dict = {}
        self.update_list = ["actions", "neglogpacs", "values", "mus", "sigmas"]
        self.tensor_list = self.update_list + ["obses", "states", "dones"]

    def cast_obs(self, obs):
        if isinstance(obs, torch.Tensor):
            self.is_tensor_obses = True
        return obs

    def obs_to_tensors(self, obs):
        upd_obs = {key: self._obs_to_tensors_internal(value) for key, value in obs.items()}
        if "obs" not in obs:
            upd_obs = {"obs": upd_obs}
        return self.block_conditioner.attach(upd_obs, evaluation=self.evaluation_mode)

    def _obs_to_tensors_internal(self, obs):
        if isinstance(obs, dict):
            return {key: self._obs_to_tensors_internal(value) for key, value in obs.items()}
        return self.cast_obs(obs)

    def preprocess_actions(self, actions):
        clamped_actions = soft_clamp(actions, torch.ones_like(actions) * -1.0, torch.ones_like(actions))
        return rescale_actions(self.actions_low, self.actions_high, clamped_actions)

    def env_step(self, actions):
        actions = self.preprocess_actions(actions)
        obs, rewards, dones, infos = self.vec_env.step(actions)
        if self.value_size == 1:
            rewards = rewards.unsqueeze(1)
        return self.obs_to_tensors(obs), rewards.to(self.ppo_device), dones.to(self.ppo_device), infos

    def env_reset(self):
        return self.obs_to_tensors(self.vec_env.reset())

    def _info_tensor_1d(self, infos, key):
        value = infos.get(key, None)
        if value is None:
            return None
        value = (
            value.to(self.ppo_device)
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value, device=self.ppo_device)
        )
        if value.ndim > 1:
            value = value.reshape(value.shape[0], -1)[:, 0]
        return value.reshape(-1).float()

    def run_frame0_eval(self, epoch_num, frame):
        if self.frame0_eval_interval <= 0 or epoch_num % self.frame0_eval_interval != 0:
            return False
        env = self.raw_env
        train_random_state_init = bool(env.random_state_init)
        if not train_random_state_init:
            return False
        done_reason_names, done_metric_names = env.done_reason_schema()
        done_reason_names = tuple(done_reason_names)
        done_metric_names = tuple(done_metric_names)
        incomplete_reason_index = done_reason_names.index("eval_max_steps_incomplete")
        old_random_state_init = env.random_state_init
        old_record_done_reasons = env.record_done_reasons
        old_eval_mode = env.eval_mode
        old_dones = self.dones.clone() if self.dones is not None else None
        gate_newly_unlocked = False
        batch_size = self.num_actors * self.num_agents
        running_rewards = torch.zeros((batch_size, self.value_size), dtype=torch.float32, device=self.ppo_device)
        running_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        episode_rewards = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        episode_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        episode_success = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        episode_fail = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        completed = torch.zeros(batch_size, dtype=torch.bool, device=self.ppo_device)
        episode_reason_bits = torch.zeros(batch_size, dtype=torch.int64, device=self.ppo_device)
        episode_done_metrics = torch.zeros(
            (batch_size, len(done_metric_names)), dtype=torch.float32, device=self.ppo_device
        )
        max_steps = self.frame0_eval_max_steps
        if max_steps <= 0:
            max_steps = int(env.max_episode_length) + 2
        eval_started = time.monotonic()
        try:
            env.random_state_init = False
            env.eval_mode = True
            env.record_done_reasons = True
            self.evaluation_mode = True
            self.model.eval()
            self.obs = self.env_reset()
            self.dones = torch.zeros_like(self.dones)
            eval_demo_ids = env.demo_ids.clone()
            with torch.no_grad():
                for _ in range(max_steps):
                    if torch.any(self.dones):
                        self.obs, _ = env.reset_done()
                        self.obs = self.obs_to_tensors(self.obs)
                        self.dones = torch.zeros_like(self.dones)
                    res_dict = self.get_action_values(self.obs)
                    self.obs, rewards, self.dones, infos = self.env_step(res_dict["mus"])
                    active = (~completed).float()
                    running_rewards += rewards * active.unsqueeze(1)
                    running_lengths += active
                    done_now = self.dones.bool() & ~completed
                    if torch.any(done_now):
                        done_indices = done_now.nonzero(as_tuple=False).reshape(-1)
                        total_rewards = self._info_tensor_1d(infos, "total_rewards")
                        total_steps = self._info_tensor_1d(infos, "total_steps")
                        if total_rewards is not None:
                            episode_rewards[done_indices] = total_rewards[done_indices]
                        else:
                            episode_rewards[done_indices] = running_rewards[done_indices, 0]
                        if total_steps is not None:
                            episode_lengths[done_indices] = total_steps[done_indices]
                        else:
                            episode_lengths[done_indices] = running_lengths[done_indices]
                        episode_success[done_indices] = (
                            self.vec_env.success_buf.to(self.ppo_device).reshape(-1).float()[done_indices]
                        )
                        episode_fail[done_indices] = (
                            self.vec_env.failure_buf.to(self.ppo_device).reshape(-1).float()[done_indices]
                        )
                        episode_reason_bits[done_indices] = (
                            env.done_reason_bits.to(self.ppo_device).reshape(-1)[done_indices].to(torch.int64)
                        )
                        episode_done_metrics[done_indices] = env.done_metrics.to(self.ppo_device)[done_indices]
                        completed[done_indices] = True
                    if bool(torch.all(completed)):
                        break
            if torch.any(~completed):
                incomplete = (~completed).nonzero(as_tuple=False).reshape(-1)
                episode_rewards[incomplete] = running_rewards[incomplete, 0]
                episode_lengths[incomplete] = running_lengths[incomplete]
                episode_done_metrics[incomplete] = env.done_metrics.to(self.ppo_device)[incomplete]
                episode_reason_bits[incomplete] |= 1 << incomplete_reason_index
            metrics = torch.stack(
                [
                    episode_success.sum(),
                    episode_fail.sum(),
                    episode_rewards.sum(),
                    episode_lengths.sum(),
                    completed.float().sum(),
                    torch.tensor(float(batch_size), device=self.ppo_device),
                ]
            )
            if self.multi_gpu:
                dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            total_episodes = torch.clamp(metrics[5], min=1.0)
            success_rate = (metrics[0] / total_episodes).item()
            fail_rate = (metrics[1] / total_episodes).item()
            mean_reward = (metrics[2] / total_episodes).item()
            mean_length = (metrics[3] / total_episodes).item()
            completed_rate = (metrics[4] / total_episodes).item()
            demo_ids = eval_demo_ids.to(self.ppo_device).reshape(-1).long()
            num_demos = int(env.num_demos)
            demo_names = list(env.data_indices[:num_demos])
            if demo_ids.numel() and (int(demo_ids.min().item()) < 0 or int(demo_ids.max().item()) >= num_demos):
                raise RuntimeError("evaluation demo id is outside the demo bank")
            demo_success = torch.zeros(num_demos, device=self.ppo_device)
            demo_count = torch.zeros(num_demos, device=self.ppo_device)
            demo_success.index_add_(0, demo_ids, episode_success)
            demo_count.index_add_(0, demo_ids, torch.ones_like(episode_success))
            bit_indices = torch.arange(len(done_reason_names), dtype=torch.int64, device=self.ppo_device)
            reason_matrix = (episode_reason_bits.unsqueeze(1) >> bit_indices.unsqueeze(0) & 1).to(torch.float32)
            demo_reason_counts = torch.zeros((num_demos, len(done_reason_names)), device=self.ppo_device)
            demo_metric_sums = torch.zeros((num_demos, len(done_metric_names)), device=self.ppo_device)
            demo_reason_counts.index_add_(0, demo_ids, reason_matrix)
            demo_metric_sums.index_add_(0, demo_ids, episode_done_metrics)
            if self.multi_gpu:
                dist.all_reduce(demo_success, op=dist.ReduceOp.SUM)
                aux = torch.cat((demo_count.unsqueeze(1), demo_reason_counts, demo_metric_sums), dim=1)
                dist.all_reduce(aux, op=dist.ReduceOp.SUM)
                reason_end = 1 + len(done_reason_names)
                demo_count = aux[:, 0]
                demo_reason_counts = aux[:, 1:reason_end]
                demo_metric_sums = aux[:, reason_end:]
            gate_newly_unlocked = bool(
                env.update_hand_velocity_gate(
                    success_rate,
                    epoch_num,
                    successful_episodes=int(metrics[0].item()),
                    total_episodes=int(metrics[5].item()),
                )
            )
            if self.global_rank == 0:
                self._report_frame0_eval(
                    env,
                    epoch_num,
                    frame,
                    success_rate,
                    fail_rate,
                    mean_reward,
                    mean_length,
                    completed_rate,
                    max_steps,
                    eval_started,
                    train_random_state_init,
                    demo_names,
                    demo_success,
                    demo_count,
                    demo_reason_counts,
                    demo_metric_sums,
                    done_reason_names,
                    done_metric_names,
                    metrics,
                    total_episodes,
                    gate_newly_unlocked,
                )
        finally:
            env.random_state_init = old_random_state_init
            env.record_done_reasons = old_record_done_reasons
            env.eval_mode = old_eval_mode
            self.evaluation_mode = False
            self.model.train()
            if old_dones is not None:
                self.dones = old_dones
            self.obs = self.env_reset()
            self.dones = torch.zeros_like(self.dones)
            self.current_rewards.zero_()
            self.current_shaped_rewards.zero_()
            self.current_lengths.zero_()
            self.current_reward_dict = {}
        return gate_newly_unlocked

    def _report_frame0_eval(
        self,
        env,
        epoch_num,
        frame,
        success_rate,
        fail_rate,
        mean_reward,
        mean_length,
        completed_rate,
        max_steps,
        eval_started,
        train_random_state_init,
        demo_names,
        demo_success,
        demo_count,
        demo_reason_counts,
        demo_metric_sums,
        done_reason_names,
        done_metric_names,
        metrics,
        total_episodes,
        gate_newly_unlocked,
    ):
        print(
            f"[frame0_eval] epoch={epoch_num} frame={frame} success_rate={success_rate:.6f} fail_rate={fail_rate:.6f} "
            f"reward={mean_reward:.6f} ep_len={mean_length:.3f} completed_rate={completed_rate:.6f} max_steps={max_steps} "
            f"wall_time_s={time.monotonic() - eval_started:.3f} train_randomStateInit={train_random_state_init} "
            f"deterministic_actions=True"
        )
        demo_parts = []
        for demo_index in range(demo_success.numel()):
            count = int(demo_count[demo_index].item())
            rate = float((demo_success[demo_index] / demo_count[demo_index]).item()) if count > 0 else float("nan")
            demo_parts.append(f"{demo_names[demo_index]}={rate:.6f}(n={count})")
        print("[frame0_eval_per_demo] " + " ".join(demo_parts), flush=True)
        total_reason_counts = demo_reason_counts.sum(dim=0)
        total_metric_sums = demo_metric_sums.sum(dim=0)
        reason_parts = [
            f"{name}={int(total_reason_counts[index].item())}" for index, name in enumerate(done_reason_names)
        ]
        print("[frame0_eval_done_reasons] " + " ".join(reason_parts), flush=True)
        per_demo_payload = {}
        for demo_index, demo_name in enumerate(demo_names):
            count = int(demo_count[demo_index].item())
            if count <= 0:
                continue
            per_demo_payload[str(demo_name)] = {
                "demo_index": demo_index,
                "episodes": count,
                "successes": int(demo_success[demo_index].item()),
                "reason_counts_inclusive": {
                    name: int(demo_reason_counts[demo_index, reason_index].item())
                    for reason_index, name in enumerate(done_reason_names)
                },
                "terminal_metric_means": {
                    name: float(demo_metric_sums[demo_index, metric_index].item() / count)
                    for metric_index, name in enumerate(done_metric_names)
                },
            }
        record = {
            "schema_version": 1,
            "eval_name": "frame0_eval",
            "epoch": int(epoch_num),
            "frame": int(frame),
            "reason_semantics": "inclusive_multilabel",
            "reason_names": list(done_reason_names),
            "terminal_metric_names": list(done_metric_names),
            "aggregate_reason_counts_inclusive": {
                name: int(total_reason_counts[index].item()) for index, name in enumerate(done_reason_names)
            },
            "aggregate_terminal_metric_means": {
                name: float(total_metric_sums[index].item() / max(float(metrics[5].item()), 1.0))
                for index, name in enumerate(done_metric_names)
            },
            "per_demo": per_demo_payload,
        }
        with open(os.path.join(self.experiment_dir, "frame0_done_reasons.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        missing = [demo_names[index] for index in range(demo_count.numel()) if int(demo_count[index].item()) == 0]
        print(f"[frame0_eval_coverage] demos={len(demo_names)} missing={','.join(missing) or 'none'}", flush=True)
        print(
            f"[hand-velocity-gate] epoch={epoch_num} frame0_eval_sr={success_rate:.6f} "
            f"threshold={env.hand_velocity_gate_threshold:.6f} unlocked={env.hand_velocity_gate_unlocked} "
            f"newly_unlocked={gate_newly_unlocked}",
            flush=True,
        )
        self.writer.add_scalar("frame0_eval/success_rate_iter", success_rate, epoch_num)
        self.writer.add_scalar("frame0_eval/fail_rate_iter", fail_rate, epoch_num)
        self.writer.add_scalar("frame0_eval/reward_iter", mean_reward, epoch_num)
        self.writer.add_scalar("frame0_eval/episode_length_iter", mean_length, epoch_num)
        self.writer.add_scalar("frame0_eval/completed_rate_iter", completed_rate, epoch_num)
        self.writer.add_scalar("frame0_eval/success_rate_step", success_rate, frame)
        self.writer.add_scalar("frame0_eval/reward_step", mean_reward, frame)
        for reason_index, reason_name in enumerate(done_reason_names):
            self.writer.add_scalar(
                f"frame0_eval_done_reason/{reason_name}_rate",
                (total_reason_counts[reason_index] / total_episodes).item(),
                epoch_num,
            )
        for metric_index, metric_name in enumerate(done_metric_names):
            self.writer.add_scalar(
                f"frame0_eval_terminal_metric/{metric_name}",
                (total_metric_sums[metric_index] / total_episodes).item(),
                epoch_num,
            )
        self.writer.add_scalar("hand_velocity_gate/unlocked", float(env.hand_velocity_gate_unlocked), epoch_num)
        self.writer.add_scalar("hand_velocity_gate/gate_success_rate", success_rate, epoch_num)
        for demo_index in range(demo_success.numel()):
            if demo_count[demo_index] > 0:
                self.writer.add_scalar(
                    f"frame0_eval_demo/demo_{demo_index}/success_rate",
                    (demo_success[demo_index] / demo_count[demo_index]).item(),
                    epoch_num,
                )

    def discount_values(self, fdones, last_extrinsic_values, mb_fdones, mb_extrinsic_values, mb_rewards):
        lastgaelam = 0
        mb_advs = torch.zeros_like(mb_rewards)
        for t in reversed(range(self.horizon_length)):
            if t == self.horizon_length - 1:
                nextnonterminal = 1.0 - fdones
                nextvalues = last_extrinsic_values
            else:
                nextnonterminal = 1.0 - mb_fdones[t + 1]
                nextvalues = mb_extrinsic_values[t + 1]
            nextnonterminal = nextnonterminal.unsqueeze(1)
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_extrinsic_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.tau * nextnonterminal * lastgaelam
        return mb_advs

    def get_full_state_weights(self):
        state = {"scaler": self.scaler.state_dict(), "model": self.model.state_dict()}
        state["epoch"] = self.epoch_num
        state["frame"] = self.frame
        state["optimizer"] = self.optimizer.state_dict()
        state["last_lr"] = self.last_lr
        state["entropy_coef"] = self.entropy_coef
        state["last_mean_rewards"] = self.last_mean_rewards
        state["env_state"] = self.vec_env.get_env_state()
        return state

    def set_full_state_weights(self, weights, set_epoch=True):
        self.model.load_state_dict(weights["model"])
        if "scaler" in weights:
            self.scaler.load_state_dict(weights["scaler"])
        if set_epoch:
            self.epoch_num = weights["epoch"]
            self.frame = weights["frame"]
        self.optimizer.load_state_dict(weights["optimizer"])
        optimizer_lrs = {float(group["lr"]) for group in self.optimizer.param_groups}
        if len(optimizer_lrs) != 1:
            raise RuntimeError(f"restored optimizer has inconsistent learning rates: {sorted(optimizer_lrs)}")
        optimizer_lr = optimizer_lrs.pop()
        self.last_lr = float(weights.get("last_lr", optimizer_lr))
        self.entropy_coef = float(weights.get("entropy_coef", self.entropy_coef))
        self.update_lr(self.last_lr)
        self.last_mean_rewards = weights.get("last_mean_rewards", -1000000000)
        self._full_state_restored = True
        self.vec_env.set_env_state(weights.get("env_state", None))

    def play_steps(self):
        step_time = 0.0
        for n in range(self.horizon_length):
            self.experience_buffer.update_data("dones", n, self.dones)
            if torch.any(self.dones):
                self.obs, _ = self.vec_env.env.reset_done()
                self.obs = self.obs_to_tensors(self.obs)
            res_dict = self.get_action_values(self.obs)
            self.experience_buffer.update_data("obses", n, self.obs["obs"])
            for k in self.update_list:
                self.experience_buffer.update_data(k, n, res_dict[k])
            step_time_start = time.time()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict["actions"])
            step_time += time.time() - step_time_start
            shaped_rewards = self.rewards_shaper(rewards)
            if "time_outs" in infos:
                shaped_rewards += (
                    self.gamma * res_dict["values"] * self.cast_obs(infos["time_outs"]).unsqueeze(1).float()
                )
            time_outs = infos.get("time_outs", torch.zeros_like(self.dones))
            self.experience_buffer.update_data("block_time_outs", n, self.cast_obs(time_outs).float())
            self.experience_buffer.update_data("rewards", n, shaped_rewards)
            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[:: self.num_agents]
            if "total_rewards" in infos:
                self.current_rewards = torch.max(self.current_rewards, infos["total_rewards"][:, None])
            if "total_steps" in infos:
                self.current_lengths = torch.max(self.current_lengths, infos["total_steps"])
            if "reward_dict" in infos:
                for k, v in infos["reward_dict"].items():
                    if k in self.current_reward_dict:
                        self.current_reward_dict[k] += v.unsqueeze(1)
                    else:
                        self.current_reward_dict[k] = v.unsqueeze(1)
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_success.update(self.vec_env.success_buf[env_done_indices])
            self._update_per_demo_stats(env_done_indices)
            self.game_fails.update(self.vec_env.failure_buf[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)
            not_dones = 1.0 - self.dones.float()
            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            for k, v in self.current_reward_dict.items():
                if k not in self.game_reward_dict:
                    self.game_reward_dict[k] = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
                self.game_reward_dict[k].update(v[env_done_indices])
                self.current_reward_dict[k] = self.current_reward_dict[k] * not_dones.unsqueeze(1)
        last_values = self.get_values(self.obs)
        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict["dones"].float()
        mb_values = self.experience_buffer.tensor_dict["values"]
        mb_rewards = self.experience_buffer.tensor_dict["rewards"]
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        batch_dict = self.experience_buffer.get_transformed_list(swap_and_flatten01, self.tensor_list)
        batch_dict["returns"] = swap_and_flatten01(mb_returns)
        batch_dict["played_frames"] = self.batch_size
        batch_dict["step_time"] = step_time
        return batch_dict

    def _update_per_demo_stats(self, env_done_indices):
        env = self.vec_env.env
        demo_ids = env.demo_ids
        if env_done_indices.numel() == 0:
            return
        if self.game_success_per_demo is None:
            self.num_demos_tracked = len(env.data_indices)
            self.game_success_per_demo = [
                torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
                for _ in range(self.num_demos_tracked)
            ]
            self.game_eplen_per_demo = [
                torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
                for _ in range(self.num_demos_tracked)
            ]
            self.demo_seq_len = env.demo_bank["seq_len"].float().reshape(-1).to(self.ppo_device)
        done_demo = demo_ids[env_done_indices].reshape(-1)
        done_succ = self.vec_env.success_buf[env_done_indices].reshape(-1, 1)
        done_len = self.current_lengths[env_done_indices].reshape(-1, 1)
        present = torch.unique(done_demo)
        present = present[(present >= 0) & (present < self.num_demos_tracked)]
        for d in present.detach().cpu().tolist():
            mask = done_demo == d
            self.game_success_per_demo[d].update(done_succ[mask])
            self.game_eplen_per_demo[d].update(done_len[mask])

    def train_epoch(self):
        self.model.eval()
        play_time_start = time.time()
        with torch.no_grad():
            batch_dict = self.postprocess_rollout_batch(self.play_steps())
        play_time_end = time.time()
        update_time_start = time.time()
        self.model.train()
        self.curr_frames = batch_dict.pop("played_frames")
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.dataset)):
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(
                    self.dataset[i]
                )
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                b_losses.append(b_loss)
                self.dataset.update_mu_sigma(cmu, csigma)
            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            self.last_lr, self.entropy_coef = self.scheduler.update(
                self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item()
            )
            self.update_lr(self.last_lr)
            kls.append(av_kls)
            self.model.running_mean_std.eval()
        update_time_end = time.time()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start
        return (
            batch_dict["step_time"],
            play_time,
            update_time,
            total_time,
            a_losses,
            c_losses,
            b_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
        )

    def prepare_dataset(self, batch_dict):
        returns = batch_dict["returns"]
        values = batch_dict["values"]
        advantages = returns - values
        self.value_mean_std.train()
        values = self.value_mean_std(values)
        returns = self.value_mean_std(returns)
        self.value_mean_std.eval()
        advantages = torch.sum(advantages, axis=1)
        advantages = synchronized_normalize(advantages, multi_gpu=self.multi_gpu)
        dataset_dict = {
            "old_values": values,
            "old_logp_actions": batch_dict["neglogpacs"],
            "advantages": advantages,
            "returns": returns,
            "actions": batch_dict["actions"],
            "obs": batch_dict["obses"],
            "dones": batch_dict["dones"],
            "rnn_states": batch_dict.get("rnn_states", None),
            "rnn_masks": batch_dict.get("rnn_masks", None),
            "mu": batch_dict["mus"],
            "sigma": batch_dict["sigmas"],
        }
        self.dataset.update_values_dict(dataset_dict)

    def train(self):
        self.init_tensors()
        if not self._full_state_restored:
            self.last_mean_rewards = -100500
        total_time = 0
        self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs
        if self.multi_gpu:
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])
        while True:
            self.epoch_num += 1
            epoch_num = self.epoch_num
            (
                step_time,
                play_time,
                update_time,
                sum_time,
                a_losses,
                c_losses,
                b_losses,
                entropies,
                kls,
                last_lr,
                lr_mul,
            ) = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents
            self.dataset.update_values_dict(None)
            should_exit = False
            if self.global_rank == 0:
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames
                if self.print_stats:
                    print(
                        f"fps step: {curr_frames / max(step_time, 1e-09):.0f} fps step and policy inference: {curr_frames / scaled_play_time:.0f} "
                        f"fps total: {curr_frames / scaled_time:.0f} epoch: {epoch_num:.0f}/{self.max_epochs:.0f} frames: {frame:.0f}"
                    )
                self.write_stats(
                    total_time,
                    epoch_num,
                    step_time,
                    play_time,
                    update_time,
                    a_losses,
                    c_losses,
                    entropies,
                    kls,
                    last_lr,
                    lr_mul,
                    frame,
                    scaled_time,
                    scaled_play_time,
                    curr_frames,
                )
                if len(b_losses) > 0:
                    self.writer.add_scalar("losses/bounds_loss", torch_ext.mean_list(b_losses).item(), frame)
                mean_success_rate = None
                if self.game_success.current_size > 0:
                    mean_success_rate = self.game_success.get_mean()
                    self.writer.add_scalar("success_rate/step", mean_success_rate, frame)
                    self.writer.add_scalar("success_rate/iter", mean_success_rate, epoch_num)
                    self.writer.add_scalar("success_rate/time", mean_success_rate, total_time)
                if self.game_success_per_demo is not None:
                    for d, meter in enumerate(self.game_success_per_demo):
                        if meter.current_size > 0:
                            v = meter.get_mean()
                            self.writer.add_scalar(f"success_rate_demo/demo_{d}/iter", v, epoch_num)
                            self.writer.add_scalar(f"success_rate_demo/demo_{d}/step", v, frame)
                            self.writer.add_scalar(f"success_rate_demo/demo_{d}/time", v, total_time)
                if self.game_eplen_per_demo is not None:
                    for d, meter in enumerate(self.game_eplen_per_demo):
                        if meter.current_size > 0:
                            v = float(meter.get_mean())
                            self.writer.add_scalar(f"episode_lengths_demo/demo_{d}/iter", v, epoch_num)
                            self.writer.add_scalar(f"episode_lengths_demo/demo_{d}/step", v, frame)
                            if d < self.demo_seq_len.numel():
                                seq_len = float(self.demo_seq_len[d].item())
                                self.writer.add_scalar(f"episode_lengths_demo/demo_{d}/seqlen", seq_len, epoch_num)
                                if seq_len > 0:
                                    self.writer.add_scalar(
                                        f"episode_lengths_demo/demo_{d}/frac", v / seq_len, epoch_num
                                    )
                mean_fail_rate = None
                if self.game_fails.current_size > 0:
                    mean_fail_rate = self.game_fails.get_mean()
                    self.writer.add_scalar("fail_rate/step", mean_fail_rate, frame)
                    self.writer.add_scalar("fail_rate/iter", mean_fail_rate, epoch_num)
                    self.writer.add_scalar("fail_rate/time", mean_fail_rate, total_time)
                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]
                    self.writer.add_scalar("rewards/step", mean_rewards[0], frame)
                    self.writer.add_scalar("rewards/iter", mean_rewards[0], epoch_num)
                    self.reward_history.append(mean_rewards[0])
                    self.writer.add_scalar("rewards/time", mean_rewards[0], total_time)
                    self.writer.add_scalar("shaped_rewards/step", mean_shaped_rewards[0], frame)
                    self.writer.add_scalar("shaped_rewards/iter", mean_shaped_rewards[0], epoch_num)
                    self.writer.add_scalar("shaped_rewards/time", mean_shaped_rewards[0], total_time)
                    game_reward_dict = {k: v.get_mean() for k, v in self.game_reward_dict.items()}
                    self.writer.add_scalars("reward_dict/step", game_reward_dict, frame)
                    self.writer.add_scalars("reward_dict/iter", game_reward_dict, epoch_num)
                    self.writer.add_scalars("reward_dict/time", game_reward_dict, total_time)
                    self.writer.add_scalar("episode_lengths/step", mean_lengths, frame)
                    self.writer.add_scalar("episode_lengths/iter", mean_lengths, epoch_num)
                    self.writer.add_scalar("episode_lengths/time", mean_lengths, total_time)
                    checkpoint_name = self.config["name"] + "_ep_" + str(epoch_num) + "_rew_" + str(mean_rewards[0])
                    if mean_success_rate is not None:
                        checkpoint_name += f"_sr_{mean_success_rate}"
                    if mean_fail_rate is not None:
                        checkpoint_name += f"_fr_{mean_fail_rate}"
                    if self.save_freq > 0 and epoch_num % self.save_freq == 0:
                        self.save(os.path.join(self.nn_dir, "last_" + checkpoint_name))
                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print("saving next best rewards: ", mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config["name"]))
                        if self.last_mean_rewards > self.config["score_to_win"]:
                            self.save(os.path.join(self.nn_dir, checkpoint_name))
                            should_exit = True
                if epoch_num >= self.early_stop_epochs:
                    slope_early_stop = True
                    for horizon in [8, 32, 64]:
                        slope, _, _, _, _ = linregress(
                            np.arange(horizon), self.reward_history[epoch_num - horizon : epoch_num]
                        )
                        if slope > 0.0001:
                            slope_early_stop = False
                            break
                    if slope_early_stop:
                        name = (
                            "last_"
                            + self.config["name"]
                            + "_ep_"
                            + str(epoch_num)
                            + "_rew_"
                            + str(mean_rewards).replace("[", "_").replace("]", "_")
                        )
                        if mean_success_rate is not None:
                            name += f"_sr_{mean_success_rate}"
                        if mean_fail_rate is not None:
                            name += f"_fr_{mean_fail_rate}"
                        self.save(os.path.join(self.nn_dir, name))
                        print("EARLY STOP!")
                        should_exit = True
                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        mean_rewards = -np.inf
                    self.save(
                        os.path.join(
                            self.nn_dir,
                            "last_"
                            + self.config["name"]
                            + "_ep_"
                            + str(epoch_num)
                            + "_rew_"
                            + str(mean_rewards).replace("[", "_").replace("]", "_"),
                        )
                    )
                    print("MAX EPOCHS NUM!")
                    should_exit = True
            gate_newly_unlocked = self.run_frame0_eval(epoch_num, self.frame if self.global_rank == 0 else frame)
            if gate_newly_unlocked and self.global_rank == 0:
                self.save(
                    os.path.join(
                        self.nn_dir, "last_" + self.config["name"] + "_ep_" + str(epoch_num) + "_eval_gate_unlocked"
                    )
                )
            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num
