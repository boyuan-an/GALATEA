import numpy as np
import torch
import torch.nn as nn

from armhand.rl.running_stats import RunningMeanStd, RunningMeanStdObs


class ContinuousPolicyModel:
    def __init__(self, network):
        self.model_class = "a2c"
        self.network_builder = network

    def build(self, config):
        return self.Network(
            self.network_builder.build(self.model_class, **config),
            obs_shape=config["input_shape"],
            value_size=config.get("value_size", 1),
            normalize_input_excluded_keys=config.get("normalize_input_excluded_keys", None),
        )

    class Network(nn.Module):
        def __init__(self, a2c_network, obs_shape, value_size, normalize_input_excluded_keys=None):
            nn.Module.__init__(self)
            self.obs_shape = obs_shape
            self.value_size = value_size
            self.value_mean_std = RunningMeanStd((self.value_size,))
            self.running_mean_std = RunningMeanStdObs(obs_shape, exclude_keys=normalize_input_excluded_keys)
            self.a2c_network = a2c_network

        def is_rnn(self):
            return False

        def norm_obs(self, observation):
            with torch.no_grad():
                return self.running_mean_std(observation)

        def denorm_value(self, value):
            with torch.no_grad():
                return self.value_mean_std(value, denorm=True)

        def forward(self, input_dict):
            is_train = input_dict.get("is_train", True)
            prev_actions = input_dict.get("prev_actions", None)
            input_dict["obs"] = self.norm_obs(input_dict["obs"])
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                return {
                    "prev_neglogp": torch.squeeze(prev_neglogp),
                    "values": value,
                    "entropy": entropy,
                    "rnn_states": states,
                    "mus": mu,
                    "sigmas": sigma,
                }
            selected_action = distr.sample()
            neglogp = self.neglogp(selected_action, mu, sigma, logstd)
            return {
                "neglogpacs": torch.squeeze(neglogp),
                "values": self.denorm_value(value),
                "actions": selected_action,
                "rnn_states": states,
                "mus": mu,
                "sigmas": sigma,
            }

        @staticmethod
        def neglogp(x, mean, std, logstd):
            return (
                0.5 * (((x - mean) / std) ** 2).sum(dim=-1)
                + 0.5 * np.log(2.0 * np.pi) * x.size()[-1]
                + logstd.sum(dim=-1)
            )
