import torch
import torch.nn as nn
from hydra.utils import instantiate
from rl_games.algos_torch.network_builder import A2CBuilder, NetworkBuilder

from armhand.rl.blocks import BLOCK_OBS_KEY, BlockSettings

CRITIC_KEYS = ("proprioception", "privileged", "target")


class AsymmetricMLPNetwork(A2CBuilder.Network):
    def __init__(self, params, **kwargs):
        actions_num = kwargs.pop("actions_num")
        self.value_size = kwargs.pop("value_size", 1)
        self.num_seqs = kwargs.pop("num_seqs", 1)
        blocks = BlockSettings.from_config(kwargs.pop("block_config"))
        NetworkBuilder.BaseNetwork.__init__(self)
        self.load(params)
        self.register_buffer(
            "block_coefficients", torch.linspace(blocks.coefficient_max, blocks.coefficient_min, blocks.num_blocks)
        )
        self.block_embedding = nn.Embedding(blocks.num_blocks, blocks.embedding_dim)
        self.actor_feature_encoder = instantiate(params["actor_feature_encoder"])
        self.critic_feature_encoder = instantiate(params["critic_feature_encoder"])
        mlp_args = {
            "units": self.units,
            "activation": self.activation,
            "norm_func_name": self.normalization,
            "dense_func": torch.nn.Linear,
            "d2rl": self.is_d2rl,
            "norm_only_first_layer": self.norm_only_first_layer,
        }
        self.actor_mlp = self._build_mlp(
            input_size=self.actor_feature_encoder.output_dim + blocks.embedding_dim, **mlp_args
        )
        self.critic_mlp = self._build_mlp(
            input_size=self.critic_feature_encoder.output_dim + blocks.embedding_dim, **mlp_args
        )
        self.value = self._build_value_layer(self.units[-1], self.value_size)
        self.value_act = self.activations_factory.create(self.value_activation)
        self.mu = torch.nn.Linear(self.units[-1], actions_num)
        self.mu_act = self.activations_factory.create(self.space_config["mu_activation"])
        self.sigma_act = self.activations_factory.create(self.space_config["sigma_activation"])
        self.sigma = nn.Parameter(torch.zeros(blocks.num_blocks, actions_num))
        mlp_init = self.init_factory.create(**self.initializer)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                mlp_init(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        self.init_factory.create(**self.space_config["mu_init"])(self.mu.weight)
        self.init_factory.create(**self.space_config["sigma_init"])(self.sigma)

    def _block_indices(self, obs):
        coefficients = obs[BLOCK_OBS_KEY].reshape(-1, 1)
        distances = torch.abs(coefficients - self.block_coefficients.to(coefficients).reshape(1, -1))
        indices = torch.argmin(distances, dim=1)
        if torch.any(torch.gather(distances, 1, indices.unsqueeze(1)) > 0.0001):
            raise ValueError("unknown block coefficient")
        return indices

    def forward(self, obs_dict):
        obs = obs_dict["obs"]
        indices = self._block_indices(obs)
        condition = self.block_embedding(indices)
        actor_out = torch.cat([self.actor_feature_encoder(obs["actor"]), condition], dim=-1)
        actor_out = self.actor_mlp(actor_out)
        critic_features = self.critic_feature_encoder({k: obs[k] for k in CRITIC_KEYS})
        critic_out = self.critic_mlp(torch.cat([critic_features, condition], dim=-1))
        value = self.value_act(self.value(critic_out))
        mu = self.mu_act(self.mu(actor_out))
        sigma = self.sigma_act(self.sigma[indices])
        return torch.nan_to_num(mu), torch.nan_to_num(mu * 0 + sigma), torch.nan_to_num(value), None


class AsymmetricMLPBuilder(A2CBuilder):
    def build(self, name, **kwargs):
        return AsymmetricMLPNetwork(self.params, **kwargs)
