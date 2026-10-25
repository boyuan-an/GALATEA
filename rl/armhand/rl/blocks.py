from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import gym
import numpy as np
import torch

BLOCK_OBS_KEY = "block_coef"


@dataclass(frozen=True)
class BlockSettings:
    num_blocks: int = 6
    coefficient_max: float = 50.0
    coefficient_min: float = 0.0
    entropy_scale: float = 0.005
    off_policy_repeats: int = 1
    embedding_dim: int = 32
    eval_coefficient: float = 0.0

    @classmethod
    def from_config(cls, section: Mapping[str, Any]) -> "BlockSettings":
        return cls(
            num_blocks=int(section["num_blocks"]),
            coefficient_max=float(section["coefficient_max"]),
            coefficient_min=float(section["coefficient_min"]),
            entropy_scale=float(section["entropy_scale"]),
            off_policy_repeats=int(section["off_policy_repeats"]),
            embedding_dim=int(section["embedding_dim"]),
            eval_coefficient=float(section["eval_coefficient"]),
        )

    def validate(self, num_actors: int | None = None) -> None:
        if self.num_blocks < 2:
            raise ValueError("num_blocks must be at least 2")
        if self.coefficient_max <= self.coefficient_min:
            raise ValueError("coefficient_max must be larger than coefficient_min")
        if self.entropy_scale < 0.0:
            raise ValueError("entropy_scale must be non-negative")
        if self.off_policy_repeats < 0 or self.off_policy_repeats >= self.num_blocks:
            raise ValueError("off_policy_repeats must be in [0, num_blocks)")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if not self.coefficient_min <= self.eval_coefficient <= self.coefficient_max:
            raise ValueError("eval_coefficient must lie inside the coefficient range")
        if num_actors is not None and num_actors % self.num_blocks != 0:
            raise ValueError(f"num_actors={num_actors} must be divisible by num_blocks={self.num_blocks}")

    def coefficient_values(self, device=None) -> torch.Tensor:
        return torch.linspace(
            self.coefficient_max, self.coefficient_min, self.num_blocks, dtype=torch.float32, device=device
        )


class BlockConditioner:
    def __init__(self, settings: BlockSettings, num_actors: int):
        settings.validate(num_actors)
        self.settings = settings
        self.num_actors = int(num_actors)
        self.block_size = self.num_actors // settings.num_blocks

    def training_coefficients(self, device) -> torch.Tensor:
        return self.settings.coefficient_values().to(device).repeat_interleave(self.block_size).unsqueeze(-1)

    def entropy_coefficients(self, device) -> torch.Tensor:
        return torch.linspace(0.5, 0.0, self.settings.num_blocks, device=device) * self.settings.entropy_scale

    def coefficient_indices(self, coefficients: torch.Tensor) -> torch.Tensor:
        values = self.settings.coefficient_values().to(device=coefficients.device, dtype=coefficients.dtype)
        distances = torch.abs(coefficients.reshape(-1, 1) - values.reshape(1, -1))
        indices = torch.argmin(distances, dim=1)
        if torch.any(torch.gather(distances, 1, indices.unsqueeze(1)) > 0.0001):
            raise ValueError("observation contains an unknown block coefficient")
        return indices

    def extend_observation_space(self, observation_space: gym.spaces.Dict) -> gym.spaces.Dict:
        spaces = dict(observation_space.spaces)
        spaces[BLOCK_OBS_KEY] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        return gym.spaces.Dict(spaces)

    def attach(self, wrapped_observation: dict, evaluation: bool = False) -> dict:
        observation = wrapped_observation["obs"]
        sample = next(iter(observation.values()))
        batch_size = int(sample.shape[0])
        if evaluation:
            selector = torch.full(
                (batch_size, 1), self.settings.eval_coefficient, device=sample.device, dtype=sample.dtype
            )
        else:
            if batch_size != self.num_actors:
                raise ValueError(f"observation batch={batch_size}, expected num_actors={self.num_actors}")
            selector = self.training_coefficients(sample.device).to(dtype=sample.dtype)
        observation[BLOCK_OBS_KEY] = selector
        return wrapped_observation


def clone_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    if isinstance(value, torch.Tensor):
        return value.clone()
    return value


def slice_tree(value: Any, selection) -> Any:
    if isinstance(value, dict):
        return {key: slice_tree(item, selection) for key, item in value.items()}
    return value[selection].clone()


def concatenate_trees(values: list) -> Any:
    first = values[0]
    if isinstance(first, dict):
        return {key: concatenate_trees([value[key] for value in values]) for key in first}
    return torch.cat(values, dim=0)


def permute_tree(value: Any, permutation: torch.Tensor) -> Any:
    if isinstance(value, dict):
        return {key: permute_tree(item, permutation) for key, item in value.items()}
    shape = value.shape
    return value.reshape(-1, 1, *shape[1:])[permutation].reshape(shape)
