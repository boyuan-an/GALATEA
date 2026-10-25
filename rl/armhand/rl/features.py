import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.output_dim = input_dim

    def forward(self, x):
        return x


class DictFeatureConcat(nn.Module):
    def __init__(self, extractors: dict):
        super().__init__()
        self._extractors = nn.ModuleDict(extractors)
        self._ordered_keys = tuple(extractors.keys())
        self.output_dim = sum(extractor.output_dim for extractor in extractors.values())

    def forward(self, obs: dict):
        return torch.cat([self._extractors[key](obs[key]) for key in self._ordered_keys], dim=-1)
