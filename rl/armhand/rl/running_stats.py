import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from gym import spaces


class RunningMeanStd(nn.Module):
    def __init__(self, insize, epsilon=1e-05):
        super().__init__()
        self.insize = insize
        self.epsilon = epsilon
        self.axis = [0]
        self.register_buffer("running_mean", torch.zeros(insize, dtype=torch.float64))
        self.register_buffer("running_var", torch.ones(insize, dtype=torch.float64))
        self.register_buffer("count", torch.ones((), dtype=torch.float64))

    @staticmethod
    def _update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count
        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * count * batch_count / tot_count
        return new_mean, m2 / tot_count, tot_count

    def forward(self, input, denorm=False):
        if self.training:
            values = input.detach().to(dtype=torch.float64)
            axes = tuple(self.axis)
            local_sum = values.sum(dim=axes)
            local_sumsq = values.square().sum(dim=axes)
            batch_count = torch.tensor(
                float(np.prod([input.shape[axis] for axis in axes])), dtype=torch.float64, device=input.device
            )
            if dist.is_available() and dist.is_initialized():
                moment_size = local_sum.numel()
                packed = torch.cat((local_sum.reshape(-1), local_sumsq.reshape(-1), batch_count.reshape(1)))
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                local_sum = packed[:moment_size].view_as(local_sum)
                local_sumsq = packed[moment_size : 2 * moment_size].view_as(local_sumsq)
                batch_count = packed[-1]
            safe_count = batch_count.clamp_min(1.0)
            mean = local_sum / safe_count
            var = (local_sumsq / safe_count - mean.square()).clamp_min(0.0)
            new_mean, new_var, new_count = self._update_mean_var_count_from_moments(
                self.running_mean, self.running_var, self.count, mean, var, safe_count
            )
            self.running_mean.copy_(new_mean)
            self.running_var.copy_(new_var)
            self.count.copy_(new_count)
        if denorm:
            y = torch.clamp(input, min=-20.0, max=20.0)
            return torch.sqrt(self.running_var.float() + self.epsilon) * y + self.running_mean.float()
        y = (input - self.running_mean.float()) / torch.sqrt(self.running_var.float() + self.epsilon)
        return torch.clamp(y, min=-20.0, max=20.0)


class RunningMeanStdObs(nn.Module):
    def __init__(self, insize: spaces.Dict, exclude_keys=None):
        super().__init__()
        exclude_keys = exclude_keys or []
        self.running_mean_std = nn.ModuleDict(
            {k: RunningMeanStd(v.shape) for k, v in insize.items() if k not in exclude_keys}
        )
        self._exclude_keys = exclude_keys

    def forward(self, input):
        return {k: v if k in self._exclude_keys else self.running_mean_std[k](v) for k, v in input.items()}
