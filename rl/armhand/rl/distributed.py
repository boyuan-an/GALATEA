import torch
import torch.distributed as dist


def synchronized_normalize(values, *, multi_gpu, epsilon=1e-08):
    detached = values.detach()
    count = torch.tensor(float(detached.numel()), dtype=detached.dtype, device=detached.device)
    local = torch.stack((detached.sum(), detached.square().sum(), count))
    if multi_gpu:
        dist.all_reduce(local, op=dist.ReduceOp.SUM)
    count = local[2].clamp_min(1.0)
    mean = local[0] / count
    variance = (local[1] / count - mean.square()).clamp_min(0.0)
    normalized = (values - mean) / torch.sqrt(variance + epsilon)
    return normalized
