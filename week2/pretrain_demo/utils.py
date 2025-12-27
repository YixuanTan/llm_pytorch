import math
import time
import torch
import torch.distributed as dist

def is_dist():
    return dist.is_available() and dist.is_initialized()

def rank0():
    return (not is_dist()) or dist.get_rank() == 0

def all_reduce_mean(x: torch.Tensor):
    if not is_dist():
        return x
    y = x.clone()
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    y /= dist.get_world_size()
    return y

def count_params(model: torch.nn.Module):
    return sum(p.nummel() for p in model.parameters())


class ThroughputMeter:
    def __init__(self):
        self.t0 = time.time()
        self.tokens = 0

    def update(self, num_tokens: int):
        self.tokens += num_tokens

    def get(self):
        dt = time.time() - self.t0
        return self.tokens / max(dt, 1e-6), dt

def cosine_lr(step, warmup, total, max_lr, min_lr):
    if step < warmup:
        return max_lr * step / max(1, warmup)
    if step >= total:
        return min_lr
    t = (step - warmup) / max(1, total - warmup)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * t))


    