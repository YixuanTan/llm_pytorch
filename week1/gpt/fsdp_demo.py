""" To run single node, 2 GPUs example:

torchrun --standalone --nproc_per_node=2 fsdp_demo.py
"""


import os

import torch
from torch.cpu import is_available
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

def setup_distributed():
    """
    如果在 torchrun 环境下（有 RANK/WORLD_SIZE），初始化分布式；
    否则退化为单进程单卡模式，方便在 notebook 里调试。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_distributed = True
    else:
        print("[setup_distributed] No RANK/WORLD_SIZE found, running in single-process mode.")
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        is_distributed = False
    return local_rank, is_distributed

def cleanup_distributed(is_distributed):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

def run_fsdp_demo():
    local_rank, is_distributed = setup_distributed()
    rank = dist.get_rank() if is_distributed else 0

    if not torch.cuda.is_available():
        raise RuntimeError("FSDP demo requires GPUs (torch.cuda.is_available() == False).")

    device = torch.device("cuda", local_rank)

    # 一个小模型做 demo，你可以替换成自己的大模型
    model = nn.Sequential(
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
    ).to(device)

    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    fsdp_model = FSDP(model, mixed_precision=mp, device_id=device)

    print(f"[FSDP] Running with world_size={dist.get_world_size() if is_distributed else 1}, "
          f"rank={rank}, local_rank={local_rank}")

    x = torch.randn(32, 4096, device=device)
    loss = fsdp_model(x).pow(2).mean()
    loss.backward()

    print(f"[FSDP] rank {rank} finished one backward step, loss={loss.item():.4f}")

    cleanup_distributed(is_distributed)


run_fsdp_demo()