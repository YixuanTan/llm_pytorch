""" To run:
torchrun --standalone --nproc_per_node=2 ddp_demo.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision

def setup_distributed():
    """
    如果在 torchrun 环境下（有 RANK/WORLD_SIZE），初始化 DDP；
    否则退化为单进程单卡模式，方便在 notebook 里调试。
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # 多进程模式：由 torchrun 设置好 env
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_distributed = True
    else:
        # notebook / 单进程模式
        print("[setup_distributed] No RANK/WORLD_SIZE found, running in single-process mode.")
        local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        is_distributed = False

    return local_rank, is_distributed


def cleanup_distributed(is_distributed: bool):
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()

def run_ddp_demo():
    local_rank, is_distributed = setup_distributed()

    if is_distributed:
        rank = dist.get_rank()
    else:
        rank = 0

    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    model = nn.Linear(4096, 4096).to(device)

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
        print(f"[DDP] Running with world_size={dist.get_world_size()}, rank={rank}, local_rank={local_rank}")
    else:
        print("[DDP] Running in single-process, non-distributed mode.")

    x = torch.randn(32, 4096, device=device)
    loss = model(x).pow(2).mean()
    loss.backward()

    print(f"[DDP] rank {rank} finished one backward step, loss={loss.item():.4f}")

    cleanup_distributed(is_distributed)

if __name__ == "__main__":
    run_ddp_demo()
