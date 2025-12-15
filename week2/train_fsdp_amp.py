import os
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, DistributedSampler

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from week1.gpt.config import GPTConfig
from week1.gpt.model import GPT, Block
from week1.gpt.data import create_dataloader

