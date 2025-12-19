import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import(
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
import torch.nn.functional as F

# Ensure project root is on sys.path for relative imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from week1.gpt.config import GPTConfig
from week1.gpt.model import GPT, Block
from week1.gpt.data import create_dataloader

def setup_distributed():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("[setup_distributed] No RANK/WORLD_SIZE found, running in single-process mode.")
        rank = 0
        world_size = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return rank, world_size, device, False  # is_distributed=False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.empty_cache()

    return rank, world_size, device, True

def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def build_fsdp_model(config: GPTConfig, device, mp_dtype=torch.bfloat16):
    torch.manual_seed(42)
    model = GPT(config).to(device)

    # auto_wrap_policy: only wrap Transformer Blocks and embeddings
    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        return transformer_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            transformer_layer_cls={Block},
        )

    mp_policy = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device,
        cpu_offload=CPUOffload(offload_params=False),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
    )
    return fsdp_model

def build_dataloader(text, block_size, batch_size, world_size, rank):
    dataloader, tokenizer = create_dataloader(
        text=text,
        block_size=block_size,
        batch_size=batch_size,
    )

    if world_size > 1:
        dataset = dataloader.dataset
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        dl = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=2,
            pin_memory=True,
        )
        return dl, tokenizer, sampler
    else:
        return dataloader, tokenizer, None


def train_one_epoch(model, dataloader, optimizer, device, rank, sampler=None, max_steps=None):
    model.train()
    if sampler is not None:
        sampler.set_epoch(0)

    step = 0
    total_loss = 0.0
    for x, _ in dataloader:
        x = x.to(device)
        logits = model(x)

        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            x[:, 1:].reshape(-1),
        )

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        # optional: gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        # if step % 10 == 0 and rank == 0:
            print(f"[rank {rank}] step {step}, loss={loss.item():.4f}")
        
        step += 1
        if max_steps is not None and step >= max_steps:
            break

    avg_loss = total_loss / max(1, step)
    if rank == 0:
        print(f"[rank 0] epoch done, avg_loss={avg_loss:.4f}")


##########################
# Trainer
##########################
rank, world_size, device, is_distributed = setup_distributed()
print(f"[rank {rank}] world_size={world_size}, device={device}")

# 玩具文本（你可以换成更大的 corpus）
toy_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""

block_size = 128
batch_size = 16

dataloader, tokenizer, sampler = build_dataloader(
    text=toy_text,
    block_size=block_size,
    batch_size=batch_size,
    world_size=world_size,
    rank=rank,
)

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=4,
    n_head=4,
    n_embd=256,
    dropout=0.0,
)

fsdp_model = build_fsdp_model(config, device, mp_dtype=torch.bfloat16)

if rank == 0:
    n_params = sum(p.numel() for p in fsdp_model.parameters())
    print(f"[rank 0] Model params: {n_params/1e6:.2f}M")

optimizer = optim.AdamW(fsdp_model.parameters(), lr=3e-4)

# 简单 train 几个 step 看看 FSDP 是否正常工作
train_one_epoch(
    model=fsdp_model,
    dataloader=dataloader,
    optimizer=optimizer,
    device=device,
    rank=rank,
    sampler=sampler,
    max_steps=50,
)

# 保存 checkpoint（只在 rank 0），但所有 rank 参与 FULL_STATE_DICT 的 all_gather。
dist.barrier()
with FSDP.state_dict_type(
    fsdp_model,
    StateDictType.FULL_STATE_DICT,
    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
):
    cpu_state = fsdp_model.state_dict()
    if rank == 0:
    os.makedirs("checkpoints_fsdp", exist_ok=True)
    torch.save(cpu_state, "checkpoints_fsdp/gpt_fsdp_rank0.pt")
    print("[rank 0] checkpoint saved.")
dist.barrier()

cleanup_distributed()