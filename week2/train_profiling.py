import os
import sys
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed import fsdp
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

# 确保项目根在 sys.path 中，方便导入 week1.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from week1.gpt.config import GPTConfig
from week1.gpt.model import GPT, Block
from week1.gpt.data import create_dataloader


def setup_distributed():
    if "RANK" not in os.environ:
        print("[setup_distributed] Running single-process mode.")
        rank, world_size = 0, 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return rank, world_size, device, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")

    device = torch.device("cuda", local_rank)
    return rank, world_size, device, True


def build_fsdp_model(config: GPTConfig, device, is_distributed: bool):
    model = GPT(config).to(device)
    
    # In single-process mode, skip FSDP wrapping
    if not is_distributed:
        return model
    
    # bf16（首选） mixed precision policy
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,      # 权重保持 bf16
        reduce_dtype=torch.bfloat16,     # 通信也用 bf16
        buffer_dtype=torch.bfloat16,     # buffer 也用 bf16
    )

    # auto_wrap_policy: only wrap Transformer Blocks
    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        return transformer_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            transformer_layer_cls={Block},
        )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        device_id=device,
        cpu_offload=CPUOffload(offload_params=False),
        use_orig_params=True,
    )
    return fsdp_model


def build_dataloader(text, block_size, batch_size, rank, world_size):
    dataloader, tokenizer = create_dataloader(
        text=text,
        block_size=block_size,
        batch_size=batch_size,
    )

    dataset = dataloader.dataset

    if world_size > 1:
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
            pin_memory=True,
            num_workers=2,
        )
        return dl, tokenizer, sampler
    else:
        return dataloader, tokenizer, None


def log_perf(step, loss, batch_size, seq_len, rank):
    """
    Log performance metrics for the training step
    
    Args:
        step: current training step
        loss: loss value
        batch_size: batch size
        seq_len: sequence length
        rank: process rank
    """
    # Get memory stats
    mem_allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)  # MB
    max_mem_allocated = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    # Calculate throughput metrics
    total_tokens = batch_size * seq_len
    
    print(f"[PERF][rank{rank}] step={step:4d} | "
          f"loss={loss:.4f} | "
          f"batch_size={batch_size} | "
          f"seq_len={seq_len} | "
          f"tokens={total_tokens} | "
          f"mem_alloc={mem_allocated:.1f}MB | "
          f"mem_reserved={mem_reserved:.1f}MB | "
          f"max_mem={max_mem_allocated:.1f}MB")


def train_one_epoch(
    model, dataloader, optimizer, device, rank, world_size,
    sampler=None, max_steps=None
):
    model.train()
    if sampler:
        sampler.set_epoch(0)

    step = 0
    
    # For timing
    step_start_time = None
    step_times = []

    for x, _ in dataloader:
        step_start_time = time.time()
        
        x = x.to(device)
        batch_size, seq_len = x.shape

        # AMP autocast (bf16)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(x)   # forward 是 bf16
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                x[:, 1:].reshape(-1),
            )

        optimizer.zero_grad(set_to_none=True)

        # backward 是 fp32 grad
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Calculate step time
        step_time = time.time() - step_start_time
        step_times.append(step_time)

        # Log performance every 2 steps
        if step % 2 == 0:
            log_perf(step, loss.item(), batch_size, seq_len, rank)
            
            # Also log timing info
            if len(step_times) >= 2:
                avg_step_time = sum(step_times[-10:]) / min(10, len(step_times))
                tokens_per_sec = (batch_size * seq_len) / avg_step_time
                if rank == 0:
                    print(f"[TIMING][rank{rank}] step={step} | "
                          f"step_time={step_time*1000:.1f}ms | "
                          f"avg_step_time={avg_step_time*1000:.1f}ms | "
                          f"tokens/sec={tokens_per_sec:.1f}")

        step += 1
        if max_steps and step >= max_steps:
            break
    
    # Final statistics
    if rank == 0 and len(step_times) > 0:
        avg_time = sum(step_times) / len(step_times)
        print(f"\n[SUMMARY][rank{rank}] Total steps: {step}")
        print(f"[SUMMARY][rank{rank}] Avg step time: {avg_time*1000:.2f}ms")
        print(f"[SUMMARY][rank{rank}] Total training time: {sum(step_times):.2f}s")


from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper, CheckpointImpl, apply_activation_checkpointing
)

def apply_ac_to_blocks(fsdp_model):
    def check_fn(m):
        return isinstance(m, Block)

    wrapper_fn = lambda m: checkpoint_wrapper(
        m,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT
    )

    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=wrapper_fn,
        check_fn=check_fn,
    )


def main():
    rank, world_size, device, is_dist = setup_distributed()
    if rank == 0:
        print(f"world_size={world_size}, device={device}")
        print("=" * 80)
        print("Training with Performance Profiling (logging every 2 steps)")
        print("=" * 80)

    # toy corpus
    toy_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""

    block_size = 64
    batch_size = 16

    dataloader, tokenizer, sampler = build_dataloader(
        text=toy_text,
        block_size=block_size,
        batch_size=batch_size,
        rank=rank,
        world_size=world_size,
    )

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=block_size,
        n_layer=6,
        n_head=6,
        n_embd=384,
    )

    fsdp_model = build_fsdp_model(config, device, is_distributed=is_dist)
    # Activation checkpointing
    apply_ac_to_blocks(fsdp_model)


    if rank == 0:
        print("Model params:", sum(p.numel() for p in fsdp_model.parameters()) / 1e6, "M")

    optimizer = optim.AdamW(fsdp_model.parameters(), lr=3e-4)

    torch.cuda.reset_peak_memory_stats()

    # train 50 steps with profiling
    if rank == 0:
        print("\n" + "=" * 80)
        print("Starting training...")
        print("=" * 80 + "\n")
        
    train_one_epoch(
        fsdp_model,
        dataloader,
        optimizer,
        device,
        rank,
        world_size,
        sampler,
        max_steps=50,
    )

    peak = torch.cuda.max_memory_allocated() / (1024**2) # unit in mb
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"[FINAL] Peak memory usage: {peak:.2f} MB")
        print("=" * 80)

    # 保存 checkpoint：所有 rank 都必须调用 state_dict()（集体操作），但只有 rank 0 保存到磁盘
    if is_dist:
        if rank == 0:
            print("\nSyncing all ranks for checkpoint...")
        dist.barrier()
        
        # All ranks must call state_dict() (collective operation)
        if rank == 0:
            print("Saving checkpoint (all ranks participate in state_dict)")
        state = fsdp_model.state_dict()
        
        # Only rank 0 saves to disk
        if rank == 0:
            os.makedirs("ckpt_fsdp_profiling", exist_ok=True)
            torch.save(state, "ckpt_fsdp_profiling/model.pt")
            print("Checkpoint saved to ckpt_fsdp_profiling/model.pt")
        
        dist.barrier()
        dist.destroy_process_group()
    else:
        # Single process: just save normally
        if rank == 0:
            print("saving checkpoint")
            state = fsdp_model.state_dict()
            os.makedirs("ckpt_fsdp_profiling", exist_ok=True)
            torch.save(state, "ckpt_fsdp_profiling/model.pt")
            print("Checkpoint saved.")


if __name__ == "__main__":
    main()
