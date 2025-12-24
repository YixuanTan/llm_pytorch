import os
import sys
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
    StateDictType,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# Import distributed checkpoint utilities
from torch.distributed.checkpoint import save, load
from torch.distributed.checkpoint.state_dict import (
    get_state_dict,
    set_state_dict,
    StateDictOptions,
)

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

def train_one_epoch(
    model, dataloader, optimizer, device, rank,
    sampler=None, max_steps=None
):
    model.train()
    if sampler:
        sampler.set_epoch(0)

    step = 0

    for x, _ in dataloader:
        x = x.to(device)

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

        if rank == 0 and step % 10 == 0:
            print(f"[rank0] step={step}, loss={loss.item():.4f}")

        step += 1
        if max_steps and step >= max_steps:
            break

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

    # train 50 steps just to test
    train_one_epoch(
        fsdp_model,
        dataloader,
        optimizer,
        device,
        rank,
        sampler,
        max_steps=50,
    )

    peak = torch.cuda.max_memory_allocated() / (1024**2) # unit in mb
    if rank == 0:
        print(f"[rank0] peak mem={peak:.2f} MB")

    # 保存 SHARDED_STATE_DICT：每个 rank 保存自己的分片，更高效
    if is_dist:
        if rank == 0:
            print("sync all ranks for sharded checkpoint")
        dist.barrier()
        
        # ============ SAVE SHARDED CHECKPOINT ============
        if rank == 0:
            print("Saving SHARDED checkpoint (each rank saves its own shard)")
        
        # 创建保存目录（所有 rank 都需要访问）
        checkpoint_dir = "ckpt_fsdp_sharded"
        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
        dist.barrier()  # 确保目录创建完成
        
        # 使用新的 distributed checkpoint API
        # 所有 rank 参与，每个 rank 保存自己的分片
        state_dict = {
            "model": fsdp_model.state_dict(),
        }
        
        save(
            state_dict=state_dict,
            storage_writer=None,  # 使用默认的文件系统存储
            checkpoint_id=checkpoint_dir,
        )
        
        if rank == 0:
            print(f"Sharded checkpoint saved to {checkpoint_dir}/")
            print(f"Each rank saved its own shard independently")
        
        dist.barrier()
        
        # ============ LOAD SHARDED CHECKPOINT ============
        if rank == 0:
            print("\nNow loading the sharded checkpoint back...")
        
        # 创建一个新的模型来加载 checkpoint
        new_model = GPT(config).to(device)
        new_fsdp_model = FSDP(
            new_model,
            auto_wrap_policy=lambda module, recurse, nonwrapped_numel: transformer_auto_wrap_policy(
                module=module,
                recurse=recurse,
                nonwrapped_numel=nonwrapped_numel,
                transformer_layer_cls={Block},
            ),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            ),
            device_id=device,
            cpu_offload=CPUOffload(offload_params=False),
            use_orig_params=True,
        )
        
        # 加载 sharded checkpoint（所有 rank 参与）
        state_dict_to_load = {
            "model": new_fsdp_model.state_dict(),
        }
        
        load(
            state_dict=state_dict_to_load,
            storage_reader=None,  # 使用默认的文件系统读取
            checkpoint_id=checkpoint_dir,
        )
        
        # 将加载的 state_dict 设置到模型
        new_fsdp_model.load_state_dict(state_dict_to_load["model"])
        
        if rank == 0:
            print("Sharded checkpoint loaded successfully!")
            
        # 可选：验证加载的模型与原模型参数一致
        if rank == 0:
            print("Verifying loaded parameters match original...")
        
        # 比较一些参数来验证
        with FSDP.summon_full_params(fsdp_model):
            with FSDP.summon_full_params(new_fsdp_model):
                orig_params = list(fsdp_model.parameters())
                loaded_params = list(new_fsdp_model.parameters())
                
                if rank == 0:
                    # 检查第一个参数
                    if len(orig_params) > 0 and len(loaded_params) > 0:
                        param_match = torch.allclose(
                            orig_params[0].cpu().float(), 
                            loaded_params[0].cpu().float(),
                            rtol=1e-4, atol=1e-4
                        )
                        print(f"First parameter match: {param_match}")
                        
                        if param_match:
                            print("✓ Checkpoint load verification passed!")
                        else:
                            print("✗ Warning: Parameters don't match exactly")
        
        dist.barrier()
        dist.destroy_process_group()
    else:
        # Single process: just save normally
        if rank == 0:
            print("saving checkpoint (single process mode)")
            state = fsdp_model.state_dict()
            os.makedirs("ckpt_fsdp_sharded", exist_ok=True)
            torch.save(state, "ckpt_fsdp_sharded/model.pt")
            print("Checkpoint saved.")


if __name__ == "__main__":
    main()
