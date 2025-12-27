import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)

from model import ModelConfig, TransformerLM, Block
from data import RandomTokenDataset, MemmapTokenDataset
from utils import rank0, all_reduce_mean, count_params, ThroughputMeter, cosine_lr

def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return True, local_rank
    return False, 0

def get_model_cfg(preset: str, vocab_size: int, seq_len: int, flash: bool, act_ckpt: bool):
    if preset == "8m":
        cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=seq_len, 
                          n_layers=4, n_heads=4, d_model=256, d_ff=1024, use_flash=flash, activation_checkpointing=act_ckpt)
    elif preset == "100m":
        cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=seq_len,
                          n_layers=12, n_heads=12, d_model=768, d_ff=3072,
                          use_flash=flash, activation_checkpointing=act_ckpt)
    elif preset == "300m":
        cfg = ModelConfig(vocab_size=vocab_size, max_seq_len=seq_len,
                          n_layers=24, n_heads=16, d_model=1024, d_ff=4096,
                          use_flash=flash, activation_checkpointing=act_ckpt)
    else:
        raise ValueError(f"unknown preset: {preset}")
    return cfg

def build_fsdp(model: torch.nn.Module, use_bf16: bool, act_ckpt: bool):
    mp = None
    if use_bf16:
        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    # FSDP 自动wrap 每个transformer Block
    # transformer_auto_wrap_policy is a partial function factory
    def auto_wrap_policy(module, recurse, nonwrapped_numel):
        return transformer_auto_wrap_policy(
            module=module,
            recurse=recurse,
            nonwrapped_numel=nonwrapped_numel,
            transformer_layer_cls={Block},
        )

    fsdp = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=torch.cuda.current_device(),
        limit_all_gathers=True,
        use_orig_params=True,
    )

    if act_ckpt:
        def check_fn(m):
            return isinstance(m, Block)

        apply_activation_checkpointing(
            fsdp,
            checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
                m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
            ),
            check_fn=check_fn,
        )
    return fsdp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", type=str, default="8m", choices=["8m", "100m", "300m"])
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=50304)

    parser.add_argument("--global_batch_tokens", type=int, default=2_097_152,
                        help="全局每 step 的 token 数（= world_size * micro_bsz * seq_len * grad_accum）")
    parser.add_argument("--micro_bsz", type=int, default=2, help="每 GPU micro batch size")
    parser.add_argument("--grad_accum", type=int, default=8)

    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max_lr", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--clip_grad", type=float, default=1.0)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--flash", action="store_true")
    parser.add_argument("--act_ckpt", action="store_true")

    parser.add_argument("--data", type=str, default="random", choices=["random", "memmap"])
    parser.add_argument("--num_tokens", type=int, default=100_000_000, help="random 数据总 token 数")
    parser.add_argument("--bin_path", type=str, default="", help="memmap: token.bin path")
    parser.add_argument("--log_every", type=int, default=10)
    
    # Profiling arguments
    parser.add_argument("--profile", action="store_true", help="启用 nsys profiling")
    parser.add_argument("--profile_start_step", type=int, default=10, help="开始 profiling 的 step")
    parser.add_argument("--profile_end_step", type=int, default=15, help="结束 profiling 的 step")
    args = parser.parse_args()

    is_ddp, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Dataset / Loader
    if args.data == "random":
        ds = RandomTokenDataset(
            num_tokens=args.num_tokens,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            seed=1234+(dist.get_rank() if is_ddp else 0),
        )
    else:
        ds = MemmapTokenDataset(args.bin_path, args.seq_len, args.vocab_size, dtype="uint16")

    sampler = DistributedSampler(ds, shuffle=True) if is_ddp else None
    dl = DataLoader(
        ds,
        batch_size=args.micro_bsz,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    it = iter(dl)

    # Model
    cfg = get_model_cfg(args.preset, args.vocab_size, args.seq_len, args.flash, args.act_ckpt)
    model = TransformerLM(cfg).to(device)
    if rank0():
        print(f"[model] preset={args.preset} params={count_params(model)/1e6:.2f}M")

    # FSDP
    if is_ddp:
        model = build_fsdp(model, use_bf16=args.bf16, act_ckpt=args.act_ckpt)

    # Optim
    # demo：AdamW，真实大训练可换 fused AdamW（依赖 apex/torch nightly/实现细节）
    opt = torch.optim.AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)

    scaler = None
    use_amp = args.bf16 and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_amp else None

    # sanity: global batch tokens (用于做 batch/lr scaling)
    world = dist.get_world_size() if is_ddp else 1
    implied_tokens = world * args.micro_bsz * args.seq_len * args.grad_accum
    if rank0():
        print(f"[batch] world={world} micro_bsz={args.micro_bsz} seq_len={args.seq_len} "
              f"grad_accum={args.grad_accum} => global_tokens/step={implied_tokens}")

    meter = ThroughputMeter()
    model.train()
    
    # nsys profiling setup
    if args.profile and rank0():
        print(f"[profiling] enabled from step {args.profile_start_step} to {args.profile_end_step}")

    for step in range(args.steps):
        # nsys profiling range markers using CUDA Profiler API
        if args.profile and torch.cuda.is_available():
            if step == args.profile_start_step:
                torch.cuda.cudart().cudaProfilerStart()
                if rank0():
                    print(f"[profiling] started at step {step}")
            elif step == args.profile_end_step:
                torch.cuda.cudart().cudaProfilerStop()
                if rank0():
                    print(f"[profiling] stopped at step {step}")
        
        if sampler is not None:
            sampler.set_epoch(step)

        # lr schedule (cosine + warmup)
        lr = cosine_lr(step, args.warmup, args.steps, args.max_lr, args.min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr

        opt.zero_grad(set_to_none=True)

        loss_accum = 0.0
        
        # Mark training iteration for profiler
        if args.profile and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(f"step_{step}")
        
        for micro in range(args.grad_accum):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(dl)
                x, y = next(it)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Mark forward/backward pass
            if args.profile and torch.cuda.is_available():
                torch.cuda.nvtx.range_push(f"micro_{micro}")
            
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                loss = loss / args.grad_accum

            loss.backward()
            loss_accum += loss.detach().float()

            meter.update(x.numel())
            
            if args.profile and torch.cuda.is_available():
                torch.cuda.nvtx.range_pop()

        # grad clip (FSDP 下建议在root上clip; 这里用通用写法)
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        opt.step()
        
        if args.profile and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()

        # reduce loss for logging 
        loss_mean = all_reduce_mean(loss_accum) if is_ddp else loss_accum
        if rank0() and (step % args.log_every == 0 or step == args.steps - 1):
            tps, dt = meter.get()
            # 简单打印显存峰值
            peak = torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            print(f"step {step:04d} | loss {loss_mean.item():.4f} | lr {lr:.2e} | "
                  f"tok/s {tps:.0f} | peak_mem {peak:.2f} GB")

        # reset peak stats periodically
        if torch.cuda.is_available() and step % args.log_every == 0:
            torch.cuda.reset_peak_memory_stats()

    if is_ddp:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()





