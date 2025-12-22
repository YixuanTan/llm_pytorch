import os
import math
import random
from typing import Tuple, Dict, Any, List

import torch

def _reset_kv_cache_by_scanning_modules(model: torch.nn.Module) -> int:
    cleared = 0
    for m in model.modules():
        if hasattr(m, "key_cache"):
            setattr(m, "key_cache", None)
            cleared += 1
        if hasattr(m, "value_cache"):
            setattr(m, "value_cache", None)
            cleared += 1

    return cleared

def reset_model_cache(model: torch.nn.Module) -> None:
    """
    尽可能清除模型内部的 KV cache。
    如果你有显式 API（比如 model.reset_cache()），你也可以在这里接入。
    """
    # 1) 若你实现了显式方法，优先用
    for meth in ["reset_kv_cache", "reset_cache", "clear_cache", "clear_kv_cache"]:
        if hasattr(model, meth) and callable(getattr(model, meth)):
            getattr(model, meth)()
            return

    # 2) 否则扫描 modules 清空常见字段
    _reset_kv_cache_by_scanning_modules(model)

def model_forward_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    use_kv_cache: bool = False,
    past_length: int = 0,
) -> torch.Tensor:
    """
    统一适配几种常见 forward 签名：
    - logits = model(input_ids)
    - logits, _ = model(input_ids, targets=None)
    - logits = model(input_ids, use_kv_cache=True)
    - logits, loss = model(input_ids, use_kv_cache=True, targets=None)
    """
    try:
        out = model(input_ids, use_kv_cache=use_kv_cache, past_length=past_length)
    except TypeError:
        out = model(input_ids)

    # out may be logits or (logits, loss/aux)
    if isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out

    if not torch.is_tensor(logits):
        raise RuntimeError(f"Model forward did not return a tensor logits. Got type={type(logits)}")

    return logits        


#--------------------
# core test
#--------------------
@torch.no_grad()
def compute_logits_full(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    reset_model_cache(model)
    logits = model_forward_logits(model, input_ids, use_kv_cache=False)
    return logits

@torch.no_grad()
def compute_logits_prefill_then_decode(
    model: torch.nn.Module, input_ids: torch.Tensor, prefill_len: int
) -> torch.Tensor:
    """
    prefill 前缀 + 逐 token decode 返回 [B, T, V] 的 logits (每个位置对应那一步输出)
    约定：第 i 位置的 logits 指的是看到 input_ids[:, :i+1] 后输出的最后一个位置 logits,
          然后我们把它放到 logits_seq[:, i, :]

    prefill_len: 前缀长度（>=1
    """
    B, T = input_ids.shape
    assert 1 <= prefill_len <= T, "prefill_len must be in [1, T]"

    reset_model_cache(model)

    # 1) prefill 
    prefix = input_ids[:, :prefill_len]  # [B, prefill_len]
    logits_prefix = model_forward_logits(model, prefix, use_kv_cache=True)
    V = logits_prefix.size(-1)

    # 2) output container
    logits_seq = torch.empty((B, T, V), device=logits_prefix.device, dtype=logits_prefix.dtype)

    # 3) prefix: 
    logits_seq[:, :prefill_len, :] = logits_prefix

    # 4) decode: 
    for t in range(prefill_len, T):
        one = input_ids[:, t:t+1] # [B, 1]
        logits_step = model_forward_logits(model, one, use_kv_cache=True, past_length=t)
        logits_seq[:, t:t+1, :] = logits_step
    return logits_seq

def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a-b).abs().max().item()

def assert_close(
    a: torch.Tensor,
    b: torch.Tensor,
    atol: float,
    rtol: float,
    msg: str,
) -> None:
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        diff = max_abs_diff(a,b)
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}, atol={atol}, rtol={rtol}")

def main():
    try:
        from week1.gpt.model import GPT
        from week1.gpt.config import GPTConfig
    except Exception as e:
        raise RuntimeError(
            "Please edit the imports in test_kv_sdpa_equivalence.py to match your repo.\n"
            "Expected to import GPT and GPTConfig from gpt.model / gpt.config.\n"
            f"Import error: {e}"
        )

    # Deterministic
    torch.manual_seed(0)
    random.seed(0)

    # Device / dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Note:
    # - CPUT testing reaches bit-level consistency (atol is extremingly small)
    # - GPU bf16 numerical error, loose atol
    use_bf16 = (device=="cuda")

    # Config: small model for test
    vocab_size = 503
    block_size = 128

    cfg = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.0,
    )

    model = GPT(cfg).to(device)
    model.eval()

    for m in model.modules():
        if hasattr(m, "use_flash"):
            try:
                setattr(m, "use_flash", True)
            except Exception:
                pass

    # Optionally run bf16 autocast on CUDA to match your training inference path
    autocast_ctx = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16) if (device=="cuda" and use_bf16) else torch.cpu.amp.autocast()
    )

    # Create random test batch
    B = 2
    T = 32
    input_ids = torch.randint(low=0, high=vocab_size, size=(B, T), device=device, dtype=torch.long)
    
    prefill_list = [1,2,8,16,T] # includes pure prefill (T)
    # Thresholds
    if device == "cpu":
        atol, rtol = 2e-6, 1e-5
    else:
        atol, rtol = 5e-3, 5e-3
    
    print(f"Device={device}, bf16_autocast={use_bf16}, atol={atol}, rtol={rtol}")

    with torch.no_grad():
        with (torch.cuda.amp.autocast(dtype=torch.bfloat16) if (device=="cuda" and use_bf16) else torch.no_grad()):
            logits_full = compute_logits_full(model, input_ids)
    
    for prefill_len in prefill_list:
        with torch.no_grad():
            with (torch.cuda.amp.autocast(dtype=torch.bfloat16) if (device=="cuda" and use_bf16) else torch.no_grad()):
                logits_kv = compute_logits_prefill_then_decode(model, input_ids, prefill_len=prefill_len)


        # We compare the whole [B, T, V]
        assert_close(
            logits_full,
            logits_kv,
            atol=atol,
            rtol=rtol,
            msg=f"KV-cache equivalence failed for refill_len={prefill_len}"
        )
        print(f"prefill_len={prefill_len}: OK (max_abs_diff={max_abs_diff(logits_full, logits_kv):.6f})")

    print("All KV-cache equivalence tests passed.")

if __name__ == "__main__":
    main()