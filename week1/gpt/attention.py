import torch
import torch.nn as nn
import torch.nn.functional as F

from week1.gpt.config import GPTConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig, use_flash: bool = True):
        super().__init__()
        self.config = config
        self.head_dim = config.n_embd // config.n_head
        self.use_flash = use_flash

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.dropout = config.dropout

        # KV cache
        self.register_buffer("key_cache", None, persistent=False)
        self.register_buffer("value_cache", None, persistent=False)

    def reset_kv_cache(self):
        self.key_cache = None
        self.value_cache = None

    def forward(self, x, use_kv_cache=False, past_length=0):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        d_h = C // self.config.n_head
        q = q.view(B, T, self.config.n_head, d_h).transpose(1,2)
        k = k.view(B, T, self.config.n_head, d_h).transpose(1,2)
        v = v.view(B, T, self.config.n_head, d_h).transpose(1,2)

        if use_kv_cache:
            cached_k = k.detach()
            cached_v = v.detach()
            cached_k = k.detach()
            cached_v = v.detach()
            if self.key_cache is None:
                self.key_cache = cached_k
                self.value_cache = cached_v
                self.key_cache = cached_k
                self.value_cache = cached_v
            else:
                self.key_cache = torch.cat([self.key_cache, cached_k], dim=2)
                self.value_cache = torch.cat([self.value_cache, cached_v], dim=2)
                self.key_cache = torch.cat([self.key_cache, cached_k], dim=2)
                self.value_cache = torch.cat([self.value_cache, cached_v], dim=2)

            k_all = self.key_cache
            v_all = self.value_cache
            v_all = self.value_cache
            T_k = k_all.size(2)
        else:
            k_all = k
            v_all = v
            T_k = T
            self.reset_kv_cache()
            self.reset_kv_cache()


        if self.use_flash:
            y = F.scaaled_dot_product_attention(
                q, k, v,
                attn_mask = None,
                dropout_p = self.dropout if self.training else 0.0,
                is_causal = True,
            )
        else:
            att = torch.matmul(q, k_all.transpose(-2, -1)) / (d_h ** 0.5)
            key_positions = torch.arange(T_k, device=x.device)
            query_positions = torch.arange(past_length, past_length + T, device=x.device)
            mask = query_positions.unsqueeze(1) >= key_positions.unsqueeze(0)
            att = att.masked_fill(~mask, float("-inf"))
            att = F.softmax(att, dim=-1)
            out = torch.matmul(att, v_all)

        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)


