import torch
import torch.nn as nn
import torch.nn.functional as F

from week1.gpt.config import GPTConfig

class MultiHeadAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.head_dim = config.n_embd // config.n_head

        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        # KV cache
        self.register_buffer("key_cache", None, persistent=False)
        self.register_buffer("value_cache", None, persistent=False)

    def forward(self, x, use_kv_cache=False):
        B, T, C = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        d_h = C // self.config.n_head
        q = q.view(B, T, self.config.n_head, d_h).transpose(1,2)
        k = k.view(B, T, self.config.n_head, d_h).transpose(1,2)
        v = v.view(B, T, self.config.n_head, d_h).transpose(1,2)

        if use_kv_cache:
            if self.key_cache is None:
                self.key_cache = k
                self.value_cache = v
            else:
                self.key_cache = torch.cat([self.key_cache, k], dim=2)
                self.value_cache = torch.cat([self.value_cache, v], dim=2)

            k_all = self.key_cache
            v_all = self.value_cahce
            T_k = k_all.size(2)

        else:
            k_all = k
            v_all = v
            T_k = T

        # scaled dot-product attention
        att = torch.matmul(q, k_all.transpose(-2, -1)) / (d_h ** 0.5)
        mask = torch.tril(torch.ones(T, T_k, device=x.device))
        att = att.masked_fill(mask==0, float("-inf"))
        att = F.softmax(att, dim=-1)
        out = torch.matmul(att, v_all)

        out = out.transpose(1,2).contiguous().view(B, T, C)
        return self.out_proj(out)


