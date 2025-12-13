import torch
import torch.nn as nn

from week1.gpt.attention import MultiHeadAttention
from week1.gpt.config import GPTConfig

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.n_embd)

    def forward(self, x):
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, use_kv_cache=False, past_length=0):
        x = x + self.attn(self.ln1(x), use_kv_cache=use_kv_cache, past_length=past_length)
        x = x + self.mlp(self.ln2(x))
        return x

    def reset_kv_cache(self):
        self.attn.reset_kv_cache()
