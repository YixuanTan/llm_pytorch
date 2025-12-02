import torch
import torch.nn as nn

from week1.gpt.config import GPTConfig
from week1.gpt.block import Block

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList([Block(config)) for _ in range(config.n_layer])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx, use_kv_cache=False):
        B, T = idx.size()
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        for block in self.blocks:
            x = block(x, use_kv_cache=use_kv_cache)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits