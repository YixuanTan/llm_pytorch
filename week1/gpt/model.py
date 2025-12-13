import torch
import torch.nn as nn
import torch.nn.functional as F

from week1.gpt.config import GPTConfig
from week1.gpt.block import Block

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, use_kv_cache=False, past_length=0):
        B, T = idx.size()

        device = idx.device
        pos = torch.arange(past_length, past_length + T, device=device)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)

        x = tok_emb + pos_emb

        x = self.drop(x)

        for block in self.blocks:
            x = block(x, use_kv_cache=use_kv_cache, past_length=past_length)

        x = self.ln_f(x)
        logits = self.head(x)

        return logits

    def reset_kv_cache(self):
        for block in self.blocks:
            block.reset_kv_cache()