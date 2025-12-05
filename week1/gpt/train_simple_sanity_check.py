import torch
import torch.optim as optim

from week1.gpt.config import GPTConfig
from week1.gpt.model import GPT
from data import create_dataloader

import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

toy_text = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them.
"""
block_size = 64
batch_size = 32

dataloader, tokenizer = create_dataloader(
    text=toy_text,
    block_size=block_size,
    batch_size=batch_size,
)

config = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    block_size=block_size,
    n_layer=2,
    n_head=2,
    n_embd=128,
    dropout=0.0,
)

model = GPT(config).to(device)
print("Model parameters:", sum(p.numel() for p in model.parameters())/ 1e6, "M")

optimizer = optim.AdamW(model.parameters(), lr=3e-4)

model.train()
max_steps = 50

step = 0
for epoch in range(10):
    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x, y)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, config.vocab_size), x[:, 1:].reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step:04d} | loss = {loss.item():.4f}")

        step += 1
        if step >= max_steps:
            break

    if step >= max_steps:
        break

model.eval()
with torch.no_grad():
    prompt = x[0:1, :16]
    out_ids = prompt.clone()

    for _ in range(32):
        logits = model(out_ids)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        out_ids = torch.cat([out_ids, next_token], dim=1)

    generated = tokenizer.decode(out_ids[0].cpu().tolist())
    print("========= Generated sample =====")
    print(generated)




