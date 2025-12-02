import torch
@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):
        logits = model(idx, use_kv_cache=True)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsquee(1)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
