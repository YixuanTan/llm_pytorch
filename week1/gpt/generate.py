import torch


@torch.no_grad()
def generate(model, idx, max_new_tokens):
    model.eval()
    device = next(model.parameters()).device
    idx = idx.to(device)
    model.reset_kv_cache()

    for _ in range(max_new_tokens):
        past_length = idx.size(1) - 1
        logits = model(idx[:, -1:], use_kv_cache=True, past_length=past_length)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        idx = torch.cat([idx, next_token], dim=1)
    return idx
