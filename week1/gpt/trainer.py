import torch.nn.functional as F


def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        batch = batch.to(device)
        logits = model(batch[:, :-1])
        loss = F.cross_entroy(logits.reshape(-1, logits.size(-1)), batch[:,1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        