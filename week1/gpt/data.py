from typing import Any


import torch 
from torch.utils.data import Dataset, DataLoader

class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

class CharDataset(Dataset):
    def __init__(self, text, block_size, tokenizer):
        self.block_size = block_size
        self.tokenizer = tokenizer 

        data_ids = tokenizer.encode(text)
        self.data = torch.tensor(data_ids, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.block_size]
        y = self.data[idx+1 : idx+1+self.block_size]
        return x, y

def create_dataloader(text, block_size, batch_size):
    tokenizer = CharTokenizer(text)
    dataset = CharDataset(text, block_size, tokenizer)
    loader = DataLoader[Any](dataset, batch_size=batch_size, shuffle=True)
    return loader, tokenizer

