"""
Dataset classes for pretraining
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    """
    Generate random tokens on-the-fly for testing/debugging purposes.
    Useful for benchmarking without real data.
    """
    def __init__(self, num_tokens: int, seq_len: int, vocab_size: int, seed: int = 42):
        """
        Args:
            num_tokens: Total number of tokens in the dataset
            seq_len: Sequence length for each sample
            vocab_size: Vocabulary size
            seed: Random seed for reproducibility
        """
        self.num_tokens = num_tokens
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.seed = seed
        
        # Number of samples = total_tokens / seq_len
        self.num_samples = num_tokens // seq_len
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns (x, y) where y is x shifted by 1 position
        """
        # Use idx as part of seed for reproducibility
        rng = np.random.default_rng(seed=self.seed + idx)
        
        # Generate random tokens
        tokens = rng.integers(0, self.vocab_size, size=self.seq_len + 1, dtype=np.int64)
        
        x = torch.from_numpy(tokens[:-1].copy())  # input: [0, seq_len-1]
        y = torch.from_numpy(tokens[1:].copy())   # target: [1, seq_len]
        
        return x, y


class MemmapTokenDataset(Dataset):
    """
    Load pre-tokenized data from a memory-mapped binary file.
    Efficient for large datasets that don't fit in memory.
    
    Expected format: binary file with tokens stored as uint16 or uint32.
    """
    def __init__(self, bin_path: str, seq_len: int, vocab_size: int, dtype: str = "uint16"):
        """
        Args:
            bin_path: Path to the binary file containing tokens
            seq_len: Sequence length for each sample
            vocab_size: Vocabulary size (for validation)
            dtype: Data type in the binary file ("uint16" or "uint32")
        """
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file not found: {bin_path}")
        
        self.bin_path = bin_path
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Map numpy dtype
        if dtype == "uint16":
            self.dtype = np.uint16
        elif dtype == "uint32":
            self.dtype = np.uint32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}. Use 'uint16' or 'uint32'")
        
        # Memory-map the file
        self.data = np.memmap(bin_path, dtype=self.dtype, mode='r')
        
        # Calculate number of samples
        # We need seq_len + 1 tokens per sample (for input and shifted target)
        self.num_samples = (len(self.data) - 1) // seq_len
        
        print(f"[MemmapTokenDataset] Loaded {len(self.data):,} tokens from {bin_path}")
        print(f"[MemmapTokenDataset] {self.num_samples:,} samples of length {seq_len}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Returns (x, y) where y is x shifted by 1 position
        """
        start_idx = idx * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        
        # Extract tokens
        tokens = self.data[start_idx:end_idx].astype(np.int64)
        
        # Validate token range
        if tokens.max() >= self.vocab_size:
            raise ValueError(f"Token {tokens.max()} exceeds vocab_size {self.vocab_size}")
        
        x = torch.from_numpy(tokens[:-1].copy())  # input: [0, seq_len-1]
        y = torch.from_numpy(tokens[1:].copy())   # target: [1, seq_len]
        
        return x, y


def create_dataloader(dataset, batch_size, is_distributed=False, num_workers=2, pin_memory=True, drop_last=True):
    """
    Create a DataLoader with optional distributed sampling
    
    Args:
        dataset: Dataset instance
        batch_size: Batch size per GPU
        is_distributed: Whether to use DistributedSampler
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory
        drop_last: Whether to drop the last incomplete batch
    
    Returns:
        DataLoader instance
    """
    sampler = None
    shuffle = True
    
    if is_distributed:
        sampler = DistributedSampler(dataset, shuffle=True)
        shuffle = False  # Sampler handles shuffling
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    
    return dataloader


if __name__ == "__main__":
    # Test RandomTokenDataset
    print("Testing RandomTokenDataset...")
    ds = RandomTokenDataset(num_tokens=100000, seq_len=128, vocab_size=50000, seed=42)
    print(f"Dataset length: {len(ds)}")
    
    x, y = ds[0]
    print(f"Sample shape: x={x.shape}, y={y.shape}")
    print(f"x[:10] = {x[:10]}")
    print(f"y[:10] = {y[:10]}")
    
    # Verify y is x shifted by 1
    x2, y2 = ds[0]
    assert torch.all(x[1:] == y[:-1]), "y should be x shifted by 1"
    print("✓ RandomTokenDataset test passed")
