import torch
from torch.utils.data import Dataset

class CopyDataset(Dataset):
    """
    Patrón repetitivo. El modelo debe aprender a predecir la secuencia.
    Ej: [1, 2, 3, 4, 1, 2, 3, 4...]
    """
    def __init__(self, vocab_size=50, seq_len=128, length=100, pattern_len=8):
        self.length = length
        self.data = []
        for _ in range(length):
            pattern = torch.randint(1, vocab_size, (pattern_len,))
            seq = pattern.repeat(seq_len // pattern_len + 1)[:seq_len+1]
            self.data.append(seq)
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]

class NeedleDataset(Dataset):
    """
    Recuperación de clave-valor.
    Formato: [PAD, ..., KEY_TOKEN, VAL_TOKEN, PAD, ..., QUERY_TOKEN, VAL_TOKEN (target)]
    """
    def __init__(self, vocab_size=50, seq_len=512, length=100, needle_distance=100):
        self.length = length
        self.data = []
        key_token = vocab_size - 1
        query_token = vocab_size - 2
        
        for _ in range(length):
            seq = torch.zeros(seq_len + 1, dtype=torch.long) # 0 is PAD
            val_token = torch.randint(1, vocab_size - 2, (1,)).item()
            
            # Place needle
            needle_pos = max(0, seq_len - needle_distance - 2)
            seq[needle_pos] = key_token
            seq[needle_pos + 1] = val_token
            
            # Place query at the end
            seq[-2] = query_token
            seq[-1] = val_token # This is the target for the query token
            
            self.data.append(seq)
            
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]
