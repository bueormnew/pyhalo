import torch
from torch.utils.data import Dataset
from halo.tokenizers.base import BaseTokenizer

class TextDataset(Dataset):
    """
    Dataset simple que lee un archivo de texto en memoria (ideal para experimentación rápida).
    Genera tensores X (input) e Y (target desplazado).
    """
    def __init__(self, file_path: str, tokenizer: BaseTokenizer, max_seq_len: int):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        self.tokens = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        
        if len(self.tokens) <= self.max_seq_len:
            raise ValueError(f"El texto debe tener al menos {self.max_seq_len + 1} tokens.")
            
    def __len__(self):
        return len(self.tokens) - self.max_seq_len
        
    def __getitem__(self, idx):
        chunk = self.tokens[idx : idx + self.max_seq_len + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
