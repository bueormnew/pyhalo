import torch.nn as nn
from halo.core.config import HaloConfig

class FeedForward(nn.Module):
    """
    Red Feed Forward estándar para Transformer.
    """
    def __init__(self, config: HaloConfig):
        super().__init__()
        hidden_dim = config.hidden_size * 4
        self.w1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))
