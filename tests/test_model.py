import torch
import pytest
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel

def test_halo_model_forward():
    config = HaloConfig(
        vocab_size=100,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=64
    )
    model = HaloSModel(config)
    
    batch_size = 2
    seq_len = 30
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits, loss = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.vocab_size)
    assert loss is None

def test_halo_model_loss():
    config = HaloConfig(vocab_size=100, hidden_size=64, num_layers=1, num_heads=2, num_kv_heads=1)
    model = HaloSModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    targets = torch.randint(0, config.vocab_size, (2, 10))
    
    logits, loss = model(input_ids, targets=targets)
    assert loss is not None
    assert loss.item() > 0
    
    # Check backward
    loss.backward()
    assert model.token_emb.weight.grad is not None

def test_halo_model_generation():
    config = HaloConfig(vocab_size=100, hidden_size=64, num_layers=1, num_heads=2, num_kv_heads=1)
    model = HaloSModel(config)
    
    input_ids = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(input_ids, max_new_tokens=10)
    
    assert generated.shape == (1, 15)
