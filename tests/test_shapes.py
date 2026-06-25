"""
Tests de validación de dimensiones de salida (shapes) para HALO-S.

Verifica que el forward pass, generación y componentes individuales
producen tensores con las dimensiones correctas bajo diversas configuraciones.

**Validates: Requirements 13.1, 13.2, 10.1**
"""

import torch
import pytest
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.attention.global_attention import GlobalFullAttention
from halo.nn.halo_block import HaloBlock
from halo.nn.rope import RotaryPositionalEmbeddings


# --- Configuraciones pequeñas para tests rápidos ---

def _small_config(**overrides):
    """Crea una configuración pequeña para tests rápidos."""
    defaults = dict(
        vocab_size=64,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        num_kv_heads=2,
        num_globals=2,
        local_window=8,
        dilated_offsets=[1, 2],
        num_random=1,
        dropout=0.0,
        max_seq_len=256,
    )
    defaults.update(overrides)
    return HaloConfig(**defaults)


# ============================================================
# 1. test_model_output_shape_default_config
# ============================================================

def test_model_output_shape_default_config():
    """Verifica logits shape (B, S, vocab_size) con configuración por defecto pequeña."""
    config = _small_config()
    model = HaloSModel(config)
    model.eval()

    B, S = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits.shape == (B, S, config.vocab_size), (
        f"Expected logits shape ({B}, {S}, {config.vocab_size}), got {logits.shape}"
    )
    assert loss is None


# ============================================================
# 2. test_model_output_shape_various_configs
# ============================================================

@pytest.mark.parametrize("hidden_size,num_heads,num_kv_heads,num_layers", [
    (64, 2, 2, 1),
    (64, 4, 2, 1),
    (128, 4, 2, 2),
    (128, 4, 4, 1),
    (64, 2, 1, 2),
])
def test_model_output_shape_various_configs(hidden_size, num_heads, num_kv_heads, num_layers):
    """Verifica logits shape con múltiples configuraciones de modelo."""
    config = _small_config(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        num_layers=num_layers,
    )
    model = HaloSModel(config)
    model.eval()

    B, S = 2, 20
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits.shape == (B, S, config.vocab_size)
    assert loss is None


# ============================================================
# 3. test_model_loss_is_scalar
# ============================================================

def test_model_loss_is_scalar():
    """Verifica que loss es un escalar (dim 0) cuando se proporcionan targets."""
    config = _small_config()
    model = HaloSModel(config)
    model.eval()

    B, S = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, S))
    targets = torch.randint(0, config.vocab_size, (B, S))

    with torch.no_grad():
        logits, loss = model(input_ids, targets=targets)

    assert loss is not None, "Loss debería existir cuando se pasan targets"
    assert loss.dim() == 0, f"Loss debería ser escalar (dim 0), got dim {loss.dim()}"
    assert loss.shape == torch.Size([]), f"Loss shape debería ser [], got {loss.shape}"


# ============================================================
# 4. test_model_logits_no_loss_without_targets
# ============================================================

def test_model_logits_no_loss_without_targets():
    """Verifica que loss es None cuando no se proporcionan targets."""
    config = _small_config()
    model = HaloSModel(config)
    model.eval()

    B, S = 1, 10
    input_ids = torch.randint(0, config.vocab_size, (B, S))

    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits is not None
    assert loss is None, "Loss debería ser None sin targets"


# ============================================================
# 5. test_generation_output_length
# ============================================================

@pytest.mark.parametrize("max_new_tokens", [5, 10, 20])
def test_generation_output_length(max_new_tokens):
    """Verifica que generate() produce el número correcto de tokens."""
    config = _small_config()
    model = HaloSModel(config)
    model.eval()

    prompt_len = 8
    input_ids = torch.randint(0, config.vocab_size, (1, prompt_len))

    with torch.no_grad():
        output = model.generate(input_ids, max_new_tokens=max_new_tokens)

    expected_len = prompt_len + max_new_tokens
    assert output.shape == (1, expected_len), (
        f"Expected output shape (1, {expected_len}), got {output.shape}"
    )


# ============================================================
# 6. test_global_attention_output_shape
# ============================================================

def test_global_attention_output_shape():
    """Verifica que GlobalFullAttention(B, G, H) + (B, N, H) → (B, G, H)."""
    config = _small_config()
    global_attn = GlobalFullAttention(config, use_flash=False)
    global_attn.eval()

    B, G, N, H = 2, config.num_globals, 20, config.hidden_size

    globals_x = torch.randn(B, G, H)
    full_seq = torch.randn(B, N, H)

    # Generar RoPE
    rope = RotaryPositionalEmbeddings(config.head_dim, config.max_seq_len)
    # RoPE necesita un dummy tensor para inferir shapes
    dummy = torch.randn(B, N, H)
    cos, sin = rope(dummy)

    with torch.no_grad():
        output = global_attn(globals_x, full_seq, cos, sin, is_causal=True)

    assert output.shape == (B, G, H), (
        f"Expected GlobalFullAttention output shape ({B}, {G}, {H}), got {output.shape}"
    )


# ============================================================
# 7. test_halo_block_output_shape
# ============================================================

def test_halo_block_output_shape():
    """Verifica que HaloBlock preserva shape (B, num_globals + seq_len, H)."""
    config = _small_config()
    block = HaloBlock(config, layer_id=0)
    block.eval()

    B = 2
    seq_len = 16
    total_len = config.num_globals + seq_len
    H = config.hidden_size

    x = torch.randn(B, total_len, H)

    # RoPE
    rope = RotaryPositionalEmbeddings(config.head_dim, config.max_seq_len)
    cos, sin = rope(x)

    with torch.no_grad():
        output = block(x, cos, sin, is_causal=True)

    assert output.shape == (B, total_len, H), (
        f"Expected HaloBlock output shape ({B}, {total_len}, {H}), got {output.shape}"
    )


# ============================================================
# 8. test_various_batch_sizes
# ============================================================

@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_various_batch_sizes(batch_size):
    """Verifica que el modelo funciona con distintos batch sizes."""
    config = _small_config()
    model = HaloSModel(config)
    model.eval()

    S = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, S))

    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits.shape == (batch_size, S, config.vocab_size), (
        f"batch_size={batch_size}: Expected ({batch_size}, {S}, {config.vocab_size}), got {logits.shape}"
    )


# ============================================================
# 9. test_various_seq_lengths
# ============================================================

@pytest.mark.parametrize("seq_len", [10, 32, 64, 128])
def test_various_seq_lengths(seq_len):
    """Verifica que el modelo funciona con distintas longitudes de secuencia."""
    config = _small_config(local_window=8, max_seq_len=256)
    model = HaloSModel(config)
    model.eval()

    B = 2
    input_ids = torch.randint(0, config.vocab_size, (B, seq_len))

    with torch.no_grad():
        logits, loss = model(input_ids)

    assert logits.shape == (B, seq_len, config.vocab_size), (
        f"seq_len={seq_len}: Expected ({B}, {seq_len}, {config.vocab_size}), got {logits.shape}"
    )
