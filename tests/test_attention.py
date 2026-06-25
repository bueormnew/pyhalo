import torch
from halo.core.config import HaloConfig
from halo.attention.halo_attention import HaloSparseAttention
from halo.nn.rope import RotaryPositionalEmbeddings

def test_halo_attention():
    # Mini configuración para probar localmente
    config = HaloConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        num_globals=2,
        local_window=16,
        dilated_offsets=[1, 2],
        num_random=2,
        max_seq_len=128
    )
    
    batch_size = 2
    seq_len = 50
    hidden_size = config.hidden_size
    
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # RoPE embeddings
    rope = RotaryPositionalEmbeddings(dim=config.head_dim, max_seq_len=128)
    cos, sin = rope(x)
    
    layer = HaloSparseAttention(config=config, layer_id=0)
    
    # Probando forward causal
    out_causal = layer(x, cos, sin, is_causal=True)
    assert out_causal.shape == (batch_size, seq_len, hidden_size)
    
    # Asegurar que los gradientes puedan propagarse
    loss = out_causal.sum()
    loss.backward()
    assert layer.q_proj.weight.grad is not None


# --- Tests de Global Attention (Task 2.3) ---
# Validates: Requirements 2.1, 2.3, 13.3

from halo.attention.global_attention import GlobalFullAttention


def test_global_attention_output_shape():
    """Verificar que GlobalFullAttention con (B, G, H) entrada → (B, G, H) salida."""
    config = HaloConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        num_globals=2,
        local_window=16,
        dilated_offsets=[1, 2],
        num_random=2,
        max_seq_len=128,
    )

    B, G, N, H = 2, config.num_globals, 50, config.hidden_size
    globals_x = torch.randn(B, G, H)
    full_seq = torch.randn(B, N, H)

    # RoPE para las posiciones
    rope = RotaryPositionalEmbeddings(dim=config.head_dim, max_seq_len=128)
    cos, sin = rope(full_seq)

    attn = GlobalFullAttention(config, use_flash=False)
    out = attn(globals_x, full_seq, cos, sin, is_causal=False)

    # La salida debe tener shape (B, G, H) — misma que globals_x
    assert out.shape == (B, G, H)


def test_global_attention_causal_mask():
    """Verificar que en modo causal la salida difiere del modo no-causal (máscara activa)."""
    config = HaloConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        num_globals=4,
        local_window=16,
        dilated_offsets=[1, 2],
        num_random=2,
        max_seq_len=128,
    )

    B, G, N, H = 2, config.num_globals, 50, config.hidden_size
    globals_x = torch.randn(B, G, H)
    full_seq = torch.randn(B, N, H)

    rope = RotaryPositionalEmbeddings(dim=config.head_dim, max_seq_len=128)
    cos, sin = rope(full_seq)

    attn = GlobalFullAttention(config, use_flash=False)
    attn.eval()

    with torch.no_grad():
        out_causal = attn(globals_x, full_seq, cos, sin, is_causal=True)
        out_noncausal = attn(globals_x, full_seq, cos, sin, is_causal=False)

    # Con G=4 y N=50, la máscara causal bloquea posiciones futuras
    # por lo que los outputs deben ser diferentes (la máscara funciona)
    assert not torch.allclose(out_causal, out_noncausal, atol=1e-6), \
        "Las salidas causal y no-causal no deberían ser iguales cuando la máscara está activa"
