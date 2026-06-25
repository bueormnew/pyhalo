"""
Tests para HaloConfig — validación de parámetros y serialización.
Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

import json
import pytest
from halo.core.config import HaloConfig


# --- Tests de validación de parámetros ---

def test_invalid_hidden_size_not_divisible():
    """Verificar que hidden_size no divisible por num_heads lanza error."""
    with pytest.raises(AssertionError):
        HaloConfig(hidden_size=65, num_heads=8)


def test_invalid_num_heads_not_divisible():
    """Verificar que num_heads no divisible por num_kv_heads lanza error."""
    with pytest.raises(AssertionError):
        HaloConfig(num_heads=3, num_kv_heads=2)


def test_invalid_num_globals_zero():
    """Verificar que num_globals=0 lanza error (se requiere al menos 1)."""
    with pytest.raises(AssertionError):
        HaloConfig(num_globals=0)


# --- Tests de propiedades ---

def test_num_neighbors_property():
    """Verificar fórmula: num_neighbors = num_globals + local_window + 2*len(dilated_offsets) + num_random."""
    config = HaloConfig(
        num_globals=3,
        local_window=32,
        dilated_offsets=[1, 2, 4],
        num_random=5,
    )
    esperado = 3 + 32 + 2 * 3 + 5  # = 46
    assert config.num_neighbors == esperado


# --- Tests de serialización ---

def test_to_dict_roundtrip():
    """Verificar que to_dict() → from_dict() produce configuración idéntica."""
    config = HaloConfig(
        vocab_size=128,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        num_globals=3,
        local_window=32,
        dilated_offsets=[1, 2, 4, 8],
        num_random=4,
        dropout=0.05,
        max_seq_len=512,
    )
    d = config.to_dict()
    restored = HaloConfig.from_dict(d)

    # Comparar todos los campos relevantes
    assert restored.vocab_size == config.vocab_size
    assert restored.hidden_size == config.hidden_size
    assert restored.num_layers == config.num_layers
    assert restored.num_heads == config.num_heads
    assert restored.num_kv_heads == config.num_kv_heads
    assert restored.num_globals == config.num_globals
    assert restored.local_window == config.local_window
    assert restored.dilated_offsets == config.dilated_offsets
    assert restored.num_random == config.num_random
    assert restored.dropout == config.dropout
    assert restored.max_seq_len == config.max_seq_len


def test_to_dict_json_serializable():
    """Verificar que to_dict() produce un dict serializable con json.dumps."""
    config = HaloConfig()
    d = config.to_dict()
    # No debe lanzar excepción
    json_str = json.dumps(d)
    assert isinstance(json_str, str)
    assert len(json_str) > 0
