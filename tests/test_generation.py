"""
Tests para generación de tokens y estimación de FLOPs.
Validates: Requirements 1.1, 1.2, 1.4, 1.5, 10.3
"""

import torch
import pytest

from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.tokenizers.char import CharacterTokenizer


def _make_small_config():
    """Configuración mínima para tests de generación en CPU."""
    return HaloConfig(
        vocab_size=64,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        num_globals=2,
        local_window=8,
        dilated_offsets=[1, 2],
        num_random=1,
        max_seq_len=64,
    )


def test_generated_tokens_in_valid_range():
    """Verificar que todos los tokens generados están en [0, vocab_size)."""
    config = _make_small_config()
    model = HaloSModel(config)
    model.eval()

    # Prompt de entrada
    prompt = torch.randint(0, config.vocab_size, (1, 5))
    output = model.generate(prompt, max_new_tokens=20, temperature=1.0)

    # Todos los tokens deben estar en rango válido
    assert (output >= 0).all()
    assert (output < config.vocab_size).all()


def test_generate_with_string_input():
    """Verificar que model.generate con string + tokenizer retorna str."""
    config = _make_small_config()
    # Usar vocab_size=256 para compatibilidad con CharacterTokenizer ASCII
    config = HaloConfig(
        vocab_size=256,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        num_globals=2,
        local_window=8,
        dilated_offsets=[1, 2],
        num_random=1,
        max_seq_len=64,
    )
    model = HaloSModel(config)
    model.eval()
    char_tok = CharacterTokenizer()

    result = model.generate("hello", max_new_tokens=10, tokenizer=char_tok)
    assert isinstance(result, str)


def test_generate_string_without_tokenizer_raises():
    """Verificar que model.generate con string sin tokenizer lanza ValueError."""
    config = _make_small_config()
    model = HaloSModel(config)
    model.eval()

    with pytest.raises(ValueError):
        model.generate("hello", max_new_tokens=5)


def test_estimate_flops_sum_invariant():
    """Verificar que total_flops == attention + global + ffn + lm_head."""
    config = _make_small_config()
    model = HaloSModel(config)

    flops = model.estimate_flops(seq_len=32)

    # La suma de componentes debe igualar el total
    suma = (
        flops["attention_flops"]
        + flops["global_flops"]
        + flops["ffn_flops"]
        + flops["lm_head_flops"]
    )
    assert flops["total_flops"] == suma, (
        f"total_flops ({flops['total_flops']}) != suma de componentes ({suma})"
    )
