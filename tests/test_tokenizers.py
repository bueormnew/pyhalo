"""
Tests para tokenizadores CharacterTokenizer y WordTokenizer.
Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
"""

import os
import tempfile

import pytest
from halo.tokenizers.char import CharacterTokenizer
from halo.tokenizers.word import WordTokenizer


# --- CharacterTokenizer ---

def test_char_tokenizer_encode_decode_roundtrip():
    """Verificar que decode(encode(text)) == text para texto ASCII."""
    tok = CharacterTokenizer()
    text = "Hello, world! 123"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


# --- WordTokenizer ---

def test_word_tokenizer_build_vocab():
    """Verificar que build_vocab con min_freq filtra correctamente."""
    tok = WordTokenizer(min_freq=2)
    texts = [
        "gato perro gato",
        "perro pez gato",
    ]
    tok.build_vocab(texts)

    # "gato" aparece 3 veces, "perro" 2 veces → ambas incluidas
    assert "gato" in tok.stoi
    assert "perro" in tok.stoi
    # "pez" aparece 1 vez → excluida con min_freq=2
    assert "pez" not in tok.stoi


def test_word_tokenizer_encode_decode_roundtrip():
    """Verificar que decode(encode(text)) == text para palabras conocidas."""
    tok = WordTokenizer(min_freq=1)
    tok.build_vocab(["hola mundo bonito"])

    text = "hola mundo bonito"
    encoded = tok.encode(text)
    decoded = tok.decode(encoded)
    assert decoded == text


def test_word_tokenizer_unk_for_unknown():
    """Verificar que palabras desconocidas se mapean a UNK (id=1)."""
    tok = WordTokenizer(min_freq=1)
    tok.build_vocab(["hola mundo"])

    encoded = tok.encode("hola desconocida mundo")
    unk_id = WordTokenizer.SPECIAL_TOKENS["<UNK>"]
    # "desconocida" no está en el vocab, debe mapearse a UNK
    assert encoded[1] == unk_id


def test_word_tokenizer_save_load_roundtrip():
    """Verificar que save/load produce mismos resultados de encode."""
    tok = WordTokenizer(min_freq=1)
    tok.build_vocab(["el gato come pescado", "el perro corre rápido"])

    text = "el gato corre"
    original_encoded = tok.encode(text)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        path = f.name

    try:
        tok.save(path)
        tok_loaded = WordTokenizer.load(path)
        loaded_encoded = tok_loaded.encode(text)
        assert original_encoded == loaded_encoded
    finally:
        os.unlink(path)


def test_word_tokenizer_special_tokens_always_present():
    """Verificar que PAD, UNK, BOS, EOS siempre están en el vocabulario."""
    tok = WordTokenizer(min_freq=1)
    tok.build_vocab(["solo una frase"])

    for token_name in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
        assert token_name in tok.stoi, f"Token especial '{token_name}' no encontrado"
