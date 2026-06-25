"""
HALO-S NN — Módulos neuronales del framework (bloques, FFN, RoPE).
"""

from halo.nn.halo_block import HaloBlock
from halo.nn.feed_forward import FeedForward
from halo.nn.rope import RotaryPositionalEmbeddings, apply_rotary_pos_emb

__all__ = [
    "HaloBlock",
    "FeedForward",
    "RotaryPositionalEmbeddings",
    "apply_rotary_pos_emb",
]
