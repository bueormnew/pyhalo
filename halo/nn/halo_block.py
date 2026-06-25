"""
HaloBlock — Bloque Transformer pre-norm optimizado con atención HALO-S.

Optimización: Los globals se procesan con atención densa (GlobalFullAttention).
Los tokens regulares se procesan con atención dispersa (HaloSparseAttention)
sobre la secuencia COMPLETA (globals + tokens) para mantener el flujo de
gradientes hacia global_memory.
"""

import torch
import torch.nn as nn
from halo.core.config import HaloConfig
from halo.attention.halo_attention import HaloSparseAttention
from halo.attention.global_attention import GlobalFullAttention
from halo.nn.feed_forward import FeedForward


class HaloBlock(nn.Module):
    """
    Bloque Transformer pre-norm con atención HALO-S y Global Attention separada.

    La secuencia de entrada tiene shape (B, num_globals + seq_len, H).
    - Globals (pos 0..G-1): atención densa a toda la secuencia via GlobalFullAttention
    - Tokens regulares (pos G..N-1): atención dispersa via HaloSparseAttention
      sobre la secuencia COMPLETA (K/V incluye globals para mantener gradientes)
    """

    def __init__(self, config: HaloConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = HaloSparseAttention(config, layer_id)
        self.global_attn = GlobalFullAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForward(config)

    def forward(self, x, cos, sin, is_causal=True):
        residual = x
        x_normed = self.ln_1(x)

        num_globals = self.config.num_globals

        # Separar globals de tokens regulares
        globals_x = x_normed[:, :num_globals, :]  # (B, G, H)

        # Global tokens: atención densa a toda la secuencia
        globals_out = self.global_attn(globals_x, x_normed, cos, sin, is_causal)

        # Tokens regulares: atención dispersa sobre secuencia COMPLETA
        # (K/V vienen de toda la secuencia incluyendo globals — necesario para gradientes)
        attn_full_out = self.attn(x_normed, cos, sin, is_causal)
        tokens_out = attn_full_out[:, num_globals:, :]  # Solo outputs de tokens regulares

        # Recombinar globals (dense) + tokens (sparse)
        attn_out = torch.cat([globals_out, tokens_out], dim=1)

        # Residual y FFN
        x = residual + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x
