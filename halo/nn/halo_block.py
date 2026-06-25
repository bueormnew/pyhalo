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
    Los primeros num_globals tokens reciben atención densa (GlobalFullAttention),
    mientras que los tokens regulares pasan por atención dispersa (HaloSparseAttention).
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

        # Separar globals de tokens regulares en la secuencia normalizada
        globals_x = x_normed[:, :num_globals, :]  # (B, G, H)

        # Global tokens: atención densa a toda la secuencia normalizada
        globals_out = self.global_attn(globals_x, x_normed, cos, sin, is_causal)

        # Tokens regulares: atención dispersa sobre secuencia completa
        # HaloSparseAttention ya incluye globals como vecinos en su neighbor list
        attn_full_out = self.attn(x_normed, cos, sin, is_causal)
        tokens_out = attn_full_out[:, num_globals:, :]  # Extraer solo tokens regulares

        # Recombinar globals (dense) + tokens (sparse)
        attn_out = torch.cat([globals_out, tokens_out], dim=1)

        # Residual y FFN
        x = residual + attn_out
        x = x + self.ffn(self.ln_2(x))
        return x
