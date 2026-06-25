"""
Atención Densa para Global Tokens — HALO-S.

Los Global Tokens son posiciones especiales (0..G-1) que atienden a TODA la
secuencia (global + regular tokens) con atención densa estándar.
Complejidad: O(G × N) donde G = num_globals, N = seq_len total.

A diferencia de la atención dispersa (gather-based), aquí se computan scores
contra todas las posiciones, permitiendo que los globals actúen como memoria
compartida accesible por cualquier token del grafo disperso.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from halo.core.config import HaloConfig


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rota la segunda mitad de las dimensiones para RoPE."""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Aplica Rotary Positional Embeddings a un tensor individual.

    Args:
        x: Tensor con shape (batch, heads, seq_len, head_dim).
        cos, sin: Embeddings rotacionales con shape broadcastable a x.

    Returns:
        Tensor con RoPE aplicado, misma shape que x.
    """
    return (x * cos) + (_rotate_half(x) * sin)


def _use_sdpa() -> bool:
    """Detecta si torch.nn.functional.scaled_dot_product_attention está disponible.

    Retorna True solo cuando SDPA existe en el entorno y hay backend
    eficiente (CUDA). En CPU no se beneficia significativamente.
    """
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False
    # Solo se beneficia con backends Flash/Memory-efficient en CUDA
    if torch.cuda.is_available():
        return True
    return False


class GlobalFullAttention(nn.Module):
    """
    Atención densa para los Global Tokens con soporte GQA y SDPA.

    Cada global token atiende a TODA la secuencia (globals + tokens regulares).
    Utiliza Grouped Query Attention: num_heads queries, num_kv_heads keys/values.

    Args:
        config: Configuración HaloConfig del modelo.
        use_flash: Si True, intenta usar SDPA como backend de atención.
                   Se desactiva automáticamente si SDPA no está disponible.
    """

    def __init__(self, config: HaloConfig, use_flash: bool = True):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size

        assert self.num_heads % self.num_kv_heads == 0, (
            "num_heads debe ser divisible por num_kv_heads"
        )
        self.num_groups = self.num_heads // self.num_kv_heads

        # Proyecciones lineales (Q desde globals, K/V desde secuencia completa)
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)

        # Flag para usar SDPA (solo si está disponible)
        self._use_flash = use_flash and _use_sdpa()

    def forward(
        self,
        globals_x: torch.Tensor,
        full_seq: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass de atención densa para global tokens.

        Args:
            globals_x: (batch, num_globals, hidden) — queries son solo los globals.
            full_seq: (batch, total_seq_len, hidden) — keys/values de toda la secuencia.
            cos, sin: Embeddings RoPE con shape (1, 1, max_seq_len, head_dim).
            is_causal: Si True, global[i] solo atiende a posiciones [0, i].

        Returns:
            (batch, num_globals, hidden) — representación actualizada de globals.
        """
        B, G, _ = globals_x.shape
        _, N, _ = full_seq.shape

        # Proyectar Q solo de globals, K/V de toda la secuencia
        q = self.q_proj(globals_x).view(B, G, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(full_seq).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(full_seq).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, G, head_dim)
        # k: (B, num_kv_heads, N, head_dim)
        # v: (B, num_kv_heads, N, head_dim)

        # Aplicar RoPE — Q usa posiciones [0, G), K usa posiciones [0, N)
        # Se aplican por separado porque Q y K tienen longitudes distintas
        cos_q = cos[:, :, :G, :]
        sin_q = sin[:, :, :G, :]
        cos_k = cos[:, :, :N, :]
        sin_k = sin[:, :, :N, :]
        q = _apply_rope(q, cos_q, sin_q)
        k = _apply_rope(k, cos_k, sin_k)

        # Expandir K, V para GQA (repeat_interleave por grupos)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)
        # k, v: (B, num_heads, N, head_dim)

        # Construir máscara causal: global[i] solo atiende a posiciones [0, i]
        attn_mask = None
        if is_causal:
            # positions_q: [0, 1, ..., G-1], positions_k: [0, 1, ..., N-1]
            positions_q = torch.arange(G, device=q.device).unsqueeze(1)  # (G, 1)
            positions_k = torch.arange(N, device=q.device).unsqueeze(0)  # (1, N)
            # True donde K está en el futuro respecto a Q
            causal_mask = positions_k > positions_q  # (G, N)
            attn_mask = causal_mask  # Se usa abajo según el path

        if self._use_flash:
            out = self._forward_sdpa(q, k, v, attn_mask)
        else:
            out = self._forward_manual(q, k, v, attn_mask)

        # Reshape: (B, num_heads, G, head_dim) → (B, G, hidden)
        out = out.transpose(1, 2).contiguous().view(B, G, -1)
        return self.o_proj(out)

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Path acelerado usando scaled_dot_product_attention (SDPA).

        Se construye una máscara explícita dado que la causal estándar (triangular)
        no aplica aquí: los globals tienen alcance causal muy limitado.
        """
        # SDPA espera attn_mask como float o bool con shape broadcastable
        # a (B, num_heads, G, N). True = posición ignorada en la versión bool.
        if causal_mask is not None:
            # Convertir a float mask: 0 donde se atiende, -inf donde se bloquea
            sdpa_mask = torch.zeros_like(causal_mask, dtype=q.dtype)
            sdpa_mask.masked_fill_(causal_mask, float('-inf'))
            # Expandir para broadcast: (1, 1, G, N)
            sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)
        else:
            sdpa_mask = None

        dropout_p = self.dropout.p if self.training else 0.0

        # SDPA: (B, num_heads, G, head_dim) @ (B, num_heads, N, head_dim) → (B, num_heads, G, head_dim)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            is_causal=False  # Usamos nuestra máscara custom
        )
        return out

    def _forward_manual(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Fallback manual: matmul scores, mask, softmax, dropout, weighted sum.

        Se usa cuando SDPA no está disponible o use_flash=False.
        """
        # Scores: (B, num_heads, G, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Aplicar máscara causal
        if causal_mask is not None:
            # causal_mask: (G, N) — expandir para (1, 1, G, N)
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax sobre la dimensión de keys
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum: (B, num_heads, G, N) @ (B, num_heads, N, head_dim) → (B, num_heads, G, head_dim)
        out = torch.matmul(attn_weights, v)
        return out
