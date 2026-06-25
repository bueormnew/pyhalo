"""
Atención Dispersa HALO-S — Optimizada.

Implementa atención gather-based con complejidad O(N × K) donde K es el número
fijo de vecinos por token. Optimizada para minimizar materialización de tensores
y uso de memoria GPU.

Optimizaciones clave:
1. GQA sin repeat_interleave — procesa por grupo con vista compartida
2. Score computation con einsum en lugar de unsqueeze + matmul
3. Cache de neighbor lists para evitar regeneración
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from halo.core.config import HaloConfig
from halo.attention.graph import generate_neighbor_lists
from halo.nn.rope import apply_rotary_pos_emb


class HaloSparseAttention(nn.Module):
    """
    Atención Dispersa HALO-S con GQA optimizado.

    Complejidad: O(N × K × d) donde K = num_neighbors (fijo).
    Memoria: O(N × K × d) para gathered KV (sin expansión GQA innecesaria).
    """

    def __init__(self, config: HaloConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim

        assert self.num_heads % self.num_kv_heads == 0
        self.num_groups = self.num_heads // self.num_kv_heads

        # Proyecciones lineales
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache de neighbor lists
        self._cached_neighbors = None
        self._cached_seq_len = -1

    def _get_neighbors(self, seq_len: int, device: torch.device):
        if self._cached_seq_len == seq_len and self._cached_neighbors is not None:
            return self._cached_neighbors

        neighbors = generate_neighbor_lists(
            seq_len=seq_len,
            local_window=self.config.local_window,
            num_globals=self.config.num_globals,
            dilated_offsets=self.config.dilated_offsets,
            num_random=self.config.num_random,
            layer_id=self.layer_id
        ).to(device)

        self._cached_neighbors = neighbors
        self._cached_seq_len = seq_len
        return neighbors

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        is_causal: bool = True
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        # Proyecciones Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        # q: (B, num_heads, S, D)
        # k: (B, num_kv_heads, S, D)
        # v: (B, num_kv_heads, S, D)

        # Aplicar RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Neighbor list: (S, K)
        neighbors = self._get_neighbors(seq_len, x.device)
        num_neighbors = neighbors.shape[1]

        # === OPTIMIZACIÓN: Gather sobre KV heads (NO expandidos) ===
        # k_gathered: (B, num_kv_heads, S, K, D) — sin expansión GQA
        k_gathered = k[:, :, neighbors, :]
        v_gathered = v[:, :, neighbors, :]

        # === OPTIMIZACIÓN: Procesar GQA por grupos sin materializar expansión ===
        # Reshape Q en grupos: (B, num_kv_heads, groups_per_head, S, D)
        q_grouped = q.view(batch_size, self.num_kv_heads, self.num_groups, seq_len, self.head_dim)

        # Scores: Q_grupo × K_gathered^T — broadcast sobre la dimensión de grupo
        # q_grouped: (B, num_kv_heads, G, S, D)
        # k_gathered: (B, num_kv_heads, 1, S, K, D) — broadcast
        # resultado: (B, num_kv_heads, G, S, K)
        scores = torch.einsum('bhgsD,bhsKD->bhgsK', q_grouped, k_gathered) * self.scale

        # Máscara causal
        if is_causal:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
            causal_mask = neighbors > positions  # (S, K) — True si es futuro
            # Expandir: (1, 1, 1, S, K)
            scores.masked_fill_(causal_mask.view(1, 1, 1, seq_len, num_neighbors), float('-inf'))

        # Softmax + Dropout
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Atención × V: (B, num_kv_heads, G, S, K) × (B, num_kv_heads, 1, S, K, D) → (B, num_kv_heads, G, S, D)
        out = torch.einsum('bhgsK,bhsKD->bhgsD', attn, v_gathered)

        # Reshape de vuelta: (B, num_heads, S, D)
        out = out.reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Re-empaquetar: (B, S, H)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        return self.o_proj(out)
