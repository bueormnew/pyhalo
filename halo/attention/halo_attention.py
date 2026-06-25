"""
Atención Dispersa HALO-S v2.0 — Hybrid SDPA + Gather.

Implementa una estrategia híbrida:
- Para CUDA con seq_len <= 2048: construye una máscara dispersa booleana y usa
  F.scaled_dot_product_attention (aprovecha memory-efficient backend).
- Para CPU o seq_len > 2048: usa el path gather-based con O(N × K) complejidad.

Optimizaciones v2.0:
1. Hybrid SDPA path con mask para secuencias cortas en GPU
2. Gather path optimizado con GQA sin repeat_interleave para secuencias largas
3. Score computation con einsum en lugar de unsqueeze + matmul
4. Cache de neighbor lists Y masks para evitar regeneración
5. torch.compile compatible (no graph breaks)

Backward compatibility: q_proj, k_proj, v_proj, o_proj mantienen los mismos nombres.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from halo.core.config import HaloConfig
from halo.attention.graph import generate_neighbor_lists
from halo.nn.rope import apply_rotary_pos_emb


# Threshold for SDPA vs gather path
_SDPA_SEQ_LEN_THRESHOLD = 2048


class HaloSparseAttention(nn.Module):
    """
    Atención Dispersa HALO-S con estrategia híbrida SDPA/Gather.

    Complejidad:
    - SDPA path: O(N² × d) pero con kernel fusionado y memory-efficient backend
    - Gather path: O(N × K × d) donde K = num_neighbors (fijo)
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

        # Proyecciones lineales — SAME NAMES for backward compatibility
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Cache de neighbor lists
        self._cached_neighbors = None
        self._cached_seq_len = -1

        # Cache de sparse mask para SDPA path
        self._cached_mask = None
        self._cached_mask_seq_len = -1

    def _get_neighbors(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or generate cached neighbor lists."""
        if self._cached_seq_len == seq_len and self._cached_neighbors is not None:
            if self._cached_neighbors.device == device:
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

    def _build_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Build a (seq_len, seq_len) float mask from neighbor lists + causal constraint.
        
        Returns a mask where 0.0 = allowed, -inf = blocked, suitable for SDPA attn_mask.
        """
        neighbors = self._get_neighbors(seq_len, device)

        # Create boolean mask: True where attention is ALLOWED
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        positions = torch.arange(seq_len, device=device).unsqueeze(1).expand(-1, neighbors.shape[1])
        mask[positions, neighbors] = True

        # Apply causal constraint: only attend to positions <= self
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
        mask = mask & causal

        # Convert to float mask for SDPA: 0 where allowed, -inf where blocked
        float_mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
        float_mask.masked_fill_(~mask, float('-inf'))
        return float_mask

    def _get_sparse_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get or build cached sparse mask."""
        if self._cached_mask_seq_len == seq_len and self._cached_mask is not None:
            if self._cached_mask.device == device:
                return self._cached_mask

        mask = self._build_sparse_mask(seq_len, device)
        self._cached_mask = mask
        self._cached_mask_seq_len = seq_len
        return mask

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

        # HYBRID: choose path based on hardware and seq_len
        use_sdpa_path = (
            x.is_cuda
            and hasattr(F, 'scaled_dot_product_attention')
            and seq_len <= _SDPA_SEQ_LEN_THRESHOLD
        )

        if use_sdpa_path:
            return self._forward_sdpa(q, k, v, batch_size, seq_len)
        else:
            return self._forward_gather(q, k, v, batch_size, seq_len, x.device, is_causal)

    def _forward_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        """Use SDPA with sparse mask — leverages memory-efficient attention backend."""
        # GQA: expand K, V to match Q heads (SDPA requires same num_heads)
        if self.num_groups > 1:
            k = k.repeat_interleave(self.num_groups, dim=1)
            v = v.repeat_interleave(self.num_groups, dim=1)

        # Get or build mask: (S, S)
        mask = self._get_sparse_mask(seq_len, q.device)

        # SDPA handles the rest efficiently
        dropout_p = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=dropout_p,
        )
        # out: (B, num_heads, S, D)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(out)

    def _forward_gather(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        is_causal: bool,
    ) -> torch.Tensor:
        """Gather-based path for very long sequences or CPU — O(N×K) complexity."""
        # Neighbor list: (S, K)
        neighbors = self._get_neighbors(seq_len, device)
        num_neighbors = neighbors.shape[1]

        # === Gather sobre KV heads (NO expandidos) ===
        # k_gathered: (B, num_kv_heads, S, K, D) — sin expansión GQA
        k_gathered = k[:, :, neighbors, :]
        v_gathered = v[:, :, neighbors, :]

        # === Procesar GQA por grupos sin materializar expansión ===
        # Reshape Q en grupos: (B, num_kv_heads, groups_per_head, S, D)
        q_grouped = q.view(batch_size, self.num_kv_heads, self.num_groups, seq_len, self.head_dim)

        # Scores: Q_grupo × K_gathered^T — broadcast sobre la dimensión de grupo
        # q_grouped: (B, num_kv_heads, G, S, D)
        # k_gathered: (B, num_kv_heads, S, K, D) — broadcast via einsum
        # resultado: (B, num_kv_heads, G, S, K)
        scores = torch.einsum('bhgsD,bhsKD->bhgsK', q_grouped, k_gathered) * self.scale

        # Máscara causal
        if is_causal:
            positions = torch.arange(seq_len, device=device).unsqueeze(1)
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
