import math
import torch
import torch.nn as nn
from typing import Optional
from halo.core.config import HaloConfig
from halo.attention.graph import generate_neighbor_lists
from halo.nn.rope import apply_rotary_pos_emb

class HaloSparseAttention(nn.Module):
    """
    Atención Dispersa HALO-S con Gather-based backend y GQA.
    Garantiza complejidad O(N * num_neighbors) en lugar de O(N^2).
    """
    def __init__(self, config: HaloConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        
        assert self.num_heads % self.num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.num_groups = self.num_heads // self.num_kv_heads
        
        # Proyecciones lineales
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Cache opcional para los vecinos si seq_len no cambia (acelera entrenamiento)
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
        
        # Proyectar Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Aplicar RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Obtener matriz de índices dispersos (Neighbor List)
        # shape: (seq_len, num_neighbors)
        neighbors = self._get_neighbors(seq_len, x.device)
        
        # Recopilación de vecinos dispersos mediante Advanced Indexing (GATHER)
        # K, V shape original: (batch, num_kv_heads, seq_len, head_dim)
        # k_gathered shape: (batch, num_kv_heads, seq_len, num_neighbors, head_dim)
        k_gathered = k[:, :, neighbors, :]
        v_gathered = v[:, :, neighbors, :]
        
        # Expansión para Grouped Query Attention (GQA)
        # Repite el KV cache para coincidir con num_heads
        if self.num_groups > 1:
            k_gathered = k_gathered.repeat_interleave(self.num_groups, dim=1)
            v_gathered = v_gathered.repeat_interleave(self.num_groups, dim=1)
            
        # Preparar Q para dot product
        q_expanded = q.unsqueeze(3) # (batch, num_heads, seq_len, 1, head_dim)
        
        # Calcular scores de atención locales
        # Matmul: (..., 1, head_dim) x (..., head_dim, num_neighbors) -> (..., 1, num_neighbors)
        scores = torch.matmul(q_expanded, k_gathered.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        # Aplicar máscara causal (autoregresiva)
        # Ningún token puede atender a una posición mayor que él mismo.
        if is_causal:
            # positions: (seq_len, 1)
            positions = torch.arange(seq_len, device=x.device).unsqueeze(1)
            causal_mask = neighbors > positions # True si es futuro
            causal_mask = causal_mask.view(1, 1, seq_len, 1, -1)
            scores.masked_fill_(causal_mask, float('-inf'))
            
        # Softmax y Dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Multiplicar por V local
        # Matmul: (..., 1, num_neighbors) x (..., num_neighbors, head_dim) -> (..., 1, head_dim)
        out = torch.matmul(attn, v_gathered)
        out = out.squeeze(3) # (batch, num_heads, seq_len, head_dim)
        
        # Re-empaquetar
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(out)
