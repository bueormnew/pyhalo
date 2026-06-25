"""
HALO-S Attention — Mecanismos de atención dispersa y global.
"""

from halo.attention.halo_attention import HaloSparseAttention
from halo.attention.global_attention import GlobalFullAttention, _use_sdpa
from halo.attention.graph import generate_neighbor_lists, estimate_graph_stats

__all__ = [
    "HaloSparseAttention",
    "GlobalFullAttention",
    "_use_sdpa",
    "generate_neighbor_lists",
    "estimate_graph_stats",
]
