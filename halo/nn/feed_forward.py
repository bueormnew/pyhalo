"""
Feed Forward Network v2.0 — SwiGLU activation.

SwiGLU: (SiLU(xW1) ⊙ xW3) W2

Backward compatible: w1, w2 keep the same names. w3 is new and initialized
to ones so that old checkpoints (without w3) produce the same output when
w3 is missing from the state_dict (identity gate).
"""

import torch
import torch.nn as nn
from halo.core.config import HaloConfig


class FeedForward(nn.Module):
    """
    Red Feed Forward con SwiGLU para HALO-S v2.0.

    Backward compatible: w1, w2 mantienen los mismos nombres de state_dict.
    w3 (gate projection) es nuevo — si no existe en un checkpoint antiguo,
    se usa inicialización a ones (neutro multiplicativamente).
    """

    def __init__(self, config: HaloConfig):
        super().__init__()
        hidden_dim = config.hidden_size * 4
        self.use_swiglu = getattr(config, 'use_swiglu', True)

        # SAME names as v1.x for state_dict compatibility
        self.w1 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.hidden_size, bias=False)

        if self.use_swiglu:
            # NEW: gate projection for SwiGLU
            self.w3 = nn.Linear(config.hidden_size, hidden_dim, bias=False)
            # Initialize w3 to produce ones initially so it's neutral if loading
            # old checkpoints that don't have w3 (the missing key will be left
            # at this initialization)
            nn.init.ones_(self.w3.weight)
            self.act = nn.SiLU()
        else:
            self.act = nn.GELU()

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            # SwiGLU: (SiLU(xW1) ⊙ xW3) W2
            return self.dropout(self.w2(self.act(self.w1(x)) * self.w3(x)))
        else:
            # Legacy GELU path
            return self.dropout(self.w2(self.act(self.w1(x))))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Override to handle loading old checkpoints that don't have w3.
        When w3 is missing, we keep the ones initialization (neutral gate).
        """
        w3_key = prefix + 'w3.weight'
        if self.use_swiglu and w3_key not in state_dict:
            # Old checkpoint without w3 — leave at ones initialization
            # Remove from missing_keys tracking to avoid warnings
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, False,
                missing_keys, unexpected_keys, error_msgs
            )
            # Filter out w3 from missing_keys since it's expected
            keys_to_remove = [k for k in missing_keys if 'w3' in k]
            for k in keys_to_remove:
                missing_keys.remove(k)
        else:
            super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict,
                missing_keys, unexpected_keys, error_msgs
            )
