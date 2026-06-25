import torch
import torch.nn as nn
import torch.nn.functional as F
from halo.core.config import HaloConfig
from halo.nn.feed_forward import FeedForward
from halo.nn.rope import RotaryPositionalEmbeddings, apply_rotary_pos_emb
from halo.attention.global_attention import _use_sdpa

class DenseAttention(nn.Module):
    """Atención densa estándar O(N²) con soporte SDPA cuando está disponible."""

    def __init__(self, config: HaloConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        # Detectar si SDPA está disponible para path acelerado
        self._use_flash = _use_sdpa()

    def forward(self, x, cos, sin, is_causal=True):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Aplicar embeddings rotacionales (RoPE)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if self._use_flash:
            # Path acelerado: usar SDPA con backend Flash/Memory-efficient
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # Fallback manual: matmul + mask + softmax + dropout
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

            if is_causal:
                mask = torch.tril(torch.ones(S, S, device=x.device)).view(1, 1, S, S)
                scores = scores.masked_fill(mask == 0, float('-inf'))

            attn = torch.softmax(scores, dim=-1)
            out = torch.matmul(self.dropout(attn), v)

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out)

class BaselineBlock(nn.Module):
    def __init__(self, config: HaloConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        self.attn = DenseAttention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForward(config)
        
    def forward(self, x, cos, sin):
        # Para el baseline simple asumimos is_causal=True siempre
        x = x + self.attn(self.ln_1(x), cos, sin, is_causal=True)
        x = x + self.ffn(self.ln_2(x))
        return x

class BaselineModel(nn.Module):
    """
    Modelo Transformer Clásico (Denso O(N^2)) para usar como Baseline de comparación.
    """
    def __init__(self, config: HaloConfig):
        super().__init__()
        self.config = config
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryPositionalEmbeddings(config.head_dim, config.max_seq_len)
        self.layers = nn.ModuleList([BaselineBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids, targets=None):
        B, S = input_ids.shape
        x = self.token_emb(input_ids)
        cos, sin = self.rope(x)
        for layer in self.layers:
            x = layer(x, cos, sin)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
        return logits, loss
