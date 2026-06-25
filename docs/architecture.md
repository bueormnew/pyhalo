# Arquitectura General — HALO-S

## Intuición y Motivación

HALO-S (Hierarchical Attention with Local Offsets — Sparse) es un modelo de lenguaje diseñado para procesar secuencias largas de forma eficiente. La motivación central es reemplazar la atención cuadrática $O(N^2)$ de los Transformers estándar por un patrón de conectividad disperso con complejidad $O(N \times K)$, donde $K \ll N$ es un número fijo de vecinos por token.

El modelo combina cuatro mecanismos de conectividad complementarios:

1. **Tokens Globales** — memoria compartida con atención densa $O(G \times N)$
2. **Ventana Local** — captura dependencias secuenciales cercanas
3. **Conexiones Dilatadas** — extiende el campo receptivo exponencialmente
4. **Conexiones Aleatorias** — garantiza propiedades de mundo pequeño en el grafo

## Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                        HaloSModel                                │
│                                                                  │
│  ┌──────────────┐   ┌──────────────────────────────────┐        │
│  │ token_emb    │   │ global_memory (nn.Parameter)      │        │
│  │ (Embedding)  │   │ shape: (num_globals, hidden_size) │        │
│  └──────┬───────┘   └──────────────┬───────────────────┘        │
│         │                          │                             │
│         └──────────┬───────────────┘                             │
│                    ▼                                              │
│         ┌──────────────────┐                                     │
│         │ cat([globals, x]) │  → (B, G+N, H)                    │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ RoPE (cos, sin)  │                                     │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│  ┌───────────────────────────────────────────────────┐           │
│  │              HaloBlock × num_layers                │           │
│  │                                                    │           │
│  │  ┌─────────────┐                                  │           │
│  │  │ LayerNorm 1 │                                  │           │
│  │  └──────┬──────┘                                  │           │
│  │         │                                          │           │
│  │    ┌────┴────────────────────────┐                │           │
│  │    ▼                             ▼                │           │
│  │ ┌────────────────┐   ┌─────────────────────┐     │           │
│  │ │GlobalFullAttn  │   │ HaloSparseAttention │     │           │
│  │ │(dense, G×N)    │   │ (gather, N×K)       │     │           │
│  │ └───────┬────────┘   └──────────┬──────────┘     │           │
│  │         │                       │                  │           │
│  │         └───────────┬───────────┘                  │           │
│  │                     ▼                              │           │
│  │           cat([globals_out, tokens_out])            │           │
│  │                     │ + residual                    │           │
│  │                     ▼                              │           │
│  │  ┌─────────────┐  ┌────────────┐                 │           │
│  │  │ LayerNorm 2 │→ │ FeedForward│ + residual      │           │
│  │  └─────────────┘  └────────────┘                 │           │
│  └───────────────────────────────────────────────────┘           │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ LayerNorm final  │                                     │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ lm_head (Linear) │  → (B, N, vocab_size)              │
│         └──────────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Formulación Matemática

Sea una secuencia de entrada $x \in \mathbb{R}^{B \times N}$ (IDs de tokens). El forward pass completo es:

$$e = \text{Embedding}(x) \in \mathbb{R}^{B \times N \times H}$$

$$\hat{x} = [\underbrace{g_1, \ldots, g_G}_{\text{globals}}; \underbrace{e_1, \ldots, e_N}_{\text{tokens}}] \in \mathbb{R}^{B \times (G+N) \times H}$$

Para cada capa $\ell = 1, \ldots, L$:

$$h^{(\ell)} = \text{LN}_1(\hat{x}^{(\ell-1)})$$

$$\hat{x}^{(\ell)} = \hat{x}^{(\ell-1)} + \text{Attn}(h^{(\ell)}) + \text{FFN}(\text{LN}_2(\hat{x}^{(\ell-1)} + \text{Attn}(h^{(\ell)})))$$

Donde $\text{Attn}$ separa globals (atención densa) y tokens regulares (atención dispersa).

Salida final:

$$\text{logits} = W_{\text{lm}} \cdot \text{LN}_f(\hat{x}^{(L)}_{G:}) \in \mathbb{R}^{B \times N \times V}$$

## Implementación en el Framework

| Componente | Archivo | Clase/Función |
|------------|---------|---------------|
| Modelo principal | `halo/models/halo_model.py` | `HaloSModel` |
| Bloque Transformer | `halo/nn/halo_block.py` | `HaloBlock` |
| Atención dispersa | `halo/attention/halo_attention.py` | `HaloSparseAttention` |
| Atención global | `halo/attention/global_attention.py` | `GlobalFullAttention` |
| Grafo de vecinos | `halo/attention/graph.py` | `generate_neighbor_lists()` |
| RoPE | `halo/nn/rope.py` | `RotaryPositionalEmbeddings` |
| FFN | `halo/nn/feed_forward.py` | `FeedForward` |
| Configuración | `halo/core/config.py` | `HaloConfig` |

## Comparación con Alternativas

| Aspecto | Transformer Denso | Longformer | BigBird | **HALO-S** |
|---------|-------------------|------------|---------|-----------|
| Complejidad temporal | $O(N^2)$ | $O(N \times w)$ | $O(N \times (w+g+r))$ | $O(N \times K)$ |
| Memoria KV | $O(N^2)$ | $O(N \times w)$ | $O(N \times w)$ | $O(N \times K)$ |
| Global tokens | No | Sí (fijos) | Sí (fijos) | Sí (aprendidos) |
| Conexiones dilatadas | No | Sí | No | Sí |
| Conexiones aleatorias | No | No | Sí | Sí |
| GQA | Configurable | No | No | Sí (integrado) |
| Backend | Dense matmul | Bloques | Bloques | Gather-based |

HALO-S combina los mecanismos de Longformer (ventana + globals + dilatación) con los de BigBird (conexiones aleatorias) y añade GQA para eficiencia en memoria KV, todo implementado con un backend gather-based que evita la necesidad de kernels CUDA personalizados.
