# Grouped Query Attention (GQA)

## Intuición y Motivación

En Multi-Head Attention (MHA) estándar, cada cabeza de atención tiene sus propias proyecciones Q, K y V. Esto significa que el **KV cache** durante la generación autoregresiva escala linealmente con el número de cabezas:

$$\text{KV cache} = 2 \times N \times H \times d_k \times \text{sizeof(float)}$$

Para modelos grandes ($H = 32$, $N = 4096$, $d_k = 128$): el KV cache ocupa 2 GB en FP32 por capa.

**Grouped Query Attention (GQA)** reduce este costo compartiendo K y V entre grupos de cabezas de query. Si tenemos $H_q$ cabezas de query y $H_{kv}$ cabezas de K/V (con $H_q / H_{kv} = G$ cabezas por grupo), el KV cache se reduce por un factor de $G$:

$$\text{KV cache (GQA)} = 2 \times N \times H_{kv} \times d_k \times \text{sizeof(float)} = \frac{\text{KV cache (MHA)}}{G}$$

## Formulación Matemática

### Definiciones

- $H_q$: número de cabezas de query (= `num_heads`)
- $H_{kv}$: número de cabezas de key/value (= `num_kv_heads`)
- $G = H_q / H_{kv}$: número de queries por grupo (ratio de compartición)
- $d_k = H / H_q$: dimensión por cabeza

### Proyecciones

$$Q = XW_Q \in \mathbb{R}^{B \times N \times H_q \times d_k}$$
$$K = XW_K \in \mathbb{R}^{B \times N \times H_{kv} \times d_k}$$
$$V = XW_V \in \mathbb{R}^{B \times N \times H_{kv} \times d_k}$$

### Expansión para Cómputo

Para computar los scores, K y V se expanden por grupo mediante `repeat_interleave`:

$$\hat{K} = \text{repeat}(K, G, \text{dim}=\text{heads}) \in \mathbb{R}^{B \times N \times H_q \times d_k}$$
$$\hat{V} = \text{repeat}(V, G, \text{dim}=\text{heads}) \in \mathbb{R}^{B \times N \times H_q \times d_k}$$

Luego se computa atención estándar:

$$\text{Attn}(Q, \hat{K}, \hat{V}) = \text{softmax}\left(\frac{Q\hat{K}^T}{\sqrt{d_k}}\right)\hat{V}$$

### Fórmula de Reducción de Memoria

| Componente | MHA | GQA | Reducción |
|-----------|-----|-----|-----------|
| Parámetros K | $H \times H_q \times d_k$ | $H \times H_{kv} \times d_k$ | $\times G$ |
| Parámetros V | $H \times H_q \times d_k$ | $H \times H_{kv} \times d_k$ | $\times G$ |
| KV cache (inferencia) | $2 \times N \times H_q \times d_k$ | $2 \times N \times H_{kv} \times d_k$ | $\times G$ |
| Parámetros Q | $H \times H_q \times d_k$ | $H \times H_q \times d_k$ | Sin cambio |

Para HALO-S con $H_q = 8$ y $H_{kv} = 2$: $G = 4$, reducción de **4×** en KV cache y parámetros de K/V.

### Espectro MHA — GQA — MQA

GQA es una generalización que interpola entre MHA y MQA:

$$\text{MHA}: H_{kv} = H_q \quad \text{(sin compartición)}$$
$$\text{GQA}: 1 < H_{kv} < H_q \quad \text{(compartición por grupos)}$$
$$\text{MQA}: H_{kv} = 1 \quad \text{(compartición total)}$$

## Implementación en el Framework

**Archivos**:
- `halo/attention/halo_attention.py` — `HaloSparseAttention` (GQA en atención dispersa)
- `halo/attention/global_attention.py` — `GlobalFullAttention` (GQA en atención densa)

La configuración se define en `HaloConfig`:

```python
num_heads: int = 8       # Cabezas de query (H_q)
num_kv_heads: int = 2    # Cabezas de K/V (H_kv)
# → num_groups = 8 / 2 = 4 queries por grupo
```

La expansión se realiza con `repeat_interleave`:

```python
self.num_groups = self.num_heads // self.num_kv_heads  # 4

# En forward:
if self.num_groups > 1:
    k = k.repeat_interleave(self.num_groups, dim=1)  # (B, H_kv, ...) → (B, H_q, ...)
    v = v.repeat_interleave(self.num_groups, dim=1)
```

La proyección de salida siempre opera sobre $H_q$ cabezas:

```python
self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
```

## Comparación con Alternativas

| Método | $H_{kv}$ | KV cache | Calidad | Uso en producción |
|--------|----------|----------|---------|-------------------|
| MHA | $H_q$ | 100% | Mejor | GPT-3, BERT |
| **GQA** | $1 < H_{kv} < H_q$ | $H_{kv}/H_q$ | ~MHA | LLaMA 2/3, Mistral |
| MQA | 1 | $1/H_q$ | Algo peor | PaLM, Falcon |

GQA logra un equilibrio óptimo: mantiene la calidad cercana a MHA (las cabezas de query siguen siendo independientes) mientras reduce significativamente la memoria. Estudios empíricos (LLaMA 2) muestran degradación mínima en perplejidad con $G = 4$ o $G = 8$.

En HALO-S, GQA se aplica tanto en la atención dispersa como en la atención densa de globals, maximizando el ahorro de memoria en ambos paths de cómputo.
