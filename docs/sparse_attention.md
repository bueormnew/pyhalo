# Atención Dispersa con Neighbor Lists

## Intuición y Motivación

La atención estándar de los Transformers computa scores entre TODOS los pares de tokens, resultando en complejidad cuadrática $O(N^2)$. Para secuencias de 4096+ tokens, esto se vuelve prohibitivo en memoria y cómputo.

HALO-S resuelve esto restringiendo cada token a atender solo a un conjunto fijo de $K$ vecinos seleccionados por criterios estructurales (localidad, dilatación, globalidad, aleatoriedad). La selección se precomputa como una **neighbor list** (lista de adyacencia) y la atención se ejecuta mediante **gather operations**, evitando instanciar la matriz densa $N \times N$.

## Formulación Matemática

### Neighbor List

Para cada token $i$, definimos su conjunto de vecinos:

$$\mathcal{N}(i) = \mathcal{G} \cup \mathcal{W}(i) \cup \mathcal{D}(i) \cup \mathcal{R}(i)$$

Donde:
- $\mathcal{G} = \{0, 1, \ldots, G-1\}$ — tokens globales (siempre accesibles)
- $\mathcal{W}(i)$ — ventana local de tamaño $w$
- $\mathcal{D}(i)$ — conexiones dilatadas con offsets exponenciales
- $\mathcal{R}(i)$ — conexiones pseudo-aleatorias

El tamaño total es fijo:

$$K = |\mathcal{N}(i)| = G + w + 2 \cdot |\text{dilated\_offsets}| + r$$

Con los defaults de HALO-S: $K = 2 + 64 + 2 \times 4 + 2 = 76$.

### Cómputo de Atención Gather-Based

Dada la neighbor list $\mathcal{N} \in \mathbb{Z}^{N \times K}$ (tensor de índices), el forward pass es:

1. **Proyecciones lineales**:
$$Q = XW_Q \in \mathbb{R}^{N \times H \times d_k}, \quad K = XW_K \in \mathbb{R}^{N \times H_{kv} \times d_k}, \quad V = XW_V \in \mathbb{R}^{N \times H_{kv} \times d_k}$$

2. **Gather de vecinos**:
$$K_{\text{gathered}}[i] = K[\mathcal{N}[i]] \in \mathbb{R}^{K \times d_k}$$
$$V_{\text{gathered}}[i] = V[\mathcal{N}[i]] \in \mathbb{R}^{K \times d_k}$$

3. **Scores locales**:
$$s_{i,j} = \frac{q_i^T \cdot k_{\mathcal{N}(i,j)}}{\sqrt{d_k}} \quad \forall j \in \{1, \ldots, K\}$$

4. **Máscara causal**: Si $\mathcal{N}(i,j) > i$, entonces $s_{i,j} = -\infty$

5. **Softmax local**:
$$\alpha_{i,j} = \text{softmax}(s_i)_j = \frac{\exp(s_{i,j})}{\sum_{m=1}^{K} \exp(s_{i,m})}$$

6. **Output**:
$$o_i = \sum_{j=1}^{K} \alpha_{i,j} \cdot v_{\mathcal{N}(i,j)}$$

### Complejidad

| Operación | Complejidad |
|-----------|-------------|
| Proyecciones Q, K, V | $O(N \times H \times d_k)$ |
| Gather de K, V | $O(N \times K \times d_k)$ |
| Scores de atención | $O(N \times H \times K \times d_k)$ |
| Softmax | $O(N \times H \times K)$ |
| Output (attn × V) | $O(N \times H \times K \times d_k)$ |
| **Total por capa** | $O(N \times K \times H \times d_k)$ |

Para HALO-S: $O(N \times 76)$ vs denso $O(N \times N)$. Con $N = 4096$: **54× reducción**.

## Implementación en el Framework

**Archivo principal**: `halo/attention/halo_attention.py` — clase `HaloSparseAttention`

**Archivo de grafo**: `halo/attention/graph.py` — función `generate_neighbor_lists()`

El gather se implementa con advanced indexing de PyTorch:

```python
# neighbors: (seq_len, num_neighbors) — índices precomputados
# k: (batch, num_kv_heads, seq_len, head_dim)
k_gathered = k[:, :, neighbors, :]  # (batch, num_kv_heads, seq_len, num_neighbors, head_dim)
v_gathered = v[:, :, neighbors, :]
```

Los scores se computan con matmul expandido:

```python
q_expanded = q.unsqueeze(3)  # (batch, heads, seq_len, 1, head_dim)
scores = torch.matmul(q_expanded, k_gathered.transpose(-2, -1))  # (..., 1, K)
```

La neighbor list se cachea por longitud de secuencia para evitar recalcularla en cada forward pass.

## Comparación con Alternativas

| Aspecto | Dense Attention | Block-Sparse | Gather-based (HALO-S) |
|---------|----------------|--------------|----------------------|
| Memoria scores | $O(N^2)$ | $O(N \times b)$ | $O(N \times K)$ |
| Kernels CUDA | No necesita | Sí (Triton) | No necesita |
| Flexibilidad patrón | Máxima | Bloques fijos | Arbitrario |
| Implementación | Trivial | Compleja | Moderada |
| GPU utilization | Óptima | Buena | Moderada |

La ventaja principal del enfoque gather-based es la **flexibilidad**: permite combinar patrones locales, dilatados, globales y aleatorios en una sola estructura sin necesidad de kernels especializados. El tradeoff es que el acceso irregular a memoria (gather) es menos eficiente en GPU que el acceso contiguo de las implementaciones block-sparse.
