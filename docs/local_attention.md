# Atención Local con Ventana Deslizante

## Intuición y Motivación

En lenguaje natural, las dependencias más fuertes tienden a ser locales: un verbo depende de su sujeto cercano, un adjetivo modifica al sustantivo adyacente. La atención local con ventana deslizante captura estas dependencias inmediatas de forma eficiente, permitiendo que cada token atienda únicamente a sus $w$ vecinos más cercanos en la secuencia.

La motivación principal es reducir la complejidad de $O(N^2)$ a $O(N \times w)$ donde $w$ es el tamaño fijo de la ventana, independiente de la longitud total de la secuencia.

## Formulación Matemática

### Definición de la Ventana

Para un token en posición $i$ y una ventana de tamaño $w$, definimos el conjunto de posiciones locales:

$$\mathcal{W}(i) = \{j : i - \lfloor w/2 \rfloor \leq j \leq i + \lfloor w/2 \rfloor\}$$

En HALO-S con `local_window = 64`, cada token atiende a 32 posiciones hacia atrás y 32 hacia adelante (en modo no-causal). En modo causal, solo se conservan las posiciones $j \leq i$.

### Cómputo de Atención Local

Dado un token $i$, las queries, keys y values se computan como:

$$q_i = W_Q \cdot x_i, \quad k_j = W_K \cdot x_j, \quad v_j = W_V \cdot x_j \quad \forall j \in \mathcal{W}(i)$$

Los scores de atención locales son:

$$\alpha_{i,j} = \frac{\exp(q_i^T k_j / \sqrt{d_k})}{\sum_{m \in \mathcal{W}(i)} \exp(q_i^T k_m / \sqrt{d_k})} \quad \forall j \in \mathcal{W}(i)$$

La salida para el token $i$:

$$o_i = \sum_{j \in \mathcal{W}(i)} \alpha_{i,j} \cdot v_j$$

### Complejidad

- **Temporal**: $O(N \times w \times d_k)$ — cada token computa $w$ scores
- **Espacial**: $O(N \times w)$ para almacenar los índices de vecinos

Para $N = 4096$ y $w = 64$: ratio de compresión $= N/w = 64\times$.

### Manejo de Bordes

Los tokens cercanos al inicio de la secuencia (posición $i < w/2$) no tienen suficientes vecinos hacia la izquierda. HALO-S maneja esto con clamping:

$$\mathcal{W}(i) = \{\text{clamp}(j, 0, N-1) : j \in \text{offsets}(i)\}$$

Los índices fuera de rango se redirigen a la posición del propio token, lo que añade peso al self-attention sin afectar la semántica.

## Implementación en el Framework

**Archivo**: `halo/attention/graph.py` — función `generate_neighbor_lists()`

La ventana local se genera como offsets relativos a cada posición:

```python
local_half = local_window // 2
local_offsets = torch.arange(-local_half, local_half + (local_window % 2))
local_idx = positions.unsqueeze(1) + local_offsets  # (seq_len, local_window)
```

Los índices fuera de rango se corrigen con:

```python
out_of_bounds = (all_idx < 0) | (all_idx >= seq_len)
all_idx = torch.where(out_of_bounds, positions.unsqueeze(1), all_idx)
```

La atención local no es un módulo separado; se integra dentro de `HaloSparseAttention` (`halo/attention/halo_attention.py`) como parte del conjunto total de vecinos por token.

## Comparación con Alternativas

| Mecanismo | Complejidad | Campo receptivo por capa | Notas |
|-----------|-------------|-------------------------|-------|
| Atención densa | $O(N^2)$ | Toda la secuencia | Gold standard en calidad |
| Ventana fija (HALO-S) | $O(N \times w)$ | $w$ tokens | Simple, eficiente |
| Longformer sliding window | $O(N \times w)$ | $w$ tokens | Similar, kernel CUDA |
| Block-sparse (BigBird) | $O(N \times b)$ | Bloques de $b$ | Requiere alineación |
| Linear Attention | $O(N \times d)$ | Global (aproximada) | Pierde expresividad |

La ventana local de HALO-S es conceptualmente idéntica a la de Longformer, pero la implementación difiere: Longformer usa kernels CUDA especializados con bloques, mientras que HALO-S usa gather-based indexing que funciona en cualquier device sin kernels custom.

En HALO-S, la ventana local se complementa con conexiones dilatadas, globales y aleatorias para alcanzar campo receptivo global en pocas capas.
