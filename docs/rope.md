# Rotary Positional Embeddings (RoPE)

## Intuición y Motivación

Los Transformers necesitan información posicional porque la atención es inherentemente invariante al orden (es una operación sobre conjuntos). RoPE codifica la posición **rotando** los vectores de query y key en el espacio de embeddings, de forma que el producto punto $q_i^T k_j$ dependa naturalmente de la distancia relativa $i - j$.

Ventajas sobre alternativas:
- **vs. Embeddings absolutos aprendidos** — RoPE generaliza a secuencias más largas que las vistas en entrenamiento
- **vs. Embeddings relativos (T5)** — RoPE no requiere bias adicional ni modifica la arquitectura de atención
- **vs. ALiBi** — RoPE preserva la expresividad completa de los scores de atención

RoPE se ha convertido en el estándar de facto para modelos de lenguaje modernos (LLaMA, Mistral, Qwen).

## Formulación Matemática

### Frecuencias Base

Para una dimensión de embedding $d_k$ (head_dim) y una base $\theta = 10000$, se definen las frecuencias inversas:

$$\omega_m = \theta^{-2m/d_k} = \frac{1}{\theta^{2m/d_k}}, \quad m = 0, 1, \ldots, d_k/2 - 1$$

Esto produce frecuencias que varían de alta (para $m = 0$) a baja (para $m = d_k/2 - 1$):

$$\omega_0 = 1, \quad \omega_1 = \theta^{-2/d_k}, \quad \ldots, \quad \omega_{d_k/2-1} = \theta^{-1+2/d_k}$$

### Ángulos de Rotación

Para una posición $t$ en la secuencia:

$$\phi_{t,m} = t \cdot \omega_m$$

### Matriz de Rotación

RoPE aplica una rotación 2D a cada par de dimensiones consecutivas. Para el par $(x_{2m}, x_{2m+1})$ en posición $t$:

$$R_t = \begin{pmatrix} \cos(\phi_{t,m}) & -\sin(\phi_{t,m}) \\ \sin(\phi_{t,m}) & \cos(\phi_{t,m}) \end{pmatrix}$$

Aplicada al vector completo de dimensión $d_k$:

$$R_t \cdot x = \begin{pmatrix} x_0 \cos\phi_{t,0} - x_1 \sin\phi_{t,0} \\ x_0 \sin\phi_{t,0} + x_1 \cos\phi_{t,0} \\ x_2 \cos\phi_{t,1} - x_3 \sin\phi_{t,1} \\ x_2 \sin\phi_{t,1} + x_3 \cos\phi_{t,1} \\ \vdots \end{pmatrix}$$

### Propiedad de Distancia Relativa

El producto punto de dos vectores rotados depende solo de la distancia relativa:

$$(R_i \cdot q)^T (R_j \cdot k) = q^T R_i^T R_j \cdot k = q^T R_{j-i} \cdot k$$

Esto se debe a que $R_i^T R_j = R_{j-i}$ (propiedad de rotaciones). Los scores de atención capturan automáticamente la posición relativa sin parámetros adicionales.

### Implementación Eficiente (rotate_half)

En lugar de construir matrices de rotación explícitas, se usa una formulación equivalente con operaciones element-wise:

$$\text{RoPE}(x, t) = x \odot \cos(\Phi_t) + \text{rotate\_half}(x) \odot \sin(\Phi_t)$$

Donde:
$$\text{rotate\_half}([x_0, x_1, \ldots, x_{d/2-1}, x_{d/2}, \ldots, x_{d-1}]) = [-x_{d/2}, \ldots, -x_{d-1}, x_0, \ldots, x_{d/2-1}]$$

Y $\Phi_t = [t\omega_0, t\omega_0, t\omega_1, t\omega_1, \ldots]$ (frecuencias duplicadas para alinear con pares).

## Implementación en el Framework

**Archivo**: `halo/nn/rope.py` — clases `RotaryPositionalEmbeddings` y función `apply_rotary_pos_emb`

### Precomputación de Cache

```python
class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, dim, max_seq_len=4096, base=10000):
        # Frecuencias inversas: θ^(-2m/d) para m = 0, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        t = torch.arange(seq_len)  # Posiciones [0, 1, ..., seq_len-1]
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim) — duplicar
        # Cache cos y sin para evitar recomputarlos
        self.cos_cached = emb.cos()[None, None, :, :]  # (1, 1, seq_len, dim)
        self.sin_cached = emb.sin()[None, None, :, :]
```

### Aplicación a Q y K

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    q, k: (batch, num_heads, seq_len, head_dim)
    cos, sin: (1, 1, seq_len, head_dim)
    """
    def rotate_half(x):
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
```

### Cache Dinámico

Si la secuencia excede `max_seq_len`, el cache se reconstruye dinámicamente:

```python
def forward(self, x, seq_len=None):
    if seq_len > self.max_seq_len:
        self._build_cache(seq_len)
        self.max_seq_len = seq_len
    return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]
```

## Comparación con Alternativas

| Método | Tipo | Extrapolación | Parámetros extra | Costo |
|--------|------|---------------|-----------------|-------|
| Sinusoidal (Vaswani) | Absoluto | Limitada | 0 | $O(N \times d)$ |
| Aprendido (GPT-2) | Absoluto | No | $N \times d$ | $O(N \times d)$ |
| Relativo (T5) | Relativo | Sí | $O(\text{bins})$ | Bias en scores |
| ALiBi | Relativo | Sí | 0 | Decay lineal |
| **RoPE** | Relativo | Sí | 0 | $O(N \times d)$ |
| YaRN (ext. RoPE) | Relativo | Excelente | 0 | $O(N \times d)$ |

RoPE combina las ventajas de los embeddings relativos (generalización a longitudes no vistas) con la simplicidad de los absolutos (no modifica la estructura de atención). Su formulación como rotación es matemáticamente elegante y computacionalmente eficiente: solo requiere multiplicaciones element-wise con tensores precomputados de cos y sin.

En HALO-S, RoPE se aplica tanto en la atención dispersa como en la atención densa de globals, con slicing apropiado del cache para manejar las diferentes longitudes de Q y K en cada módulo.
