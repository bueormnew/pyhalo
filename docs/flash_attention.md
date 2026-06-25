# Integración SDPA (Flash Attention)

## Intuición y Motivación

PyTorch 2.0+ incluye `torch.nn.functional.scaled_dot_product_attention` (SDPA), una interfaz unificada que despacha automáticamente a backends optimizados:

- **Flash Attention** (Dao et al.) — fusión de operaciones en SRAM, sin materializar la matriz $N \times N$
- **Memory-Efficient Attention** (xFormers) — chunked computation para reducir peak memory
- **Math backend** — implementación estándar como fallback

HALO-S integra SDPA donde es aplicable para acelerar la atención densa sin cambiar la semántica del modelo.

### Dónde se aplica SDPA

| Componente | Usa SDPA | Razón |
|-----------|----------|-------|
| `GlobalFullAttention` | ✅ Sí | Atención densa estándar (G×N) |
| `DenseAttention` (Baseline) | ✅ Sí | Atención densa estándar (N×N) |
| `HaloSparseAttention` | ❌ No | Patrón gather-based irregular, incompatible con SDPA |

SDPA requiere patrones de atención regulares (densos o causales triangulares). La atención dispersa gather-based de HALO-S usa indices arbitrarios por token, lo que la hace incompatible con las optimizaciones de Flash Attention.

## Formulación Matemática

### SDPA Estándar

La operación que SDPA computa:

$$\text{SDPA}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

Donde $M$ es una máscara aditiva opcional ($-\infty$ para posiciones bloqueadas, $0$ para las permitidas).

### Con Máscara Causal para Globals

Para `GlobalFullAttention`, la máscara no es la triangular estándar (porque Q tiene $G$ filas y K tiene $N$ columnas):

$$M_{i,j} = \begin{cases} 0 & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases} \quad i \in [0, G), \; j \in [0, N)$$

Se construye una máscara explícita en lugar de usar `is_causal=True` de SDPA.

### Con Dropout en Training

$$\text{SDPA}(Q, K, V, p) = \text{dropout}\left(\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right), p\right) V$$

Donde $p = \text{dropout\_rate}$ durante training y $p = 0$ en inferencia.

## Implementación en el Framework

### Detección de Disponibilidad

**Archivo**: `halo/attention/global_attention.py`

```python
def _use_sdpa() -> bool:
    """Detecta si SDPA está disponible y es eficiente."""
    if not hasattr(F, 'scaled_dot_product_attention'):
        return False
    if torch.cuda.is_available():
        return True
    return False  # CPU no se beneficia significativamente
```

### Uso en GlobalFullAttention

```python
class GlobalFullAttention(nn.Module):
    def __init__(self, config, use_flash=True):
        self._use_flash = use_flash and _use_sdpa()

    def forward(self, globals_x, full_seq, cos, sin, is_causal):
        # ... proyecciones y RoPE ...

        if self._use_flash:
            out = self._forward_sdpa(q, k, v, attn_mask)
        else:
            out = self._forward_manual(q, k, v, attn_mask)
```

### Path SDPA

```python
def _forward_sdpa(self, q, k, v, causal_mask):
    # Convertir bool mask a float mask para SDPA
    sdpa_mask = torch.zeros_like(causal_mask, dtype=q.dtype)
    sdpa_mask.masked_fill_(causal_mask, float('-inf'))
    sdpa_mask = sdpa_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, G, N)

    dropout_p = self.dropout.p if self.training else 0.0

    out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=sdpa_mask,
        dropout_p=dropout_p,
        is_causal=False  # Máscara custom, no triangular estándar
    )
    return out
```

### Fallback Manual

```python
def _forward_manual(self, q, k, v, causal_mask):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
    if causal_mask is not None:
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = self.dropout(attn_weights)
    return torch.matmul(attn_weights, v)
```

### Uso en BaselineModel (DenseAttention)

**Archivo**: `halo/models/baseline_model.py`

El modelo baseline usa SDPA con `is_causal=True` directamente (la máscara triangular estándar aplica):

```python
if _use_sdpa():
    out = F.scaled_dot_product_attention(
        q, k, v,
        is_causal=is_causal,
        dropout_p=self.dropout.p if self.training else 0.0
    )
```

## Comparación con Alternativas

| Backend | Complejidad Memoria | Velocidad | Requisitos |
|---------|-------------------|-----------|-----------|
| Manual (matmul + softmax) | $O(N^2)$ | Base | Ninguno |
| **SDPA Flash** | $O(N)$ | 2-4× más rápido | CUDA, sm80+ |
| SDPA Memory-efficient | $O(\sqrt{N})$ | 1.5-2× | CUDA |
| SDPA Math (fallback) | $O(N^2)$ | ~Base | Ninguno |
| xFormers | $O(\sqrt{N})$ | 2× | Paquete extra |
| Custom Triton kernel | $O(N)$ | Máximo | Triton + desarrollo |

La ventaja de usar la interfaz SDPA es la **portabilidad**: PyTorch selecciona automáticamente el backend más eficiente disponible en el hardware actual, sin necesidad de código condicional por plataforma. HALO-S obtiene la aceleración de Flash Attention en CUDA sin ninguna dependencia externa adicional.
