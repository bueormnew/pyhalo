# Tokens Globales con Atención Densa

## Intuición y Motivación

En un modelo con atención dispersa, cada token solo "ve" a sus $K$ vecinos locales. Esto crea un problema: la información de un extremo de la secuencia necesita múltiples capas para propagarse al otro extremo (como un teléfono descompuesto).

Los **tokens globales** resuelven esto actuando como una **memoria compartida centralizada**. Son posiciones especiales ($G$ tokens aprendidos) que:

1. **Atienden a toda la secuencia** — con atención densa $O(G \times N)$
2. **Son atendidos por todos los tokens** — aparecen como vecinos en todas las neighbor lists
3. **Actúan como "hub" de información** — cualquier token puede depositar y leer información global a través de ellos

Esto garantiza que en solo 2 capas, cualquier par de tokens puede comunicarse: token A → global → token B.

## Formulación Matemática

### Posicionamiento

Los globals ocupan las primeras $G$ posiciones de la secuencia:

$$\hat{x} = [\underbrace{g_1, \ldots, g_G}_{\text{globales aprendidos}}; \underbrace{e_1, \ldots, e_N}_{\text{tokens de entrada}}]$$

Donde $g_i \in \mathbb{R}^H$ son parámetros aprendidos (`nn.Parameter`).

### Atención Densa de Globals

Para cada global en posición $i \in \{0, \ldots, G-1\}$, se computa atención densa contra toda la secuencia $S = G + N$:

$$q_i = W_Q \cdot g_i \in \mathbb{R}^{H_q \times d_k}$$
$$K = W_K \cdot \hat{x} \in \mathbb{R}^{S \times H_{kv} \times d_k}$$
$$V = W_V \cdot \hat{x} \in \mathbb{R}^{S \times H_{kv} \times d_k}$$

Scores con máscara causal (global $i$ solo atiende a posiciones $\leq i$):

$$s_{i,j} = \begin{cases} \frac{q_i^T k_j}{\sqrt{d_k}} & \text{si } j \leq i \\ -\infty & \text{si } j > i \end{cases}$$

$$\alpha_{i,j} = \text{softmax}(s_i)_j$$

$$\text{out}_i = \sum_{j=0}^{S-1} \alpha_{i,j} \cdot v_j$$

### Interacción con Tokens Dispersos

Simultáneamente, los tokens regulares acceden a los globals a través de su neighbor list:

$$\mathcal{N}(i) = \underbrace{\{0, 1, \ldots, G-1\}}_{\text{siempre incluidos}} \cup \mathcal{W}(i) \cup \mathcal{D}(i) \cup \mathcal{R}(i)$$

Esto crea un **flujo bidireccional**:
- Globals ← toda la secuencia (atención densa)
- Tokens → globals (incluidos en neighbor lists)

### Complejidad

| Componente | Costo |
|-----------|-------|
| Proyecciones Q (globals) | $O(G \times H \times d_k)$ |
| Proyecciones K, V (secuencia) | $O(S \times H_{kv} \times d_k)$ |
| Scores | $O(G \times S \times H \times d_k)$ |
| Output | $O(G \times S \times H \times d_k)$ |
| **Total por capa** | $O(G \times S \times H \times d_k)$ |

Con $G = 2$ y $S = 4098$: costo de globals $\approx 2/4098 \approx 0.05\%$ del costo de un Transformer denso completo. Es negligible pero proporciona conectividad global.

## Implementación en el Framework

**Archivo**: `halo/attention/global_attention.py` — clase `GlobalFullAttention`

**Integración**: `halo/nn/halo_block.py` — clase `HaloBlock`

El flujo en `HaloBlock.forward()`:

```python
# Separar globals de tokens
globals_x = x_normed[:, :num_globals, :]  # (B, G, H)

# Atención densa para globals
globals_out = self.global_attn(globals_x, x_normed, cos, sin, is_causal)

# Atención dispersa para tokens (globals ya están en neighbor lists)
attn_full_out = self.attn(x_normed, cos, sin, is_causal)
tokens_out = attn_full_out[:, num_globals:, :]

# Recombinar
attn_out = torch.cat([globals_out, tokens_out], dim=1)
```

Los globals se inicializan como `nn.Parameter` en `HaloSModel`:

```python
self.global_memory = nn.Parameter(torch.randn(config.num_globals, config.hidden_size))
```

Se inyectan al inicio de cada secuencia en el forward del modelo.

## Comparación con Alternativas

| Enfoque | Tipo | Aprendible | Complejidad | Notas |
|---------|------|-----------|-------------|-------|
| CLS token (BERT) | 1 token especial | Sí | $O(N)$ extra | Solo para clasificación |
| Longformer globals | Tokens fijos de entrada | No (fijos) | $O(G \times N)$ | El usuario elige cuáles |
| BigBird random globals | Tokens aleatorios | No | $O(G \times N)$ | Aleatorios por batch |
| **HALO-S globals** | Parámetros aprendidos | Sí | $O(G \times N)$ | Memoria compartida aprendida |
| Perceiver latents | Array aprendido | Sí | $O(L \times N)$ | Más latents, más caro |

La ventaja de HALO-S es que los globals son **parámetros aprendidos** (no dependen de la entrada) que se especializan durante el entrenamiento para almacenar información útil a nivel de secuencia. Esto los diferencia de los globals de Longformer que simplemente son tokens de entrada seleccionados por el usuario.
