# Análisis de Complejidad — HALO-S vs Transformer Denso

## Intuición y Motivación

El cuello de botella computacional de los Transformers es la atención: computar scores entre todos los pares de tokens cuesta $O(N^2)$ en tiempo y memoria. HALO-S reemplaza esta operación con un patrón disperso que mantiene un número fijo de vecinos por token, logrando complejidad lineal $O(N \times K)$.

Este documento presenta un análisis riguroso de la complejidad temporal y espacial de cada componente, comparando HALO-S con un Transformer denso equivalente.

## Formulación Matemática

### Parámetros de Referencia

| Símbolo | Significado | Default HALO-S |
|---------|-------------|----------------|
| $N$ | Longitud de secuencia | 4096 |
| $H$ | Dimensión oculta | 512 |
| $H_q$ | Cabezas de query | 8 |
| $H_{kv}$ | Cabezas de K/V (GQA) | 2 |
| $d_k$ | Dimensión por cabeza ($H/H_q$) | 64 |
| $G$ | Tokens globales | 2 |
| $w$ | Ventana local | 64 |
| $d$ | Número de dilated offsets | 4 |
| $r$ | Conexiones aleatorias | 2 |
| $K$ | Total vecinos ($G + w + 2d + r$) | 76 |
| $L$ | Número de capas | 6 |
| $V$ | Tamaño vocabulario | 256 |

### Complejidad Temporal (FLOPs por capa)

#### Transformer Denso

| Operación | FLOPs |
|-----------|-------|
| Proyecciones Q, K, V | $6 \times N \times H^2$ |
| Scores ($QK^T$) | $2 \times H_q \times N^2 \times d_k$ |
| Attn × V | $2 \times H_q \times N^2 \times d_k$ |
| Proyección O | $2 \times N \times H^2$ |
| FFN (2 capas) | $16 \times N \times H^2$ |
| **Total atención** | $\mathbf{O(N^2 \times H_q \times d_k)}$ |
| **Total por capa** | $O(N^2 \times H + N \times H^2)$ |

#### HALO-S (Atención Dispersa)

| Operación | FLOPs |
|-----------|-------|
| Proyecciones Q, K, V | $2N(H \cdot H_q d_k + 2H \cdot H_{kv} d_k)$ |
| Gather K, V | $O(N \times K \times d_k)$ (memory-bound) |
| Scores dispersos ($Q \cdot K_{\text{gathered}}^T$) | $2 \times H_q \times N \times K \times d_k$ |
| Attn × V gathered | $2 \times H_q \times N \times K \times d_k$ |
| Proyección O | $2 \times N \times H^2$ |
| FFN (2 capas) | $16 \times (N+G) \times H^2$ |
| **Total atención dispersa** | $\mathbf{O(N \times K \times H_q \times d_k)}$ |

#### HALO-S (Atención Global — adicional)

| Operación | FLOPs |
|-----------|-------|
| Proyecciones Q (globals) | $2 \times G \times H \times H_q \times d_k$ |
| Proyecciones K, V (full seq) | $2 \times (N+G) \times H \times H_{kv} \times d_k \times 2$ |
| Scores globales | $2 \times H_q \times G \times (N+G) \times d_k$ |
| Attn × V | $2 \times H_q \times G \times (N+G) \times d_k$ |
| **Total global** | $O(G \times N \times H_q \times d_k)$ |

### Complejidad Total por Capa

$$\text{HALO-S}_{\text{total}} = O(N \times K \times H_q \times d_k) + O(G \times N \times H_q \times d_k) + O(N \times H^2)$$

$$= O\big(N \times (K + G) \times H_q \times d_k + N \times H^2\big)$$

$$\text{Denso}_{\text{total}} = O(N^2 \times H_q \times d_k + N \times H^2)$$

**Ratio de reducción en atención**:

$$\frac{\text{Denso}}{\text{HALO-S}} \approx \frac{N}{K + G} = \frac{4096}{76 + 2} \approx \mathbf{52.5\times}$$

### Complejidad Espacial (Memoria)

#### Memoria de Activaciones por Capa

| Componente | Transformer Denso | HALO-S |
|-----------|-------------------|---------|
| Scores de atención | $B \times H_q \times N \times N$ | $B \times H_q \times N \times K$ |
| K, V cache | $B \times H_q \times N \times d_k \times 2$ | $B \times H_{kv} \times N \times d_k \times 2$ |
| K, V gathered | — | $B \times H_q \times N \times K \times d_k \times 2$ |
| Neighbor list | — | $N \times K$ (int64) |
| Scores globales | — | $B \times H_q \times G \times N$ |

#### Peak Memory (FP32, por capa)

Con $B = 1$, $N = 4096$, defaults de HALO-S:

| Componente | Denso | HALO-S |
|-----------|-------|--------|
| Scores atención | $8 \times 4096^2 \times 4 = 512$ MB | $8 \times 4096 \times 76 \times 4 = 9.5$ MB |
| KV cache | $8 \times 4096 \times 64 \times 2 \times 4 = 16$ MB | $2 \times 4096 \times 64 \times 2 \times 4 = 4$ MB |
| Gathered KV | — | $8 \times 4096 \times 76 \times 64 \times 2 \times 4 = 1216$ MB |
| Global scores | — | $8 \times 2 \times 4096 \times 4 = 0.25$ MB |

**Nota**: La operación gather crea tensores grandes. En la práctica, el peak memory de HALO-S está dominado por el gathered KV. Sin embargo, para $N$ grande ($>$ 8192), el Transformer denso supera a HALO-S porque $N^2 > N \times K$ crece más rápido.

### Punto de Cruce

HALO-S es más eficiente en memoria de scores cuando:

$$N^2 > N \times K \implies N > K = 76$$

Siempre es ventajoso para cualquier secuencia mayor a 76 tokens.

Para el gathered KV vs dense scores:

$$N \times K \times d_k \times 2 < N^2 \implies K \times d_k \times 2 < N$$

$$76 \times 64 \times 2 = 9728 < N$$

Para $N > 9728$: HALO-S usa menos memoria total que denso incluyendo el gathered KV.

## Implementación en el Framework

**Cálculo de FLOPs**: `halo/models/halo_model.py` — método `HaloSModel.estimate_flops()`

**Estadísticas del grafo**: `halo/attention/graph.py` — función `estimate_graph_stats()`

**Benchmarks**: `halo/utils/benchmarks.py` — funciones `benchmark_speed()`, `benchmark_memory()`

## Comparación con Alternativas

### Tabla Resumen de Complejidad

| Modelo | Temporal (atención) | Espacial (scores) | Escalado |
|--------|--------------------|--------------------|----------|
| Transformer | $O(N^2 d)$ | $O(N^2)$ | Cuadrático |
| Longformer | $O(N \times w \times d)$ | $O(N \times w)$ | Lineal |
| BigBird | $O(N \times (w+g+r) \times d)$ | $O(N \times (w+g+r))$ | Lineal |
| Linear Attention | $O(N \times d^2)$ | $O(d^2)$ | Lineal |
| **HALO-S** | $O(N \times K \times d)$ | $O(N \times K)$ | **Lineal** |

### Escalado Empírico (Forward Pass)

| $N$ | Denso ($N^2$) | HALO-S ($N \times 76$) | Speedup |
|-----|---------------|------------------------|---------|
| 512 | 262,144 | 38,912 | 6.7× |
| 1024 | 1,048,576 | 77,824 | 13.5× |
| 2048 | 4,194,304 | 155,648 | 26.9× |
| 4096 | 16,777,216 | 311,296 | 53.9× |
| 8192 | 67,108,864 | 622,592 | 107.8× |

El speedup crece linealmente con $N$ porque la complejidad del Transformer denso es cuadrática mientras que HALO-S es lineal.
