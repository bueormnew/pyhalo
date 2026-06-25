# Conexiones Dilatadas

## Intuición y Motivación

La ventana local captura dependencias cercanas, pero el campo receptivo por capa es limitado a $w$ tokens. Para que un token en posición 0 se comunique con uno en posición 1000 a través de solo ventanas locales, necesitaría $\lceil 1000/w \rceil$ capas de propagación.

Las **conexiones dilatadas** resuelven esto añadiendo saltos exponencialmente crecientes: cada token se conecta directamente a posiciones a distancias $\pm 1, \pm 2, \pm 4, \pm 8, \ldots$ de sí mismo. Esto expande el campo receptivo exponencialmente con pocas conexiones adicionales:

- Con offsets $[1, 2, 4, 8]$: un token alcanza posiciones hasta $\pm 8$ directamente
- Tras 2 capas: alcance transitivo hasta $\pm 16$ (por composición)
- Tras $L$ capas: alcance $\sim 2^L \times \max(\text{offset})$

## Formulación Matemática

### Definición del Conjunto Dilatado

Para un token en posición $i$ y offsets dilatados $\Delta = [\delta_1, \delta_2, \ldots, \delta_d]$:

$$\mathcal{D}(i) = \{i + \delta_k : \delta_k \in \Delta\} \cup \{i - \delta_k : \delta_k \in \Delta\}$$

Con $\Delta = [1, 2, 4, 8]$ (defaults de HALO-S):

$$\mathcal{D}(i) = \{i \pm 1, i \pm 2, i \pm 4, i \pm 8\}$$

Tamaño: $|\mathcal{D}(i)| = 2 \times |\Delta| = 8$ conexiones por token.

### Expansión Exponencial del Campo Receptivo

Definimos el campo receptivo $\mathcal{R}_\ell(i)$ como el conjunto de posiciones alcanzables desde $i$ tras $\ell$ capas. Con solo conexiones dilatadas:

$$\mathcal{R}_1(i) = \mathcal{D}(i) = \{i \pm 1, \pm 2, \pm 4, \pm 8\}$$

$$\mathcal{R}_2(i) = \bigcup_{j \in \mathcal{R}_1(i)} \mathcal{D}(j) \supseteq \{i \pm k : k \leq 16\}$$

En general, con $\Delta = [2^0, 2^1, \ldots, 2^{d-1}]$:

$$\text{Alcance tras } \ell \text{ capas} \geq \ell \times 2^{d-1}$$

Para HALO-S con 6 capas y $\Delta = [1,2,4,8]$: alcance teórico $\geq 6 \times 8 = 48$ posiciones solo con dilatación (más de 4000 con la composición de ventana local + dilatación + globals).

### Selección de Offsets

Los offsets siguen una progresión geométrica base 2:

$$\delta_k = 2^{k-1}, \quad k = 1, \ldots, d$$

Esto maximiza la cobertura con mínimas conexiones. La razón:

$$\text{Posiciones cubiertas en 1 capa} = \sum_{k=0}^{d-1} 2 = 2d \text{ (conexiones directas)}$$
$$\text{Posiciones alcanzables por composición} \approx 2^d \text{ (exponencial)}$$

### Manejo de Bordes con Clamping

Para posiciones cercanas a los extremos ($i < \max(\Delta)$ o $i > N - \max(\Delta)$):

$$\hat{\delta}_{i,k} = \text{clamp}(i + \delta_k, 0, N-1)$$

Si el índice resultante está fuera de rango, se redirige a la posición del propio token.

## Implementación en el Framework

**Archivo**: `halo/attention/graph.py` — función `generate_neighbor_lists()`

```python
dilated_offsets = [1, 2, 4, 8]  # Progresión geométrica base 2

# Para cada offset, crear conexiones bidireccionales
dilated_list = []
for offset in dilated_offsets:
    dilated_list.append(positions + offset)   # Hacia adelante
    dilated_list.append(positions - offset)   # Hacia atrás

dilated_idx = torch.stack(dilated_list, dim=1)  # (seq_len, 2*len(offsets))
```

Las conexiones dilatadas se combinan con las demás en la neighbor list final:

```python
all_idx = torch.cat([global_idx, local_idx, dilated_idx, random_idx], dim=1)
```

## Comparación con Alternativas

| Mecanismo | Costo adicional | Alcance por capa | Alcance total ($L$ capas) |
|-----------|----------------|------------------|--------------------------|
| Solo ventana local | 0 | $w$ | $L \times w$ |
| **Dilatación (HALO-S)** | $2d$ conexiones | $2^{d-1}$ | $\sim L \times 2^{d-1}$ |
| Stride (fixed) | $N/s$ conexiones | $N$ (subsampled) | $N$ |
| Chunk attention | Variable | Tamaño del chunk | $N$ |

| Aspecto | Dilated Convolutions | Dilated Attention (HALO-S) | Hierarchical Pooling |
|---------|---------------------|---------------------------|---------------------|
| Inspiración | WaveNet | WaveNet + Graph Theory | U-Net |
| Parámetros extra | Filtros por nivel | 0 (reusa proyecciones) | Capas de pooling |
| Implementación | CNN dilatada | Indices en neighbor list | Módulos de downsample |
| Overhead | Moderado | Mínimo (8 índices) | Alto |

La ventaja de las conexiones dilatadas en HALO-S es que **no añaden parámetros**: simplemente son índices adicionales en la neighbor list que comparten las mismas proyecciones Q, K, V con las conexiones locales y globales. El costo marginal es solo 8 posiciones más en el gather operation.
