# Documentación Técnica — HALO-S

Documentación matemática y arquitectónica del framework HALO-S: un modelo de lenguaje basado en atención dispersa con complejidad $O(N \times K)$ en lugar de la cuadrática $O(N^2)$ de los Transformers convencionales.

## Índice de Documentos

| Documento | Descripción |
|-----------|-------------|
| [architecture.md](architecture.md) | Arquitectura general del sistema HALO-S, diagrama de componentes y flujo de datos |
| [local_attention.md](local_attention.md) | Atención local con ventana deslizante: intuición, formulación y complejidad $O(N \times w)$ |
| [sparse_attention.md](sparse_attention.md) | Atención dispersa con neighbor lists: formulación, gather-based backend, $O(N \times K)$ |
| [global_tokens.md](global_tokens.md) | Tokens globales con atención densa: rol de memoria compartida, $O(G \times N)$ |
| [dilated_connections.md](dilated_connections.md) | Conexiones dilatadas: expansión exponencial del campo receptivo con offsets $[1,2,4,8]$ |
| [gqa.md](gqa.md) | Grouped Query Attention: compartición de K/V, reducción de memoria, comparación con MHA/MQA |
| [flash_attention.md](flash_attention.md) | Integración SDPA: uso de `scaled_dot_product_attention`, fallback y compatibilidad |
| [complexity.md](complexity.md) | Análisis completo de complejidad temporal y espacial: HALO-S vs Transformer denso |
| [rope.md](rope.md) | Rotary Positional Embeddings: formulación de la matriz de rotación y frecuencias |

## Convenciones

- Notación LaTeX: `$...$` para inline, `$$...$$` para bloques
- Cada documento sigue la estructura: Intuición → Formulación Matemática → Implementación → Comparación
- Las referencias a código apuntan a archivos del directorio `halo/`

## Referencia Rápida de Parámetros

| Parámetro | Default | Descripción |
|-----------|---------|-------------|
| `hidden_size` | 512 | Dimensión de los embeddings |
| `num_heads` | 8 | Cabezas de atención (query) |
| `num_kv_heads` | 2 | Cabezas de K/V (GQA) |
| `num_globals` | 2 | Tokens globales con atención densa |
| `local_window` | 64 | Ventana local de atención |
| `dilated_offsets` | [1,2,4,8] | Offsets para conexiones dilatadas |
| `num_random` | 2 | Conexiones pseudo-aleatorias por token |
| `max_seq_len` | 4096 | Longitud máxima de secuencia |
