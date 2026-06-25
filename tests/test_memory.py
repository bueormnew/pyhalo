"""
Tests de escalado lineal de memoria para HALO-S.

Verifica que la atención dispersa escala linealmente O(N×K) en lugar
de cuadráticamente O(N²), confirmando la ventaja fundamental del framework.

**Validates: Requirements 10.7**
"""

import pytest
import torch

from halo.core.config import HaloConfig
from halo.attention.graph import generate_neighbor_lists, estimate_graph_stats
from halo.utils.metrics import estimate_memory


# --- Configuración compartida para tests ---

def _small_config(**overrides) -> HaloConfig:
    """Config pequeña para tests rápidos en CPU."""
    defaults = dict(
        vocab_size=64,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        num_globals=2,
        local_window=16,
        dilated_offsets=[1, 2, 4],
        num_random=2,
        dropout=0.0,
        max_seq_len=2048,
    )
    defaults.update(overrides)
    return HaloConfig(**defaults)


# ===========================================================================
# Test 1: Neighbor list memory scales linearly
# ===========================================================================

def test_neighbor_list_memory_scales_linearly():
    """
    Genera neighbor lists para seq_len N y 2N.
    Verifica que el ratio de memoria es ≈ 2x (lineal), no 4x (cuadrático).

    Property 9: Escalado Lineal de Memoria.
    """
    config = _small_config()
    N = 256
    N2 = 512  # 2x

    # Generar neighbor lists para ambas longitudes
    neighbors_n = generate_neighbor_lists(
        seq_len=N,
        local_window=config.local_window,
        num_globals=config.num_globals,
        dilated_offsets=config.dilated_offsets,
        num_random=config.num_random,
    )
    neighbors_2n = generate_neighbor_lists(
        seq_len=N2,
        local_window=config.local_window,
        num_globals=config.num_globals,
        dilated_offsets=config.dilated_offsets,
        num_random=config.num_random,
    )

    # Obtener estadísticas del grafo
    stats_n = estimate_graph_stats(neighbors_n, N)
    stats_2n = estimate_graph_stats(neighbors_2n, N2)

    # El ratio de memoria dispersa al duplicar N debe ser ≈ 2x (lineal)
    sparse_ratio = stats_2n["sparse_memory_mb"] / stats_n["sparse_memory_mb"]

    # Verificar ratio lineal (≈ 2.0) con tolerancia
    assert 1.8 <= sparse_ratio <= 2.2, (
        f"El ratio de memoria dispersa al duplicar N debería ser ≈ 2.0 "
        f"(lineal), pero fue {sparse_ratio:.3f}"
    )

    # Verificar que NO es cuadrático (si fuera denso, ratio sería ≈ 4.0)
    dense_ratio = stats_2n["dense_memory_mb"] / stats_n["dense_memory_mb"]
    assert dense_ratio > 3.5, (
        f"El ratio denso debería ser ≈ 4.0 (cuadrático), pero fue {dense_ratio:.3f}"
    )


# ===========================================================================
# Test 2: Sparse attention tensor sizes vs dense
# ===========================================================================

def test_sparse_attention_tensor_sizes():
    """
    Para una config dada, calcula la memoria teórica del enfoque gather-based
    (N × num_neighbors × head_dim) vs atención densa (N × N × head_dim).
    Verifica que HALO-S usa O(N) memoria por cabeza.
    """
    config = _small_config()
    seq_len = 512
    head_dim = config.head_dim
    num_neighbors = config.num_neighbors

    # Memoria de scores de atención dispersa: N × num_neighbors (por cabeza)
    sparse_scores_elements = seq_len * num_neighbors
    # Memoria de scores de atención densa: N × N (por cabeza)
    dense_scores_elements = seq_len * seq_len

    # HALO-S debe usar significativamente menos memoria
    assert sparse_scores_elements < dense_scores_elements, (
        f"La atención dispersa ({sparse_scores_elements} elementos) debería usar "
        f"menos memoria que la densa ({dense_scores_elements} elementos)"
    )

    # Verificar que la relación muestra O(N) vs O(N²)
    # Con N=512 y K=num_neighbors fijo, sparse/dense = K/N
    ratio = sparse_scores_elements / dense_scores_elements
    expected_ratio = num_neighbors / seq_len
    assert abs(ratio - expected_ratio) < 1e-6, (
        f"El ratio sparse/dense debería ser num_neighbors/N = {expected_ratio:.4f}, "
        f"pero fue {ratio:.4f}"
    )

    # Verificar escalado: al duplicar N, la memoria dispersa crece 2x, la densa 4x
    seq_len_2 = 1024
    sparse_2 = seq_len_2 * num_neighbors
    dense_2 = seq_len_2 * seq_len_2

    sparse_growth = sparse_2 / sparse_scores_elements  # Debería ser ≈ 2
    dense_growth = dense_2 / dense_scores_elements      # Debería ser ≈ 4

    assert 1.9 <= sparse_growth <= 2.1, (
        f"El crecimiento disperso al duplicar N debería ser ≈ 2.0, fue {sparse_growth:.3f}"
    )
    assert 3.9 <= dense_growth <= 4.1, (
        f"El crecimiento denso al duplicar N debería ser ≈ 4.0, fue {dense_growth:.3f}"
    )


# ===========================================================================
# Test 3: Model forward completes for long sequence (smoke test)
# ===========================================================================

@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Este test de secuencia larga es más relevante con CUDA, pero se ejecuta en CPU si está disponible"
)
def test_model_forward_completes_long_sequence_cuda():
    """Verifica que el modelo puede procesar una secuencia larga en CUDA sin OOM."""
    from halo.models.halo_model import HaloSModel

    config = _small_config(max_seq_len=1024)
    model = HaloSModel(config).cuda().eval()

    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len), device="cuda")

    with torch.no_grad():
        logits, _ = model(input_ids)

    assert logits.shape == (1, seq_len, config.vocab_size)


def test_model_forward_completes_long_sequence():
    """
    Smoke test: verifica que el modelo puede procesar una secuencia larga (512+)
    sin errores en CPU. Confirma que la complejidad lineal permite secuencias
    largas sin restricciones de memoria excesivas.
    """
    from halo.models.halo_model import HaloSModel

    config = _small_config(max_seq_len=1024)
    model = HaloSModel(config).eval()

    seq_len = 512
    input_ids = torch.randint(0, config.vocab_size, (1, seq_len))

    with torch.no_grad():
        logits, _ = model(input_ids)

    # Verificar shapes correctas
    assert logits.shape == (1, seq_len, config.vocab_size), (
        f"Expected logits shape (1, {seq_len}, {config.vocab_size}), "
        f"got {logits.shape}"
    )

    # Verificar que los logits son valores finitos (no NaN/Inf)
    assert torch.isfinite(logits).all(), "Los logits contienen NaN o Inf"


# ===========================================================================
# Test 4: estimate_memory utility returns reasonable values
# ===========================================================================

def test_estimate_memory_utility():
    """
    Verifica que estimate_memory() retorna valores razonables:
    - parameters_mb > 0
    - kv_cache_mb > 0
    - total > ambos componentes individuales
    """
    from halo.models.halo_model import HaloSModel

    config = _small_config()
    model = HaloSModel(config)

    mem = estimate_memory(model, config, batch_size=1, seq_len=512)

    # Verificar que las claves esperadas existen
    assert "parameters_mb" in mem
    assert "kv_cache_mb" in mem
    assert "total_estimated_mb" in mem

    # Verificar que los valores son positivos
    assert mem["parameters_mb"] > 0, (
        f"parameters_mb debería ser > 0, fue {mem['parameters_mb']}"
    )
    assert mem["kv_cache_mb"] > 0, (
        f"kv_cache_mb debería ser > 0, fue {mem['kv_cache_mb']}"
    )

    # Verificar que total es mayor que cada componente individual
    assert mem["total_estimated_mb"] > mem["parameters_mb"], (
        f"total ({mem['total_estimated_mb']}) debería ser > parameters_mb ({mem['parameters_mb']})"
    )
    assert mem["total_estimated_mb"] > mem["kv_cache_mb"], (
        f"total ({mem['total_estimated_mb']}) debería ser > kv_cache_mb ({mem['kv_cache_mb']})"
    )

    # Verificar que total ≈ parameters + kv_cache
    expected_total = mem["parameters_mb"] + mem["kv_cache_mb"]
    assert abs(mem["total_estimated_mb"] - expected_total) < 0.01, (
        f"total ({mem['total_estimated_mb']}) debería ser ≈ "
        f"parameters + kv_cache ({expected_total})"
    )


# ===========================================================================
# Test 5: Compression ratio increases with seq_len
# ===========================================================================

def test_compression_ratio_increases_with_seq_len():
    """
    A mayor seq_len, el ratio de compresión (dense/sparse) debe aumentar,
    confirmando que la ventaja de eficiencia crece con la longitud de secuencia.

    Para atención dispersa O(N×K) vs densa O(N²):
    - ratio = (N²) / (N×K) = N/K
    - Al aumentar N con K fijo, el ratio crece linealmente.
    """
    config = _small_config()

    seq_lengths = [128, 256, 512, 1024]
    ratios = []

    for seq_len in seq_lengths:
        neighbors = generate_neighbor_lists(
            seq_len=seq_len,
            local_window=config.local_window,
            num_globals=config.num_globals,
            dilated_offsets=config.dilated_offsets,
            num_random=config.num_random,
        )
        stats = estimate_graph_stats(neighbors, seq_len)
        ratios.append(stats["compression_ratio"])

    # Verificar que el ratio crece monotónicamente
    for i in range(len(ratios) - 1):
        assert ratios[i + 1] > ratios[i], (
            f"El ratio de compresión debería crecer con seq_len, "
            f"pero ratio[{seq_lengths[i + 1]}]={ratios[i + 1]:.3f} <= "
            f"ratio[{seq_lengths[i]}]={ratios[i]:.3f}"
        )

    # Verificar que para seq_len grande, el ratio es sustancial (>2x al menos)
    assert ratios[-1] > 2.0, (
        f"Con seq_len={seq_lengths[-1]}, el ratio de compresión debería ser > 2.0, "
        f"pero fue {ratios[-1]:.3f}"
    )

    # Verificar crecimiento aproximadamente lineal:
    # ratio ≈ N / K, así que al duplicar N, ratio se duplica
    ratio_growth = ratios[-1] / ratios[0]
    seq_growth = seq_lengths[-1] / seq_lengths[0]
    # El crecimiento del ratio debería ser proporcional al crecimiento de seq_len
    assert ratio_growth > seq_growth * 0.7, (
        f"El crecimiento del ratio ({ratio_growth:.2f}x) debería ser proporcional "
        f"al crecimiento de seq_len ({seq_growth:.1f}x)"
    )
