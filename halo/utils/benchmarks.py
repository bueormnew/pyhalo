"""
HALO-S Benchmarks — Utilidades para medir rendimiento del modelo.

Funciones para evaluar velocidad (latencia), generación (throughput),
consumo de memoria (VRAM), y estimación teórica de FLOPs.
"""

import time
import torch
from typing import Optional

from halo.core.config import HaloConfig


def benchmark_speed(
    model: torch.nn.Module,
    config: HaloConfig,
    seq_lengths: list = None,
    batch_size: int = 1,
    warmup_runs: int = 3,
    timed_runs: int = 10,
    device: Optional[str] = None,
) -> list:
    """
    Mide la latencia del forward pass para múltiples longitudes de secuencia.

    Ejecuta warmup_runs iteraciones de calentamiento y luego timed_runs
    iteraciones cronometradas para cada longitud de secuencia.

    Args:
        model: Modelo nn.Module compatible con input_ids (batch, seq_len).
        config: Configuración del modelo HALO-S.
        seq_lengths: Lista de longitudes de secuencia a evaluar.
        batch_size: Tamaño del batch para las pruebas.
        warmup_runs: Número de iteraciones de calentamiento (no cronometradas).
        timed_runs: Número de iteraciones cronometradas para promediar.
        device: Dispositivo a usar ('cpu', 'cuda'). Auto-detecta si None.

    Returns:
        Lista de diccionarios con keys: seq_len, avg_ms, tokens_per_sec.
    """
    if seq_lengths is None:
        seq_lengths = [128, 512, 1024, 2048]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for seq_len in seq_lengths:
            # Generar entrada aleatoria con tokens válidos
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_len), device=device
            )

            # Fase de calentamiento (no se cronometra)
            for _ in range(warmup_runs):
                _ = model(input_ids)

            # Sincronizar CUDA antes de medir
            if device == "cuda":
                torch.cuda.synchronize()

            # Fase cronometrada
            start = time.perf_counter()
            for _ in range(timed_runs):
                _ = model(input_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
            end = time.perf_counter()

            # Calcular métricas
            total_time_ms = (end - start) * 1000.0
            avg_ms = total_time_ms / timed_runs
            total_tokens = batch_size * seq_len
            tokens_per_sec = total_tokens / (avg_ms / 1000.0)

            results.append({
                "seq_len": seq_len,
                "avg_ms": round(avg_ms, 4),
                "tokens_per_sec": round(tokens_per_sec, 2),
            })

    return results


def benchmark_generation(
    model: torch.nn.Module,
    config: HaloConfig,
    prompt_len: int = 10,
    max_new_tokens: int = 100,
    num_runs: int = 5,
    device: Optional[str] = None,
) -> dict:
    """
    Mide el throughput de generación autoregresiva.

    Crea un prompt aleatorio y ejecuta la generación num_runs veces,
    promediando el tiempo total.

    Args:
        model: Modelo con método generate() compatible.
        config: Configuración del modelo HALO-S.
        prompt_len: Longitud del prompt inicial en tokens.
        max_new_tokens: Número máximo de tokens nuevos a generar.
        num_runs: Número de ejecuciones para promediar.
        device: Dispositivo a usar. Auto-detecta si None.

    Returns:
        Diccionario con tokens_generated, total_time_s, tokens_per_sec.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    # Crear prompt aleatorio
    prompt = torch.randint(0, config.vocab_size, (1, prompt_len), device=device)

    total_time = 0.0
    tokens_generated = 0

    with torch.no_grad():
        for _ in range(num_runs):
            # Sincronizar antes de medir
            if device == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            output = model.generate(prompt, max_new_tokens=max_new_tokens)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            # Contar tokens generados (output incluye prompt)
            if isinstance(output, torch.Tensor):
                generated = output.shape[-1] - prompt_len
            else:
                generated = max_new_tokens

            total_time += (end - start)
            tokens_generated += generated

    # Promediar sobre todas las ejecuciones
    avg_time = total_time / num_runs
    avg_tokens = tokens_generated / num_runs
    tokens_per_sec = avg_tokens / avg_time if avg_time > 0 else 0.0

    return {
        "tokens_generated": int(avg_tokens),
        "total_time_s": round(avg_time, 4),
        "tokens_per_sec": round(tokens_per_sec, 2),
    }


def benchmark_memory(
    model: torch.nn.Module,
    config: HaloConfig,
    seq_lengths: list = None,
    batch_size: int = 1,
    device: str = "cuda",
) -> list:
    """
    Mide el uso real de VRAM (memoria pico y asignada) durante forward pass.

    Solo funciona en dispositivos CUDA. Retorna lista vacía si CUDA no está
    disponible.

    Args:
        model: Modelo nn.Module compatible con input_ids (batch, seq_len).
        config: Configuración del modelo HALO-S.
        seq_lengths: Lista de longitudes de secuencia a evaluar.
        batch_size: Tamaño del batch para las pruebas.
        device: Dispositivo CUDA a usar (default "cuda").

    Returns:
        Lista de diccionarios con keys: seq_len, peak_memory_mb, allocated_mb.
        Lista vacía si CUDA no está disponible.
    """
    # Verificar disponibilidad de CUDA
    if not torch.cuda.is_available():
        return []

    if seq_lengths is None:
        seq_lengths = [512, 1024, 2048, 4096]

    model = model.to(device)
    model.eval()

    results = []

    with torch.no_grad():
        for seq_len in seq_lengths:
            # Resetear estadísticas de memoria
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

            # Generar entrada aleatoria
            input_ids = torch.randint(
                0, config.vocab_size, (batch_size, seq_len), device=device
            )

            # Forward pass para registrar uso de memoria
            _ = model(input_ids)
            torch.cuda.synchronize()

            # Registrar métricas de memoria
            peak_memory = torch.cuda.max_memory_allocated(device)
            allocated_memory = torch.cuda.memory_allocated(device)

            results.append({
                "seq_len": seq_len,
                "peak_memory_mb": round(peak_memory / (1024 * 1024), 2),
                "allocated_mb": round(allocated_memory / (1024 * 1024), 2),
            })

    return results


def estimate_flops(config: HaloConfig, seq_len: int = 1024) -> dict:
    """
    Estimación teórica de FLOPs para un forward pass del modelo HALO-S.

    Función standalone que calcula los FLOPs sin necesidad de instanciar
    el modelo. Usa la misma fórmula que HaloSModel.estimate_flops().

    Componentes calculados:
    - Atención dispersa: proyecciones QKV + scores + attn×V + proyección O
    - Atención global densa: para los global tokens que atienden toda la secuencia
    - FFN: dos capas lineales (hidden → 4×hidden → hidden)
    - LM Head: proyección final al vocabulario

    Args:
        config: Configuración del modelo HALO-S.
        seq_len: Longitud de la secuencia de entrada (sin contar globals).

    Returns:
        Diccionario con desglose de FLOPs por componente:
        - attention_flops: FLOPs de la atención dispersa (todas las capas)
        - global_flops: FLOPs de la atención densa global (todas las capas)
        - ffn_flops: FLOPs de las capas FFN (todas las capas)
        - lm_head_flops: FLOPs de la cabeza de lenguaje
        - total_flops: Suma total de FLOPs
        - total_gflops: Total en GigaFLOPs (redondeado a 4 decimales)
    """
    hidden = config.hidden_size
    num_heads = config.num_heads
    num_kv_heads = config.num_kv_heads
    head_dim = config.head_dim
    num_globals = config.num_globals
    num_neighbors = config.num_neighbors
    num_layers = config.num_layers
    vocab_size = config.vocab_size

    # --- FLOPs de atención dispersa por capa ---
    # Proyecciones Q, K, V: Q usa num_heads, K/V usan num_kv_heads
    qkv_flops = 2 * seq_len * hidden * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
    # Scores de atención dispersa: Q(seq_len, heads, head_dim) × K(num_neighbors, head_dim)
    scores_flops = 2 * num_heads * seq_len * num_neighbors * head_dim
    # Atención × V: scores(seq_len, num_neighbors) × V(num_neighbors, head_dim)
    attn_v_flops = 2 * num_heads * seq_len * num_neighbors * head_dim
    # Proyección de salida O
    o_proj_flops = 2 * seq_len * (num_heads * head_dim) * hidden

    attention_flops = qkv_flops + scores_flops + attn_v_flops + o_proj_flops

    # --- FLOPs de atención densa para Global Tokens por capa ---
    # Proyecciones Q, K, V para globals: Q de globals, K/V de secuencia completa
    total_seq = seq_len + num_globals
    global_qkv_flops = (
        2 * num_globals * hidden * (num_heads * head_dim)       # Q de globals
        + 2 * total_seq * hidden * (num_kv_heads * head_dim)    # K de toda la seq
        + 2 * total_seq * hidden * (num_kv_heads * head_dim)    # V de toda la seq
    )
    # Scores: Q(globals) × K(total_seq)^T + Atención × V
    global_scores_flops = 2 * num_heads * num_globals * total_seq * head_dim * 2
    # Proyección O para globals
    global_o_proj_flops = 2 * num_globals * (num_heads * head_dim) * hidden

    global_flops = global_qkv_flops + global_scores_flops + global_o_proj_flops

    # --- FLOPs de FFN por capa ---
    # Dos capas lineales: hidden → 4×hidden → hidden, sobre toda la secuencia
    ffn_flops = 2 * total_seq * hidden * (4 * hidden) * 2

    # --- FLOPs del LM Head ---
    lm_head_flops = 2 * seq_len * hidden * vocab_size

    # --- Total ---
    total_flops = (attention_flops + global_flops + ffn_flops) * num_layers + lm_head_flops
    total_gflops = total_flops / 1e9

    return {
        "attention_flops": attention_flops * num_layers,
        "global_flops": global_flops * num_layers,
        "ffn_flops": ffn_flops * num_layers,
        "lm_head_flops": lm_head_flops,
        "total_flops": total_flops,
        "total_gflops": round(total_gflops, 4),
    }
