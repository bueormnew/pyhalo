import torch

def count_parameters(model: torch.nn.Module) -> int:
    """Devuelve el número total de parámetros entrenables del modelo."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def estimate_memory(model: torch.nn.Module, config, batch_size: int = 1, seq_len: int = 1024) -> dict:
    """
    Estima el uso de VRAM (en MB) para inferencia/entrenamiento.
    Incluye los pesos del modelo y el KV Cache estimado para GQA.
    """
    # Aproximación de parámetros a FP32 (4 bytes)
    param_mem_mb = count_parameters(model) * 4 / (1024 ** 2)
    
    # KV Cache (K y V), precisión FP16 (2 bytes) por defecto para estimación
    # Tamaño: (batch * seq_len * num_kv_heads * head_dim * 2 * num_layers * 2 bytes)
    head_dim = config.head_dim
    kv_cache_mb = (batch_size * seq_len * config.num_kv_heads * head_dim * 2 * config.num_layers * 2) / (1024 ** 2)
    
    return {
        "parameters_mb": round(param_mem_mb, 2),
        "kv_cache_mb": round(kv_cache_mb, 2),
        "total_estimated_mb": round(param_mem_mb + kv_cache_mb, 2)
    }
