"""
HALO-S Phase 0: Graph Prototype
Módulo para la generación de la conectividad dispersa (sparse) del modelo HALO-S.
Utiliza Neighbor Lists (listas de adyacencia) para evitar la instanciación de matrices densas NxN.
"""

import torch

def generate_neighbor_lists(
    seq_len: int,
    local_window: int = 64,
    num_globals: int = 2,
    dilated_offsets: list[int] = [1, 2, 4, 8],
    num_random: int = 2,
    layer_id: int = 0
) -> torch.Tensor:
    """
    Genera una lista de vecinos (neighbor list) de tamaño fijo para cada token en la secuencia.
    
    Args:
        seq_len: Longitud total de la secuencia (incluyendo globales).
        local_window: Tamaño de la ventana local total (mitad hacia atrás, mitad hacia adelante).
        num_globals: Número de tokens globales al inicio de la secuencia.
        dilated_offsets: Desplazamientos para las conexiones dilatadas.
        num_random: Número de conexiones pseudoaleatorias.
        layer_id: Identificador de la capa para el hash pseudoaleatorio.
        
    Returns:
        neighbors: Tensor de enteros de shape (seq_len, num_neighbors)
                   donde num_neighbors = num_globals + local_window + 2 * len(dilated_offsets) + num_random
    """
    if seq_len <= num_globals:
        raise ValueError("Sequence length must be greater than num_globals")
        
    local_half = local_window // 2
    num_dilated = len(dilated_offsets) * 2
    num_neighbors = num_globals + local_window + num_dilated + num_random
    
    device = torch.device('cpu') # For prototype, CPU is fine to build indices
    positions = torch.arange(seq_len, device=device)
    
    # 1. Global Tokens (0 to num_globals - 1)
    global_idx = torch.arange(num_globals, device=device).unsqueeze(0).expand(seq_len, -1)
    
    # 2. Local Window (-local_half to +local_half - 1)
    # Asegura tamaño exacto `local_window`
    local_offsets = torch.arange(-local_half, local_half + (local_window % 2), device=device).unsqueeze(0)
    local_idx = positions.unsqueeze(1) + local_offsets
    
    # 3. Dilated Connections
    dilated_list = []
    for offset in dilated_offsets:
        dilated_list.append(positions + offset)
        dilated_list.append(positions - offset)
    
    if dilated_list:
        dilated_idx = torch.stack(dilated_list, dim=1)
    else:
        dilated_idx = torch.empty((seq_len, 0), dtype=torch.long, device=device)
        
    # 4. Pseudo-random Connections (Deterministic based on layer_id and position)
    # Using a fast vectorizable integer hash
    pos_expanded = positions.unsqueeze(1).expand(-1, num_random)
    rand_offsets = torch.arange(num_random, device=device).unsqueeze(0).expand(seq_len, -1)
    
    # LCG-like hash parameters
    hash_val = (pos_expanded * 2654435761 + layer_id * 805459861 + rand_offsets * 3266489917)
    hash_val = hash_val.to(torch.int64) & 0xFFFFFFFF
    
    # Map back to valid sequence indices, avoiding globals
    valid_range = max(1, seq_len - num_globals)
    random_idx = num_globals + (hash_val % valid_range)
    
    # Concatenate all indices
    all_idx = torch.cat([global_idx, local_idx, dilated_idx, random_idx], dim=1)
    
    # Clamp out-of-bounds indices
    # Si un token apunta fuera de la secuencia, se redirige a sí mismo.
    # Matemáticamente añade más peso al propio token en self-attention, lo cual es inofensivo.
    out_of_bounds = (all_idx < 0) | (all_idx >= seq_len)
    all_idx = torch.where(out_of_bounds, positions.unsqueeze(1), all_idx)
    
    return all_idx

def estimate_graph_stats(neighbor_lists: torch.Tensor, seq_len: int):
    """
    Calcula estadísticas del grafo generado para comparar con representaciones densas.
    """
    num_neighbors = neighbor_lists.shape[1]
    
    # Memoria de la representación rala (int64)
    sparse_memory_bytes = neighbor_lists.numel() * 8
    
    # Memoria teórica de una máscara densa (bool)
    dense_memory_bytes = seq_len * seq_len * 1
    
    # Factor de compresión
    compression_ratio = dense_memory_bytes / sparse_memory_bytes if sparse_memory_bytes > 0 else float('inf')
    
    # Conexiones totales
    total_connections = seq_len * num_neighbors
    
    return {
        "seq_len": seq_len,
        "num_neighbors": num_neighbors,
        "sparse_memory_mb": sparse_memory_bytes / (1024 * 1024),
        "dense_memory_mb": dense_memory_bytes / (1024 * 1024),
        "compression_ratio": compression_ratio,
        "total_connections": total_connections,
        "dense_connections": seq_len * seq_len
    }
