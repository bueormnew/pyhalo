import torch
import time
from halo.attention.graph import generate_neighbor_lists, estimate_graph_stats

def main():
    print("=" * 50)
    print(" HALO-S Graph Prototype Benchmark ")
    print("=" * 50)
    
    # Configuraciones típicas para un modelo
    local_window = 64
    num_globals = 2
    dilated_offsets = [1, 2, 4, 8]
    num_random = 2
    layer_id = 0
    
    seq_lengths = [1024, 4096, 16384, 65536, 131072]
    
    print(f"Configuración de vecinos:")
    print(f"- Globales: {num_globals}")
    print(f"- Local: {local_window}")
    print(f"- Dilatados: {len(dilated_offsets) * 2} (offsets: {dilated_offsets})")
    print(f"- Aleatorios (Deterministas): {num_random}")
    num_neighbors = num_globals + local_window + len(dilated_offsets) * 2 + num_random
    print(f"- Total vecinos por token: {num_neighbors}\n")
    
    print(f"{'Seq Len':<10} | {'Sparse Mem (MB)':<16} | {'Dense Mem (MB)':<15} | {'Compression':<12} | {'Time (ms)'}")
    print("-" * 75)
    
    for seq_len in seq_lengths:
        start_t = time.time()
        # Generar conectividad (neighbor lists)
        neighbors = generate_neighbor_lists(
            seq_len=seq_len,
            local_window=local_window,
            num_globals=num_globals,
            dilated_offsets=dilated_offsets,
            num_random=num_random,
            layer_id=layer_id
        )
        end_t = time.time()
        
        stats = estimate_graph_stats(neighbors, seq_len)
        
        gen_time_ms = (end_t - start_t) * 1000
        
        print(f"{seq_len:<10} | {stats['sparse_memory_mb']:<16.2f} | {stats['dense_memory_mb']:<15.2f} | {stats['compression_ratio']:<12.1f}x | {gen_time_ms:.2f} ms")

if __name__ == "__main__":
    main()
