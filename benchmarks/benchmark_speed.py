import time
import torch
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.utils.metrics import count_parameters

def main():
    print("==================================================")
    print(" HALO-S Forward & Generation Benchmark ")
    print("==================================================")
    
    config = HaloConfig(
        vocab_size=1000,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_kv_heads=2,
        local_window=64,
        num_globals=2,
        max_seq_len=4096
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    
    model = HaloSModel(config).to(device)
    print(f"Parámetros totales: {count_parameters(model):,}")
    
    # Benchmark Forward
    seq_lengths = [128, 512, 1024, 2048]
    batch_size = 1
    
    print("\n--- Forward Latency ---")
    for seq in seq_lengths:
        x = torch.randint(0, config.vocab_size, (batch_size, seq), device=device)
        
        # Warmup
        for _ in range(2):
            model(x)
            
        start_t = time.time()
        with torch.no_grad():
            for _ in range(5):
                model(x)
        end_t = time.time()
        
        avg_ms = ((end_t - start_t) / 5) * 1000
        print(f"Seq Len: {seq:<4} | Latency: {avg_ms:.2f} ms")
        
    print("\n--- Generation Throughput ---")
    x = torch.randint(0, config.vocab_size, (1, 10), device=device)
    start_t = time.time()
    generated = model.generate(x, max_new_tokens=50)
    end_t = time.time()
    
    gen_time = end_t - start_t
    tps = 50 / gen_time
    print(f"Generados 50 tokens en {gen_time:.2f} s | Velocidad: {tps:.2f} tokens/s")
    print("==================================================")

if __name__ == "__main__":
    main()
