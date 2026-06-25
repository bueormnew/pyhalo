import time
import torch
import math
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.datasets.synthetic import CopyDataset
from halo.training.trainer import Trainer

def run_exp2():
    print("="*50)
    print(" EXPERIMENT 2: ABLATION STUDY ")
    print("="*50)
    
    dataset = CopyDataset(vocab_size=50, seq_len=256, length=80, pattern_len=8)
    
    def train_variant(name, override_kwargs):
        config = HaloConfig(
            vocab_size=50, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=256,
            local_window=32, num_globals=2, dilated_offsets=[1,2,4,8], num_random=2
        )
        for k, v in override_kwargs.items():
            setattr(config, k, v)
            
        model = HaloSModel(config)
        trainer = Trainer(model, learning_rate=3e-3)
        start = time.time()
        trainer.fit(dataset, epochs=5, batch_size=8)
        total_time = time.time() - start
        final_loss = trainer.evaluate(dataset, batch_size=8)
        return {"name": name, "time": total_time, "loss": final_loss, "ppl": math.exp(final_loss)}

    variants = [
        ("HALO-S Completo", {}),
        ("Sin Globales", {"num_globals": 0}),
        ("Sin Dilatadas", {"dilated_offsets": []}),
        ("Sin Aleatorias", {"num_random": 0}),
        ("Solo Local", {"num_globals": 0, "dilated_offsets": [], "num_random": 0})
    ]
    
    results = []
    for name, kwargs in variants:
        print(f"\nTraining variant: {name}")
        res = train_variant(name, kwargs)
        results.append(res)
        
    print("\n--- RESULTS ---")
    for r in results:
        print(f"{r['name']:<18} | Time: {r['time']:.2f}s | Loss: {r['loss']:.4f} | PPL: {r['ppl']:.4f}")

if __name__ == "__main__":
    run_exp2()
