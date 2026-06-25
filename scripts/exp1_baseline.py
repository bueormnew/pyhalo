import time
import torch
import math
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.models.baseline_model import BaselineModel
from halo.datasets.synthetic import CopyDataset
from halo.training.trainer import Trainer

def run_exp1():
    print("="*50)
    print(" EXPERIMENT 1: HALO-S vs DENSE TRANSFORMER ")
    print("="*50)
    
    config = HaloConfig(
        vocab_size=50,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=256,
        local_window=32,
        num_globals=2,
        dilated_offsets=[1,2,4,8],
        num_random=2
    )
    
    dataset = CopyDataset(vocab_size=50, seq_len=256, length=80, pattern_len=8)
    
    def train_model(model_cls, name):
        model = model_cls(config)
        trainer = Trainer(model, learning_rate=3e-3)
        print(f"\nTraining {name}...")
        start = time.time()
        trainer.fit(dataset, epochs=5, batch_size=8)
        total_time = time.time() - start
        final_loss = trainer.evaluate(dataset, batch_size=8)
        perplexity = math.exp(final_loss)
        return {"time": total_time, "loss": final_loss, "ppl": perplexity}

    res_halo = train_model(HaloSModel, "HALO-S")
    res_dense = train_model(BaselineModel, "Dense Transformer")
    
    print("\n--- RESULTS ---")
    print(f"Dense Transformer : Time={res_dense['time']:.2f}s | Loss={res_dense['loss']:.4f} | PPL={res_dense['ppl']:.4f}")
    print(f"HALO-S            : Time={res_halo['time']:.2f}s | Loss={res_halo['loss']:.4f} | PPL={res_halo['ppl']:.4f}")

if __name__ == "__main__":
    run_exp1()
