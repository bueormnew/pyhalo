import time
import torch
from torch.utils.data import DataLoader
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.models.baseline_model import BaselineModel
from halo.datasets.synthetic import NeedleDataset
from halo.training.trainer import Trainer

def run_exp3():
    print("="*50)
    print(" EXPERIMENT 3: LONG CONTEXT (Needle in a Haystack) ")
    print("="*50)
    
    config = HaloConfig(
        vocab_size=50, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2, max_seq_len=256,
        local_window=32, num_globals=2, dilated_offsets=[1,2,4,8], num_random=2
    )
    
    distances = [10, 50, 100, 200]
    
    def test_needle(model_cls, name):
        accuracies = []
        for dist in distances:
            # Training dataset to memorize needle task
            dataset = NeedleDataset(vocab_size=50, seq_len=256, length=80, needle_distance=dist)
            model = model_cls(config)
            trainer = Trainer(model, learning_rate=5e-3)
            trainer.fit(dataset, epochs=10, batch_size=8)
            
            dl = DataLoader(dataset, batch_size=8)
            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for x, y in dl:
                    x, y = x.to(trainer.device), y.to(trainer.device)
                    logits, _ = model(x)
                    preds = logits[:, -1, :].argmax(dim=-1)
                    targets = y[:, -1]
                    correct += (preds == targets).sum().item()
                    total += len(targets)
            acc = correct / total
            accuracies.append(acc)
            print(f"[{name}] Dist: {dist} -> Acc: {acc:.2f}")
        return accuracies

    print("--- Evaluating HALO-S ---")
    acc_halo = test_needle(HaloSModel, "HALO-S")
    
    print("\n--- Evaluating Dense Transformer ---")
    acc_dense = test_needle(BaselineModel, "Dense Transformer")
    
    print("\n--- RESULTS (Accuracy vs Distance) ---")
    print(f"{'Distance':<10} | {'HALO-S':<10} | {'Dense Transformer'}")
    for i, d in enumerate(distances):
        print(f"{d:<10} | {acc_halo[i]:<10.2f} | {acc_dense[i]:.2f}")

if __name__ == "__main__":
    run_exp3()
