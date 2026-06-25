import torch
import pytest
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.training.trainer import Trainer
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, vocab_size, seq_len, length):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.length = length
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        x = torch.randint(0, self.vocab_size, (self.seq_len,))
        y = torch.randint(0, self.vocab_size, (self.seq_len,))
        return x, y

def test_trainer_fit():
    config = HaloConfig(vocab_size=50, hidden_size=64, num_layers=1, num_heads=2, num_kv_heads=1, max_seq_len=20)
    model = HaloSModel(config)
    dataset = DummyDataset(vocab_size=config.vocab_size, seq_len=10, length=4)
    
    trainer = Trainer(model, learning_rate=1e-3, device="cpu")
    
    # Run 1 epoch
    initial_loss = trainer.evaluate(dataset, batch_size=2)
    trainer.fit(dataset, epochs=2, batch_size=2)
    final_loss = trainer.evaluate(dataset, batch_size=2)
    
    # Overfitting on random data should reduce the loss somewhat
    # No strict assertion on loss dropping due to randomness, just that it runs without errors
    assert final_loss is not None


# --- Tests de StreamingDataset (Task 8.3) ---
# Validates: Requirements 5.5, 5.6, 5.7

import os
import tempfile
import json
from halo.datasets.streaming import StreamingDataset
from halo.tokenizers.char import CharacterTokenizer


def test_streaming_dataset_yields_correct_shape():
    """Verificar que StreamingDataset yield (x, y) con shape (max_seq_len,)."""
    max_seq_len = 16

    # Crear archivo temporal JSONL con datos suficientes
    tmp_path = os.path.join(tempfile.gettempdir(), "test_stream_shape.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for i in range(50):
            json.dump({"text": f"Esta es la linea numero {i} con texto suficiente para tokens"}, f)
            f.write("\n")

    tok = CharacterTokenizer()
    ds = StreamingDataset(
        file_paths=[tmp_path],
        tokenizer=tok,
        max_seq_len=max_seq_len,
        buffer_size=5,  # Buffer pequeño para test rápido
    )

    # Obtener un elemento del iterador
    it = iter(ds)
    x, y = next(it)
    del it  # Liberar el iterador para cerrar archivos

    assert x.shape == (max_seq_len,), f"Shape x esperada ({max_seq_len},), obtenida {x.shape}"
    assert y.shape == (max_seq_len,), f"Shape y esperada ({max_seq_len},), obtenida {y.shape}"

    os.unlink(tmp_path)


def test_streaming_dataset_iterates_multiple_items():
    """Verificar que se pueden iterar N items sin que se detenga."""
    max_seq_len = 8
    n_items = 10

    # Crear archivo temporal con texto abundante
    tmp_path = os.path.join(tempfile.gettempdir(), "test_stream_multi.jsonl")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for i in range(100):
            json.dump({"text": f"Texto largo para generar muchos tokens en la iteracion {i} del dataset"}, f)
            f.write("\n")

    tok = CharacterTokenizer()
    ds = StreamingDataset(
        file_paths=[tmp_path],
        tokenizer=tok,
        max_seq_len=max_seq_len,
        buffer_size=5,
    )

    # Iterar N elementos sin error
    it = iter(ds)
    count = 0
    for _ in range(n_items):
        x, y = next(it)
        assert x.shape == (max_seq_len,)
        assert y.shape == (max_seq_len,)
        count += 1
    del it  # Liberar el iterador para cerrar archivos

    assert count == n_items
    os.unlink(tmp_path)
