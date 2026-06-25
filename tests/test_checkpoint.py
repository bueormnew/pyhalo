"""
Tests para checkpoint save/load del Trainer.
Validates: Requirements 3.1, 3.2, 3.4
"""

import os
import tempfile

import torch
import pytest
from torch.utils.data import Dataset

from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.training.trainer import Trainer


class _MiniDataset(Dataset):
    """Dataset mínimo para tests de checkpoint."""
    def __init__(self, vocab_size, seq_len, length=8):
        self.data = [
            (torch.randint(0, vocab_size, (seq_len,)),
             torch.randint(0, vocab_size, (seq_len,)))
            for _ in range(length)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _make_small_config():
    """Configuración mínima para tests rápidos en CPU."""
    return HaloConfig(
        vocab_size=32,
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        num_kv_heads=1,
        num_globals=2,
        local_window=8,
        dilated_offsets=[1, 2],
        num_random=1,
        max_seq_len=32,
    )


def test_checkpoint_save_load_roundtrip():
    """Entrenar 1 paso, guardar, cargar en nuevo trainer; verificar epoch y global_step."""
    config = _make_small_config()
    model = HaloSModel(config)
    dataset = _MiniDataset(vocab_size=config.vocab_size, seq_len=10, length=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Entrenar 1 época y guardar
        trainer = Trainer(model, learning_rate=1e-3, device="cpu", checkpoint_dir=tmpdir)
        trainer.fit(dataset, epochs=1, batch_size=2)
        ckpt_path = trainer.save_checkpoint()

        # Crear nuevo modelo y trainer, luego cargar
        model2 = HaloSModel(config)
        trainer2 = Trainer(model2, learning_rate=1e-3, device="cpu")
        meta = trainer2.load_checkpoint(ckpt_path)

        # Verificar que el estado se restauró correctamente
        assert trainer2.current_epoch == trainer.current_epoch
        assert trainer2.global_step == trainer.global_step


def test_checkpoint_contains_all_keys():
    """Verificar que el archivo guardado tiene todas las claves necesarias."""
    config = _make_small_config()
    model = HaloSModel(config)
    dataset = _MiniDataset(vocab_size=config.vocab_size, seq_len=10, length=4)

    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(model, learning_rate=1e-3, device="cpu", checkpoint_dir=tmpdir)
        trainer.fit(dataset, epochs=1, batch_size=2)
        ckpt_path = trainer.save_checkpoint()

        # Cargar el checkpoint como dict raw
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Claves requeridas
        required_keys = {
            "model_state_dict",
            "optimizer_state_dict",
            "epoch",
            "global_step",
            "halo_version",
            "config",
        }
        for key in required_keys:
            assert key in checkpoint, f"Falta la clave '{key}' en el checkpoint"
