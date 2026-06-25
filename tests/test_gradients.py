"""
Tests de flujo de gradientes para HaloSModel y BaselineModel.

Verifica que todos los parámetros entrenables reciban gradientes válidos
después de un forward + backward pass.

**Validates: Requirement 10.2**
"""

import torch
import pytest
from halo.core.config import HaloConfig
from halo.models.halo_model import HaloSModel
from halo.models.baseline_model import BaselineModel


# --- Configuraciones pequeñas para testing rápido ---

def _small_config(num_layers: int = 1) -> HaloConfig:
    """Config mínima para tests de gradientes."""
    return HaloConfig(
        vocab_size=64,
        hidden_size=64,
        num_layers=num_layers,
        num_heads=2,
        num_kv_heads=1,
        num_globals=2,
        local_window=8,
        dilated_offsets=[1, 2],
        num_random=1,
        dropout=0.0,
        max_seq_len=128,
    )


def _run_forward_backward(model, config, seq_len: int = 16):
    """Ejecuta forward + backward con datos aleatorios y retorna la pérdida."""
    batch_size = 2
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits, loss = model(input_ids, targets=targets)
    loss.backward()
    return loss


# --- Property 8: Gradient Flow ---


class TestGradientFlow:
    """
    Property 8: Gradient Flow — todos los parámetros entrenables tienen
    gradientes no None y magnitud > 0 después de backward().

    **Validates: Requirement 10.2**
    """

    def test_all_parameters_receive_gradients(self):
        """Todos los parámetros con requires_grad=True deben tener grad != None y grad.abs().sum() > 0.

        Nota: La última capa tiene global_attn con gradiente 0 por diseño,
        ya que los globals del último layer se descartan antes de calcular logits.
        Solo las capas intermedias (0..N-2) propagan gradientes a global_attn
        porque sus globals alimentan la siguiente capa.
        """
        config = _small_config(num_layers=2)
        model = HaloSModel(config)
        model.train()

        _run_forward_backward(model, config)

        last_layer_idx = config.num_layers - 1

        for name, param in model.named_parameters():
            if param.requires_grad:
                # global_attn de la última capa no recibe gradientes
                # porque los globals se descartan al calcular logits.
                is_last_layer_global_attn = (
                    f"layers.{last_layer_idx}.global_attn" in name
                )
                if is_last_layer_global_attn:
                    continue

                assert param.grad is not None, (
                    f"Parámetro '{name}' no recibió gradientes (grad es None)"
                )
                grad_magnitude = param.grad.abs().sum().item()
                assert grad_magnitude > 0, (
                    f"Parámetro '{name}' tiene gradiente con magnitud 0"
                )

    def test_global_memory_receives_gradients(self):
        """self.global_memory debe recibir gradientes (prueba que globals están conectados al loss)."""
        config = _small_config(num_layers=2)
        model = HaloSModel(config)
        model.train()

        _run_forward_backward(model, config)

        assert model.global_memory.grad is not None, (
            "global_memory no recibió gradientes — los globals no están conectados al loss"
        )
        grad_magnitude = model.global_memory.grad.abs().sum().item()
        assert grad_magnitude > 0, (
            "global_memory tiene gradiente con magnitud 0 — no fluye información del loss"
        )

    def test_gradient_magnitude_not_exploding(self):
        """Ningún parámetro debe tener gradientes con inf o nan después de backward."""
        config = _small_config(num_layers=2)
        model = HaloSModel(config)
        model.train()

        _run_forward_backward(model, config)

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.any(torch.isinf(param.grad)), (
                    f"Parámetro '{name}' tiene gradientes con valores infinitos"
                )
                assert not torch.any(torch.isnan(param.grad)), (
                    f"Parámetro '{name}' tiene gradientes con valores NaN"
                )

    def test_gradient_flow_through_multiple_layers(self):
        """Con num_layers=4, los gradientes deben propagarse a través de todas las capas.

        Las capas intermedias (0..N-2) tienen gradientes en todos los parámetros,
        incluyendo global_attn. La última capa excluye global_attn completo
        porque los globals finales no contribuyen al loss.
        """
        config = _small_config(num_layers=4)
        model = HaloSModel(config)
        model.train()

        _run_forward_backward(model, config)

        last_layer_idx = config.num_layers - 1

        # Verificar que todas las capas reciben gradientes
        for layer_idx in range(config.num_layers):
            layer = model.layers[layer_idx]
            for name, param in layer.named_parameters():
                if param.requires_grad:
                    # Excluir global_attn de la última capa (gradiente 0 por diseño)
                    is_last_layer_global_attn = (
                        layer_idx == last_layer_idx and "global_attn" in name
                    )
                    if is_last_layer_global_attn:
                        continue

                    assert param.grad is not None, (
                        f"Capa {layer_idx}, parámetro '{name}' no recibió gradientes"
                    )
                    grad_magnitude = param.grad.abs().sum().item()
                    assert grad_magnitude > 0, (
                        f"Capa {layer_idx}, parámetro '{name}' tiene gradiente con magnitud 0"
                    )

    def test_baseline_model_gradient_flow(self):
        """BaselineModel también debe tener flujo de gradientes correcto."""
        config = _small_config(num_layers=2)
        model = BaselineModel(config)
        model.train()

        _run_forward_backward(model, config)

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"BaselineModel: parámetro '{name}' no recibió gradientes"
                )
                grad_magnitude = param.grad.abs().sum().item()
                assert grad_magnitude > 0, (
                    f"BaselineModel: parámetro '{name}' tiene gradiente con magnitud 0"
                )

        # Verificar que no hay inf/nan en gradientes
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                assert not torch.any(torch.isinf(param.grad)), (
                    f"BaselineModel: '{name}' tiene gradientes infinitos"
                )
                assert not torch.any(torch.isnan(param.grad)), (
                    f"BaselineModel: '{name}' tiene gradientes NaN"
                )
