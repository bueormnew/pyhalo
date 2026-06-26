"""
HALO-S Device Optimization — Automatic hardware detection and optimization.

Detects if running on CPU or CUDA and applies appropriate optimizations:
- CPU: torch.set_num_threads, inference mode hints
- CUDA: TF32 math, memory-efficient attention preference, autocast hints

This module NEVER breaks functionality — all optimizations have fallback paths.
"""

import os
import torch
from halo.core.logging import get_logger

logger = get_logger("halo.device")


def optimize_for_device(model, device=None, mode="auto"):
    """
    Apply device-specific optimizations to a model.

    Args:
        model: nn.Module to optimize
        device: "cpu", "cuda", or None (auto-detect from model parameters)
        mode: "auto", "inference", "training"

    Returns:
        The model (possibly compiled or optimized)

    This function NEVER raises — on failure it logs a warning and returns the model unchanged.
    """
    try:
        if device is None:
            device = str(next(model.parameters()).device)

        if "cuda" in device:
            return _optimize_cuda(model, mode)
        else:
            return _optimize_cpu(model, mode)
    except Exception as e:
        logger.warning(f"Optimización de dispositivo falló (continuando sin optimizar): {e}")
        return model


def _optimize_cuda(model, mode):
    """Optimizaciones para GPU NVIDIA."""
    # Habilitar TF32 para matmul (Ampere+)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Preferir backend memory-efficient para SDPA
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)

    if mode == "inference":
        model.eval()
        # Try torch.compile for inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("torch.compile aplicado (reduce-overhead)")
        except Exception as e:
            logger.warning(f"torch.compile no disponible: {e}")

    logger.info("Optimización CUDA aplicada (TF32 + Flash SDP)")
    return model


def _optimize_cpu(model, mode):
    """Optimizaciones para CPU."""
    # Usar todos los threads disponibles
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)

    if mode == "inference":
        model.eval()
        # Try torch.compile for CPU
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="default")
                logger.info("torch.compile aplicado (CPU default)")
        except Exception as e:
            logger.warning(f"torch.compile no disponible en CPU: {e}")

    logger.info(f"Optimización CPU aplicada (threads={num_threads})")
    return model


def get_optimal_device():
    """Retorna el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info():
    """Retorna información del dispositivo como dict."""
    info = {
        "device": get_optimal_device(),
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_threads": os.cpu_count(),
    }
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    return info
