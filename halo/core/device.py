"""
HALO-S Device Optimization v2.2 — Hardware profiles and automatic optimization.

Supports: T4, P100, L4, L40, RTX 6000, A100, CPU, Multi-GPU, TPU (best-effort).
All optimizations have fallback paths — NEVER breaks functionality.
"""

import os
import torch
import torch.nn as nn
from typing import Optional
from halo.core.logging import get_logger

logger = get_logger("halo.device")

# --- Device Profiles ---

DEVICE_PROFILES = {
    "t4": {
        "name": "NVIDIA Tesla T4",
        "memory_gb": 16,
        "compute_capability": (7, 5),
        "supports_tf32": False,
        "supports_flash": True,
        "supports_bf16": False,
        "optimal_batch_seq": {256: 8, 512: 4, 1024: 2, 2048: 1},
        "compile_mode": "reduce-overhead",
    },
    "p100": {
        "name": "NVIDIA Tesla P100",
        "memory_gb": 16,
        "compute_capability": (6, 0),
        "supports_tf32": False,
        "supports_flash": False,
        "supports_bf16": False,
        "optimal_batch_seq": {256: 8, 512: 4, 1024: 2, 2048: 1},
        "compile_mode": "default",
    },
    "l4": {
        "name": "NVIDIA L4",
        "memory_gb": 24,
        "compute_capability": (8, 9),
        "supports_tf32": True,
        "supports_flash": True,
        "supports_bf16": True,
        "optimal_batch_seq": {256: 16, 512: 8, 1024: 4, 2048: 2},
        "compile_mode": "reduce-overhead",
    },
    "l40": {
        "name": "NVIDIA L40",
        "memory_gb": 48,
        "compute_capability": (8, 9),
        "supports_tf32": True,
        "supports_flash": True,
        "supports_bf16": True,
        "optimal_batch_seq": {256: 32, 512: 16, 1024: 8, 2048: 4},
        "compile_mode": "max-autotune",
    },
    "rtx_6000": {
        "name": "NVIDIA RTX 6000 Ada",
        "memory_gb": 48,
        "compute_capability": (8, 9),
        "supports_tf32": True,
        "supports_flash": True,
        "supports_bf16": True,
        "optimal_batch_seq": {256: 32, 512: 16, 1024: 8, 2048: 4},
        "compile_mode": "max-autotune",
    },
    "a100": {
        "name": "NVIDIA A100",
        "memory_gb": 80,
        "compute_capability": (8, 0),
        "supports_tf32": True,
        "supports_flash": True,
        "supports_bf16": True,
        "optimal_batch_seq": {256: 64, 512: 32, 1024: 16, 2048: 8},
        "compile_mode": "max-autotune",
    },
    "cpu": {
        "name": "CPU",
        "memory_gb": 0,
        "compute_capability": (0, 0),
        "supports_tf32": False,
        "supports_flash": False,
        "supports_bf16": True,  # Most modern CPUs support bf16
        "optimal_batch_seq": {256: 4, 512: 2, 1024: 1, 2048: 1},
        "compile_mode": "default",
    },
}


def detect_device_profile() -> dict:
    """
    Auto-detect the current device and return its optimization profile.
    Matches GPU name against known profiles.
    """
    if not torch.cuda.is_available():
        return DEVICE_PROFILES["cpu"]
    
    gpu_name = torch.cuda.get_device_name(0).lower()
    props = torch.cuda.get_device_properties(0)
    cc = (props.major, props.minor)
    mem_gb = props.total_memory / 1e9
    
    # Match against known profiles
    if "t4" in gpu_name:
        return DEVICE_PROFILES["t4"]
    elif "p100" in gpu_name:
        return DEVICE_PROFILES["p100"]
    elif "l4" in gpu_name and "l40" not in gpu_name:
        return DEVICE_PROFILES["l4"]
    elif "l40" in gpu_name:
        return DEVICE_PROFILES["l40"]
    elif "rtx 6000" in gpu_name or "rtx6000" in gpu_name:
        return DEVICE_PROFILES["rtx_6000"]
    elif "a100" in gpu_name:
        return DEVICE_PROFILES["a100"]
    else:
        # Generic profile based on capabilities
        return {
            "name": torch.cuda.get_device_name(0),
            "memory_gb": mem_gb,
            "compute_capability": cc,
            "supports_tf32": cc >= (8, 0),
            "supports_flash": cc >= (7, 5),
            "supports_bf16": cc >= (8, 0),
            "optimal_batch_seq": {256: max(1, int(mem_gb/2)), 512: max(1, int(mem_gb/4)), 1024: max(1, int(mem_gb/8))},
            "compile_mode": "reduce-overhead" if cc >= (7, 0) else "default",
        }


def optimize_for_device(model: nn.Module, device: Optional[str] = None, mode: str = "auto") -> nn.Module:
    """
    Apply device-specific optimizations to a model.
    
    Args:
        model: nn.Module to optimize
        device: "cpu", "cuda", or None (auto-detect)
        mode: "auto", "inference", "training"
    
    Returns:
        The model (possibly with optimizations applied)
    
    This function NEVER raises — on failure it logs a warning and returns the model unchanged.
    """
    try:
        if device is None:
            try:
                device = str(next(model.parameters()).device)
            except StopIteration:
                device = "cpu"
        
        profile = detect_device_profile()
        logger.info(f"Dispositivo detectado: {profile['name']}")
        
        if "cuda" in device:
            return _optimize_cuda(model, mode, profile)
        else:
            return _optimize_cpu(model, mode, profile)
    except Exception as e:
        logger.warning(f"Optimización falló (continuando sin optimizar): {e}")
        return model


def _optimize_cuda(model: nn.Module, mode: str, profile: dict) -> nn.Module:
    """Apply CUDA-specific optimizations based on device profile."""
    # TF32 (Ampere+)
    if profile.get("supports_tf32", False):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 habilitado")
    
    # Flash/Memory-efficient SDP
    if profile.get("supports_flash", False):
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            logger.info("Flash SDP habilitado")
        except Exception:
            pass
    
    # torch.compile for inference
    if mode == "inference":
        model.eval()
        try:
            compile_mode = profile.get("compile_mode", "default")
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode=compile_mode)
                logger.info(f"torch.compile aplicado (mode={compile_mode})")
        except Exception as e:
            logger.warning(f"torch.compile no disponible: {e}")
    
    return model


def _optimize_cpu(model: nn.Module, mode: str, profile: dict) -> nn.Module:
    """Apply CPU-specific optimizations."""
    num_threads = os.cpu_count() or 4
    torch.set_num_threads(num_threads)
    logger.info(f"CPU threads: {num_threads}")
    
    if mode == "inference":
        model.eval()
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model, mode="default")
                logger.info("torch.compile aplicado (CPU)")
        except Exception as e:
            logger.warning(f"torch.compile no disponible: {e}")
    
    return model


def get_optimal_device() -> str:
    """Return the best available device string."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_info() -> dict:
    """Return comprehensive device information."""
    info = {
        "device": get_optimal_device(),
        "cuda_available": torch.cuda.is_available(),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_threads": os.cpu_count(),
        "profile": detect_device_profile(),
    }
    if info["cuda_available"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    return info


def get_optimal_batch_size(config, seq_len: int = 1024) -> int:
    """
    Get the recommended batch size for the current device and sequence length.
    
    Args:
        config: HaloConfig instance
        seq_len: Sequence length to use
    
    Returns:
        Recommended batch size
    """
    profile = detect_device_profile()
    batch_map = profile.get("optimal_batch_seq", {})
    
    # Find closest seq_len in map
    closest = min(batch_map.keys(), key=lambda k: abs(k - seq_len), default=None)
    if closest is None:
        return 1
    
    return batch_map[closest]
