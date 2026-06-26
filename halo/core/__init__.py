"""
HALO-S Core — Configuración y utilidades fundamentales del framework.
"""

from halo.core.config import HaloConfig
from halo.core.device import optimize_for_device, get_optimal_device, device_info

__all__ = [
    "HaloConfig",
    "optimize_for_device",
    "get_optimal_device",
    "device_info",
]
