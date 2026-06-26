"""
HALO-S: Framework de Atención Dispersa para Modelos de Lenguaje
================================================================

Arquitectura transformer con complejidad O(N×K) basada en neighbor lists,
global tokens, conexiones dilatadas y Grouped Query Attention (GQA).

Uso básico:
    >>> from halo import HaloConfig, HaloSModel, set_seed
    >>> set_seed(42)
    >>> config = HaloConfig(vocab_size=256, hidden_size=512)
    >>> model = HaloSModel(config)

Autor: BUEORM (dalusx64@gmail.com)
"""

__version__ = "2.2.1"

# --- Core ---
from halo.core.config import HaloConfig
from halo.core.device import (
    optimize_for_device,
    get_optimal_device,
    device_info,
    detect_device_profile,
    get_optimal_batch_size,
)

# --- Hub ---
from halo.hub import save_for_hub, load_from_hub, push_to_hub

# --- Models ---
from halo.models.halo_model import HaloSModel
from halo.models.baseline_model import BaselineModel

# --- Training ---
from halo.training.trainer import Trainer

# --- Tokenizers ---
from halo.tokenizers.char import CharacterTokenizer

# WordTokenizer se importa condicionalmente porque se crea en una tarea posterior
try:
    from halo.tokenizers.word import WordTokenizer
except ImportError:
    WordTokenizer = None

# --- Generation ---
from halo.generation.samplers import generate

# --- Utils ---
from halo.utils.random import set_seed
from halo.utils.metrics import count_parameters

__all__ = [
    # Core
    "HaloConfig",
    # Device
    "optimize_for_device",
    "get_optimal_device",
    "device_info",
    "detect_device_profile",
    "get_optimal_batch_size",
    # Hub
    "save_for_hub",
    "load_from_hub",
    "push_to_hub",
    # Models
    "HaloSModel",
    "BaselineModel",
    # Training
    "Trainer",
    # Tokenizers
    "CharacterTokenizer",
    "WordTokenizer",
    # Generation
    "generate",
    # Utils
    "set_seed",
    "count_parameters",
]
