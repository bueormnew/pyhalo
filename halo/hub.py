"""
HALO-S HuggingFace Hub Integration.

Provides seamless saving/loading from HuggingFace Hub format:
- Saves config.json (HF format) alongside model weights
- Loads models from HuggingFace Hub repositories
- Compatible with all HALO-S model versions (v1.x, v2.x)

Usage:
    from halo.hub import save_for_hub, load_from_hub, push_to_hub
    
    # Save in HF format
    save_for_hub(model, config, "path/to/model_dir")
    
    # Load from local HF dir or hub
    model = load_from_hub("path/to/model_dir", device="cuda")
    model = load_from_hub("bueormnew/halo-s-70m", device="cuda")
"""

import os
import json
import torch

try:
    from safetensors.torch import save_file, load_file
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download, HfApi
    _HF_HUB_AVAILABLE = True
except ImportError:
    _HF_HUB_AVAILABLE = False

from halo.core.config import HaloConfig
from halo.core.logging import get_logger

logger = get_logger("halo.hub")


def save_for_hub(model, config: HaloConfig, save_dir: str, safe_serialization: bool = True):
    """
    Save model in HuggingFace-compatible format.
    
    Creates:
    - config.json (HaloConfig in HF format)
    - model.safetensors (or pytorch_model.bin)
    
    Args:
        model: HaloSModel instance
        config: HaloConfig instance
        save_dir: Directory to save files
        safe_serialization: Use safetensors (True) or pytorch (False)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config.json
    config_dict = {
        "model_type": "halo-s",
        "architectures": ["HaloSModel"],
        "halo_version": "2.2.0",
        **config.to_dict()
    }
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    logger.info(f"config.json guardado en {config_path}")
    
    # Save weights
    state_dict = model.state_dict()
    if safe_serialization and _ST_AVAILABLE:
        weights_path = os.path.join(save_dir, "model.safetensors")
        save_file(state_dict, weights_path)
    else:
        weights_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(state_dict, weights_path)
    logger.info(f"Pesos guardados en {weights_path}")


def load_from_hub(path_or_repo: str, device: str = "cpu", revision: str = None) -> "HaloSModel":
    """
    Load HALO-S model from a local directory or HuggingFace Hub repo.
    
    Supports:
    - Local directories with config.json + model.safetensors/pytorch_model.bin
    - HuggingFace Hub repositories (e.g., "bueormnew/halo-s-70m")
    - Old format .pt files (backward compatible)
    
    Args:
        path_or_repo: Local path or HF repo ID
        device: Device to load onto
        revision: Git revision for HF Hub
    
    Returns:
        Loaded HaloSModel instance
    """
    from halo.models.halo_model import HaloSModel
    
    # Determine if local or hub
    if os.path.isdir(path_or_repo):
        return _load_from_local_dir(path_or_repo, device)
    elif os.path.isfile(path_or_repo):
        # Single file — use from_pretrained
        return HaloSModel.from_pretrained(path_or_repo, device=device)
    else:
        # Try HuggingFace Hub
        return _load_from_hf_hub(path_or_repo, device, revision)


def _load_from_local_dir(model_dir: str, device: str) -> "HaloSModel":
    """Load from a local directory with config.json + weights."""
    from halo.models.halo_model import HaloSModel
    
    # Load config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json no encontrado en {model_dir}")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    # Remove HF-specific keys before creating HaloConfig
    hf_keys = {"model_type", "architectures", "halo_version", "torch_dtype"}
    filtered = {k: v for k, v in config_dict.items() if k not in hf_keys}
    config = HaloConfig.from_dict(filtered)
    
    # Load weights (prefer safetensors)
    st_path = os.path.join(model_dir, "model.safetensors")
    pt_path = os.path.join(model_dir, "pytorch_model.bin")
    
    model = HaloSModel(config)
    
    if os.path.exists(st_path) and _ST_AVAILABLE:
        state_dict = load_file(st_path, device=device)
        model.load_state_dict(state_dict, strict=False)
    elif os.path.exists(pt_path):
        state_dict = torch.load(pt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    else:
        raise FileNotFoundError(f"No se encontraron pesos en {model_dir}")
    
    model.to(device)
    logger.info(f"Modelo cargado desde {model_dir} ({config.num_layers} capas, {model.count_parameters():,} params)")
    return model


def _load_from_hf_hub(repo_id: str, device: str, revision: str = None) -> "HaloSModel":
    """Load from HuggingFace Hub repository."""
    if not _HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub no instalado. pip install huggingface_hub"
        )
    
    from halo.models.halo_model import HaloSModel
    
    logger.info(f"Descargando modelo desde HuggingFace Hub: {repo_id}")
    
    # Download config
    try:
        config_path = hf_hub_download(repo_id, "config.json", revision=revision)
    except Exception as e:
        raise ValueError(f"No se pudo descargar config.json desde {repo_id}: {e}")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    hf_keys = {"model_type", "architectures", "halo_version", "torch_dtype"}
    filtered = {k: v for k, v in config_dict.items() if k not in hf_keys}
    config = HaloConfig.from_dict(filtered)
    
    # Download weights (try safetensors first, then pytorch)
    model = HaloSModel(config)
    
    try:
        weights_path = hf_hub_download(repo_id, "model.safetensors", revision=revision)
        if _ST_AVAILABLE:
            state_dict = load_file(weights_path, device=device)
        else:
            raise ImportError("safetensors needed")
    except Exception:
        try:
            weights_path = hf_hub_download(repo_id, "pytorch_model.bin", revision=revision)
            state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        except Exception as e:
            raise ValueError(f"No se pudieron descargar pesos desde {repo_id}: {e}")
    
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    logger.info(f"Modelo descargado desde {repo_id} ({model.count_parameters():,} params)")
    return model


def push_to_hub(model, config: HaloConfig, repo_id: str, token: str = None, private: bool = False):
    """
    Push model to HuggingFace Hub.
    
    Args:
        model: HaloSModel instance
        config: HaloConfig instance
        repo_id: Repository ID (e.g., "username/model-name")
        token: HF API token (or set HF_TOKEN env var)
        private: Whether the repo should be private
    """
    if not _HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub no instalado. pip install huggingface_hub")
    
    import tempfile
    
    api = HfApi(token=token)
    
    # Create repo if needed
    try:
        api.create_repo(repo_id, private=private, exist_ok=True)
    except Exception as e:
        logger.warning(f"No se pudo crear repo (puede que ya exista): {e}")
    
    # Save to temp dir and upload
    with tempfile.TemporaryDirectory() as tmpdir:
        save_for_hub(model, config, tmpdir)
        api.upload_folder(folder_path=tmpdir, repo_id=repo_id)
    
    logger.info(f"Modelo subido a https://huggingface.co/{repo_id}")
