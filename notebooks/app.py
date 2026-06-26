"""
HALO-S Playground — Gradio App para HuggingFace Spaces.

Carga modelos HALO-S desde HuggingFace Hub usando safetensors
y permite generación interactiva de texto.

Compatible con:
- HALO-S v1.x (CharacterTokenizer, vocab_size=256)
- HALO-S v2.x (tiktoken GPT-2 BPE, vocab_size=50257)
"""

import torch
import gradio as gr

try:
    import tiktoken
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    _TIKTOKEN_AVAILABLE = False

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None

from huggingface_hub import hf_hub_download

from halo import HaloConfig, HaloSModel, CharacterTokenizer
from halo.core.device import optimize_for_device, get_optimal_device

# ============================================================
# MODELOS DISPONIBLES
# ============================================================

MODELS = {
    "HALO-S Usmall (3.7M)": {
        "repo": "BUEORM/HALO-S-Usmall",
        "checkpoint": "model.safetensors",
        "tokenizer": "char",
    },
    "HALO-S V1 (20M)": {
        "repo": "BUEORM/HALO-S-V1",
        "checkpoint": "model.safetensors",
        "tokenizer": "char",
    },
    "HALO-S V2 (70M)": {
        "repo": "BUEORM/HALO-S-large",
        "checkpoint": "model.safetensors",
        "tokenizer": "gpt2",
    },
}

# ============================================================
# CONFIGURACIONES — deben coincidir EXACTAMENTE con las usadas al entrenar
# ============================================================

CONFIGS = {
    "HALO-S Usmall (3.7M)": HaloConfig(
        vocab_size=256,
        hidden_size=256,
        num_layers=4,
        num_heads=8,
        num_kv_heads=2,
        num_globals=2,
        local_window=32,
        dilated_offsets=[1, 2, 4, 8, 16],
        num_random=2,
        dropout=0.0,
        max_seq_len=256,
        use_swiglu=False,
    ),
    "HALO-S V1 (20M)": HaloConfig(
        vocab_size=256,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        num_globals=2,
        local_window=64,
        dilated_offsets=[1, 2, 4, 8, 16, 32, 64],
        num_random=2,
        dropout=0.0,
        max_seq_len=1024,
        use_swiglu=False,
    ),
    "HALO-S V2 (70M)": HaloConfig(
        vocab_size=50257,
        hidden_size=512,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        num_globals=2,
        local_window=64,
        dilated_offsets=[1, 2, 4, 8, 16, 32, 64, 128],
        num_random=2,
        dropout=0.0,
        max_seq_len=1024,
        use_swiglu=False,
    ),
}

# ============================================================
# CACHE DE MODELOS Y TOKENIZERS
# ============================================================

_model_cache = {}
_tokenizer_cache = {}

# ============================================================
# FUNCIONES DE CARGA
# ============================================================


def get_tokenizer(model_name):
    """Obtiene el tokenizer correcto para cada modelo."""
    if model_name in _tokenizer_cache:
        return _tokenizer_cache[model_name]

    tok_type = MODELS[model_name]["tokenizer"]

    if tok_type == "char":
        tokenizer = CharacterTokenizer()
    elif tok_type == "gpt2":
        if not _TIKTOKEN_AVAILABLE:
            raise ImportError("tiktoken no instalado. pip install tiktoken")
        tokenizer = tiktoken.get_encoding("gpt2")
    else:
        raise ValueError(f"Tokenizer desconocido: {tok_type}")

    _tokenizer_cache[model_name] = tokenizer
    return tokenizer


def load_model(model_name):
    """Carga un modelo desde HuggingFace Hub con safetensors."""
    if model_name in _model_cache:
        return _model_cache[model_name]

    info = MODELS[model_name]
    config = CONFIGS[model_name]

    # Descargar el archivo safetensors desde HF Hub
    checkpoint_path = hf_hub_download(
        repo_id=info["repo"],
        filename=info["checkpoint"],
    )

    # Crear modelo con la configuración correcta
    model = HaloSModel(config)

    # Cargar pesos desde safetensors
    if load_safetensors is not None and checkpoint_path.endswith(".safetensors"):
        state_dict = load_safetensors(checkpoint_path, device="cpu")
    else:
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # strict=False para compatibilidad entre versiones (w3 de SwiGLU puede faltar)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Optimizar para el dispositivo actual (CPU en Spaces, GPU si disponible)
    # NO usar torch.compile — en generación autoregresiva la compilación
    # del grafo tarda más que la inferencia misma para modelos pequeños
    device = get_optimal_device()
    model = model.to(device)

    # Solo aplicar optimizaciones ligeras (TF32, threads) sin compile
    import os
    if "cuda" in device:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        torch.set_num_threads(os.cpu_count() or 4)

    _model_cache[model_name] = model
    return model


# ============================================================
# GENERACIÓN
# ============================================================


def generate_text(prompt, model_name, max_new_tokens, temperature):
    """Genera texto con el modelo seleccionado."""
    if not prompt or not prompt.strip():
        return "⚠️ Por favor ingresa un prompt."

    try:
        model = load_model(model_name)
        tokenizer = get_tokenizer(model_name)
        device = next(model.parameters()).device

        # Tokenizar el prompt
        tok_type = MODELS[model_name]["tokenizer"]
        if tok_type == "char":
            token_ids = tokenizer.encode(prompt)
        else:
            token_ids = tokenizer.encode(prompt)

        if len(token_ids) == 0:
            return "⚠️ El prompt no produjo tokens válidos."

        # Limitar longitud del prompt al max_seq_len del modelo
        config = CONFIGS[model_name]
        max_ctx = config.max_seq_len - config.num_globals - int(max_new_tokens)
        if max_ctx < 1:
            max_ctx = config.max_seq_len // 2
        token_ids = token_ids[-max_ctx:]

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

        # Generar
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                max_new_tokens=int(max_new_tokens),
                temperature=float(temperature),
                top_k=50,
                top_p=0.9,
            )

        # Decodificar
        output_ids = output[0].tolist()
        if tok_type == "char":
            text = tokenizer.decode(output_ids)
        else:
            text = tokenizer.decode(output_ids)

        return text

    except Exception as e:
        return f"❌ Error durante la generación:\n\n{type(e).__name__}: {str(e)}"


# ============================================================
# INTERFAZ GRADIO
# ============================================================

DESCRIPTION = """
# 🌀 HALO-S Playground

**Generación de texto con modelos HALO-S** — Atención dispersa O(N×K)

### Modelos disponibles:
| Modelo | Parámetros | Tokenizer | Entrenado en |
|--------|-----------|-----------|--------------|
| HALO-S Usmall | 3.7M | Character (256) | WikiText-2 |
| HALO-S V1 | 20M | Character (256) | WikiText-103 |
| HALO-S V2 | 70M | GPT-2 BPE (50257) | WikiText-103 |

### Notas:
- Los modelos son experimentales y fueron entrenados con datos limitados
- La calidad de generación depende del modelo y los hiperparámetros
- Para V2 usa prompts en inglés (entrenado con BPE en texto inglés)
- Compatible con HALO-S v1.x y v2.x

[📦 PyPI](https://pypi.org/project/pyhalos/) | [💻 GitHub](https://github.com/bueormnew/pyhalo) | [📖 Docs](https://github.com/bueormnew/pyhalo/tree/main/docs)
"""

demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(
            lines=5,
            label="📝 Prompt",
            placeholder="The history of artificial intelligence...",
            value="The history of",
        ),
        gr.Dropdown(
            choices=list(MODELS.keys()),
            value="HALO-S Usmall (3.7M)",
            label="🧠 Modelo",
        ),
        gr.Slider(
            minimum=10,
            maximum=512,
            value=100,
            step=10,
            label="📏 Max New Tokens",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=2.0,
            value=0.8,
            step=0.1,
            label="🌡️ Temperature",
        ),
    ],
    outputs=gr.Textbox(
        lines=15,
        label="✨ Texto Generado",
    ),
    title="🌀 HALO-S Playground",
    description=DESCRIPTION,
    examples=[
        ["The history of", "HALO-S Usmall (3.7M)", 100, 0.8],
        ["Machine learning is", "HALO-S V1 (20M)", 150, 0.7],
        ["The United States of America", "HALO-S V2 (70M)", 200, 0.8],
    ],
    cache_examples=False,
)

# ============================================================
# LANZAR
# ============================================================

if __name__ == "__main__":
    demo.launch()
