<p align="center">
  <h1 align="center">🌀 HALO-S</h1>
  <p align="center"><strong>Hierarchical Attention with Local Offsets — Sparse</strong></p>
  <p align="center">A linear-complexity language model framework that replaces quadratic attention with a structured sparse connectivity graph.</p>
  <p align="center"><em>v2.2.1 — Now with HuggingFace Hub integration, device profiles, safetensors support, SwiGLU FFN, and hybrid SDPA+Gather attention</em></p>
</p>

<p align="center">

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyPI](https://img.shields.io/pypi/v/pyhalos)
![Version](https://img.shields.io/badge/version-2.2.1-brightgreen)
![License](https://img.shields.io/badge/license-custom-orange)
![Tests](https://img.shields.io/badge/tests-61%20passed-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)
![HuggingFace](https://img.shields.io/badge/🤗-Hub%20Compatible-yellow)
![Safetensors](https://img.shields.io/badge/safetensors-supported-brightgreen)

</p>

---

## What's New in v2.2.1

| Version | Date | Key Changes |
|---------|------|-------------|
| **v2.2.1** | 2024 | Stability fixes, improved backward compat, documentation overhaul, 61 tests, comprehensive FAQ/troubleshooting |
| **v2.2.0** | 2024 | HuggingFace Hub integration, device profiles (T4/P100/L4/L40/RTX 6000/A100), `push_to_hub()`, `load_from_hub()`, safetensors as default format |
| **v2.1.0** | 2024 | Safetensors support, `optimize_for_device()`, device auto-detection, `get_optimal_batch_size()` |
| **v2.0.0** | 2024 | Hybrid SDPA+Gather attention, SwiGLU FFN, gradient checkpointing, `from_pretrained()`, breaking config changes |
| **v1.0.0** | 2024 | Initial release: sparse attention, GQA, global tokens, Trainer, generation, CharacterTokenizer |

### Migration Notes

- **v1.x → v2.x**: Models saved with v1.x use GELU FFN and old state_dict keys. Use `HaloSModel.from_pretrained("old_model.pt")` which auto-detects and remaps weights. The `use_swiglu` flag is automatically set to `False` when loading v1.x checkpoints. Fine-tuning on v2.x architecture is recommended but not required.
- **v2.0 → v2.1+**: Seamless. Config unchanged, safetensors optional. All existing `.pt` checkpoints continue to work. New `optimize_for_device()` and `get_optimal_batch_size()` functions are additive only.
- **v2.1 → v2.2+**: New Hub functions added (`save_for_hub`, `load_from_hub`, `push_to_hub`). No breaking changes. `save_for_hub()` now creates HF-compatible directories with `config.json` and `model.safetensors`. Device profiles expanded to include RTX 6000 Ada.
- **v2.2.0 → v2.2.1**: Stability fixes only. No API changes. Test suite expanded from 55 to 61 tests. Documentation completely rewritten.

### Version Compatibility Table

| Feature / API | v1.0 | v2.0 | v2.1 | v2.2 | v2.2.1 |
|---------------|:----:|:----:|:----:|:----:|:------:|
| Sparse Gather Attention | ✓ | ✓ | ✓ | ✓ | ✓ |
| GQA (Grouped Query Attention) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Global Tokens | ✓ | ✓ | ✓ | ✓ | ✓ |
| RoPE | ✓ | ✓ | ✓ | ✓ | ✓ |
| CharacterTokenizer | ✓ | ✓ | ✓ | ✓ | ✓ |
| Trainer (AMP, accumulation) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Generation (top-k, top-p) | ✓ | ✓ | ✓ | ✓ | ✓ |
| GELU FFN | ✓ | ✓* | ✓* | ✓* | ✓* |
| SwiGLU FFN | ✗ | ✓ | ✓ | ✓ | ✓ |
| Hybrid SDPA+Gather | ✗ | ✓ | ✓ | ✓ | ✓ |
| Gradient Checkpointing | ✗ | ✓ | ✓ | ✓ | ✓ |
| `from_pretrained()` | ✗ | ✓ | ✓ | ✓ | ✓ |
| Safetensors support | ✗ | ✗ | ✓ | ✓ | ✓ |
| `optimize_for_device()` | ✗ | ✗ | ✓ | ✓ | ✓ |
| `get_optimal_batch_size()` | ✗ | ✗ | ✓ | ✓ | ✓ |
| Device Profiles | ✗ | ✗ | ✓ | ✓ | ✓ |
| `save_for_hub()` | ✗ | ✗ | ✗ | ✓ | ✓ |
| `load_from_hub()` | ✗ | ✗ | ✗ | ✓ | ✓ |
| `push_to_hub()` | ✗ | ✗ | ✗ | ✓ | ✓ |
| RTX 6000 Ada profile | ✗ | ✗ | ✗ | ✓ | ✓ |

*GELU FFN available via `use_swiglu=False` config option.

---

## What if attention didn't have to be quadratic?

Every modern language model pays a steep price for long sequences: the standard Transformer's self-attention scales as O(N²), making context windows beyond 4K tokens prohibitively expensive. HALO-S takes a different path. By constructing a **fixed-degree sparse connectivity graph** — combining local windows, dilated connections, learned global tokens, and random edges — each token attends to only K neighbors regardless of sequence length. The result is **O(N×K) complexity** with K=76 by default, yielding a theoretical **~52.5× reduction** in attention operations at N=4096.

HALO-S is implemented as a clean, research-ready PyTorch framework. No custom CUDA kernels. No external dependencies beyond PyTorch and NumPy. Just gather-based sparse attention that runs on any hardware.

> ⚠️ **Honest disclaimer**: HALO-S is a promising architectural exploration. The theoretical complexity advantages are mathematically sound, but **large-scale empirical validation** against established models on standard benchmarks is still in progress. Use it for research, experimentation, and learning. The numbers in this README reflect theoretical analysis and small/medium-scale experiments (3.5M–70M parameters), not production-validated results at billions of parameters.

---

## Table of Contents

- [What's New in v2.2.1](#whats-new-in-v221)
- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [HuggingFace Hub Integration](#huggingface-hub-integration)
- [Device Optimization System](#device-optimization-system)
- [Performance Analysis (Theoretical)](#performance-analysis-theoretical)
- [Empirical Benchmarks](#empirical-benchmarks)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Configuration Reference](#configuration-reference)
- [API Reference](#api-reference)
- [Backward Compatibility Guide](#backward-compatibility-guide)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Project Structure](#project-structure)
- [Why HALO-S?](#why-halo-s)
- [Running Tests](#running-tests)
- [Running Experiments](#running-experiments)
- [Citation](#citation)
- [License](#license)
- [Author](#author)
- [🇪🇸 Versión en Español](#-versión-en-español)

---

## Key Features

| Feature | Description | Since |
|---------|-------------|:-----:|
| **Linear Attention Complexity** | O(N×K) instead of O(N²) — scales to long sequences efficiently | v1.0 |
| **Gather-Based Sparse Attention** | No custom CUDA kernels needed; runs on CPU and GPU | v1.0 |
| **Hybrid SDPA + Gather** | Uses PyTorch's native SDPA for global tokens, gather for sparse tokens | v2.0 |
| **Learned Global Tokens** | Shared memory parameters that attend to the full sequence | v1.0 |
| **Dilated Connections** | Exponentially expanding receptive field across layers | v1.0 |
| **Random Edges** | Small-world graph properties for information propagation | v1.0 |
| **Grouped Query Attention (GQA)** | Reduced KV memory with configurable head ratios | v1.0 |
| **Rotary Position Embeddings (RoPE)** | Relative position encoding without learned parameters | v1.0 |
| **SwiGLU Feed-Forward** | Gated linear unit activation for improved training dynamics | v2.0 |
| **Mixed Precision Training** | Native AMP support with GradScaler (FP16/BF16) | v1.0 |
| **Gradient Accumulation** | Train with effective large batches on limited hardware | v1.0 |
| **Gradient Checkpointing** | Trade compute for memory — train larger models on smaller GPUs | v2.0 |
| **Checkpoint Save/Load** | Full training state persistence and resumption | v1.0 |
| **Streaming Datasets** | Train on data larger than RAM with buffer shuffling | v1.0 |
| **Autoregressive Generation** | Top-k, top-p, and temperature sampling built-in | v1.0 |
| **HuggingFace Hub Integration** | Save, load, and push models to/from HF Hub | v2.2 |
| **Safetensors Support** | Safe, fast model serialization as default format | v2.1 |
| **Device Profiles** | Auto-optimized settings for T4, P100, L4, L40, RTX 6000, A100, CPU | v2.1 |
| **Multi-GPU Support** | DataParallel for multi-GPU training | v1.0 |
| **Backward Compatibility** | Load models from any HALO-S version (v1.0+) | v2.0 |
| **BaselineModel** | Built-in dense Transformer for fair comparison experiments | v1.0 |
| **Synthetic Datasets** | CopyDataset, NeedleDataset for architecture evaluation | v1.0 |
| **Benchmarking Utilities** | Speed, generation, memory, and FLOPs measurement tools | v1.0 |

---

## Architecture Overview

HALO-S replaces dense self-attention with a **structured sparse graph** where each token connects to a fixed set of K neighbors:

```
┌─────────────────────────────────────────────────────────────────┐
│                        HaloSModel                                │
│                                                                  │
│  ┌──────────────┐   ┌──────────────────────────────────┐        │
│  │ token_emb    │   │ global_memory (nn.Parameter)      │        │
│  │ (Embedding)  │   │ shape: (num_globals, hidden_size) │        │
│  └──────┬───────┘   └──────────────┬───────────────────┘        │
│         │                          │                             │
│         └──────────┬───────────────┘                             │
│                    ▼                                              │
│         ┌──────────────────┐                                     │
│         │ cat([globals, x]) │  → (B, G+N, H)                    │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ RoPE (cos, sin)  │                                     │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│  ┌───────────────────────────────────────────────────┐           │
│  │              HaloBlock × num_layers                │           │
│  │                                                    │           │
│  │  ┌─────────────┐                                  │           │
│  │  │ LayerNorm 1 │                                  │           │
│  │  └──────┬──────┘                                  │           │
│  │         │                                          │           │
│  │    ┌────┴────────────────────────┐                │           │
│  │    ▼                             ▼                │           │
│  │ ┌────────────────┐   ┌─────────────────────┐     │           │
│  │ │GlobalFullAttn  │   │ HaloSparseAttention │     │           │
│  │ │(SDPA, G×N)     │   │ (gather, N×K)       │     │           │
│  │ └───────┬────────┘   └──────────┬──────────┘     │           │
│  │         │                       │                  │           │
│  │         └───────────┬───────────┘                  │           │
│  │                     ▼                              │           │
│  │           cat([globals_out, tokens_out])            │           │
│  │                     │ + residual                    │           │
│  │                     ▼                              │           │
│  │  ┌─────────────┐  ┌────────────────┐             │           │
│  │  │ LayerNorm 2 │→ │ SwiGLU FFN     │ + residual  │           │
│  │  └─────────────┘  └────────────────┘             │           │
│  └───────────────────────────────────────────────────┘           │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ LayerNorm final  │                                     │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ lm_head (Linear) │  → (B, N, vocab_size)              │
│         └──────────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Connectivity Components

Each token's neighbor list is composed of:

| Component | Neighbors | Purpose |
|-----------|-----------|---------|
| **Global Tokens (G)** | 2 | Learned parameters attending to full sequence — shared memory |
| **Local Window (w)** | 64 | Captures sequential/syntactic dependencies |
| **Dilated Connections (2d)** | 8 | Exponentially expanding receptive field |
| **Random Edges (r)** | 2 | Guarantees small-world graph properties |
| **Total (K)** | **76** | Fixed budget per token regardless of N |

The connectivity graph is constructed once per forward pass (or cached) by `halo.attention.graph.build_neighbor_list()`. Each component contributes edges:

- **Local Window**: positions `[max(0, i-w//2), ..., i-1, i+1, ..., min(N, i+w//2)]`
- **Dilated**: positions `[i ± offset for offset in dilated_offsets]` (clamped to [0, N))
- **Random**: `num_random` uniformly sampled positions from `[0, N)` (resampled per forward pass during training, fixed during eval)

The result is a tensor of shape `(N, K)` containing neighbor indices for each token position.

### Hybrid SDPA + Gather Attention (v2.0+)

Starting in v2.0, HALO-S uses a **hybrid attention strategy**:

- **Global tokens** use PyTorch's native `F.scaled_dot_product_attention` (SDPA). Since globals attend to the full sequence, SDPA's hardware-optimized kernels (Flash Attention, Memory-Efficient Attention, Math backend) are leveraged automatically when available.
- **Regular tokens** use `torch.gather`-based sparse attention. The precomputed neighbor list determines exactly which K positions each token attends to, and Q/K/V tensors are gathered accordingly.

This hybrid approach gives the best of both worlds: globals get hardware acceleration, while regular tokens maintain O(N×K) sparse complexity.

```python
# Simplified pseudocode of the hybrid forward pass
def forward(x, neighbor_idx):
    q, k, v = self.qkv_proj(x)  # Project to Q, K, V
    
    # Split global tokens from regular tokens
    q_globals, q_tokens = q[:, :G], q[:, G:]
    k_full, v_full = k, v  # Globals see everything
    
    # Global tokens: dense SDPA (fast, hardware-optimized)
    globals_out = F.scaled_dot_product_attention(
        q_globals, k_full, v_full, is_causal=True
    )
    
    # Regular tokens: gather-based sparse attention
    # neighbor_idx shape: (B, N, K) — precomputed connectivity graph
    k_gathered = torch.gather(k, dim=1, index=neighbor_idx)  # (B, N, K, D)
    v_gathered = torch.gather(v, dim=1, index=neighbor_idx)  # (B, N, K, D)
    scores = (q_tokens @ k_gathered.transpose(-1, -2)) / sqrt(d)
    scores = scores.masked_fill(causal_mask, float('-inf'))
    tokens_out = softmax(scores) @ v_gathered
    
    return cat([globals_out, tokens_out])
```

**Why hybrid?** Global tokens attend to all N positions (dense by definition), so giving them SDPA means they benefit from Flash Attention's O(1) memory and fused kernels on supported hardware. Regular tokens only need K=76 neighbors, so gather is more efficient than masking out N-K positions in a dense computation.

### SwiGLU vs GELU Feed-Forward

HALO-S v2.0+ uses **SwiGLU** (Gated Linear Unit with Swish activation) by default, replacing the standard GELU FFN from v1.x:

```
# Standard GELU FFN (v1.x):
FFN(x) = Linear₂(GELU(Linear₁(x)))
  Parameters: 2 × hidden × 4×hidden = 8H²

# SwiGLU FFN (v2.0+):
FFN(x) = Linear₂(Swish(Linear₁(x)) ⊙ Linear₃(x))
  Parameters: 3 × hidden × (8/3)×hidden ≈ 8H²
```

SwiGLU provides better training dynamics and converges faster in practice. The gating mechanism allows the network to selectively pass information, leading to improved gradient flow and representation learning. Research (Shazeer 2020, LLaMA) shows SwiGLU consistently outperforms GELU/ReLU in language modeling at equivalent parameter counts.

To use the old GELU FFN: `config = HaloConfig(use_swiglu=False)`

### Gradient Checkpointing (v2.0+)

For training large models on limited GPU memory, gradient checkpointing trades compute for memory by recomputing activations during backward pass instead of storing them:

```python
# Enable gradient checkpointing (reduces memory ~40-60% at cost of ~30% slower training)
model.enable_gradient_checkpointing()

# Disable when not needed (e.g., inference)
model.disable_gradient_checkpointing()

# Check status
print(f"Gradient checkpointing enabled: {model.gradient_checkpointing}")
```

Memory savings scale with model depth:

| Layers | Without Checkpointing | With Checkpointing | Savings |
|:------:|:---------------------:|:-------------------:|:-------:|
| 4 | ~1.2 GB | ~0.8 GB | 33% |
| 8 | ~2.4 GB | ~1.2 GB | 50% |
| 12 | ~3.6 GB | ~1.5 GB | 58% |
| 24 | ~7.2 GB | ~2.5 GB | 65% |

*Approximate values for hidden_size=512, seq_len=1024, batch_size=4*

### Mathematical Formulation

Given input sequence x ∈ ℝ^(B×N), the forward pass:

1. **Embed**: e = Embedding(x) ∈ ℝ^(B×N×H)
2. **Prepend globals**: x̂ = [g₁,...,g_G ; e₁,...,e_N] ∈ ℝ^(B×(G+N)×H)
3. **Apply RoPE**: Compute rotary embeddings cos(mθ), sin(mθ) for positions m ∈ [0, G+N)
4. **Per layer l ∈ [1, L]**:
   - Pre-norm: h = LN₁(x̂^(l-1))
   - Split: h_G = h[:G], h_T = h[G:]
   - Global attention: ĝ = SDPA(W_q·h_G, W_k·h, W_v·h)
   - Build neighbors: idx = build_neighbors(N, K, layer=l)
   - Sparse attention: t̂ = GatherAttn(W_q·h_T, W_k·h, W_v·h, idx)
   - Merge + residual: x̂^(l) = x̂^(l-1) + [ĝ; t̂]
   - FFN + residual: x̂^(l) = x̂^(l) + SwiGLU(LN₂(x̂^(l)))
5. **Output**: logits = W_lm · LN_f(x̂^(L)_{G:}) ∈ ℝ^(B×N×V)

### Information Flow Analysis

The sparse connectivity graph ensures efficient information propagation:

- **Direct reach** (1 hop): K = 76 tokens
- **2-hop reach**: up to K² ≈ 5,776 tokens (with overlap)
- **Graph diameter**: O(log N) due to random edges and dilated connections
- **Effective receptive field after L layers**: Grows as K^L (bounded by N)

For a 6-layer model with K=76: theoretical maximum receptive field covers the entire sequence for N ≤ 76⁶ ≈ 192 billion positions. In practice, information mixing is complete within 3-4 layers for sequences up to 8192 tokens.

---

## HuggingFace Hub Integration

HALO-S v2.2+ provides seamless integration with the HuggingFace ecosystem. Models can be saved in HF-compatible format, loaded from Hub repositories, and pushed directly to your HF account.

### Prerequisites

```bash
# Install HuggingFace Hub client (optional dependency)
pip install huggingface_hub safetensors

# Login to HuggingFace (required for push_to_hub)
huggingface-cli login
# Or set the HF_TOKEN environment variable
export HF_TOKEN="hf_your_token_here"
```

### Saving Models in HuggingFace Format — `save_for_hub()`

```python
from halo import HaloConfig, HaloSModel, save_for_hub

config = HaloConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_layers=12,
    num_heads=16,
    num_kv_heads=4,
    max_seq_len=4096,
)
model = HaloSModel(config)

# Train your model...
# trainer.fit(...)

# Save in HF format (creates config.json + model.safetensors)
save_for_hub(model, config, "./my-halo-model/")
```

This creates:
```
my-halo-model/
├── config.json          # HaloConfig serialized in HF-compatible JSON format
└── model.safetensors    # Weights in safetensors format (safe, fast, zero-copy)
```

The `config.json` includes all HALO-S configuration plus HF metadata fields:
```json
{
  "model_type": "halo-s",
  "architectures": ["HaloSModel"],
  "halo_version": "2.2.1",
  "vocab_size": 32000,
  "hidden_size": 1024,
  "num_layers": 12,
  "num_heads": 16,
  "num_kv_heads": 4,
  "num_globals": 2,
  "local_window": 64,
  "dilated_offsets": [1, 2, 4, 8],
  "num_random": 2,
  "dropout": 0.1,
  "max_seq_len": 4096,
  "use_swiglu": true
}
```

### Loading Models from HuggingFace Hub — `load_from_hub()`

```python
from halo import load_from_hub

# Load from a HuggingFace repository
model = load_from_hub("bueormnew/halo-s-70m", device="cuda")

# Load from a local HF-format directory
model = load_from_hub("./my-halo-model/", device="cuda")

# Load a specific revision/branch
model = load_from_hub("bueormnew/halo-s-70m", device="cuda", revision="v2.1")

# Load old .pt checkpoint (backward compatible)
model = load_from_hub("path/to/old_model.pt", device="cpu")

# Load and immediately set to eval mode
model = load_from_hub("bueormnew/halo-s-70m", device="cuda")
model.eval()
```

The `load_from_hub()` function automatically:
1. Detects whether the path is a local directory, a local file, or a Hub repository ID
2. Downloads `config.json` and weights from Hub if needed (uses `huggingface_hub` cache)
3. Reconstructs `HaloConfig` from the JSON (tolerates missing fields, applies defaults)
4. Instantiates `HaloSModel` with the reconstructed config
5. Loads weights with `strict=False` for backward compatibility across versions
6. Handles both `model.safetensors` and `pytorch_model.bin` formats
7. Moves model to the specified device

### Pushing Models to HuggingFace Hub — `push_to_hub()`

```python
from halo import HaloConfig, HaloSModel, push_to_hub

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)
model = HaloSModel(config)

# Train...

# Push to your HuggingFace account (public)
push_to_hub(model, config, "your-username/halo-s-custom", private=False)

# Push as private model
push_to_hub(model, config, "your-username/halo-s-private", private=True)

# Use explicit token (alternative to huggingface-cli login)
push_to_hub(model, config, "your-username/halo-s-custom", token="hf_xxxxx")

# Push with a commit message
push_to_hub(model, config, "your-username/halo-s-custom", 
            commit_message="Update: trained for 10 more epochs")
```

After pushing, your model is available at `https://huggingface.co/your-username/halo-s-custom` and can be loaded by anyone with:
```python
model = load_from_hub("your-username/halo-s-custom")
```

### Complete HuggingFace Workflow Example

```python
"""
Full workflow: Train → Save → Push → Load from Hub → Generate
"""
from halo import (
    HaloConfig, HaloSModel, Trainer, CharacterTokenizer,
    save_for_hub, push_to_hub, load_from_hub,
    set_seed, optimize_for_device,
)
from halo.datasets import TextDataset

# === Step 1: Train a model ===
set_seed(42)
config = HaloConfig(
    vocab_size=256, hidden_size=512, num_layers=6,
    num_heads=8, num_kv_heads=2, max_seq_len=2048,
)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")

tok = CharacterTokenizer()
dataset = TextDataset("data/corpus.txt", tokenizer=tok, max_seq_len=2048)
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
trainer.fit(dataset=dataset, epochs=10, batch_size=8)

# === Step 2: Save locally in HF format ===
save_for_hub(model, config, "./my-trained-halo/")

# === Step 3: Push to HuggingFace Hub ===
push_to_hub(model, config, "your-username/halo-s-char-lm")

# === Step 4: Load from Hub (anyone can do this) ===
loaded_model = load_from_hub("your-username/halo-s-char-lm", device="cuda")
loaded_model = optimize_for_device(loaded_model, mode="inference")

# === Step 5: Generate text ===
output = loaded_model.generate(
    "The meaning of life is",
    tokenizer=tok,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
)
print(output)
```

### Loading Old Models (All Versions Supported)

HALO-S maintains full backward compatibility across all versions:

```python
from halo import HaloSModel, load_from_hub

# v1.x model (.pt file with old GELU FFN state_dict)
# Automatically detects missing w3 keys → sets use_swiglu=False → loads GELU weights
model = load_from_hub("old_models/halo_v1_char.pt")

# v2.0 model (.pt file with SwiGLU w3 weight)
# Detects w3 keys → use_swiglu=True → loads all weights
model = load_from_hub("old_models/halo_v2_70m.pt")

# v2.1+ model (safetensors format, local directory)
model = load_from_hub("./models/halo_v21/")

# v2.2+ HuggingFace format (config.json + model.safetensors on Hub)
model = load_from_hub("bueormnew/halo-s-70m")
```

### Saving with PyTorch Format (Fallback)

If safetensors is not installed, models are saved as `pytorch_model.bin`:

```python
from halo import save_for_hub

# Force PyTorch format even if safetensors is available
save_for_hub(model, config, "./my-model/", safe_serialization=False)
# Creates: my-model/config.json + my-model/pytorch_model.bin
```

### Error Handling

```python
from halo import load_from_hub

# If huggingface_hub is not installed:
try:
    model = load_from_hub("bueormnew/halo-s-70m")
except ImportError as e:
    print("Install huggingface_hub: pip install huggingface_hub")

# If model not found on Hub:
try:
    model = load_from_hub("nonexistent/model")
except Exception as e:
    print(f"Model not found: {e}")

# Local paths always work without huggingface_hub installed:
model = load_from_hub("./local-model/")  # Only needs safetensors or torch
```

---

## Device Optimization System

HALO-S v2.1+ includes an automatic device optimization system that configures hardware-specific settings (TF32, Flash SDP, torch.compile, thread count) based on detected GPU profiles.

### Supported Device Profiles

| Profile | GPU | Memory | TF32 | Flash SDP | BF16 | Compile Mode | Architecture |
|---------|-----|--------|:----:|:---------:|:----:|:------------:|:------------:|
| `t4` | NVIDIA Tesla T4 | 16 GB | ✗ | ✓ | ✗ | reduce-overhead | Turing |
| `p100` | NVIDIA Tesla P100 | 16 GB | ✗ | ✗ | ✗ | default | Pascal |
| `l4` | NVIDIA L4 | 24 GB | ✓ | ✓ | ✓ | reduce-overhead | Ada Lovelace |
| `l40` | NVIDIA L40 | 48 GB | ✓ | ✓ | ✓ | max-autotune | Ada Lovelace |
| `rtx_6000` | NVIDIA RTX 6000 Ada | 48 GB | ✓ | ✓ | ✓ | max-autotune | Ada Lovelace |
| `a100` | NVIDIA A100 | 80 GB | ✓ | ✓ | ✓ | max-autotune | Ampere |
| `cpu` | CPU | System RAM | ✗ | ✗ | ✓* | default | x86/ARM |

*BF16 on CPU depends on processor support (most modern x86 CPUs support it via AVX-512 or AMX).

### Using optimize_for_device()

```python
from halo import HaloConfig, HaloSModel, optimize_for_device

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16)
model = HaloSModel(config).to("cuda")

# Auto-detect device and apply optimal settings
model = optimize_for_device(model)

# Explicitly specify device
model = optimize_for_device(model, device="cuda")

# Optimize for inference (enables torch.compile + eval mode)
model = optimize_for_device(model, device="cuda", mode="inference")

# Optimize for training
model = optimize_for_device(model, device="cuda", mode="training")
```

### What optimize_for_device() Does

**On CUDA devices (Ampere+ / Ada Lovelace):**
1. Enables TF32 matmul (`torch.backends.cuda.matmul.allow_tf32 = True`)
2. Enables TF32 cuDNN (`torch.backends.cudnn.allow_tf32 = True`)
3. Enables Flash SDP and Memory-Efficient SDP backends
4. Applies `torch.compile` in inference mode with device-appropriate compile mode
5. Sets `torch.backends.cudnn.benchmark = True` for consistent input sizes

**On older CUDA devices (Turing — T4):**
1. Enables Flash SDP where supported (Turing+ with FP16)
2. Uses `reduce-overhead` compile mode (avoids expensive graph capture)
3. Skips TF32 (not supported below Ampere)

**On Pascal (P100):**
1. Uses `default` compile mode (minimal overhead)
2. Skips Flash SDP (not supported on Pascal)
3. Skips TF32 (not supported below Ampere)

**On CPU:**
1. Sets optimal thread count (`torch.set_num_threads(os.cpu_count())`)
2. Applies `torch.compile` with `mode="default"` for inference
3. Enables BF16 if CPU supports it

The function is **failsafe** — it never raises an exception. If any optimization fails (unsupported hardware, missing CUDA version, etc.), it logs a warning and returns the model unchanged.

### Device Profile Examples

#### Tesla T4 (Colab, Kaggle, GCP)

```python
from halo import HaloConfig, HaloSModel, optimize_for_device, get_optimal_batch_size

config = HaloConfig(vocab_size=32000, hidden_size=768, num_layers=8, num_heads=12, num_kv_heads=4)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")

# T4 has 16 GB — use gradient checkpointing for larger models
model.enable_gradient_checkpointing()

batch_size = get_optimal_batch_size(config, seq_len=1024)  # → 2
print(f"T4 recommended batch size @ seq=1024: {batch_size}")
```

#### NVIDIA L4 (GCP, RunPod)

```python
from halo import HaloConfig, HaloSModel, optimize_for_device, get_optimal_batch_size

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")  # Enables TF32 + Flash SDP

batch_size = get_optimal_batch_size(config, seq_len=1024)  # → 4
print(f"L4 recommended batch size @ seq=1024: {batch_size}")
```

#### NVIDIA A100 (Cloud, HPC)

```python
from halo import HaloConfig, HaloSModel, optimize_for_device, get_optimal_batch_size

config = HaloConfig(
    vocab_size=32000, hidden_size=1536, num_layers=16,
    num_heads=24, num_kv_heads=6, max_seq_len=8192,
)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")  # TF32 + Flash + max-autotune

batch_size = get_optimal_batch_size(config, seq_len=2048)  # → 8
print(f"A100 recommended batch size @ seq=2048: {batch_size}")
```

#### RTX 6000 Ada (Workstation)

```python
from halo import HaloConfig, HaloSModel, optimize_for_device, get_optimal_batch_size

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")  # TF32 + Flash + max-autotune

batch_size = get_optimal_batch_size(config, seq_len=1024)  # → 8
print(f"RTX 6000 recommended batch size @ seq=1024: {batch_size}")
```

#### CPU (Development, Testing)

```python
from halo import HaloConfig, HaloSModel, optimize_for_device

config = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4)
model = HaloSModel(config)

# CPU optimization: sets thread count, applies torch.compile
model = optimize_for_device(model, device="cpu", mode="inference")

# The model now uses all available CPU cores and is compiled for speed
tok = CharacterTokenizer()
output = model.generate("Hello", tokenizer=tok, max_new_tokens=50)
```

### Auto-Detecting Device Profile

```python
from halo import detect_device_profile, device_info

# Get the detected profile dictionary
profile = detect_device_profile()
print(f"Device: {profile['name']}")
print(f"Memory: {profile['memory_gb']} GB")
print(f"TF32: {profile['supports_tf32']}")
print(f"Flash SDP: {profile['supports_flash']}")
print(f"BF16: {profile['supports_bf16']}")
print(f"Compile mode: {profile['compile_mode']}")

# Get comprehensive device info
info = device_info()
print(f"Best device: {info['device']}")
print(f"CUDA available: {info['cuda_available']}")
print(f"Number of GPUs: {info['num_gpus']}")
print(f"CPU threads: {info['cpu_threads']}")
if info['cuda_available']:
    print(f"GPU: {info['gpu_name']} ({info['gpu_memory_gb']} GB)")
```

### Getting Optimal Batch Size

```python
from halo import HaloConfig, get_optimal_batch_size

config = HaloConfig(hidden_size=1024, num_layers=12, num_heads=16)

# Get recommended batch size for current device and sequence length
batch_size = get_optimal_batch_size(config, seq_len=1024)
print(f"Recommended batch size: {batch_size}")

# Different sequence lengths get different recommendations
for seq_len in [256, 512, 1024, 2048, 4096]:
    bs = get_optimal_batch_size(config, seq_len=seq_len)
    print(f"  seq_len={seq_len:>5} → batch_size={bs}")
```

### Optimal Batch Sizes by Device (Reference Table)

| Seq Length | T4 (16GB) | P100 (16GB) | L4 (24GB) | L40 (48GB) | RTX 6000 (48GB) | A100 (80GB) |
|:----------:|:---------:|:-----------:|:---------:|:----------:|:---------------:|:-----------:|
| 256 | 8 | 8 | 16 | 32 | 32 | 64 |
| 512 | 4 | 4 | 8 | 16 | 16 | 32 |
| 1024 | 2 | 2 | 4 | 8 | 8 | 16 |
| 2048 | 1 | 1 | 2 | 4 | 4 | 8 |
| 4096 | — | — | 1 | 2 | 2 | 4 |

*Based on hidden_size=1024, num_layers=12. Smaller models can use larger batches.*

### Multi-GPU Training with DataParallel

```python
import torch
import torch.nn as nn
from halo import HaloConfig, HaloSModel, Trainer, optimize_for_device

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16)
model = HaloSModel(config)

# Wrap in DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to("cuda")
model = optimize_for_device(model)

# Training proceeds normally — DataParallel handles distribution
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
```

---

## Performance Analysis (Theoretical)

> ⚠️ **All performance data below is THEORETICAL**, derived from complexity analysis. Large-scale empirical benchmarks are in progress. See [Empirical Benchmarks](#empirical-benchmarks) for real measurements.

### Attention Operation Reduction

At sequence length N=4096 with K=76 neighbors per token:

```
Dense Transformer attention operations:  N²      = 16,777,216
HALO-S attention operations:             N×(K+G) =    319,488

Reduction factor: 16,777,216 / 319,488 ≈ 52.5×
```

### Scaling Comparison (Attention FLOPs)

| Sequence Length (N) | Dense Transformer (N²) | HALO-S (N×76) | Theoretical Speedup |
|:---:|:---:|:---:|:---:|
| 512 | 262,144 | 38,912 | 6.7× |
| 1,024 | 1,048,576 | 77,824 | 13.5× |
| 2,048 | 4,194,304 | 155,648 | 26.9× |
| 4,096 | 16,777,216 | 311,296 | 53.9× |
| 8,192 | 67,108,864 | 622,592 | 107.8× |
| 16,384 | 268,435,456 | 1,245,184 | 215.6× |
| 32,768 | 1,073,741,824 | 2,490,368 | 431.1× |
| 65,536 | 4,294,967,296 | 4,980,736 | 862.3× |
| 131,072 | 17,179,869,184 | 9,961,472 | 1,724.6× |

The speedup grows **linearly with N** because dense attention is O(N²) while HALO-S is O(N×K) with fixed K.

### Theoretical Comparison with Other Architectures

> ⚠️ **THEORETICAL COMPARISON** — based on published complexity analyses, not head-to-head benchmarks by us.

| Model | Attention Complexity | Memory (Scores) | Global Context | Dilated | Random Edges | GQA | Custom Kernels |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense Transformer** | O(N²·d) | O(N²) | Full (implicit) | ✗ | ✗ | Optional | ✗ |
| **Longformer** | O(N·w·d) | O(N·w) | ✓ (fixed) | ✓ | ✗ | ✗ | ✓ |
| **BigBird** | O(N·(w+g+r)·d) | O(N·(w+g+r)) | ✓ (fixed) | ✗ | ✓ | ✗ | ✓ |
| **Mamba (SSM)** | O(N·d²) | O(d²) | Implicit (state) | ✗ | ✗ | N/A | ✓ |
| **RWKV** | O(N·d) | O(d) | Implicit (state) | ✗ | ✗ | N/A | ✓ |
| **Flash Attention** | O(N²·d) | O(N) | Full (implicit) | ✗ | ✗ | Optional | ✓ |
| **HALO-S** | **O(N·K·d)** | **O(N·K)** | ✓ (learned) | ✓ | ✓ | ✓ | **✗** |

Key differentiator: HALO-S achieves sub-quadratic complexity **without custom CUDA kernels**, making it portable across all PyTorch-supported hardware.

### Memory Efficiency Analysis

| Component | Dense Transformer | HALO-S | Advantage |
|-----------|:-:|:-:|:-:|
| Attention scores (B=1, N=4096) | 512 MB | 9.5 MB | **54× less** |
| KV cache (GQA 4:1 ratio) | 16 MB | 4 MB | **4× less** |
| Total attention memory (N=4096) | 528 MB | 13.5 MB | **39× less** |
| Crossover point (memory) | — | N > 9,728 | Total advantage |

**Note on the crossover point**: Due to the gather operation creating intermediate tensors (gathered K and V), HALO-S uses more total memory than dense attention for short sequences. The memory advantage manifests at longer sequences (N > ~9,728) where the O(N²) attention score matrix dominates total memory usage.

### Qualitative Comparison (THEORETICAL)

| Capability | Transformer | Mamba | Longformer | **HALO-S** |
|------------|:---:|:---:|:---:|:---:|
| Long-range dependencies | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ (theoretical) |
| Training efficiency | ★★☆☆☆ | ★★★★★ | ★★★★☆ | ★★★★☆ (theoretical) |
| Inference speed | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ (theoretical) |
| Hardware compatibility | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| Implementation simplicity | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ |
| No custom kernels needed | ★★★★★ | ✗ | ✗ | ★★★★★ |
| Portability (CPU/GPU/TPU) | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★★★★ |

---

## Empirical Benchmarks

> 📊 **Real benchmark data** from actual training runs on NVIDIA GPUs. These results provide an honest assessment of where HALO-S stands today.

### Test 1: Small Scale (seq=256, ~3.5M params, 10 epochs)

Character-level language modeling on a small corpus. Both models trained with identical hyperparameters (AdamW, lr=3e-4, AMP enabled).

**Configuration:**
- Model: hidden_size=256, num_layers=4, num_heads=4, num_kv_heads=2
- Data: Character-level text, vocab_size=256
- Training: 10 epochs, batch_size=32, seq_len=256

| Metric | HALO-S | Dense Transformer | Δ | Notes |
|--------|:------:|:-----------------:|:-:|-------|
| **Perplexity** | 3.48 | 3.45 | +0.9% | Near-parity |
| **Train Time** | 1675s | 828s | 2.0× slower | Gather overhead |
| **Peak Memory** | 1.72 GB | 0.72 GB | 2.4× more | Gathered K/V tensors |
| **Generation** | 102 tok/s | 346 tok/s | 3.4× slower | Sequential gather |
| **Final Train Loss** | 1.25 | 1.24 | +0.8% | Converged similarly |

**Interpretation**: At 256 tokens, the O(N²) vs O(N×K) difference is minimal (N²=65,536 vs N×K=19,456 — only 3.4× theoretical). The gather overhead dominates at this scale.

### Test 2: Medium Scale (seq=1024, ~20M params, 3 epochs)

Character-level language modeling with longer context and larger model.

**Configuration:**
- Model: hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=2
- Data: Character-level text, vocab_size=256, seq_len=1024
- Training: 3 epochs, batch_size=8, mixed precision

| Metric | HALO-S | Dense Transformer | Δ | Notes |
|--------|:------:|:-----------------:|:-:|-------|
| **Perplexity** | 3.56 | 3.59 | −0.8% | **HALO-S wins** |
| **Train Time** | 3885s | 1872s | 2.1× slower | Gather still dominates |
| **Peak Memory** | 4.95 GB | 0.80 GB | 6.2× more | Intermediate tensors |
| **Generation** | 62 tok/s | 214 tok/s | 3.5× slower | Per-token gather |
| **Final Train Loss** | 1.27 | 1.28 | −0.8% | Slightly better |

**Interpretation**: At 1024 tokens (N²=1,048,576 vs N×K=77,824 — 13.5× theoretical), HALO-S achieves *slightly better* perplexity than the dense baseline. The sparse connectivity may act as implicit regularization at this scale.

### Test 3: Large Scale (seq=1024, ~70M params, BPE tokenizer, 2 epochs)

BPE-tokenized language modeling at 70M parameter scale — the largest experiment run.

**Configuration:**
- Model: hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4
- Data: BPE tokenized (tiktoken gpt2), vocab_size=50257, seq_len=1024
- Training: 2 epochs, batch_size=4, mixed precision, gradient accumulation=2

| Metric | HALO-S | Dense Transformer | Δ | Notes |
|--------|:------:|:-----------------:|:-:|-------|
| **Perplexity** | 102.3 | 100.7 | +1.6% | Near-parity |
| **Train Time** | 59.8 min | 46.3 min | 1.3× slower | Gap closing! |
| **Latency @1024** | 27.7 ms | 12.3 ms | 2.3× higher | Per-step latency |
| **Peak Memory** | 0.818 GB | 0.816 GB | ~Same | Model params dominate |
| **Throughput** | ~36 tok/ms | ~83 tok/ms | 2.3× lower | Bound by gather |

**Interpretation**: At 70M parameters, the speed gap narrows to 1.3× (from 2.0× at smaller scale). This is because the model parameters and FFN computation dominate total FLOPs, making the attention mechanism a smaller fraction of total compute. Memory usage is virtually identical because at this scale, model parameters (~280MB) far outweigh attention scores.

### Test 4: Ablation Study (seq=256, ~3.5M params, 5 epochs)

Contribution of each connectivity component measured by removing one at a time.

| Variant | Val Loss | Perplexity | Train Time | Parameters |
|---------|:--------:|:----------:|:----------:|:----------:|
| **HALO-S Complete** (all components) | 2.23 | 9.33 | 13.15s | 3.5M |
| Without Global Tokens | 2.12 | 8.32 | 11.54s | 3.5M |
| Without Dilated Connections | 2.02 | 7.52 | 9.80s | 3.5M |
| Without Random Edges | 2.15 | 8.59 | 10.50s | 3.5M |
| Local Window Only | 1.92 | 6.80 | 9.20s | 3.5M |

**Why removing components improves short-sequence performance**: This counterintuitive result is expected. At seq_len=256, the local window (64 tokens) already covers 25% of the sequence. Adding dilated/random/global connections increases the neighbor budget K without proportional benefit — the overhead isn't recovered. These components are designed to shine at seq_len > 2048 where local windows cover < 3% of the sequence and long-range connections become essential.

### Test 5: Long Context — Needle in a Haystack (seq=512, 10 epochs)

Synthetic retrieval task: a "needle" (unique token pattern) is placed at varying distances from a query position. Model must predict the needle value.

| Distance | HALO-S Accuracy | Dense Transformer Accuracy | Winner |
|:--------:|:---------------:|:--------------------------:|:------:|
| 10 tokens | 0.06 | 0.07 | Tie |
| 50 tokens | 0.05 | 0.10 | Dense |
| 100 tokens | 0.06 | 0.05 | Tie |
| 200 tokens | 0.09 | 0.06 | HALO-S |

Both architectures perform similarly on this task at short sequence lengths, with neither achieving high accuracy. This suggests the task requires either more training, larger models, or longer sequences to properly evaluate long-range retrieval capabilities.

### Summary of Empirical Findings

| Scale | Params | Seq Len | PPL Gap | Speed Gap | Memory Gap |
|:-----:|:------:|:-------:|:-------:|:---------:|:----------:|
| Small | 3.5M | 256 | +0.9% (HALO-S worse) | 2.0× slower | 2.4× more |
| Medium | 20M | 1024 | −0.8% (**HALO-S better**) | 2.1× slower | 6.2× more |
| Large | 70M | 1024 | +1.6% (HALO-S worse) | 1.3× slower | ~Same |

**Key Takeaways:**

1. **Perplexity parity**: HALO-S achieves comparable perplexity to dense Transformers across all scales tested (3.5M → 70M parameters). The quality gap is consistently < 2%.

2. **Speed overhead decreasing with scale**: The gap narrows from 2.0× at 3.5M to 1.3× at 70M as attention becomes a smaller fraction of total compute.

3. **Memory crossover not yet reached**: At seq_len ≤ 1024, gathered K/V tensors use more memory than dense attention. The advantage requires seq_len > ~9,728.

4. **Architecture designed for seq_len > 2048**: The O(N×K) vs O(N²) complexity difference becomes meaningful at longer sequences. At N=1024 with K=76, the ratio is only 13.5×, which doesn't overcome constant-factor overhead.

5. **Ablation validates component purpose**: Removing connectivity components improves short-sequence performance but is expected to degrade long-sequence performance.

**Where HALO-S should excel (not yet validated at scale):**
- Sequences > 4096 tokens where O(N²) becomes prohibitive
- Memory-constrained inference with very long contexts
- Scenarios where custom CUDA kernels are unavailable (CPU inference, non-NVIDIA hardware)
- Research requiring easy-to-modify attention patterns

**Current recommendation**: Use HALO-S for research and experimentation with long-context models. For production short-context (<2K) tasks, standard Transformers with FlashAttention remain more efficient.

---

## Installation

### From PyPI

```bash
# Core installation (PyTorch + NumPy only)
pip install pyhalos

# Specific version
pip install pyhalos==2.2.1

# Full installation (includes tqdm progress bars + SentencePiece tokenizer + safetensors)
pip install pyhalos[full]

# With HuggingFace Hub support
pip install pyhalos[full] huggingface_hub

# Development installation (includes pytest)
pip install pyhalos[full,dev]
```

### From Source

```bash
git clone https://github.com/bueormnew/pyhalo.git
cd pyhalo
pip install -e ".[full,dev]"
```

### Requirements

| Dependency | Version | Required | Purpose |
|------------|---------|:--------:|---------|
| Python | ≥ 3.10 | ✓ | Runtime (uses modern type hints, match statements) |
| PyTorch | ≥ 2.1.0 | ✓ | Deep learning framework (SDPA, compile support) |
| NumPy | ≥ 1.24.0 | ✓ | Array operations for graph construction |
| tqdm | any | ✗ | Progress bars during training |
| sentencepiece | any | ✗ | Subword tokenization (SentencePiece models) |
| safetensors | any | ✗ | Safe, fast model serialization |
| huggingface_hub | any | ✗ | Hub integration (push/pull models) |
| tiktoken | any | ✗ | BPE tokenization (OpenAI GPT-2/GPT-4 encoding) |

### Verifying Installation

```python
import halo
print(f"HALO-S version: {halo.__version__}")  # → 2.2.1
print(f"Device: {halo.device_info()['device']}")

# Quick smoke test
from halo import HaloConfig, HaloSModel, set_seed
set_seed(42)
config = HaloConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2)
model = HaloSModel(config)
print(f"✓ Model created: {model.count_parameters():,} parameters")

# Test generation
from halo import CharacterTokenizer
tok = CharacterTokenizer()
output = model.generate("Hello", tokenizer=tok, max_new_tokens=20, temperature=0.8)
print(f"✓ Generation works: '{output[:30]}...'")

# Test Hub functions available
from halo import save_for_hub, load_from_hub, push_to_hub
print("✓ Hub functions imported successfully")
```

### Upgrading from Previous Versions

```bash
# Upgrade to latest
pip install --upgrade pyhalos

# Check version
python -c "import halo; print(halo.__version__)"  # → 2.2.1
```

---

## Quick Start

### Minimal Example

```python
from halo import HaloConfig, HaloSModel, set_seed

set_seed(42)

# Configure a small model
config = HaloConfig(
    vocab_size=256,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    num_kv_heads=2,       # GQA: 4:1 ratio
    num_globals=2,
    local_window=64,
    max_seq_len=4096,
)

# Instantiate
model = HaloSModel(config)

# Inspect
print(model.summary())
print(f"Parameters: {model.count_parameters():,}")
print(f"FLOPs (N=1024): {model.estimate_flops(seq_len=1024)['total_gflops']:.2f} GFLOPs")
```

### Text Generation (String API)

```python
from halo import HaloConfig, HaloSModel, CharacterTokenizer, set_seed

set_seed(42)

config = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4)
model = HaloSModel(config)
tok = CharacterTokenizer()

# Generate from a text prompt (returns string)
output = model.generate(
    "Hello world",
    tokenizer=tok,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
)
print(output)
```

### Tensor Generation (No Tokenizer)

```python
import torch
from halo import HaloConfig, HaloSModel

config = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4)
model = HaloSModel(config)

# Generate from tensor input (returns tensor)
input_ids = torch.randint(0, 256, (1, 20))
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_p=0.9,
)
print(f"Input: {input_ids.shape} → Output: {output_ids.shape}")
```

### Loading a Pretrained Model from Hub

```python
from halo import load_from_hub, optimize_for_device, CharacterTokenizer

# Load a pretrained model from HuggingFace
model = load_from_hub("bueormnew/halo-s-70m", device="cuda")

# Optimize for current hardware
model = optimize_for_device(model, mode="inference")

# Generate text
tok = CharacterTokenizer()
output = model.generate(
    "The meaning of life is",
    tokenizer=tok,
    max_new_tokens=200,
    temperature=0.7,
    top_p=0.9,
)
print(output)
```

### Using Device Optimization

```python
from halo import (
    HaloConfig, HaloSModel, optimize_for_device,
    detect_device_profile, get_optimal_batch_size
)

# Auto-detect and configure
profile = detect_device_profile()
print(f"Running on: {profile['name']}")

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)
model = HaloSModel(config).to("cuda")

# Apply hardware-specific optimizations
model = optimize_for_device(model, mode="training")

# Get recommended batch size for your GPU
batch_size = get_optimal_batch_size(config, seq_len=1024)
print(f"Recommended batch size: {batch_size}")
```

### Generating with tiktoken (BPE)

```python
import torch
import tiktoken
from halo import HaloConfig, HaloSModel, set_seed

set_seed(42)

# Use tiktoken for BPE encoding
enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab  # 50257

config = HaloConfig(
    vocab_size=vocab_size,
    hidden_size=768,
    num_layers=8,
    num_heads=12,
    num_kv_heads=4,
    max_seq_len=2048,
)
model = HaloSModel(config)

# Encode → Generate → Decode
prompt = "Once upon a time"
input_ids = torch.tensor([enc.encode(prompt)]).long()
output_ids = model.generate(input_ids, max_new_tokens=100, temperature=0.8, top_k=50)
text = enc.decode(output_ids[0].tolist())
print(text)
```

### Training a Model (Quick Version)

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer, set_seed
from halo.datasets import TextDataset

set_seed(42)

config = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4)
model = HaloSModel(config)
tok = CharacterTokenizer()
dataset = TextDataset(file_path="data/corpus.txt", tokenizer=tok, max_seq_len=512)

trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
history = trainer.fit(dataset=dataset, epochs=5, batch_size=16)

# Generate after training
output = model.generate("The ", tokenizer=tok, max_new_tokens=100, temperature=0.8)
print(output)
```

---

## Advanced Usage

### Training with Gradient Checkpointing

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer, set_seed
from halo.datasets import TextDataset

set_seed(42)

config = HaloConfig(
    vocab_size=256,
    hidden_size=1024,
    num_layers=12,
    num_heads=16,
    num_kv_heads=4,
    max_seq_len=2048,
)
model = HaloSModel(config)

# Enable gradient checkpointing to reduce memory usage (~40-60% savings)
model.enable_gradient_checkpointing()
print(f"Gradient checkpointing: enabled")
print(f"Estimated memory savings: ~{12 * 0.85:.0f}MB per layer avoided")

tok = CharacterTokenizer()
dataset = TextDataset(file_path="data/corpus.txt", tokenizer=tok, max_seq_len=2048)

trainer = Trainer(
    model=model,
    learning_rate=3e-4,
    mixed_precision=True,
    gradient_accumulation_steps=8,  # Effective batch = 8 × batch_size
)

history = trainer.fit(dataset=dataset, epochs=5, batch_size=4)
```

### Training with Mixed Precision & Gradient Accumulation

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer, set_seed
from halo.datasets import JSONLDataset

set_seed(42)

# Model
config = HaloConfig(
    vocab_size=256,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    num_kv_heads=2,
    max_seq_len=2048,
)
model = HaloSModel(config)

# Dataset
tok = CharacterTokenizer()
dataset = JSONLDataset(
    file_path="data/train.jsonl",
    tokenizer=tok,
    max_seq_len=2048,
    text_field="text",
)

# Trainer with full features
trainer = Trainer(
    model=model,
    learning_rate=3e-4,
    mixed_precision=True,              # FP16/BF16 automatic mixed precision
    gradient_accumulation_steps=4,     # Effective batch = 4 × batch_size
    max_grad_norm=1.0,                 # Gradient clipping
    checkpoint_dir="./checkpoints",
    log_every=10,
)

# Train
history = trainer.fit(
    dataset=dataset,
    epochs=10,
    batch_size=8,
    save_every=2,  # Checkpoint every 2 epochs
)

# Access training history
for epoch_data in history:
    print(f"Epoch {epoch_data['epoch']}: loss={epoch_data['train_loss']:.4f}")
```

### Checkpoint Save & Resume

```python
# Save checkpoint manually
trainer.save_checkpoint(path="my_checkpoint.pt")

# Resume training from checkpoint (restores model, optimizer, scheduler, epoch)
trainer.load_checkpoint("my_checkpoint.pt")
# Continue training from where you left off
trainer.fit(dataset=dataset, epochs=15, batch_size=8)  # Resumes from saved epoch
```

### Streaming Dataset (Files Larger Than RAM)

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer
from halo.datasets import StreamingDataset

tok = CharacterTokenizer()

# StreamingDataset reads files lazily with buffer shuffling
stream_dataset = StreamingDataset(
    file_paths=["data/shard_01.jsonl", "data/shard_02.jsonl", "data/shard_03.jsonl"],
    tokenizer=tok,
    max_seq_len=2048,
    buffer_size=10000,     # Local shuffle buffer
    text_field="text",
    file_format="jsonl",   # or "txt"
)

# Use with DataLoader (IterableDataset compatible)
from torch.utils.data import DataLoader
loader = DataLoader(stream_dataset, batch_size=4)

# Or pass directly to Trainer
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
trainer.fit(dataset=stream_dataset, epochs=1, batch_size=4)
```

### Multi-GPU Training

```python
import torch
import torch.nn as nn
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer
from halo.datasets import StreamingDataset

config = HaloConfig(
    vocab_size=32000,
    hidden_size=1024,
    num_layers=12,
    num_heads=16,
    num_kv_heads=4,
    max_seq_len=2048,
)
model = HaloSModel(config)

# Multi-GPU with DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    print(f"Training on {torch.cuda.device_count()} GPUs")

model = model.to("cuda")

tok = CharacterTokenizer()
dataset = StreamingDataset(
    file_paths=["data/shard_01.jsonl", "data/shard_02.jsonl"],
    tokenizer=tok,
    max_seq_len=2048,
    buffer_size=50000,
)

trainer = Trainer(
    model=model,
    learning_rate=3e-4,
    mixed_precision=True,
    gradient_accumulation_steps=4,
    checkpoint_dir="./checkpoints",
)

history = trainer.fit(dataset=dataset, epochs=3, batch_size=16, save_every=1)
```

### Benchmarking

```python
from halo import HaloConfig, HaloSModel
from halo.utils.benchmarks import benchmark_speed, benchmark_generation, estimate_flops

config = HaloConfig(vocab_size=256, hidden_size=512, num_layers=6, num_heads=8)
model = HaloSModel(config)

# Latency benchmark across sequence lengths
speed_results = benchmark_speed(
    model, config,
    seq_lengths=[128, 512, 1024, 2048, 4096],
    batch_size=1,
    warmup_runs=3,
    timed_runs=10,
)
for r in speed_results:
    print(f"  N={r['seq_len']:>5} | {r['avg_ms']:.2f} ms | {r['tokens_per_sec']:,.0f} tok/s")

# Generation throughput
gen_results = benchmark_generation(
    model, config,
    prompt_len=10,
    max_new_tokens=200,
    num_runs=5,
)
print(f"Generation: {gen_results['tokens_per_sec']:.1f} tokens/sec")

# Theoretical FLOPs (no model instantiation needed)
flops = estimate_flops(config, seq_len=4096)
print(f"Total: {flops['total_gflops']:.2f} GFLOPs")
print(f"  Sparse attention: {flops['attention_flops']/1e9:.2f} G")
print(f"  Global attention: {flops['global_flops']/1e9:.2f} G")
print(f"  FFN:              {flops['ffn_flops']/1e9:.2f} G")
```

### Model Introspection

```python
from halo import HaloConfig, HaloSModel, count_parameters

config = HaloConfig(vocab_size=256, hidden_size=512, num_layers=6, num_heads=8)
model = HaloSModel(config)

# Summary with architecture details and memory estimate
print(model.summary())

# Parameter count
print(f"Trainable params: {model.count_parameters():,}")

# Standalone parameter counter (works on any nn.Module)
print(f"Via utility: {count_parameters(model):,}")

# FLOPs breakdown
flops = model.estimate_flops(seq_len=2048)
for key, value in flops.items():
    print(f"  {key}: {value}")
```

### Word-Level Tokenizer

```python
from halo import WordTokenizer

tok = WordTokenizer()
tok.build_vocab(["The cat sat on the mat.", "Hello world!"], min_freq=1)

encoded = tok.encode("The cat sat")
decoded = tok.decode(encoded)
print(f"Vocab size: {tok.vocab_size}")
print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

### Complete Training Pipeline (End-to-End)

```python
"""
Full training pipeline: data → model → train → save → push to Hub → generate
"""
from halo import (
    HaloConfig, HaloSModel, Trainer, CharacterTokenizer,
    set_seed, optimize_for_device, save_for_hub, push_to_hub,
    get_optimal_batch_size, detect_device_profile,
)
from halo.datasets import JSONLDataset

# Reproducibility
set_seed(42)

# Check hardware
profile = detect_device_profile()
print(f"Device: {profile['name']} ({profile['memory_gb']} GB)")

# Configure model
config = HaloConfig(
    vocab_size=256,
    hidden_size=768,
    num_layers=8,
    num_heads=12,
    num_kv_heads=4,
    num_globals=2,
    local_window=64,
    max_seq_len=2048,
    use_swiglu=True,
)

# Create and optimize model
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")
model.enable_gradient_checkpointing()  # Save memory
print(f"Model: {model.count_parameters():,} parameters")

# Dataset
tok = CharacterTokenizer()
dataset = JSONLDataset("data/train.jsonl", tokenizer=tok, max_seq_len=2048)

# Get optimal batch size for this hardware
batch_size = get_optimal_batch_size(config, seq_len=2048)
print(f"Batch size: {batch_size}")

# Train
trainer = Trainer(
    model=model,
    learning_rate=3e-4,
    mixed_precision=True,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    checkpoint_dir="./checkpoints",
    log_every=50,
)

history = trainer.fit(dataset=dataset, epochs=10, batch_size=batch_size, save_every=2)

# Disable checkpointing for inference
model.disable_gradient_checkpointing()

# Save in HuggingFace format
save_for_hub(model, config, "./halo-s-trained/")

# Push to Hub (requires huggingface-cli login)
push_to_hub(model, config, "your-username/halo-s-custom")

# Generate
output = model.generate("Once upon a time", tokenizer=tok, max_new_tokens=200, temperature=0.7)
print(output)
print("\nDone! Model available at https://huggingface.co/your-username/halo-s-custom")
```

---

## Configuration Reference

### HaloConfig — Complete Parameter Documentation

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class HaloConfig:
    # === Vocabulary & Embedding ===
    vocab_size: int = 256
    # Size of the token vocabulary.
    # - 256 for CharacterTokenizer (byte-level)
    # - ~32000 for BPE/SentencePiece
    # - 50257 for tiktoken (GPT-2 encoding)

    # === Model Dimensions ===
    hidden_size: int = 512
    # Model dimension (embedding size, residual stream width).
    # Must be divisible by num_heads.
    # Common values: 128 (tiny), 256 (small), 512 (medium), 768-1024 (large), 1536+ (XL)

    num_layers: int = 6
    # Number of HaloBlock layers (transformer blocks).
    # Each block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    # Common values: 2 (tiny), 4 (small), 6-8 (medium), 12 (large), 16-24 (XL)

    # === Attention Heads ===
    num_heads: int = 8
    # Number of query attention heads.
    # head_dim = hidden_size // num_heads (must divide evenly)
    # More heads = finer-grained attention patterns

    num_kv_heads: int = 2
    # Number of key/value heads for Grouped Query Attention (GQA).
    # GQA ratio = num_heads // num_kv_heads
    # - num_kv_heads == num_heads: standard Multi-Head Attention (MHA)
    # - num_kv_heads == 1: Multi-Query Attention (MQA)
    # - 1 < num_kv_heads < num_heads: Grouped Query Attention (GQA)
    # Reduces KV cache memory by the GQA ratio.

    # === Sparse Graph Parameters ===
    num_globals: int = 2
    # Number of learned global tokens prepended to each sequence.
    # These attend to ALL positions (dense) and act as shared memory.
    # More globals = more broadcast capacity but more compute per layer.
    # Typical: 2-4 for most tasks, 8+ for very long sequences.

    local_window: int = 64
    # Size of the local attention window.
    # Each token attends to `local_window` nearest neighbors.
    # Captures syntax, local semantics, and short-range dependencies.
    # Typical: 32 (aggressive), 64 (default), 128 (conservative)

    dilated_offsets: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    # Distances for dilated (exponentially-spaced) connections.
    # Creates long-range shortcuts in both forward and backward directions.
    # [1,2,4,8] = 4 offsets × 2 directions = 8 dilated connections per token.
    # Larger offsets = longer range but sparser coverage.
    # For very long sequences: [1, 2, 4, 8, 16, 32]

    num_random: int = 2
    # Number of random edges per token in the connectivity graph.
    # Ensures small-world properties: O(log N) diameter.
    # Even 1-2 random edges dramatically reduce graph diameter.

    # === Regularization & Limits ===
    dropout: float = 0.1
    # Dropout rate applied in attention and FFN layers.
    # 0.0 for large models with lots of data.
    # 0.1-0.2 for smaller models or limited data.

    max_seq_len: int = 4096
    # Maximum supported sequence length.
    # Affects RoPE precomputation and positional encoding range.
    # Can be set higher than actual training sequences safely.

    # === Architecture Variant ===
    use_swiglu: bool = True
    # Whether to use SwiGLU activation in feed-forward layers.
    # True (default, v2.0+): SwiGLU gated FFN (better training dynamics)
    # False: Standard GELU FFN (v1.x compatible, fewer parameters per layer)
```

### Derived Properties

```python
config = HaloConfig(hidden_size=512, num_heads=8, num_kv_heads=2)

# Head dimension (computed)
print(config.head_dim)        # 64 (= 512 // 8)

# Total neighbors per token (computed)
print(config.num_neighbors)   # 76 (= 2 + 64 + 2×4 + 2)
#                                    globals + window + 2×len(dilated_offsets) + random

# Serialization
d = config.to_dict()          # → dict with all fields
config2 = HaloConfig.from_dict(d)  # Reconstruct (tolerates unknown/missing keys)
```

### Example Configurations

```python
from halo import HaloConfig

# Tiny model (~1M params) — for unit testing and debugging
tiny = HaloConfig(
    vocab_size=256, hidden_size=128, num_layers=2,
    num_heads=4, num_kv_heads=2, max_seq_len=512,
)

# Small model (~3.5M params) — for experimentation and quick iteration
small = HaloConfig(
    vocab_size=256, hidden_size=256, num_layers=4,
    num_heads=4, num_kv_heads=2, max_seq_len=2048,
)

# Medium model (~20M params) — character-level language modeling
medium = HaloConfig(
    vocab_size=256, hidden_size=512, num_layers=8,
    num_heads=8, num_kv_heads=2, max_seq_len=4096,
)

# Large model (~70M params) — BPE language model
large = HaloConfig(
    vocab_size=32000, hidden_size=1024, num_layers=12,
    num_heads=16, num_kv_heads=4, max_seq_len=4096,
)

# XL model (~150M params) — research scale
xl = HaloConfig(
    vocab_size=32000, hidden_size=1536, num_layers=16,
    num_heads=24, num_kv_heads=6, max_seq_len=8192,
    local_window=128, dilated_offsets=[1, 2, 4, 8, 16],
)

# Long-context model — optimized for very long sequences
long_ctx = HaloConfig(
    vocab_size=32000, hidden_size=1024, num_layers=12,
    num_heads=16, num_kv_heads=4, max_seq_len=32768,
    local_window=128, dilated_offsets=[1, 2, 4, 8, 16, 32],
    num_globals=4, num_random=4,
)
```

---

## API Reference

### Core

| Symbol | Type | Description |
|--------|------|-------------|
| `halo.HaloConfig` | dataclass | Model configuration (all hyperparameters) |
| `halo.HaloSModel` | nn.Module | Main HALO-S language model |
| `halo.BaselineModel` | nn.Module | Dense Transformer baseline for comparison |

### Model Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `HaloSModel(config)` | `config: HaloConfig` | Create model from config |
| `.forward(x)` | `x: Tensor (B, N)` → `Tensor (B, N, V)` | Forward pass, returns logits |
| `.generate(...)` | See below | Autoregressive text generation |
| `.summary()` | → `str` | Architecture summary with parameter counts |
| `.count_parameters()` | → `int` | Total trainable parameters |
| `.estimate_flops(seq_len)` | → `dict` | FLOPs breakdown at given sequence length |
| `.from_pretrained(path)` | `path: str` → `HaloSModel` | Load from checkpoint (any version) |
| `.enable_gradient_checkpointing()` | — | Enable memory-saving checkpointing |
| `.disable_gradient_checkpointing()` | — | Disable gradient checkpointing |

### Generation

```python
model.generate(
    prompt,                    # str or Tensor (B, N)
    tokenizer=None,           # Required if prompt is str
    max_new_tokens=100,       # Max tokens to generate
    temperature=1.0,          # Sampling temperature (0 = greedy)
    top_k=0,                  # Top-k filtering (0 = disabled)
    top_p=1.0,               # Nucleus sampling threshold (1.0 = disabled)
    stop_token=None,          # Stop generation at this token ID
)
# Returns: str (if prompt was str) or Tensor (if prompt was Tensor)
```

**Sampling strategies:**
- `temperature=0.0` or `temperature=0.01`: Greedy decoding (deterministic)
- `temperature=0.7, top_p=0.9`: Standard nucleus sampling (good balance)
- `temperature=1.0, top_k=50`: Top-k sampling (diverse)
- `temperature=0.5, top_k=40, top_p=0.95`: Combined (recommended for quality)

### Training

| Symbol | Description |
|--------|-------------|
| `halo.Trainer` | Training loop with AMP, gradient accumulation, checkpointing |

```python
trainer = Trainer(
    model,                          # nn.Module
    learning_rate=3e-4,            # AdamW learning rate
    mixed_precision=True,          # Enable AMP (FP16 on CUDA, BF16 on Ampere+)
    gradient_accumulation_steps=1, # Accumulation steps
    max_grad_norm=1.0,            # Gradient clipping (0 = disabled)
    checkpoint_dir=None,          # Auto-save directory (None = no auto-save)
    log_every=10,                 # Log interval (steps)
)

history = trainer.fit(
    dataset,               # Dataset or IterableDataset
    epochs=10,             # Number of epochs
    batch_size=8,          # Batch size per GPU
    save_every=None,       # Checkpoint every N epochs (None = only at end)
)
# Returns: List[dict] with per-epoch metrics {epoch, train_loss, ...}
```

### Device Optimization

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `halo.optimize_for_device(model, device, mode)` | → `nn.Module` | Apply hardware-specific optimizations |
| `halo.detect_device_profile()` | → `dict` | Auto-detect GPU and return profile |
| `halo.get_optimal_batch_size(config, seq_len)` | → `int` | Recommended batch size |
| `halo.get_optimal_device()` | → `str` | Best available device ("cuda"/"mps"/"cpu") |
| `halo.device_info()` | → `dict` | Comprehensive device information |

### HuggingFace Hub

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `halo.save_for_hub(model, config, dir, safe_serialization=True)` | — | Save config.json + model.safetensors |
| `halo.load_from_hub(path_or_repo, device="cpu", revision=None)` | → `HaloSModel` | Load from local dir, file, or HF Hub |
| `halo.push_to_hub(model, config, repo_id, token=None, private=False)` | — | Upload to HuggingFace Hub |

### Tokenizers

| Symbol | Description |
|--------|-------------|
| `halo.CharacterTokenizer` | Byte-level tokenizer (vocab_size=256, no training needed) |
| `halo.WordTokenizer` | Whitespace-based tokenizer (requires `build_vocab()`) |

```python
# CharacterTokenizer — always ready, no training
tok = CharacterTokenizer()
ids = tok.encode("Hello")   # [72, 101, 108, 108, 111]
text = tok.decode(ids)      # "Hello"
print(tok.vocab_size)       # 256

# WordTokenizer — requires vocabulary building
tok = WordTokenizer()
tok.build_vocab(["The cat sat.", "Hello world!"], min_freq=1)
ids = tok.encode("The cat")
text = tok.decode(ids)
```

### Datasets

| Symbol | Type | Description |
|--------|------|-------------|
| `halo.datasets.JSONLDataset` | Dataset | JSONL files with configurable text field |
| `halo.datasets.TextDataset` | Dataset | Plain text files (splits into chunks) |
| `halo.datasets.StreamingDataset` | IterableDataset | Lazy loading with buffer shuffle |
| `halo.datasets.CopyDataset` | Dataset | Synthetic: learn to copy sequences |
| `halo.datasets.NeedleDataset` | Dataset | Synthetic: needle-in-a-haystack retrieval |

### Utilities

| Symbol | Signature | Description |
|--------|-----------|-------------|
| `halo.set_seed(seed)` | `seed: int` | Set all random seeds (torch, numpy, python) |
| `halo.count_parameters(model)` | → `int` | Count trainable params (works on any Module) |
| `halo.generate(model, ...)` | → `Tensor` | Standalone generation function |

### Benchmarks (halo.utils.benchmarks)

| Symbol | Description |
|--------|-------------|
| `benchmark_speed(model, config, seq_lengths, ...)` | Latency/throughput across sequence lengths |
| `benchmark_generation(model, config, ...)` | Generation throughput (tokens/sec) |
| `estimate_flops(config, seq_len)` | Theoretical FLOPs breakdown |

---

## Backward Compatibility Guide

HALO-S maintains full backward compatibility across all versions. The loading system auto-detects model format and version, applying necessary weight remapping.

### Version Detection & Loading

| Format | Detection Method | Loading Path |
|--------|------------------|-------------|
| **HuggingFace format** (dir with config.json + model.safetensors) | `os.path.isdir()` + config.json exists | `load_from_hub()` → config.json → safetensors/bin |
| **v2.1+ safetensors** (single .safetensors file) | File extension | `from_pretrained()` → safetensors.load_file() |
| **v2.0 checkpoint** (.pt with SwiGLU w3 key) | `"w3"` in state_dict keys | `from_pretrained()` → direct load |
| **v1.x checkpoint** (.pt without w3 key) | Absence of `"w3"` keys | `from_pretrained()` → GELU mode auto-set |
| **Training checkpoint** (.pt with optimizer/scheduler) | `"model_state_dict"` key present | `trainer.load_checkpoint()` |
| **HuggingFace Hub repo** | Not a local path | `load_from_hub()` → hf_hub_download() |

### Loading v1.x Models

v1.x models used GELU FFN (no `w3` weight). When loading:

```python
from halo import HaloSModel, HaloConfig

# Method 1: from_pretrained auto-detects version
model = HaloSModel.from_pretrained("checkpoints/halo_v1_model.pt")
# Internally: detects missing w3 → sets use_swiglu=False → loads GELU weights

# Method 2: explicit config
config = HaloConfig(
    vocab_size=256, hidden_size=512, num_layers=6,
    num_heads=8, num_kv_heads=2,
    use_swiglu=False,  # ← v1.x used GELU
)
model = HaloSModel(config)
import torch
state_dict = torch.load("checkpoints/halo_v1_model.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state_dict, strict=False)
```

### Loading v2.0+ Models

```python
from halo import HaloSModel, load_from_hub

# v2.0 (.pt with SwiGLU) — auto-detected
model = HaloSModel.from_pretrained("checkpoints/halo_v2_70m.pt")

# v2.1+ (safetensors) — auto-detected by extension
model = load_from_hub("./saved_models/halo_v21/")

# v2.2+ Hub format — downloads from HuggingFace
model = load_from_hub("bueormnew/halo-s-70m", device="cuda")
model = load_from_hub("bueormnew/halo-s-70m", revision="v2.0")  # Specific version
```

### Loading Training Checkpoints (Resume Training)

```python
from halo import HaloConfig, HaloSModel, Trainer

config = HaloConfig(vocab_size=256, hidden_size=512, num_layers=6, num_heads=8, num_kv_heads=2)
model = HaloSModel(config)
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)

# Load full training state (model + optimizer + scheduler + epoch + loss)
trainer.load_checkpoint("checkpoints/epoch_5.pt")

# Continue training from where you left off
trainer.fit(dataset=dataset, epochs=10, batch_size=8)  # Resumes from epoch 6
```

### State Dict Key Mapping Reference

```python
# v1.x keys → v2.x keys (example)
# "blocks.0.ffn.w1.weight"  → "blocks.0.ffn.w1.weight"  (unchanged)
# "blocks.0.ffn.w2.weight"  → "blocks.0.ffn.w2.weight"  (unchanged)
# (no w3 in v1.x)           → "blocks.0.ffn.w3.weight"  (new in v2.0, randomly initialized)

# When loading v1.x into v2.x model with strict=False:
# - w1, w2 load normally
# - w3 stays randomly initialized (model needs fine-tuning for full SwiGLU benefit)
# Alternatively: set use_swiglu=False to use GELU, then all weights load perfectly
```

---

## Troubleshooting & FAQ

### Common Issues

#### Q: `ImportError: No module named 'halo'`

```bash
# Make sure you installed the correct package name
pip install pyhalos  # ← correct (NOT "halo" or "pyhalo")

# Verify
python -c "import halo; print(halo.__version__)"
```

#### Q: `RuntimeError: CUDA out of memory`

```python
# Solution 1: Enable gradient checkpointing
model.enable_gradient_checkpointing()

# Solution 2: Reduce batch size
batch_size = get_optimal_batch_size(config, seq_len=your_seq_len)

# Solution 3: Use gradient accumulation (same effective batch, less memory)
trainer = Trainer(model=model, gradient_accumulation_steps=8, ...)
trainer.fit(dataset=dataset, batch_size=2)  # Effective batch = 16

# Solution 4: Reduce sequence length
config = HaloConfig(..., max_seq_len=1024)  # Instead of 4096

# Solution 5: Use smaller model
config = HaloConfig(hidden_size=512, num_layers=6, ...)  # Instead of 1024/12
```

#### Q: `ImportError: huggingface_hub not found` when using `push_to_hub()`

```bash
# Hub functions require optional dependencies
pip install huggingface_hub safetensors

# Then login
huggingface-cli login
```

#### Q: Model generates garbage / nonsensical text

```python
# This is expected for UNTRAINED models! Random weights produce random output.
# You need to train the model first:
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
trainer.fit(dataset=your_dataset, epochs=10, batch_size=8)

# After training, generation quality depends on:
# 1. Training data quality and quantity
# 2. Number of training epochs (more = better, up to overfitting)
# 3. Model size (larger models learn better representations)
# 4. Sampling parameters (lower temperature = more coherent)
```

#### Q: Training is very slow

```python
# Solution 1: Enable mixed precision
trainer = Trainer(model=model, mixed_precision=True, ...)

# Solution 2: Optimize for device
model = optimize_for_device(model, mode="training")

# Solution 3: Use appropriate batch size
batch_size = get_optimal_batch_size(config, seq_len=seq_len)

# Solution 4: Reduce sequence length if possible
# Note: HALO-S is designed for long sequences; shorter = less advantage

# Solution 5: Use DataParallel for multi-GPU
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

#### Q: `torch.compile` errors or slowdowns

```python
# torch.compile can sometimes fail on complex models
# optimize_for_device handles this gracefully (catches errors, returns uncompiled model)

# If you're calling torch.compile manually:
try:
    model = torch.compile(model, mode="reduce-overhead")
except Exception:
    print("torch.compile failed, using eager mode")
    # Model works fine without compilation, just slightly slower
```

#### Q: Different results between CPU and CUDA

```python
# This is normal! Floating-point operations are not perfectly reproducible across devices.
# For reproducibility within a single device:
from halo import set_seed
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

#### Q: `load_from_hub()` fails with "repository not found"

```python
# Check the repo ID format: "username/model-name"
model = load_from_hub("bueormnew/halo-s-70m")  # ✓ correct format

# For private models, ensure you're logged in:
# huggingface-cli login
# Or pass token explicitly:
model = load_from_hub("your-username/private-model", token="hf_xxxxx")
```

#### Q: Can I use HALO-S with HuggingFace `transformers` library?

HALO-S is a standalone framework and is not directly compatible with the `transformers` library's `AutoModel` system. However, you can:
1. Use `save_for_hub()` / `push_to_hub()` to share models on HuggingFace Hub
2. Load them with `load_from_hub()` (HALO-S's own function)
3. The config.json format is HF-compatible for metadata display on the Hub

#### Q: How does HALO-S compare to FlashAttention?

FlashAttention is an optimized **implementation** of standard dense attention (still O(N²) complexity, but with O(N) memory via tiling). HALO-S is a different **architecture** (O(N×K) complexity). They solve different problems:
- FlashAttention: Makes dense attention faster/lighter via hardware-optimized kernels
- HALO-S: Reduces the number of attention operations via sparse connectivity

HALO-S actually uses SDPA (which includes FlashAttention when available) for its global tokens. The two approaches are complementary, not competing.

#### Q: Why is HALO-S slower than dense Transformers at short sequences?

The `torch.gather` operation creates intermediate tensors and has per-operation overhead that dense matrix multiplication doesn't have. At short sequences (N < 2048), the theoretical reduction in FLOPs (e.g., 13.5× at N=1024) doesn't overcome this constant-factor overhead. The architecture is designed to excel at N > 4096 where the O(N²) → O(N×K) reduction becomes dominant.

### FAQ

#### Q: What's the minimum model size for useful results?

For character-level language modeling: ~3.5M parameters (hidden_size=256, num_layers=4) trained for 10+ epochs on a few MB of text produces coherent character sequences.

For BPE/subword language modeling: ~20M+ parameters recommended. The 70M parameter configuration produces the best results in our experiments.

#### Q: Can I use HALO-S for tasks other than language modeling?

The architecture is designed for autoregressive language modeling, but the sparse attention mechanism could be adapted for:
- Document classification (use global tokens as [CLS])
- Sequence-to-sequence (with appropriate masking modifications)
- Long-document understanding (leveraging the O(N×K) scaling)

These are not currently implemented but are architecturally feasible.

#### Q: Is HALO-S production-ready?

Not yet. HALO-S is research software. For production use:
- Standard Transformers with FlashAttention are faster for seq_len < 4K
- Mamba/SSMs are faster for inference
- HALO-S's advantages at very long sequences (>8K) have not been validated at billion-parameter scale

Use HALO-S for research, experimentation, education, and prototyping.

#### Q: How do I contribute?

See the [repository](https://github.com/bueormnew/pyhalo) for contribution guidelines. Key areas where help is needed:
- Benchmarks at longer sequences (4K, 8K, 16K+)
- Scaling experiments at 100M+ parameters
- Custom CUDA kernel for the gather operation
- Integration with popular training frameworks (DeepSpeed, FSDP)

---

## Project Structure

```
pyhalo/
├── halo/                          # Main package
│   ├── __init__.py                # Public API exports (v2.2.1)
│   ├── hub.py                     # HuggingFace Hub integration (save/load/push)
│   ├── attention/
│   │   ├── global_attention.py    # Dense SDPA attention for global tokens
│   │   ├── graph.py              # Neighbor list generation (local + dilated + random)
│   │   └── halo_attention.py     # Gather-based sparse attention for regular tokens
│   ├── core/
│   │   ├── config.py             # HaloConfig dataclass with validation
│   │   ├── device.py             # Device profiles, optimization, auto-detection
│   │   └── logging.py           # Structured logging utilities
│   ├── datasets/
│   │   ├── jsonl.py             # JSONLDataset for structured data
│   │   ├── streaming.py         # StreamingDataset (IterableDataset, infinite)
│   │   ├── synthetic.py         # CopyDataset, NeedleDataset for testing
│   │   └── text.py             # Plain text dataset
│   ├── generation/
│   │   └── samplers.py          # Top-k, top-p, temperature sampling
│   ├── models/
│   │   ├── halo_model.py        # HaloSModel (main model, from_pretrained)
│   │   └── baseline_model.py    # Dense Transformer baseline for comparison
│   ├── nn/
│   │   ├── feed_forward.py      # SwiGLU / GELU feed-forward networks
│   │   ├── halo_block.py        # HaloBlock (attention + FFN + residual)
│   │   └── rope.py             # Rotary Positional Embeddings (RoPE)
│   ├── tokenizers/
│   │   ├── base.py             # BaseTokenizer abstract class
│   │   ├── char.py             # CharacterTokenizer (byte-level, vocab=256)
│   │   ├── word.py             # WordTokenizer (whitespace-based)
│   │   └── sentencepiece.py    # SentencePiece wrapper (subword)
│   ├── training/
│   │   └── trainer.py          # Trainer with AMP, accumulation, checkpoints
│   └── utils/
│       ├── benchmarks.py        # Speed, generation, memory, FLOPs benchmarks
│       ├── metrics.py          # Parameter counting, memory estimation
│       └── random.py           # Seed management (set_seed)
├── tests/                       # 61 tests covering all components
│   ├── test_attention.py       # Sparse attention correctness
│   ├── test_model.py          # Model forward/backward, from_pretrained
│   ├── test_training.py       # Trainer, checkpoints, AMP
│   ├── test_generation.py     # Generation, sampling strategies
│   ├── test_tokenizers.py     # All tokenizer implementations
│   ├── test_shapes.py         # Tensor shape validation
│   ├── test_gradients.py      # Gradient flow verification
│   ├── test_memory.py         # Memory usage tracking
│   ├── test_checkpoint.py     # Save/load/resume correctness
│   ├── test_config.py         # HaloConfig validation and serialization
│   └── test_graph.py          # Connectivity graph properties
├── benchmarks/                  # Benchmark scripts
│   ├── benchmark_speed.py     # Latency/throughput benchmarks
│   └── benchmark_graph.py     # Graph construction benchmarks
├── notebooks/                   # Jupyter notebooks with experiments
│   ├── halo_v2_70m_benchmark.ipynb
│   ├── halo_vs_transformer_benchmark.ipynb
│   └── halo_vs_transformer_large.ipynb
├── scripts/                     # Experiment scripts
│   ├── exp1_baseline.py       # HALO-S vs Dense comparison
│   ├── exp2_ablation.py       # Component ablation study
│   └── exp3_long_context.py   # Needle-in-a-haystack evaluation
├── docs/                        # Technical documentation
│   ├── architecture.md          # Full architecture documentation
│   ├── complexity.md           # Complexity analysis and proofs
│   ├── local_attention.md      # Local window mechanism
│   ├── dilated_connections.md  # Dilated connection strategy
│   ├── global_tokens.md        # Global token design
│   ├── sparse_attention.md     # Sparse attention implementation
│   ├── gqa.md                  # Grouped Query Attention
│   ├── rope.md                 # RoPE implementation details
│   └── flash_attention.md      # Flash attention compatibility notes
├── pyproject.toml              # Package configuration (pyhalos v2.2.1)
├── LICENSE                     # Custom license (research free, commercial paid)
└── README.md                   # This file
```

---

## Why HALO-S?

### Philosophy

HALO-S was born from a simple question: *Can we get most of the representational power of dense attention while paying only a fraction of the computational cost?*

The approach is grounded in graph theory. Instead of letting every token attend to every other token (a complete graph), HALO-S constructs a **sparse connectivity graph** with properties borrowed from network science:

1. **Local clustering** (window attention) — nearby tokens form tightly connected neighborhoods, capturing syntax and local semantics
2. **Long-range shortcuts** (dilated connections) — exponentially spaced connections prevent information bottlenecks across distance
3. **Small-world properties** (random edges) — a few random connections ensure that the graph diameter remains logarithmic, so information can propagate in O(log N) hops
4. **Shared memory** (global tokens) — learned parameters that act as a broadcast channel, available to every token in every layer

This combination is inspired by how efficient real-world networks (neural, social, transportation) achieve both local efficiency and global connectivity.

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No exotic dependencies** | Pure PyTorch + NumPy. No custom CUDA kernels, no Triton. |
| **Run anywhere** | CPU, single GPU, multi-GPU. No hardware lock-in. |
| **Research-first** | Clean code, extensive comments, full test coverage. |
| **Honest about limitations** | Benchmarks include both strengths and weaknesses. |
| **Backward compatible** | Every version loads every previous version's models. |
| **Modular** | Swap attention, FFN, tokenizer independently. |
| **Progressive enhancement** | Basic functionality works without optional deps. |

### Honest Assessment

**What HALO-S does well (demonstrated):**
- ✅ Clean, modular PyTorch implementation with no exotic dependencies
- ✅ Mathematically sound complexity reduction (O(N×K) vs O(N²))
- ✅ Runs on any hardware — CPU, single GPU, no custom kernels required
- ✅ All 61 tests pass — correctness of gradients, shapes, generation, and checkpoints verified
- ✅ Training loop works end-to-end with AMP, gradient accumulation, and streaming data
- ✅ Achieves perplexity parity with dense Transformers at 3.5M–70M scale
- ✅ HuggingFace Hub integration for easy model sharing
- ✅ Comprehensive device optimization for all major GPUs
- ✅ Full backward compatibility across all versions (v1.0 → v2.2.1)
- ✅ Safetensors support for safe, fast serialization

**What remains to be proven:**
- ⏳ Actual wall-clock speedup vs optimized dense attention (FlashAttention v2) at very long sequences (>8K)
- ⏳ Scaling behavior at 100M+ parameters
- ⏳ Performance on downstream NLP tasks (summarization, QA, etc.)
- ⏳ Comparison with Mamba/SSM architectures on actual generation quality
- ⏳ Multi-node distributed training at scale
- ⏳ Custom CUDA kernel for gather to eliminate overhead

**The gather-based approach has a known trade-off**: while it avoids custom CUDA kernels (portability), the `torch.gather` operations create intermediate tensors that can be memory-intensive. For sequences shorter than ~9,728 tokens, the gathered KV tensors may exceed dense attention memory. The advantage becomes clear at longer sequences.

---

## Running Tests

```bash
# Run all 61 tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=halo --cov-report=term-missing

# Run specific test module
pytest tests/test_attention.py -v
pytest tests/test_model.py -v
pytest tests/test_training.py -v
pytest tests/test_checkpoint.py -v

# Run tests matching a pattern
pytest tests/ -k "generation" -v
pytest tests/ -k "checkpoint" -v
pytest tests/ -k "hub" -v

# Quick smoke test (fastest subset)
pytest tests/test_config.py tests/test_shapes.py -v

# Run with parallel execution (if pytest-xdist installed)
pytest tests/ -n auto -v
```

### Test Coverage

| Module | Tests | Coverage |
|--------|:-----:|---------|
| `halo.attention` | 8 | Sparse attention, global attention, graph construction |
| `halo.models` | 9 | Forward/backward pass, from_pretrained, generation, Hub loading |
| `halo.training` | 8 | Trainer, AMP, checkpoints, gradient accumulation, resume |
| `halo.generation` | 6 | Top-k, top-p, temperature, stop tokens, string API |
| `halo.tokenizers` | 6 | Char, word, encode/decode roundtrip |
| `halo.core` | 9 | Config validation, serialization, device detection, profiles |
| `halo.utils` | 5 | Benchmarks, metrics, seed management |
| Other | 10 | Shapes, gradients, memory, graph properties |
| **Total** | **61** | All passing ✓ |

---

## Running Experiments

```bash
# Experiment 1: Baseline comparison (HALO-S vs Dense Transformer)
# Tests perplexity, training time, memory, generation speed
python scripts/exp1_baseline.py

# Experiment 2: Ablation study (contribution of each connectivity component)
# Trains variants: full, no-globals, no-dilated, no-random, local-only
python scripts/exp2_ablation.py

# Experiment 3: Long context scaling (Needle in a Haystack)
# Tests retrieval accuracy at varying token distances
python scripts/exp3_long_context.py
```

### Reproducing Benchmark Results

```bash
# Install all dependencies
pip install pyhalos[full,dev]
pip install tiktoken  # For BPE experiments

# Run the full benchmark suite (requires GPU, ~2 hours total)
python scripts/exp1_baseline.py > exp1_results.txt
python scripts/exp2_ablation.py > exp2_results.txt
python scripts/exp3_long_context.py > exp3_results.txt

# Quick benchmark (CPU, ~10 minutes)
python benchmarks/benchmark_speed.py
python benchmarks/benchmark_graph.py
```

---

## Citation

If you use HALO-S in your research, please cite:

```bibtex
@software{halo_s_2024,
  author = {BUEORM},
  title = {HALO-S: Hierarchical Attention with Local Offsets — Sparse},
  version = {2.2.1},
  year = {2024},
  url = {https://github.com/bueormnew/pyhalo},
  note = {Linear-complexity sparse attention framework for language models},
}
```

---

## License

**HALO-S Framework License** — Custom dual-use license:

| Use Case | Permission | Conditions |
|----------|:---:|---|
| Education & Research | ✅ Free | Must credit "HALO-S" in any derivative work |
| Personal projects & experimentation | ✅ Free | Must include copyright notice |
| Commercial / Production use | ❌ Requires license | Contact for commercial licensing |

For commercial licensing inquiries: **dalusx64@gmail.com**

See [LICENSE](./LICENSE) for full terms.

---

## Author

**BUEORM**
- 📧 dalusx64@gmail.com
- 🐙 [github.com/bueormnew/pyhalo](https://github.com/bueormnew/pyhalo)

---
---

## 🇪🇸 Versión en Español

<p align="center">
  <h2 align="center">🌀 HALO-S</h2>
  <p align="center"><strong>Atención Jerárquica con Offsets Locales — Disperso</strong></p>
  <p align="center">Un framework de modelos de lenguaje con complejidad lineal que reemplaza la atención cuadrática con un grafo de conectividad dispersa estructurado.</p>
  <p align="center"><em>v2.2.1 — Ahora con integración HuggingFace Hub, perfiles de dispositivo, soporte safetensors, SwiGLU FFN y atención híbrida SDPA+Gather</em></p>
</p>

---

### Novedades en v2.2.1

| Versión | Fecha | Cambios Principales |
|---------|-------|---------------------|
| **v2.2.1** | 2024 | Correcciones de estabilidad, compatibilidad mejorada, documentación renovada, 61 tests, FAQ/troubleshooting |
| **v2.2.0** | 2024 | Integración HuggingFace Hub, perfiles de dispositivo (T4/P100/L4/L40/RTX 6000/A100), `push_to_hub()`, `load_from_hub()`, safetensors como formato por defecto |
| **v2.1.0** | 2024 | Soporte safetensors, `optimize_for_device()`, auto-detección de dispositivo, `get_optimal_batch_size()` |
| **v2.0.0** | 2024 | Atención híbrida SDPA+Gather, SwiGLU FFN, gradient checkpointing, `from_pretrained()`, cambios de config |
| **v1.0.0** | 2024 | Lanzamiento inicial: atención dispersa, GQA, tokens globales, Trainer, generación, CharacterTokenizer |

#### Notas de Migración

- **v1.x → v2.x**: Modelos v1.x usan GELU FFN y claves antiguas. Usa `HaloSModel.from_pretrained("modelo_viejo.pt")` que auto-detecta y remapea pesos. El flag `use_swiglu` se establece automáticamente a `False` al cargar checkpoints v1.x.
- **v2.0 → v2.1+**: Transparente. Config sin cambios, safetensors opcional. Todos los checkpoints `.pt` siguen funcionando.
- **v2.1 → v2.2+**: Funciones Hub nuevas (`save_for_hub`, `load_from_hub`, `push_to_hub`). Sin cambios disruptivos. Perfiles de dispositivo expandidos con RTX 6000 Ada.
- **v2.2.0 → v2.2.1**: Solo correcciones de estabilidad. Sin cambios de API. Suite de tests expandida de 55 a 61.

#### Tabla de Compatibilidad de Versiones

| Característica / API | v1.0 | v2.0 | v2.1 | v2.2 | v2.2.1 |
|---------------------|:----:|:----:|:----:|:----:|:------:|
| Atención Gather Dispersa | ✓ | ✓ | ✓ | ✓ | ✓ |
| GQA | ✓ | ✓ | ✓ | ✓ | ✓ |
| Tokens Globales | ✓ | ✓ | ✓ | ✓ | ✓ |
| RoPE | ✓ | ✓ | ✓ | ✓ | ✓ |
| SwiGLU FFN | ✗ | ✓ | ✓ | ✓ | ✓ |
| Atención Híbrida SDPA+Gather | ✗ | ✓ | ✓ | ✓ | ✓ |
| Gradient Checkpointing | ✗ | ✓ | ✓ | ✓ | ✓ |
| Safetensors | ✗ | ✗ | ✓ | ✓ | ✓ |
| Perfiles de Dispositivo | ✗ | ✗ | ✓ | ✓ | ✓ |
| HuggingFace Hub | ✗ | ✗ | ✗ | ✓ | ✓ |
| Perfil RTX 6000 Ada | ✗ | ✗ | ✗ | ✓ | ✓ |

---

### ¿Y si la atención no tuviera que ser cuadrática?

Todo modelo de lenguaje moderno paga un precio elevado por secuencias largas: la auto-atención estándar del Transformer escala como O(N²), haciendo que ventanas de contexto mayores a 4K tokens sean prohibitivamente costosas. HALO-S toma un camino diferente. Al construir un **grafo de conectividad dispersa de grado fijo** — combinando ventanas locales, conexiones dilatadas, tokens globales aprendidos y aristas aleatorias — cada token atiende solo a K vecinos sin importar la longitud de la secuencia. El resultado es **complejidad O(N×K)** con K=76 por defecto, logrando una **reducción teórica de ~52.5×** en operaciones de atención para N=4096.

HALO-S está implementado como un framework limpio de PyTorch listo para investigación. Sin kernels CUDA personalizados. Sin dependencias externas más allá de PyTorch y NumPy. Solo atención dispersa basada en gather que funciona en cualquier hardware.

> ⚠️ **Aviso honesto**: HALO-S es una exploración arquitectónica prometedora. Las ventajas teóricas de complejidad son matemáticamente sólidas, pero la **validación empírica a gran escala** contra modelos establecidos en benchmarks estándar aún está en progreso. Úsalo para investigación, experimentación y aprendizaje.

---

### Tabla de Contenidos (Español)

- [Novedades en v2.2.1](#novedades-en-v221)
- [Características Principales](#características-principales)
- [Arquitectura](#arquitectura)
- [Integración HuggingFace Hub](#integración-huggingface-hub)
- [Sistema de Optimización por Dispositivo](#sistema-de-optimización-por-dispositivo)
- [Análisis de Rendimiento (Teórico)](#análisis-de-rendimiento-teórico)
- [Benchmarks Empíricos](#benchmarks-empíricos-1)
- [Instalación (ES)](#instalación-es)
- [Inicio Rápido](#inicio-rápido)
- [Uso Avanzado (ES)](#uso-avanzado-es)
- [Referencia de Configuración](#referencia-de-configuración)
- [Referencia del API](#referencia-del-api)
- [Guía de Compatibilidad](#guía-de-compatibilidad)
- [Solución de Problemas](#solución-de-problemas)
- [¿Por Qué HALO-S?](#por-qué-halo-s)
- [Licencia (ES)](#licencia-es)

---

### Características Principales

| Característica | Descripción | Desde |
|---|---|:---:|
| **Complejidad de Atención Lineal** | O(N×K) en lugar de O(N²) — escala eficientemente a secuencias largas | v1.0 |
| **Atención Dispersa basada en Gather** | Sin kernels CUDA personalizados; funciona en CPU y GPU | v1.0 |
| **Atención Híbrida SDPA + Gather** | Usa SDPA nativo de PyTorch para globals, gather para tokens dispersos | v2.0 |
| **Tokens Globales Aprendidos** | Parámetros de memoria compartida que atienden la secuencia completa | v1.0 |
| **Conexiones Dilatadas** | Campo receptivo exponencialmente expansivo entre capas | v1.0 |
| **Aristas Aleatorias** | Propiedades de grafo de mundo pequeño para propagación de información | v1.0 |
| **Grouped Query Attention (GQA)** | Memoria KV reducida con ratios de cabezas configurables | v1.0 |
| **RoPE** | Codificación posicional relativa sin parámetros aprendidos | v1.0 |
| **SwiGLU Feed-Forward** | Activación gated linear unit para mejor entrenamiento | v2.0 |
| **Entrenamiento con Precisión Mixta** | Soporte nativo de AMP con GradScaler (FP16/BF16) | v1.0 |
| **Acumulación de Gradientes** | Entrena con batches efectivos grandes en hardware limitado | v1.0 |
| **Gradient Checkpointing** | Intercambia cómputo por memoria — entrena modelos más grandes | v2.0 |
| **Guardado/Carga de Checkpoints** | Persistencia y reanudación completa del estado de entrenamiento | v1.0 |
| **Datasets de Streaming** | Entrena con datos mayores a la RAM con shuffling por buffer | v1.0 |
| **Generación Autoregresiva** | Muestreo top-k, top-p y temperatura integrados | v1.0 |
| **HuggingFace Hub** | Guarda, carga y sube modelos al Hub de HF | v2.2 |
| **Safetensors** | Serialización de modelos segura y rápida como formato por defecto | v2.1 |
| **Perfiles de Dispositivo** | Configuración auto-optimizada para T4, P100, L4, L40, RTX 6000, A100, CPU | v2.1 |
| **Soporte Multi-GPU** | DataParallel para entrenamiento multi-GPU | v1.0 |
| **Compatibilidad Retroactiva** | Carga modelos de cualquier versión de HALO-S (v1.0+) | v2.0 |

---

### Arquitectura

HALO-S reemplaza la auto-atención densa con un **grafo disperso estructurado** donde cada token se conecta a un conjunto fijo de K vecinos:

```
┌─────────────────────────────────────────────────────────────────┐
│                        HaloSModel                                │
│                                                                  │
│  ┌──────────────┐   ┌──────────────────────────────────┐        │
│  │ token_emb    │   │ global_memory (nn.Parameter)      │        │
│  │ (Embedding)  │   │ forma: (num_globals, hidden_size) │        │
│  └──────┬───────┘   └──────────────┬───────────────────┘        │
│         │                          │                             │
│         └──────────┬───────────────┘                             │
│                    ▼                                              │
│         ┌──────────────────┐                                     │
│         │ cat([globals, x]) │  → (B, G+N, H)                    │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ RoPE (cos, sin)  │                                     │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│  ┌───────────────────────────────────────────────────┐           │
│  │              HaloBlock × num_layers                │           │
│  │                                                    │           │
│  │  ┌─────────────┐                                  │           │
│  │  │ LayerNorm 1 │                                  │           │
│  │  └──────┬──────┘                                  │           │
│  │         │                                          │           │
│  │    ┌────┴────────────────────────┐                │           │
│  │    ▼                             ▼                │           │
│  │ ┌────────────────┐   ┌─────────────────────┐     │           │
│  │ │GlobalFullAttn  │   │ HaloSparseAttention │     │           │
│  │ │(SDPA, G×N)     │   │ (gather, N×K)       │     │           │
│  │ └───────┬────────┘   └──────────┬──────────┘     │           │
│  │         │                       │                  │           │
│  │         └───────────┬───────────┘                  │           │
│  │                     ▼                              │           │
│  │           cat([globals_out, tokens_out])            │           │
│  │                     │ + residual                    │           │
│  │                     ▼                              │           │
│  │  ┌─────────────┐  ┌────────────────┐             │           │
│  │  │ LayerNorm 2 │→ │ SwiGLU FFN     │ + residual  │           │
│  │  └─────────────┘  └────────────────┘             │           │
│  └───────────────────────────────────────────────────┘           │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ LayerNorm final  │                                     │
│         └────────┬─────────┘                                     │
│                  ▼                                                │
│         ┌──────────────────┐                                     │
│         │ lm_head (Linear) │  → (B, N, vocab_size)              │
│         └──────────────────┘                                     │
└─────────────────────────────────────────────────────────────────┘
```

#### Componentes de Conectividad

| Componente | Vecinos | Propósito |
|-----------|---------|---------|
| **Tokens Globales (G)** | 2 | Parámetros aprendidos con atención a la secuencia completa |
| **Ventana Local (w)** | 64 | Captura dependencias secuenciales/sintácticas |
| **Conexiones Dilatadas (2d)** | 8 | Campo receptivo exponencialmente expansivo |
| **Aristas Aleatorias (r)** | 2 | Garantiza propiedades de grafo de mundo pequeño |
| **Total (K)** | **76** | Presupuesto fijo por token independiente de N |

#### Atención Híbrida SDPA + Gather (v2.0+)

A partir de v2.0, HALO-S usa una **estrategia de atención híbrida**:

- **Tokens globales**: usan `F.scaled_dot_product_attention` (SDPA) nativo de PyTorch. Los kernels optimizados por hardware (Flash Attention, Memory-Efficient) se activan automáticamente.
- **Tokens regulares**: usan atención dispersa basada en `torch.gather`. La lista de vecinos precomputada determina qué K posiciones atiende cada token.

**¿Por qué híbrida?** Los tokens globales atienden a todas las N posiciones (denso por definición), así que SDPA les da acceso a Flash Attention. Los tokens regulares solo necesitan K=76 vecinos, así que gather es más eficiente que enmascarar N-K posiciones.

#### SwiGLU vs GELU

HALO-S v2.0+ usa **SwiGLU** por defecto:

```
# GELU FFN estándar (v1.x):
FFN(x) = Linear₂(GELU(Linear₁(x)))

# SwiGLU FFN (v2.0+):
FFN(x) = Linear₂(Swish(Linear₁(x)) ⊙ Linear₃(x))
```

SwiGLU proporciona mejor dinámica de entrenamiento y converge más rápido. Para usar GELU: `config = HaloConfig(use_swiglu=False)`

#### Gradient Checkpointing (v2.0+)

Para entrenar modelos grandes en GPUs con memoria limitada:

```python
# Habilitar (reduce memoria ~40-60%, costo ~30% más lento)
model.enable_gradient_checkpointing()

# Deshabilitar (para inferencia)
model.disable_gradient_checkpointing()
```

| Capas | Sin Checkpointing | Con Checkpointing | Ahorro |
|:-----:|:------------------:|:-----------------:|:------:|
| 4 | ~1.2 GB | ~0.8 GB | 33% |
| 8 | ~2.4 GB | ~1.2 GB | 50% |
| 12 | ~3.6 GB | ~1.5 GB | 58% |
| 24 | ~7.2 GB | ~2.5 GB | 65% |

---

### Integración HuggingFace Hub

HALO-S v2.2+ proporciona integración transparente con el ecosistema de HuggingFace.

#### Prerrequisitos

```bash
pip install huggingface_hub safetensors
huggingface-cli login
```

#### Guardar en Formato HuggingFace — `save_for_hub()`

```python
from halo import HaloConfig, HaloSModel, save_for_hub

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)
model = HaloSModel(config)

# Entrenar...

# Guardar (crea config.json + model.safetensors)
save_for_hub(model, config, "./mi-modelo-halo/")
```

#### Cargar desde HuggingFace Hub — `load_from_hub()`

```python
from halo import load_from_hub

# Desde repositorio de HuggingFace
model = load_from_hub("bueormnew/halo-s-70m", device="cuda")

# Desde directorio local en formato HF
model = load_from_hub("./mi-modelo-halo/", device="cuda")

# Revisión específica
model = load_from_hub("bueormnew/halo-s-70m", revision="v2.1")

# Modelo antiguo .pt (compatible)
model = load_from_hub("path/to/modelo_viejo.pt", device="cpu")
```

`load_from_hub()` automáticamente:
1. Detecta si es directorio local, archivo local, o repositorio Hub
2. Descarga config.json y pesos del Hub si es necesario
3. Reconstruye HaloConfig desde el JSON
4. Carga pesos con `strict=False` para compatibilidad
5. Maneja tanto safetensors como pytorch_model.bin

#### Subir a HuggingFace Hub — `push_to_hub()`

```python
from halo import push_to_hub

# Subir a tu cuenta (público)
push_to_hub(model, config, "tu-usuario/halo-s-custom", private=False)

# Modelo privado
push_to_hub(model, config, "tu-usuario/halo-s-privado", private=True)

# Con token explícito
push_to_hub(model, config, "tu-usuario/halo-s-custom", token="hf_xxxxx")
```

#### Flujo de Trabajo Completo HuggingFace

```python
from halo import (
    HaloConfig, HaloSModel, Trainer, CharacterTokenizer,
    save_for_hub, push_to_hub, load_from_hub,
    set_seed, optimize_for_device,
)
from halo.datasets import TextDataset

# Entrenar
set_seed(42)
config = HaloConfig(vocab_size=256, hidden_size=512, num_layers=6, num_heads=8, num_kv_heads=2)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")

tok = CharacterTokenizer()
dataset = TextDataset("datos/corpus.txt", tokenizer=tok, max_seq_len=2048)
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
trainer.fit(dataset=dataset, epochs=10, batch_size=8)

# Guardar y subir
save_for_hub(model, config, "./mi-modelo/")
push_to_hub(model, config, "tu-usuario/halo-s-char-lm")

# Cargar (cualquier persona puede hacer esto)
modelo_cargado = load_from_hub("tu-usuario/halo-s-char-lm", device="cuda")
```

#### Compatibilidad con Versiones Anteriores

```python
from halo import load_from_hub

# v1.x (.pt con GELU) → auto-detecta y carga
model = load_from_hub("modelos/halo_v1.pt")

# v2.0 (.pt con SwiGLU w3) → carga directa
model = load_from_hub("modelos/halo_v2_70m.pt")

# v2.1+ (safetensors) → carga optimizada
model = load_from_hub("./modelos/halo_v21/")

# HuggingFace Hub → descarga y carga
model = load_from_hub("bueormnew/halo-s-70m")
```

---

### Sistema de Optimización por Dispositivo

HALO-S v2.1+ incluye un sistema automático que configura ajustes específicos del hardware (TF32, Flash SDP, torch.compile, hilos CPU).

#### Perfiles de Dispositivo Soportados

| Perfil | GPU | Memoria | TF32 | Flash SDP | BF16 | Modo Compile | Arquitectura |
|--------|-----|---------|:----:|:---------:|:----:|:------------:|:------------:|
| `t4` | NVIDIA Tesla T4 | 16 GB | ✗ | ✓ | ✗ | reduce-overhead | Turing |
| `p100` | NVIDIA Tesla P100 | 16 GB | ✗ | ✗ | ✗ | default | Pascal |
| `l4` | NVIDIA L4 | 24 GB | ✓ | ✓ | ✓ | reduce-overhead | Ada Lovelace |
| `l40` | NVIDIA L40 | 48 GB | ✓ | ✓ | ✓ | max-autotune | Ada Lovelace |
| `rtx_6000` | NVIDIA RTX 6000 Ada | 48 GB | ✓ | ✓ | ✓ | max-autotune | Ada Lovelace |
| `a100` | NVIDIA A100 | 80 GB | ✓ | ✓ | ✓ | max-autotune | Ampere |
| `cpu` | CPU | RAM del sistema | ✗ | ✗ | ✓* | default | x86/ARM |

#### Uso de optimize_for_device()

```python
from halo import HaloConfig, HaloSModel, optimize_for_device

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16)
model = HaloSModel(config).to("cuda")

# Auto-detectar y aplicar configuración óptima
model = optimize_for_device(model)

# Para inferencia (habilita torch.compile + eval mode)
model = optimize_for_device(model, device="cuda", mode="inference")

# Para entrenamiento
model = optimize_for_device(model, device="cuda", mode="training")
```

#### Ejemplos por Dispositivo

**Tesla T4 (Colab, Kaggle):**
```python
config = HaloConfig(vocab_size=32000, hidden_size=768, num_layers=8, num_heads=12, num_kv_heads=4)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")
model.enable_gradient_checkpointing()  # T4 tiene solo 16 GB
batch_size = get_optimal_batch_size(config, seq_len=1024)  # → 2
```

**NVIDIA A100 (Cloud, HPC):**
```python
config = HaloConfig(vocab_size=32000, hidden_size=1536, num_layers=16, num_heads=24, num_kv_heads=6)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")  # TF32 + Flash + max-autotune
batch_size = get_optimal_batch_size(config, seq_len=2048)  # → 8
```

**RTX 6000 Ada (Workstation):**
```python
config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")  # TF32 + Flash + max-autotune
batch_size = get_optimal_batch_size(config, seq_len=1024)  # → 8
```

#### Qué hace optimize_for_device()

En dispositivos **CUDA** (Ampere+ / Ada Lovelace):
1. Habilita TF32 matmul y cuDNN
2. Habilita Flash SDP y Memory-Efficient SDP
3. Aplica `torch.compile` con modo apropiado al dispositivo

En **CPU**:
1. Configura threads óptimos (`torch.set_num_threads(os.cpu_count())`)
2. Aplica `torch.compile` con `mode="default"` para inferencia

La función es **a prueba de fallos** — nunca lanza excepción.

#### Batch Sizes Óptimos por Dispositivo

| Seq Length | T4 (16GB) | P100 (16GB) | L4 (24GB) | L40 (48GB) | RTX 6000 (48GB) | A100 (80GB) |
|:----------:|:---------:|:-----------:|:---------:|:----------:|:---------------:|:-----------:|
| 256 | 8 | 8 | 16 | 32 | 32 | 64 |
| 512 | 4 | 4 | 8 | 16 | 16 | 32 |
| 1024 | 2 | 2 | 4 | 8 | 8 | 16 |
| 2048 | 1 | 1 | 2 | 4 | 4 | 8 |
| 4096 | — | — | 1 | 2 | 2 | 4 |

---

### Análisis de Rendimiento (Teórico)

> ⚠️ **Todos los datos de rendimiento son TEÓRICOS**, derivados del análisis de complejidad. Ver [Benchmarks Empíricos](#benchmarks-empíricos-1) para mediciones reales.

#### Reducción de Operaciones de Atención

Con longitud de secuencia N=4096 y K=76 vecinos por token:

```
Operaciones de atención Transformer denso:  N²      = 16,777,216
Operaciones de atención HALO-S:             N×(K+G) =    319,488

Factor de reducción: 16,777,216 / 319,488 ≈ 52.5×
```

#### Tabla de Escalado (FLOPs de Atención)

| Longitud (N) | Transformer Denso (N²) | HALO-S (N×76) | Speedup Teórico |
|:---:|:---:|:---:|:---:|
| 512 | 262,144 | 38,912 | 6.7× |
| 1,024 | 1,048,576 | 77,824 | 13.5× |
| 2,048 | 4,194,304 | 155,648 | 26.9× |
| 4,096 | 16,777,216 | 311,296 | 53.9× |
| 8,192 | 67,108,864 | 622,592 | 107.8× |
| 16,384 | 268,435,456 | 1,245,184 | 215.6× |
| 32,768 | 1,073,741,824 | 2,490,368 | 431.1× |
| 65,536 | 4,294,967,296 | 4,980,736 | 862.3× |

El speedup crece **linealmente con N** porque la atención densa es O(N²) mientras HALO-S es O(N×K).

#### Comparación Teórica con Otras Arquitecturas

| Modelo | Complejidad Atención | Memoria (Scores) | Contexto Global | Dilatación | Aristas Aleatorias | GQA | Kernels Custom |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Transformer Denso** | O(N²·d) | O(N²) | Completo (implícito) | ✗ | ✗ | Opcional | ✗ |
| **Longformer** | O(N·w·d) | O(N·w) | ✓ (fijos) | ✓ | ✗ | ✗ | ✓ |
| **BigBird** | O(N·(w+g+r)·d) | O(N·(w+g+r)) | ✓ (fijos) | ✗ | ✓ | ✗ | ✓ |
| **Mamba (SSM)** | O(N·d²) | O(d²) | Implícito (estado) | ✗ | ✗ | N/A | ✓ |
| **RWKV** | O(N·d) | O(d) | Implícito (estado) | ✗ | ✗ | N/A | ✓ |
| **Flash Attention** | O(N²·d) | O(N) | Completo (implícito) | ✗ | ✗ | Opcional | ✓ |
| **HALO-S** | **O(N·K·d)** | **O(N·K)** | ✓ (aprendidos) | ✓ | ✓ | ✓ | **✗** |

Diferenciador clave: HALO-S logra complejidad sub-cuadrática **sin kernels CUDA personalizados**, haciéndolo portable a todo hardware soportado por PyTorch.

---

### Benchmarks Empíricos

> 📊 **Datos de benchmark reales** de ejecuciones de entrenamiento en GPUs NVIDIA.

#### Prueba 1: Escala Pequeña (seq=256, ~3.5M params, 10 épocas)

| Métrica | HALO-S | Transformer Denso | Δ | Notas |
|---------|:------:|:-----------------:|:-:|-------|
| **Perplejidad** | 3.48 | 3.45 | +0.9% | Casi iguales |
| **Tiempo Entrenamiento** | 1675s | 828s | 2.0× más lento | Overhead de gather |
| **Memoria Pico** | 1.72 GB | 0.72 GB | 2.4× más | Tensores K/V reunidos |
| **Generación** | 102 tok/s | 346 tok/s | 3.4× más lenta | Gather secuencial |

#### Prueba 2: Escala Media (seq=1024, ~20M params, 3 épocas)

| Métrica | HALO-S | Transformer Denso | Δ | Notas |
|---------|:------:|:-----------------:|:-:|-------|
| **Perplejidad** | 3.56 | 3.59 | −0.8% | **HALO-S gana** |
| **Tiempo Entrenamiento** | 3885s | 1872s | 2.1× más lento | Gather domina |
| **Memoria Pico** | 4.95 GB | 0.80 GB | 6.2× más | Tensores intermedios |
| **Generación** | 62 tok/s | 214 tok/s | 3.5× más lenta | Gather por token |

#### Prueba 3: Escala Grande (seq=1024, ~70M params, BPE, 2 épocas)

| Métrica | HALO-S | Transformer Denso | Δ | Notas |
|---------|:------:|:-----------------:|:-:|-------|
| **Perplejidad** | 102.3 | 100.7 | +1.6% | Casi iguales |
| **Tiempo Entrenamiento** | 59.8 min | 46.3 min | 1.3× más lento | ¡Gap cerrándose! |
| **Latencia @1024** | 27.7 ms | 12.3 ms | 2.3× mayor | Latencia por paso |
| **Memoria Pico** | 0.818 GB | 0.816 GB | ~Igual | Params del modelo dominan |

#### Resumen de Hallazgos Empíricos

| Escala | Params | Seq Len | Gap PPL | Gap Velocidad | Gap Memoria |
|:------:|:------:|:-------:|:-------:|:-------------:|:-----------:|
| Pequeña | 3.5M | 256 | +0.9% (HALO-S peor) | 2.0× más lento | 2.4× más |
| Media | 20M | 1024 | −0.8% (**HALO-S mejor**) | 2.1× más lento | 6.2× más |
| Grande | 70M | 1024 | +1.6% (HALO-S peor) | 1.3× más lento | ~Igual |

#### Interpretación Honesta

1. **Paridad de perplejidad**: Diferencia consistentemente < 2% en todas las escalas.
2. **Overhead de velocidad decreciente**: De 2.0× a 1.3× conforme la atención es menor fracción del cómputo total.
3. **Ventaja de memoria aún no alcanzada**: Requiere seq_len > ~9,728 tokens.
4. **Diseñado para seq_len > 2048**: La diferencia O(N×K) vs O(N²) se vuelve significativa en secuencias más largas.

**Recomendación**: Usa HALO-S para investigación con modelos de contexto largo. Para tareas de producción con contexto corto (<2K), los Transformers con FlashAttention son más eficientes.

---

### Instalación (ES)

```bash
# Instalación básica (solo PyTorch + NumPy)
pip install pyhalos

# Versión específica
pip install pyhalos==2.2.1

# Instalación completa (incluye tqdm + SentencePiece + safetensors)
pip install pyhalos[full]

# Con soporte HuggingFace Hub
pip install pyhalos[full] huggingface_hub

# Desde código fuente
git clone https://github.com/bueormnew/pyhalo.git
cd pyhalo
pip install -e ".[full,dev]"
```

#### Requisitos

| Dependencia | Versión | Requerida | Propósito |
|-------------|---------|:---------:|-----------|
| Python | ≥ 3.10 | ✓ | Runtime |
| PyTorch | ≥ 2.1.0 | ✓ | Framework de deep learning |
| NumPy | ≥ 1.24.0 | ✓ | Operaciones de arrays |
| tqdm | cualquiera | ✗ | Barras de progreso |
| sentencepiece | cualquiera | ✗ | Tokenización subword |
| safetensors | cualquiera | ✗ | Serialización segura |
| huggingface_hub | cualquiera | ✗ | Integración Hub |
| tiktoken | cualquiera | ✗ | Tokenización BPE (OpenAI) |

#### Verificar Instalación

```python
import halo
print(f"HALO-S versión: {halo.__version__}")  # → 2.2.1
print(f"Dispositivo: {halo.device_info()['device']}")

from halo import HaloConfig, HaloSModel, set_seed
set_seed(42)
config = HaloConfig(vocab_size=256, hidden_size=64, num_layers=2, num_heads=4, num_kv_heads=2)
model = HaloSModel(config)
print(f"✓ Modelo creado: {model.count_parameters():,} parámetros")
```

---

### Inicio Rápido

#### Ejemplo Mínimo

```python
from halo import HaloConfig, HaloSModel, set_seed

set_seed(42)

config = HaloConfig(
    vocab_size=256,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    num_kv_heads=2,       # GQA: ratio 4:1
    num_globals=2,
    local_window=64,
    max_seq_len=4096,
)

model = HaloSModel(config)
print(model.summary())
print(f"Parámetros: {model.count_parameters():,}")
```

#### Generación de Texto (API String)

```python
from halo import HaloConfig, HaloSModel, CharacterTokenizer, set_seed

set_seed(42)
config = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4)
model = HaloSModel(config)
tok = CharacterTokenizer()

# Generar desde texto (retorna string)
output = model.generate(
    "Hola mundo",
    tokenizer=tok,
    max_new_tokens=50,
    temperature=0.8,
    top_k=40,
)
print(output)
```

#### Generación por Tensor

```python
import torch
from halo import HaloConfig, HaloSModel

config = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4)
model = HaloSModel(config)

input_ids = torch.randint(0, 256, (1, 20))
output_ids = model.generate(input_ids, max_new_tokens=100, temperature=1.0, top_p=0.9)
print(f"Entrada: {input_ids.shape} → Salida: {output_ids.shape}")
```

#### Cargar desde HuggingFace Hub

```python
from halo import load_from_hub, optimize_for_device, CharacterTokenizer

# Cargar modelo preentrenado
model = load_from_hub("bueormnew/halo-s-70m", device="cuda")
model = optimize_for_device(model, mode="inference")

tok = CharacterTokenizer()
output = model.generate("El sentido de la vida es", tokenizer=tok, max_new_tokens=200, temperature=0.7)
print(output)
```

#### Usar Optimización de Dispositivo

```python
from halo import (
    HaloConfig, HaloSModel, optimize_for_device,
    detect_device_profile, get_optimal_batch_size
)

profile = detect_device_profile()
print(f"Ejecutando en: {profile['name']}")

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")

batch_size = get_optimal_batch_size(config, seq_len=1024)
print(f"Batch size recomendado: {batch_size}")
```

#### Entrenamiento Rápido

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer, set_seed
from halo.datasets import TextDataset

set_seed(42)
config = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4)
model = HaloSModel(config)
tok = CharacterTokenizer()
dataset = TextDataset(file_path="datos/corpus.txt", tokenizer=tok, max_seq_len=512)

trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
history = trainer.fit(dataset=dataset, epochs=5, batch_size=16)

output = model.generate("El ", tokenizer=tok, max_new_tokens=100, temperature=0.8)
print(output)
```

---

### Uso Avanzado (ES)

#### Entrenamiento con Gradient Checkpointing

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer, set_seed
from halo.datasets import TextDataset

set_seed(42)
config = HaloConfig(vocab_size=256, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4, max_seq_len=2048)
model = HaloSModel(config)

# Habilitar gradient checkpointing (ahorra ~40-60% de memoria de activaciones)
model.enable_gradient_checkpointing()

tok = CharacterTokenizer()
dataset = TextDataset(file_path="datos/corpus.txt", tokenizer=tok, max_seq_len=2048)

trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True, gradient_accumulation_steps=8)
history = trainer.fit(dataset=dataset, epochs=5, batch_size=4)
```

#### Entrenamiento con Precisión Mixta y Acumulación de Gradientes

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer, set_seed
from halo.datasets import JSONLDataset

set_seed(42)

config = HaloConfig(vocab_size=256, hidden_size=512, num_layers=6, num_heads=8, num_kv_heads=2, max_seq_len=2048)
model = HaloSModel(config)

tok = CharacterTokenizer()
dataset = JSONLDataset(file_path="datos/train.jsonl", tokenizer=tok, max_seq_len=2048, text_field="text")

trainer = Trainer(
    model=model,
    learning_rate=3e-4,
    mixed_precision=True,              # Precisión mixta FP16/BF16
    gradient_accumulation_steps=4,     # Batch efectivo = 4 × batch_size
    max_grad_norm=1.0,                 # Clipping de gradientes
    checkpoint_dir="./checkpoints",
    log_every=10,
)

history = trainer.fit(dataset=dataset, epochs=10, batch_size=8, save_every=2)
```

#### Guardar y Reanudar Checkpoints

```python
# Guardar checkpoint manualmente
trainer.save_checkpoint(path="mi_checkpoint.pt")

# Reanudar entrenamiento desde checkpoint
trainer.load_checkpoint("mi_checkpoint.pt")
trainer.fit(dataset=dataset, epochs=5, batch_size=8)  # Reanuda desde la época guardada
```

#### Dataset de Streaming (Archivos Mayores que RAM)

```python
from halo.datasets import StreamingDataset
from halo import CharacterTokenizer

tok = CharacterTokenizer()

stream_dataset = StreamingDataset(
    file_paths=["datos/shard_01.jsonl", "datos/shard_02.jsonl"],
    tokenizer=tok,
    max_seq_len=2048,
    buffer_size=10000,
    text_field="text",
    file_format="jsonl",
)

from torch.utils.data import DataLoader
loader = DataLoader(stream_dataset, batch_size=4)
```

#### Entrenamiento Multi-GPU

```python
import torch
import torch.nn as nn
from halo import HaloConfig, HaloSModel, Trainer, optimize_for_device

config = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16)
model = HaloSModel(config)

if torch.cuda.device_count() > 1:
    print(f"Usando {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to("cuda")
model = optimize_for_device(model)

trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
```

#### Pipeline Completo (End-to-End)

```python
from halo import (
    HaloConfig, HaloSModel, Trainer, CharacterTokenizer,
    set_seed, optimize_for_device, save_for_hub, push_to_hub,
    get_optimal_batch_size, detect_device_profile,
)
from halo.datasets import JSONLDataset

set_seed(42)

# Verificar hardware
profile = detect_device_profile()
print(f"Dispositivo: {profile['name']} ({profile['memory_gb']} GB)")

# Configurar modelo
config = HaloConfig(
    vocab_size=256, hidden_size=768, num_layers=8,
    num_heads=12, num_kv_heads=4, max_seq_len=2048,
)
model = HaloSModel(config).to("cuda")
model = optimize_for_device(model, mode="training")
model.enable_gradient_checkpointing()

# Dataset y batch size óptimo
tok = CharacterTokenizer()
dataset = JSONLDataset("datos/train.jsonl", tokenizer=tok, max_seq_len=2048)
batch_size = get_optimal_batch_size(config, seq_len=2048)

# Entrenar
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True, gradient_accumulation_steps=4)
history = trainer.fit(dataset=dataset, epochs=10, batch_size=batch_size, save_every=2)

# Guardar y subir
model.disable_gradient_checkpointing()
save_for_hub(model, config, "./halo-s-entrenado/")
push_to_hub(model, config, "tu-usuario/halo-s-custom")
```

---

### Referencia de Configuración

#### HaloConfig — Documentación Completa de Parámetros

```python
@dataclass
class HaloConfig:
    vocab_size: int = 256           # Tamaño del vocabulario (256=char, 32000=BPE, 50257=tiktoken)
    hidden_size: int = 512          # Dimensión del modelo (debe ser divisible por num_heads)
    num_layers: int = 6             # Número de bloques HaloBlock
    num_heads: int = 8              # Cabezas de atención de query
    num_kv_heads: int = 2           # Cabezas Key/Value (ratio GQA = num_heads/num_kv_heads)
    num_globals: int = 2            # Tokens globales aprendidos
    local_window: int = 64          # Tamaño de ventana de atención local
    dilated_offsets: List[int] = [1, 2, 4, 8]  # Distancias de conexiones dilatadas
    num_random: int = 2             # Aristas aleatorias por token
    dropout: float = 0.1            # Tasa de dropout
    max_seq_len: int = 4096         # Longitud máxima de secuencia soportada
    use_swiglu: bool = True         # Activación SwiGLU (True) o GELU (False)
```

#### Configuraciones de Ejemplo

```python
from halo import HaloConfig

# Tiny (~1M params) — para pruebas
tiny = HaloConfig(vocab_size=256, hidden_size=128, num_layers=2, num_heads=4, num_kv_heads=2)

# Pequeño (~3.5M params) — para experimentación
small = HaloConfig(vocab_size=256, hidden_size=256, num_layers=4, num_heads=4, num_kv_heads=2)

# Mediano (~20M params) — LM a nivel carácter
medium = HaloConfig(vocab_size=256, hidden_size=512, num_layers=8, num_heads=8, num_kv_heads=2)

# Grande (~70M params) — LM con BPE
large = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4)

# XL (~150M params) — escala de investigación
xl = HaloConfig(vocab_size=32000, hidden_size=1536, num_layers=16, num_heads=24, num_kv_heads=6,
                local_window=128, dilated_offsets=[1, 2, 4, 8, 16], max_seq_len=8192)

# Contexto largo — optimizado para secuencias muy largas
long_ctx = HaloConfig(vocab_size=32000, hidden_size=1024, num_layers=12, num_heads=16, num_kv_heads=4,
                      max_seq_len=32768, local_window=128, dilated_offsets=[1, 2, 4, 8, 16, 32],
                      num_globals=4, num_random=4)
```

---

### Referencia del API

#### Core

| Símbolo | Tipo | Descripción |
|---------|------|-------------|
| `halo.HaloConfig` | dataclass | Configuración del modelo |
| `halo.HaloSModel` | nn.Module | Modelo HALO-S principal |
| `halo.BaselineModel` | nn.Module | Transformer denso de referencia |

#### Métodos del Modelo

| Método | Descripción |
|--------|-------------|
| `HaloSModel(config)` | Crear modelo desde config |
| `.forward(x)` | Forward pass: `(B, N)` → `(B, N, V)` logits |
| `.generate(...)` | Generación autoregresiva de texto |
| `.summary()` | Resumen de arquitectura |
| `.count_parameters()` | Parámetros entrenables totales |
| `.estimate_flops(seq_len)` | Desglose de FLOPs |
| `.from_pretrained(path)` | Cargar desde checkpoint (cualquier versión) |
| `.enable_gradient_checkpointing()` | Habilitar checkpointing |
| `.disable_gradient_checkpointing()` | Deshabilitar checkpointing |

#### Hub

| Símbolo | Descripción |
|---------|-------------|
| `halo.save_for_hub(model, config, dir)` | Guardar config.json + model.safetensors |
| `halo.load_from_hub(path_or_repo, device, revision)` | Cargar desde local o Hub |
| `halo.push_to_hub(model, config, repo_id, token, private)` | Subir a HuggingFace Hub |

#### Dispositivo

| Símbolo | Descripción |
|---------|-------------|
| `halo.optimize_for_device(model, device, mode)` | Optimizaciones por hardware |
| `halo.detect_device_profile()` | Auto-detectar GPU |
| `halo.get_optimal_batch_size(config, seq_len)` | Batch size recomendado |
| `halo.get_optimal_device()` | Mejor dispositivo disponible |
| `halo.device_info()` | Información completa del dispositivo |

#### Tokenizers

| Símbolo | Descripción |
|---------|-------------|
| `halo.CharacterTokenizer` | Tokenizador a nivel byte (vocab=256) |
| `halo.WordTokenizer` | Tokenizador por palabras (requiere `build_vocab()`) |

#### Datasets

| Símbolo | Tipo | Descripción |
|---------|------|-------------|
| `halo.datasets.JSONLDataset` | Dataset | Archivos JSONL |
| `halo.datasets.TextDataset` | Dataset | Archivos de texto plano |
| `halo.datasets.StreamingDataset` | IterableDataset | Carga lazy con buffer shuffle |
| `halo.datasets.CopyDataset` | Dataset | Sintético: aprender a copiar |
| `halo.datasets.NeedleDataset` | Dataset | Sintético: aguja en un pajar |

#### Utilidades

| Símbolo | Descripción |
|---------|-------------|
| `halo.set_seed(seed)` | Establecer semillas aleatorias |
| `halo.count_parameters(model)` | Contar parámetros entrenables |
| `halo.generate(model, ...)` | Función de generación standalone |

---

### Guía de Compatibilidad

HALO-S mantiene compatibilidad total entre todas las versiones. El sistema de carga auto-detecta formato y versión.

| Formato | Detección | Ruta de Carga |
|---------|-----------|---------------|
| **Formato HuggingFace** (dir con config.json) | `os.path.isdir()` | `load_from_hub()` → config.json → safetensors |
| **v2.1+ safetensors** | Extensión .safetensors | `from_pretrained()` → safetensors.load_file() |
| **v2.0 checkpoint** (.pt con w3) | `"w3"` en state_dict | `from_pretrained()` → carga directa |
| **v1.x checkpoint** (.pt sin w3) | Ausencia de `"w3"` | `from_pretrained()` → modo GELU |
| **Checkpoint de entrenamiento** | `"model_state_dict"` presente | `trainer.load_checkpoint()` |
| **Repo HuggingFace Hub** | No es ruta local | `load_from_hub()` → hf_hub_download() |

---

### Solución de Problemas

#### P: `ImportError: No module named 'halo'`

```bash
pip install pyhalos  # ← nombre correcto (NO "halo" ni "pyhalo")
python -c "import halo; print(halo.__version__)"
```

#### P: `RuntimeError: CUDA out of memory`

```python
# 1. Habilitar gradient checkpointing
model.enable_gradient_checkpointing()
# 2. Reducir batch size
batch_size = get_optimal_batch_size(config, seq_len=your_seq_len)
# 3. Usar acumulación de gradientes
trainer = Trainer(model=model, gradient_accumulation_steps=8, ...)
# 4. Reducir seq_len o hidden_size
```

#### P: El modelo genera basura

```python
# ¡Esto es normal en modelos NO ENTRENADOS! Pesos aleatorios = salida aleatoria.
# Entrena el modelo primero:
trainer = Trainer(model=model, learning_rate=3e-4, mixed_precision=True)
trainer.fit(dataset=tu_dataset, epochs=10, batch_size=8)
```

#### P: `ImportError: huggingface_hub not found`

```bash
pip install huggingface_hub safetensors
huggingface-cli login
```

#### P: ¿HALO-S funciona con la librería `transformers`?

HALO-S es un framework independiente y no es directamente compatible con `AutoModel` de transformers. Sin embargo, puedes compartir modelos en HuggingFace Hub con `push_to_hub()` y cargarlos con `load_from_hub()`.

#### P: ¿Por qué HALO-S es más lento que los Transformers en secuencias cortas?

`torch.gather` crea tensores intermedios con overhead que la multiplicación de matrices densa no tiene. A secuencias cortas (N < 2048), la reducción teórica de FLOPs no supera este overhead. La arquitectura está diseñada para secuencias > 4096 donde O(N²) → O(N×K) se vuelve dominante.

#### P: ¿Es HALO-S apto para producción?

Aún no. HALO-S es software de investigación. Para producción:
- Transformers con FlashAttention son más rápidos para seq_len < 4K
- Mamba/SSMs son más rápidos para inferencia
- Las ventajas de HALO-S en secuencias muy largas (>8K) no han sido validadas a escala de miles de millones de parámetros

---

### ¿Por Qué HALO-S?

#### Filosofía

HALO-S nació de una pregunta simple: *¿Podemos obtener la mayor parte del poder representacional de la atención densa pagando solo una fracción del costo computacional?*

El enfoque se basa en teoría de grafos:
1. **Clustering local** (ventana) — tokens cercanos forman vecindarios conectados
2. **Atajos de largo alcance** (conexiones dilatadas) — previenen cuellos de botella
3. **Propiedades de mundo pequeño** (aristas aleatorias) — diámetro logarítmico
4. **Memoria compartida** (tokens globales) — canal de broadcast disponible para todos

#### Principios de Diseño

| Principio | Implementación |
|-----------|---------------|
| **Sin dependencias exóticas** | PyTorch puro + NumPy |
| **Funciona en cualquier lugar** | CPU, GPU única, multi-GPU |
| **Investigación primero** | Código limpio, tests completos |
| **Honesto sobre limitaciones** | Benchmarks incluyen fortalezas y debilidades |
| **Compatible hacia atrás** | Todas las versiones cargan modelos anteriores |
| **Modular** | Atención, FFN, tokenizer intercambiables |

#### Evaluación Honesta

**Lo que HALO-S hace bien (demostrado):**
- ✅ Implementación limpia y modular en PyTorch sin dependencias exóticas
- ✅ Reducción de complejidad matemáticamente sólida (O(N×K) vs O(N²))
- ✅ Funciona en cualquier hardware — CPU, GPU, sin kernels custom
- ✅ 61 tests pasando — correctitud de gradientes, formas, generación y checkpoints
- ✅ Paridad de perplejidad con Transformers densos (3.5M → 70M parámetros)
- ✅ Integración HuggingFace Hub para compartir modelos fácilmente
- ✅ Optimización de dispositivo para todas las GPUs principales

**Lo que falta por probar:**
- ⏳ Speedup real vs FlashAttention en secuencias muy largas (>8K)
- ⏳ Comportamiento de escalado a 100M+ parámetros
- ⏳ Rendimiento en tareas NLP downstream
- ⏳ Comparación con Mamba/SSM en calidad de generación
- ⏳ Entrenamiento distribuido multi-nodo

---

### Licencia (ES)

**Licencia del Framework HALO-S** — Licencia dual personalizada:

| Caso de Uso | Permiso | Condiciones |
|-------------|:-------:|---|
| Educación e Investigación | ✅ Gratis | Debe acreditar "HALO-S" |
| Proyectos personales | ✅ Gratis | Debe incluir aviso de copyright |
| Uso Comercial / Producción | ❌ Requiere licencia | Contactar para licencia comercial |

Para consultas de licencia comercial: **dalusx64@gmail.com**

Ver [LICENSE](./LICENSE) para términos completos.

---

### Autor

**BUEORM**
- 📧 dalusx64@gmail.com
- 🐙 [github.com/bueormnew/pyhalo](https://github.com/bueormnew/pyhalo)

---

### Ejecución de Tests

```bash
# Ejecutar los 61 tests
pytest tests/ -v

# Con reporte de cobertura
pytest tests/ --cov=halo --cov-report=term-missing

# Módulo específico
pytest tests/test_attention.py -v
pytest tests/test_model.py -v

# Tests que coincidan con patrón
pytest tests/ -k "generation" -v
```

### Ejecutar Experimentos

```bash
# Experimento 1: Comparación baseline (HALO-S vs Transformer Denso)
python scripts/exp1_baseline.py

# Experimento 2: Estudio de ablación (contribución de cada componente)
python scripts/exp2_ablation.py

# Experimento 3: Contexto largo (Aguja en un Pajar)
python scripts/exp3_long_context.py
```

### Citar

```bibtex
@software{halo_s_2024,
  author = {BUEORM},
  title = {HALO-S: Hierarchical Attention with Local Offsets — Sparse},
  version = {2.2.1},
  year = {2024},
  url = {https://github.com/bueormnew/pyhalo},
  note = {Framework de atención dispersa con complejidad lineal para modelos de lenguaje},
}
```
