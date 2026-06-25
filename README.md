<p align="center">
  <h1 align="center">🌀 HALO-S</h1>
  <p align="center"><strong>Hierarchical Attention with Local Offsets — Sparse</strong></p>
  <p align="center">A linear-complexity language model framework that replaces quadratic attention with a structured sparse connectivity graph.</p>
</p>

<p align="center">

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyPI](https://img.shields.io/pypi/v/pyhalo)
![License](https://img.shields.io/badge/license-custom-orange)
![Tests](https://img.shields.io/badge/tests-55%20passed-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red)

</p>

---

## What if attention didn't have to be quadratic?

Every modern language model pays a steep price for long sequences: the standard Transformer's self-attention scales as O(N²), making context windows beyond 4K tokens prohibitively expensive. HALO-S takes a different path. By constructing a **fixed-degree sparse connectivity graph** — combining local windows, dilated connections, learned global tokens, and random edges — each token attends to only K neighbors regardless of sequence length. The result is **O(N×K) complexity** with K=76 by default, yielding a theoretical **~52.5× reduction** in attention operations at N=4096.

HALO-S is implemented as a clean, research-ready PyTorch framework. No custom CUDA kernels. No external dependencies beyond PyTorch and NumPy. Just gather-based sparse attention that runs on any hardware.

> ⚠️ **Honest disclaimer**: HALO-S is a promising architectural exploration. The theoretical complexity advantages are mathematically sound, but **large-scale empirical validation** against established models on standard benchmarks is still in progress. Use it for research, experimentation, and learning. The numbers in this README reflect theoretical analysis and small-scale experiments, not production-validated results.

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture Overview](#architecture-overview)
- [Performance Analysis (Theoretical)](#performance-analysis-theoretical)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Advanced Usage](#advanced-usage)
- [Project Structure](#project-structure)
- [Why HALO-S?](#why-halo-s)
- [License](#license)
- [🇪🇸 Versión en Español](#-versión-en-español)

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Linear Attention Complexity** | O(N×K) instead of O(N²) — scales to long sequences efficiently |
| **Gather-Based Sparse Attention** | No custom CUDA kernels needed; runs on CPU and GPU |
| **Learned Global Tokens** | Shared memory parameters that attend to the full sequence |
| **Dilated Connections** | Exponentially expanding receptive field across layers |
| **Random Edges** | Small-world graph properties for information propagation |
| **Grouped Query Attention (GQA)** | Reduced KV memory with configurable head ratios |
| **Rotary Position Embeddings (RoPE)** | Relative position encoding without learned parameters |
| **Mixed Precision Training** | Native AMP support with GradScaler |
| **Gradient Accumulation** | Train with effective large batches on limited hardware |
| **Checkpoint Save/Load** | Full training state persistence and resumption |
| **Streaming Datasets** | Train on data larger than RAM with buffer shuffling |
| **Autoregressive Generation** | Top-k, top-p, and temperature sampling built-in |

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
│  │ │(dense, G×N)    │   │ (gather, N×K)       │     │           │
│  │ └───────┬────────┘   └──────────┬──────────┘     │           │
│  │         │                       │                  │           │
│  │         └───────────┬───────────┘                  │           │
│  │                     ▼                              │           │
│  │           cat([globals_out, tokens_out])            │           │
│  │                     │ + residual                    │           │
│  │                     ▼                              │           │
│  │  ┌─────────────┐  ┌────────────┐                 │           │
│  │  │ LayerNorm 2 │→ │ FeedForward│ + residual      │           │
│  │  └─────────────┘  └────────────┘                 │           │
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

### Mathematical Formulation

Given input sequence x ∈ ℝ^(B×N), the forward pass:

1. **Embed**: e = Embedding(x) ∈ ℝ^(B×N×H)
2. **Prepend globals**: x̂ = [g₁,...,g_G ; e₁,...,e_N] ∈ ℝ^(B×(G+N)×H)
3. **Per layer**: Pre-norm → Split attention (global dense + token sparse) → Residual → Pre-norm → FFN → Residual
4. **Output**: logits = W_lm · LN_f(x̂^(L)_{G:}) ∈ ℝ^(B×N×V)

---

## Performance Analysis (Theoretical)

> ⚠️ **All performance data below is THEORETICAL**, derived from complexity analysis. Large-scale empirical benchmarks are in progress.

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

The speedup grows **linearly with N** because dense attention is O(N²) while HALO-S is O(N×K).

### Theoretical Comparison with Other Architectures

> ⚠️ **THEORETICAL COMPARISON** — based on published complexity analyses, not head-to-head benchmarks.

| Model | Attention Complexity | Memory (Scores) | Global Context | Dilated | Random Edges | GQA |
|-------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dense Transformer** | O(N²·d) | O(N²) | Full (implicit) | ✗ | ✗ | Optional |
| **Longformer** | O(N·w·d) | O(N·w) | ✓ (fixed) | ✓ | ✗ | ✗ |
| **BigBird** | O(N·(w+g+r)·d) | O(N·(w+g+r)) | ✓ (fixed) | ✗ | ✓ | ✗ |
| **Mamba (SSM)** | O(N·d²) | O(d²) | Implicit (state) | ✗ | ✗ | N/A |
| **HALO-S** | **O(N·K·d)** | **O(N·K)** | ✓ (learned) | ✓ | ✓ | ✓ |

### Memory Efficiency

| Component | Dense Transformer | HALO-S | Advantage |
|-----------|:-:|:-:|:-:|
| Attention scores (B=1, N=4096) | 512 MB | 9.5 MB | **54× less** |
| KV cache (GQA effect) | 16 MB | 4 MB | **4× less** |
| Crossover point | — | N > 9,728 | Total memory advantage |

### Qualitative Comparison (THEORETICAL)

| Capability | Transformer | Mamba | Longformer | **HALO-S** |
|------------|:---:|:---:|:---:|:---:|
| Long-range dependencies | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★☆ (theoretical) |
| Training efficiency | ★★☆☆☆ | ★★★★★ | ★★★★☆ | ★★★★☆ (theoretical) |
| Inference speed | ★★☆☆☆ | ★★★★★ | ★★★☆☆ | ★★★★☆ (theoretical) |
| Hardware compatibility | ★★★★★ | ★★★☆☆ | ★★★☆☆ | ★★★★★ |
| Implementation simplicity | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ |

---

## Installation

### From PyPI

```bash
# Core installation (PyTorch + NumPy only)
pip install pyhalo

# Full installation (includes tqdm progress bars + SentencePiece tokenizer)
pip install pyhalo[full]
```

### From Source

```bash
git clone https://github.com/bueormnew/pyhalo.git
cd pyhalo
pip install -e ".[full,dev]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1.0
- NumPy ≥ 1.24.0
- (Optional) tqdm, sentencepiece

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

### Tensor Generation

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

---

## Advanced Usage

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

# Resume training from checkpoint
trainer.load_checkpoint("my_checkpoint.pt")
# Continue training...
trainer.fit(dataset=dataset, epochs=5, batch_size=8)
```

### Streaming Dataset (Files Larger Than RAM)

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer
from halo.datasets import StreamingDataset

tok = CharacterTokenizer()

# StreamingDataset reads files lazily with buffer shuffling
stream_dataset = StreamingDataset(
    file_paths=["data/shard_01.jsonl", "data/shard_02.jsonl"],
    tokenizer=tok,
    max_seq_len=2048,
    buffer_size=10000,     # Local shuffle buffer
    text_field="text",
    file_format="jsonl",   # or "txt"
)

# Use with DataLoader (IterableDataset compatible)
from torch.utils.data import DataLoader
loader = DataLoader(stream_dataset, batch_size=4)
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

---

## Project Structure

```
pyhalo/
├── halo/                          # Main package
│   ├── __init__.py                # Public API exports
│   ├── attention/
│   │   ├── global_attention.py    # Dense attention for global tokens
│   │   ├── graph.py              # Neighbor list generation (local + dilated + random)
│   │   └── halo_attention.py     # Gather-based sparse attention
│   ├── core/
│   │   ├── config.py             # HaloConfig dataclass
│   │   └── logging.py           # Structured logging utilities
│   ├── datasets/
│   │   ├── jsonl.py             # JSONLDataset for structured data
│   │   ├── streaming.py         # StreamingDataset (IterableDataset, infinite)
│   │   ├── synthetic.py         # CopyDataset, NeedleDataset for testing
│   │   └── text.py             # Plain text dataset
│   ├── generation/
│   │   └── samplers.py          # Top-k, top-p, temperature sampling
│   ├── models/
│   │   ├── halo_model.py        # HaloSModel (main model)
│   │   └── baseline_model.py    # Dense baseline for comparison
│   ├── nn/
│   │   ├── feed_forward.py      # SwiGLU / GELU feed-forward
│   │   ├── halo_block.py        # HaloBlock (attention + FFN + residual)
│   │   └── rope.py             # Rotary Positional Embeddings
│   ├── tokenizers/
│   │   ├── base.py             # BaseTokenizer abstract class
│   │   ├── char.py             # CharacterTokenizer (byte-level)
│   │   ├── word.py             # WordTokenizer (whitespace-based)
│   │   └── sentencepiece.py    # SentencePiece wrapper
│   ├── training/
│   │   └── trainer.py          # Trainer with AMP, accumulation, checkpoints
│   └── utils/
│       ├── benchmarks.py        # Speed, generation, memory, FLOPs benchmarks
│       ├── metrics.py          # Parameter counting, memory estimation
│       └── random.py           # Seed management
├── docs/
│   ├── architecture.md          # Full architecture documentation
│   ├── complexity.md           # Complexity analysis and proofs
│   ├── local_attention.md      # Local window mechanism
│   ├── dilated_connections.md  # Dilated connection strategy
│   ├── global_tokens.md        # Global token design
│   ├── sparse_attention.md     # Sparse attention implementation
│   ├── gqa.md                  # Grouped Query Attention
│   ├── rope.md                 # RoPE implementation details
│   └── flash_attention.md      # Flash attention compatibility notes
├── tests/                       # 55 tests covering all components
│   ├── test_attention.py
│   ├── test_model.py
│   ├── test_training.py
│   ├── test_generation.py
│   ├── test_tokenizers.py
│   ├── test_shapes.py
│   ├── test_gradients.py
│   ├── test_memory.py
│   ├── test_checkpoint.py
│   ├── test_config.py
│   └── test_graph.py
├── benchmarks/                  # Benchmark scripts
│   ├── benchmark_speed.py
│   └── benchmark_graph.py
├── scripts/                     # Experiment scripts
│   ├── exp1_baseline.py
│   ├── exp2_ablation.py
│   └── exp3_long_context.py
├── pyproject.toml              # Package configuration
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

### Honest Assessment

**What HALO-S does well (demonstrated):**
- ✅ Clean, modular PyTorch implementation with no exotic dependencies
- ✅ Mathematically sound complexity reduction (O(N×K) vs O(N²))
- ✅ Runs on any hardware — CPU, single GPU, no custom kernels required
- ✅ All 55 tests pass — correctness of gradients, shapes, generation, and checkpoints verified
- ✅ Training loop works end-to-end with AMP, gradient accumulation, and streaming data

**What remains to be proven:**
- ⏳ Perplexity parity with dense Transformers at equivalent parameter count on standard benchmarks (WikiText-103, C4, etc.)
- ⏳ Scaling behavior at 100M+ parameters
- ⏳ Actual wall-clock speedup vs optimized dense attention (FlashAttention v2)
- ⏳ Performance on downstream NLP tasks (summarization, QA, etc.)
- ⏳ Comparison with Mamba/SSM architectures on actual generation quality

**The gather-based approach has a known trade-off**: while it avoids custom CUDA kernels, the `torch.gather` operations create intermediate tensors that can be memory-intensive. For sequences shorter than ~9,728 tokens, the gathered KV tensors may exceed dense attention memory. The advantage becomes clear at longer sequences.

---

## Running Tests

```bash
# Run all 55 tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=halo --cov-report=term-missing

# Run specific test module
pytest tests/test_attention.py -v
```

---

## Running Experiments

```bash
# Experiment 1: Baseline comparison (HALO-S vs Dense)
python scripts/exp1_baseline.py

# Experiment 2: Ablation study (contribution of each connectivity component)
python scripts/exp2_ablation.py

# Experiment 3: Long context scaling behavior
python scripts/exp3_long_context.py
```

---

## Citation

If you use HALO-S in your research, please cite:

```bibtex
@software{halo_s_2024,
  author = {BUEORM},
  title = {HALO-S: Hierarchical Attention with Local Offsets — Sparse},
  year = {2024},
  url = {https://github.com/bueormnew/pyhalo},
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
</p>

---

### ¿Y si la atención no tuviera que ser cuadrática?

Todo modelo de lenguaje moderno paga un precio elevado por secuencias largas: la auto-atención estándar del Transformer escala como O(N²), haciendo que ventanas de contexto mayores a 4K tokens sean prohibitivamente costosas. HALO-S toma un camino diferente. Al construir un **grafo de conectividad dispersa de grado fijo** — combinando ventanas locales, conexiones dilatadas, tokens globales aprendidos y aristas aleatorias — cada token atiende solo a K vecinos sin importar la longitud de la secuencia. El resultado es **complejidad O(N×K)** con K=76 por defecto, logrando una **reducción teórica de ~52.5×** en operaciones de atención para N=4096.

HALO-S está implementado como un framework limpio de PyTorch listo para investigación. Sin kernels CUDA personalizados. Sin dependencias externas más allá de PyTorch y NumPy. Solo atención dispersa basada en gather que funciona en cualquier hardware.

> ⚠️ **Aviso honesto**: HALO-S es una exploración arquitectónica prometedora. Las ventajas teóricas de complejidad son matemáticamente sólidas, pero la **validación empírica a gran escala** contra modelos establecidos en benchmarks estándar aún está en progreso. Úsalo para investigación, experimentación y aprendizaje.

---

### Características Principales

| Característica | Descripción |
|---|---|
| **Complejidad de Atención Lineal** | O(N×K) en lugar de O(N²) — escala eficientemente a secuencias largas |
| **Atención Dispersa basada en Gather** | Sin kernels CUDA personalizados; funciona en CPU y GPU |
| **Tokens Globales Aprendidos** | Parámetros de memoria compartida que atienden la secuencia completa |
| **Conexiones Dilatadas** | Campo receptivo exponencialmente expansivo entre capas |
| **Aristas Aleatorias** | Propiedades de grafo de mundo pequeño para propagación de información |
| **Grouped Query Attention (GQA)** | Memoria KV reducida con ratios de cabezas configurables |
| **RoPE (Rotary Position Embeddings)** | Codificación posicional relativa sin parámetros aprendidos |
| **Entrenamiento con Precisión Mixta** | Soporte nativo de AMP con GradScaler |
| **Acumulación de Gradientes** | Entrena con batches efectivos grandes en hardware limitado |
| **Guardado/Carga de Checkpoints** | Persistencia y reanudación completa del estado de entrenamiento |
| **Datasets de Streaming** | Entrena con datos mayores a la RAM con shuffling por buffer |
| **Generación Autoregresiva** | Muestreo top-k, top-p y temperatura integrados |

---

### Análisis de Rendimiento (Teórico)

> ⚠️ **Todos los datos de rendimiento son TEÓRICOS**, derivados del análisis de complejidad.

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

#### Comparación Teórica con Otras Arquitecturas

> ⚠️ **COMPARACIÓN TEÓRICA** — basada en análisis de complejidad publicados, no en benchmarks directos.

| Modelo | Complejidad Atención | Memoria (Scores) | Contexto Global | Dilatación | Aristas Aleatorias | GQA |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **Transformer Denso** | O(N²·d) | O(N²) | Completo (implícito) | ✗ | ✗ | Opcional |
| **Longformer** | O(N·w·d) | O(N·w) | ✓ (fijos) | ✓ | ✗ | ✗ |
| **BigBird** | O(N·(w+g+r)·d) | O(N·(w+g+r)) | ✓ (fijos) | ✗ | ✓ | ✗ |
| **Mamba (SSM)** | O(N·d²) | O(d²) | Implícito (estado) | ✗ | ✗ | N/A |
| **HALO-S** | **O(N·K·d)** | **O(N·K)** | ✓ (aprendidos) | ✓ | ✓ | ✓ |

---

### Instalación

```bash
# Instalación básica (solo PyTorch + NumPy)
pip install pyhalo

# Instalación completa (incluye tqdm + SentencePiece)
pip install pyhalo[full]

# Desde código fuente
git clone https://github.com/bueormnew/pyhalo.git
cd pyhalo
pip install -e ".[full,dev]"
```

**Requisitos:** Python ≥ 3.10, PyTorch ≥ 2.1.0, NumPy ≥ 1.24.0

---

### Inicio Rápido

```python
from halo import HaloConfig, HaloSModel, set_seed

set_seed(42)

# Configurar modelo
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

#### Generación de Texto

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

#### Entrenamiento Completo

```python
from halo import HaloConfig, HaloSModel, Trainer, CharacterTokenizer, set_seed
from halo.datasets import JSONLDataset

set_seed(42)

config = HaloConfig(vocab_size=256, hidden_size=512, num_layers=6, num_heads=8, num_kv_heads=2)
model = HaloSModel(config)
tok = CharacterTokenizer()

dataset = JSONLDataset(
    file_path="datos/train.jsonl",
    tokenizer=tok,
    max_seq_len=2048,
    text_field="text",
)

trainer = Trainer(
    model=model,
    learning_rate=3e-4,
    mixed_precision=True,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    checkpoint_dir="./checkpoints",
)

history = trainer.fit(dataset=dataset, epochs=10, batch_size=8, save_every=2)
```

#### Benchmarks

```python
from halo import HaloConfig, HaloSModel
from halo.utils.benchmarks import benchmark_speed, benchmark_generation, estimate_flops

config = HaloConfig(vocab_size=256, hidden_size=512, num_layers=6, num_heads=8)
model = HaloSModel(config)

# Benchmark de latencia
resultados = benchmark_speed(model, config, seq_lengths=[512, 1024, 2048, 4096])
for r in resultados:
    print(f"  N={r['seq_len']:>5} | {r['avg_ms']:.2f} ms | {r['tokens_per_sec']:,.0f} tok/s")

# FLOPs teóricos
flops = estimate_flops(config, seq_len=4096)
print(f"Total: {flops['total_gflops']:.2f} GFLOPs")
```

---

### ¿Por Qué HALO-S?

#### Filosofía

HALO-S nació de una pregunta simple: *¿Podemos obtener la mayor parte del poder representacional de la atención densa pagando solo una fracción del costo computacional?*

El enfoque se basa en teoría de grafos. En lugar de permitir que cada token atienda a todos los demás (un grafo completo), HALO-S construye un **grafo de conectividad dispersa** con propiedades de ciencia de redes:

1. **Clustering local** (atención de ventana) — tokens cercanos forman vecindarios densamente conectados
2. **Atajos de largo alcance** (conexiones dilatadas) — conexiones espaciadas exponencialmente previenen cuellos de botella
3. **Propiedades de mundo pequeño** (aristas aleatorias) — garantizan que el diámetro del grafo sea logarítmico
4. **Memoria compartida** (tokens globales) — parámetros aprendidos que actúan como canal de broadcast

#### Evaluación Honesta

**Lo que HALO-S hace bien (demostrado):**
- ✅ Implementación limpia y modular en PyTorch sin dependencias exóticas
- ✅ Reducción de complejidad matemáticamente sólida (O(N×K) vs O(N²))
- ✅ Funciona en cualquier hardware — CPU, GPU, sin kernels personalizados
- ✅ 55 tests pasan — correctitud de gradientes, formas, generación y checkpoints
- ✅ Loop de entrenamiento funciona end-to-end con AMP y streaming

**Lo que queda por demostrar:**
- ⏳ Paridad de perplejidad con Transformers densos a parámetros equivalentes
- ⏳ Comportamiento de escalado a 100M+ parámetros
- ⏳ Speedup real de wall-clock vs atención densa optimizada (FlashAttention v2)
- ⏳ Rendimiento en tareas NLP downstream
- ⏳ Comparación con Mamba/SSM en calidad de generación

---

### Licencia

**Licencia HALO-S Framework** — Licencia dual personalizada:

| Caso de Uso | Permiso | Condiciones |
|---|:---:|---|
| Educación e Investigación | ✅ Gratis | Debe acreditar "HALO-S" en cualquier trabajo derivado |
| Proyectos personales | ✅ Gratis | Debe incluir aviso de copyright |
| Uso Comercial / Producción | ❌ Requiere licencia | Contactar para licencia comercial |

Para consultas de licencia comercial: **dalusx64@gmail.com**

---

### Autor

**BUEORM**
- 📧 dalusx64@gmail.com
- 🐙 [github.com/bueormnew/pyhalo](https://github.com/bueormnew/pyhalo)

---

<p align="center">
  <em>Built with focus on clarity, correctness, and computational efficiency.</em><br>
  <em>Construido con enfoque en claridad, correctitud y eficiencia computacional.</em>
</p>
