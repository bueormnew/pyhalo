import torch
import torch.nn as nn
from halo.core.config import HaloConfig
from halo.nn.halo_block import HaloBlock
from halo.nn.rope import RotaryPositionalEmbeddings
from halo.generation.samplers import generate
from halo.tokenizers.base import BaseTokenizer
from halo.utils.metrics import estimate_memory

class HaloSModel(nn.Module):
    """
    Modelo de Lenguaje HALO-S completo.
    Incorpora los Global Tokens, inyección de RoPE y el stack de HaloBlocks.
    """
    def __init__(self, config: HaloConfig):
        super().__init__()
        self.config = config
        
        # Embeddings de vocabulario
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # GLOBAL TOKENS
        # Se inyectan en las primeras 'num_globals' posiciones de cada secuencia
        self.global_memory = nn.Parameter(torch.randn(config.num_globals, config.hidden_size))
        
        # Rotary Positional Embeddings
        self.rope = RotaryPositionalEmbeddings(config.head_dim, config.max_seq_len)
        
        # Capas Transformadoras
        self.layers = nn.ModuleList([
            HaloBlock(config, layer_id=i) for i in range(config.num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
    def forward(self, input_ids: torch.Tensor, targets: torch.Tensor = None):
        batch_size, seq_len = input_ids.shape
        
        x = self.token_emb(input_ids)
        
        # Inyectar Global Tokens
        # globals: (batch, num_globals, hidden)
        globals_expanded = self.global_memory.unsqueeze(0).expand(batch_size, -1, -1)
        x = torch.cat([globals_expanded, x], dim=1) # (batch, num_globals + seq_len, hidden)
        
        # Calcular RoPE
        cos, sin = self.rope(x)
        
        # Propagación transitiva a través de capas locales y dispersas
        for layer in self.layers:
            x = layer(x, cos, sin, is_causal=True)
            
        x = self.ln_f(x)
        
        # Extraer solo las posiciones de los tokens (descartando los globals) para calcular la pérdida
        x_out = x[:, self.config.num_globals:, :]
        
        logits = self.lm_head(x_out)
        
        loss = None
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), targets.view(-1))
            
        return logits, loss
        
    def generate(self, input_ids, max_new_tokens: int = 100, temperature: float = 1.0, top_k: int = None, top_p: float = None, tokenizer: BaseTokenizer = None):
        """
        Generación autoregresiva. Acepta un Tensor de IDs o un string.
        
        Si input_ids es un str, se requiere un tokenizer para codificar/decodificar.
        Retorna str cuando la entrada es str, Tensor cuando la entrada es Tensor.
        """
        if isinstance(input_ids, str):
            if tokenizer is None:
                raise ValueError("Se requiere un tokenizer para generar desde texto")
            # Codificar texto a tensor
            token_ids = tokenizer.encode(input_ids)
            input_tensor = torch.tensor([token_ids], dtype=torch.long, device=next(self.parameters()).device)
            # Generar tokens
            output_tensor = generate(self, input_tensor, max_new_tokens, temperature, top_k, top_p)
            # Decodificar resultado completo a string
            output_ids = output_tensor[0].tolist()
            return tokenizer.decode(output_ids)
        else:
            return generate(self, input_ids, max_new_tokens, temperature, top_k, top_p)
        
    def save(self, path: str):
        torch.save(self.state_dict(), path)
        
    def load(self, path: str, device="cpu"):
        self.load_state_dict(torch.load(path, map_location=device, weights_only=True))

    def count_parameters(self) -> int:
        """Devuelve el número total de parámetros entrenables del modelo."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_flops(self, seq_len: int = 1024) -> dict:
        """
        Estima los FLOPs teóricos para un forward pass con la longitud de secuencia dada.
        
        Calcula los FLOPs por componente basándose en la configuración del modelo
        y el patrón de atención dispersa HALO-S.
        
        Args:
            seq_len: Longitud de la secuencia de entrada (sin contar globals).
            
        Returns:
            Diccionario con el desglose de FLOPs por componente y totales.
        """
        config = self.config
        hidden = config.hidden_size
        num_heads = config.num_heads
        num_kv_heads = config.num_kv_heads
        head_dim = config.head_dim
        num_globals = config.num_globals
        num_neighbors = config.num_neighbors
        num_layers = config.num_layers
        vocab_size = config.vocab_size

        # --- FLOPs de atención dispersa por capa ---
        # Proyecciones Q, K, V: Q usa num_heads, K/V usan num_kv_heads
        qkv_flops = 2 * seq_len * hidden * (num_heads * head_dim + 2 * num_kv_heads * head_dim)
        # Scores de atención dispersa: Q(seq_len, heads, head_dim) × K(num_neighbors, head_dim)
        scores_flops = 2 * num_heads * seq_len * num_neighbors * head_dim
        # Atención × V: scores(seq_len, num_neighbors) × V(num_neighbors, head_dim)
        attn_v_flops = 2 * num_heads * seq_len * num_neighbors * head_dim
        # Proyección de salida O
        o_proj_flops = 2 * seq_len * (num_heads * head_dim) * hidden

        attention_flops = qkv_flops + scores_flops + attn_v_flops + o_proj_flops

        # --- FLOPs de atención densa para Global Tokens por capa ---
        # Proyecciones Q, K, V para globals: Q de globals, K/V de secuencia completa
        total_seq = seq_len + num_globals
        global_qkv_flops = (
            2 * num_globals * hidden * (num_heads * head_dim)  # Q de globals
            + 2 * total_seq * hidden * (num_kv_heads * head_dim)  # K de toda la seq
            + 2 * total_seq * hidden * (num_kv_heads * head_dim)  # V de toda la seq
        )
        # Scores: Q(globals) × K(total_seq)^T + Atención × V
        global_scores_flops = 2 * num_heads * num_globals * total_seq * head_dim * 2  # scores + attn_v
        # Proyección O para globals
        global_o_proj_flops = 2 * num_globals * (num_heads * head_dim) * hidden

        global_flops = global_qkv_flops + global_scores_flops + global_o_proj_flops

        # --- FLOPs de FFN por capa ---
        # Dos capas lineales: hidden → 4*hidden → hidden, aplicadas a toda la secuencia
        ffn_flops = 2 * total_seq * hidden * (4 * hidden) * 2

        # --- FLOPs del LM Head ---
        lm_head_flops = 2 * seq_len * hidden * vocab_size

        # --- Total ---
        total_flops = (attention_flops + global_flops + ffn_flops) * num_layers + lm_head_flops
        total_gflops = total_flops / 1e9

        return {
            "attention_flops": attention_flops * num_layers,
            "global_flops": global_flops * num_layers,
            "ffn_flops": ffn_flops * num_layers,
            "lm_head_flops": lm_head_flops,
            "total_flops": total_flops,
            "total_gflops": round(total_gflops, 4),
        }

    def summary(self) -> str:
        """
        Devuelve un resumen legible del modelo con configuración, parámetros y memoria estimada.
        
        Returns:
            String formateado con la información principal del modelo.
        """
        config = self.config
        total_params = self.count_parameters()
        mem_info = estimate_memory(self, config)

        lines = [
            "=" * 60,
            "HALO-S Model",
            "=" * 60,
            f"  vocab_size:      {config.vocab_size}",
            f"  hidden_size:     {config.hidden_size}",
            f"  num_layers:      {config.num_layers}",
            f"  num_heads:       {config.num_heads}",
            f"  num_kv_heads:    {config.num_kv_heads}",
            f"  num_globals:     {config.num_globals}",
            f"  local_window:    {config.local_window}",
            f"  num_neighbors:   {config.num_neighbors}",
            f"  max_seq_len:     {config.max_seq_len}",
            "-" * 60,
            f"  Total parámetros: {total_params:,}",
            f"  Memoria estimada: {mem_info['total_estimated_mb']:.2f} MB",
            "=" * 60,
        ]
        return "\n".join(lines)
