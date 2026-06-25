from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class HaloConfig:
    """Configuración principal para el modelo HALO-S."""
    vocab_size: int = 256  # Por defecto para CharTokenizer
    hidden_size: int = 512
    num_layers: int = 6
    num_heads: int = 8
    num_kv_heads: int = 2  # Implementando Grouped Query Attention (GQA)
    
    # Parámetros del grafo disperso HALO-S
    num_globals: int = 2
    local_window: int = 64
    dilated_offsets: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    num_random: int = 2
    
    # Hiperparámetros de regularización e inferencia
    dropout: float = 0.1
    max_seq_len: int = 4096
    
    # v2.0 flags
    use_swiglu: bool = True
    
    def __post_init__(self):
        """Validación de parámetros al crear la instancia."""
        assert self.hidden_size % self.num_heads == 0, (
            f"hidden_size ({self.hidden_size}) debe ser divisible por "
            f"num_heads ({self.num_heads})"
        )
        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) debe ser divisible por "
            f"num_kv_heads ({self.num_kv_heads})"
        )
        assert self.num_globals >= 1, (
            f"Se requiere al menos 1 global token, se recibió num_globals={self.num_globals}"
        )
        assert self.local_window > 0, (
            f"local_window debe ser > 0, se recibió {self.local_window}"
        )
        assert self.max_seq_len > self.num_globals, (
            f"max_seq_len ({self.max_seq_len}) debe ser mayor que "
            f"num_globals ({self.num_globals})"
        )
        assert 0 <= self.dropout < 1, (
            f"dropout debe estar en [0, 1), se recibió {self.dropout}"
        )
    
    @property
    def head_dim(self):
        """Dimensión por cabeza de atención."""
        return self.hidden_size // self.num_heads
    
    @property
    def num_neighbors(self) -> int:
        """Número total de vecinos por token en el grafo disperso.
        
        Incluye: globals + ventana local + conexiones dilatadas (ida y vuelta) + aleatorios.
        """
        return (
            self.num_globals
            + self.local_window
            + 2 * len(self.dilated_offsets)
            + self.num_random
        )
    
    def to_dict(self) -> dict:
        """Serializa la configuración a un diccionario con todos los campos."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: dict) -> "HaloConfig":
        """Reconstruye una instancia de HaloConfig desde un diccionario.
        
        Tolerates old configs that don't include v2.0 fields by filtering
        unknown keys and relying on defaults.
        """
        import inspect
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)
