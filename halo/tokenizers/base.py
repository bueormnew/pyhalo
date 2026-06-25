class BaseTokenizer:
    """Interfaz común para todos los tokenizadores del framework HALO-S."""
    
    def encode(self, text: str) -> list[int]:
        raise NotImplementedError
        
    def decode(self, tokens: list[int]) -> str:
        raise NotImplementedError
        
    @property
    def vocab_size(self) -> int:
        raise NotImplementedError
