from halo.tokenizers.base import BaseTokenizer

class CharacterTokenizer(BaseTokenizer):
    """
    Tokenizador básico a nivel de carácter para experimentación rápida
    y pruebas internas de bajo consumo sin dependencias externas.
    """
    def __init__(self, chars: str = None):
        if chars is None:
            # Vocabulario de ASCII por defecto
            chars = "".join(chr(i) for i in range(256))
            
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self._vocab_size = len(chars)
        
    def encode(self, text: str) -> list[int]:
        # Si un carácter no existe, cae en 0 (usualmente NULL o UNK)
        return [self.stoi.get(c, 0) for c in text]
        
    def decode(self, tokens: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in tokens)
        
    @property
    def vocab_size(self) -> int:
        return self._vocab_size
