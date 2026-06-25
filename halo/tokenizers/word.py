"""
WordTokenizer — Tokenizador a nivel de palabra para HALO-S.

Construye un vocabulario desde texto plano y mapea palabras a IDs enteros.
Incluye tokens especiales: <PAD>, <UNK>, <BOS>, <EOS>.
"""

import json
from collections import Counter

from halo.tokenizers.base import BaseTokenizer


class WordTokenizer(BaseTokenizer):
    """
    Tokenizador a nivel de palabra con vocabulario construido desde texto.
    Hereda de BaseTokenizer e implementa encode/decode/vocab_size.
    
    Tokens especiales siempre presentes en el vocabulario:
        - <PAD> (id=0): Relleno para secuencias de longitud variable
        - <UNK> (id=1): Palabra desconocida (no presente en vocabulario)
        - <BOS> (id=2): Inicio de secuencia
        - <EOS> (id=3): Fin de secuencia
    """

    # Tokens especiales con IDs fijos
    SPECIAL_TOKENS = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}

    def __init__(self, vocab: dict[str, int] = None, min_freq: int = 1):
        """
        Inicializa el tokenizador con un vocabulario opcional.

        Args:
            vocab: Diccionario {palabra: id} pre-construido. Si es None,
                   se debe llamar a build_vocab() antes de usar encode/decode.
            min_freq: Frecuencia mínima para incluir una palabra al construir
                      vocabulario con build_vocab().
        """
        self.min_freq = min_freq

        if vocab is not None:
            # Usar vocabulario proporcionado directamente
            self.stoi = dict(vocab)
            self.itos = {i: w for w, i in self.stoi.items()}
        else:
            # Inicializar solo con tokens especiales
            self.stoi = dict(self.SPECIAL_TOKENS)
            self.itos = {i: w for w, i in self.stoi.items()}

    def build_vocab(self, texts: list[str]) -> None:
        """
        Construye vocabulario desde una lista de textos.

        Divide cada texto por espacios en blanco, cuenta frecuencias,
        y solo incluye palabras con frecuencia >= min_freq.
        Los tokens especiales siempre se incluyen con IDs fijos.

        Args:
            texts: Lista de strings de donde extraer el vocabulario.
        """
        # Contar frecuencias de todas las palabras
        counter = Counter()
        for text in texts:
            words = text.split()
            counter.update(words)

        # Reiniciar vocabulario con tokens especiales
        self.stoi = dict(self.SPECIAL_TOKENS)

        # Añadir palabras que cumplen el umbral de frecuencia mínima
        next_id = len(self.SPECIAL_TOKENS)
        for word, freq in sorted(counter.items()):
            if freq >= self.min_freq and word not in self.SPECIAL_TOKENS:
                self.stoi[word] = next_id
                next_id += 1

        # Construir mapeo inverso
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text: str) -> list[int]:
        """
        Convierte texto a una lista de IDs enteros.

        Divide el texto por espacios en blanco y mapea cada palabra
        a su ID correspondiente. Palabras desconocidas se mapean a UNK.

        Args:
            text: Texto a codificar.

        Returns:
            Lista de IDs enteros.
        """
        unk_id = self.SPECIAL_TOKENS["<UNK>"]
        return [self.stoi.get(word, unk_id) for word in text.split()]

    def decode(self, tokens: list[int]) -> str:
        """
        Convierte una lista de IDs a texto.

        Mapea cada ID a su palabra correspondiente y une con espacios.
        Los tokens especiales (PAD, UNK, BOS, EOS) se omiten en la salida.

        Args:
            tokens: Lista de IDs enteros a decodificar.

        Returns:
            String con las palabras unidas por espacios.
        """
        # IDs de tokens especiales que se omiten en la decodificación
        special_ids = set(self.SPECIAL_TOKENS.values())

        words = []
        for token_id in tokens:
            if token_id in special_ids:
                continue
            word = self.itos.get(token_id, "")
            if word:
                words.append(word)

        return " ".join(words)

    @property
    def vocab_size(self) -> int:
        """Retorna el tamaño total del vocabulario (incluyendo tokens especiales)."""
        return len(self.stoi)

    def save(self, path: str) -> None:
        """
        Serializa el vocabulario completo a un archivo JSON.

        El archivo contiene un diccionario con:
            - "stoi": mapeo palabra → ID
            - "min_freq": frecuencia mínima configurada
            - "special_tokens": tokens especiales con sus IDs

        Args:
            path: Ruta del archivo donde guardar el vocabulario.
        """
        data = {
            "stoi": self.stoi,
            "min_freq": self.min_freq,
            "special_tokens": self.SPECIAL_TOKENS,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "WordTokenizer":
        """
        Reconstruye un WordTokenizer desde un archivo JSON.

        Args:
            path: Ruta al archivo JSON previamente guardado con save().

        Returns:
            Instancia de WordTokenizer con el vocabulario restaurado.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab = data["stoi"]
        min_freq = data.get("min_freq", 1)

        return cls(vocab=vocab, min_freq=min_freq)
