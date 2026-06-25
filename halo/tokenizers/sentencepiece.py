"""
Tokenizador SentencePiece para HALO-S.

Wrapper sobre la librería sentencepiece (dependencia opcional) que proporciona
tokenización subword (BPE/Unigram) compatible con la interfaz BaseTokenizer.

Requiere: pip install sentencepiece
"""

import os

from halo.tokenizers.base import BaseTokenizer

# Import condicional de sentencepiece con mensaje descriptivo
try:
    import sentencepiece as spm
except ImportError:
    spm = None


def _check_sentencepiece_installed() -> None:
    """Verifica que sentencepiece esté instalado, lanza ImportError si no."""
    if spm is None:
        raise ImportError(
            "SentencePiece no está instalado. "
            "Instalar con: pip install halo-s[full] o pip install sentencepiece"
        )


class SentencePieceTokenizer(BaseTokenizer):
    """
    Tokenizador basado en SentencePiece para tokenización subword.

    Soporta modelos BPE y Unigram pre-entrenados, delegando todas las
    operaciones de codificación/decodificación al procesador de SentencePiece.

    Requiere la dependencia opcional `sentencepiece`.
    """

    def __init__(self, model_path: str):
        """
        Inicializa el tokenizador cargando un modelo .model pre-entrenado.

        Args:
            model_path: Ruta al archivo .model de SentencePiece.

        Raises:
            ImportError: Si sentencepiece no está instalado.
            FileNotFoundError: Si model_path no existe.
        """
        _check_sentencepiece_installed()

        # Validar existencia del archivo del modelo
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo SentencePiece en: {model_path}"
            )

        # Cargar el procesador de SentencePiece
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._model_path = model_path

    def encode(self, text: str) -> list[int]:
        """Codifica texto en una lista de IDs de tokens usando SentencePiece."""
        return self._sp.EncodeAsIds(text)

    def decode(self, tokens: list[int]) -> str:
        """Decodifica una lista de IDs de tokens a texto."""
        return self._sp.DecodeIds(tokens)

    @property
    def vocab_size(self) -> int:
        """Retorna el tamaño del vocabulario del modelo cargado."""
        return self._sp.GetPieceSize()

    @classmethod
    def train(
        cls,
        input_file: str,
        model_prefix: str,
        vocab_size: int = 8000,
        model_type: str = "bpe",
    ) -> "SentencePieceTokenizer":
        """
        Entrena un modelo SentencePiece y retorna una instancia del tokenizador.

        Args:
            input_file: Archivo de texto plano para entrenamiento.
            model_prefix: Prefijo para los archivos de salida (.model, .vocab).
            vocab_size: Tamaño del vocabulario objetivo (default: 8000).
            model_type: Tipo de modelo, "bpe" o "unigram" (default: "bpe").

        Returns:
            Instancia de SentencePieceTokenizer con el modelo recién entrenado.

        Raises:
            ImportError: Si sentencepiece no está instalado.
            FileNotFoundError: Si input_file no existe.
        """
        _check_sentencepiece_installed()

        # Validar existencia del archivo de entrada
        if not os.path.isfile(input_file):
            raise FileNotFoundError(
                f"No se encontró el archivo de entrenamiento: {input_file}"
            )

        # Entrenar modelo con SentencePieceTrainer
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
        )

        # Retornar instancia con el modelo entrenado
        model_path = f"{model_prefix}.model"
        return cls(model_path)
