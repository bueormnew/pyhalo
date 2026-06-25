"""
StreamingDataset: Dataset iterativo con buffer shuffle para archivos grandes.

Permite entrenar con datos que no caben en RAM, leyendo archivos de forma
cíclica e infinita, acumulando tokens y aplicando shuffling local por buffer.
Compatible con torch.utils.data.DataLoader como IterableDataset.
"""

import json
import random
from typing import Iterator

import torch
from torch.utils.data import IterableDataset

from halo.tokenizers.base import BaseTokenizer


class StreamingDataset(IterableDataset):
    """
    Dataset con streaming para archivos muy grandes que no caben en RAM.
    Implementa un iterador infinito con shuffling por buffer.

    Hereda de torch.utils.data.IterableDataset para compatibilidad con DataLoader.
    Itera cíclicamente sobre los archivos fuente, acumula tokens entre líneas,
    corta en secuencias de max_seq_len+1, y aplica shuffling local por buffer.
    """

    def __init__(
        self,
        file_paths: list[str],
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        buffer_size: int = 10000,
        text_field: str = "text",
        file_format: str = "jsonl",
    ):
        """
        Args:
            file_paths: Lista de archivos fuente para leer.
            tokenizer: Tokenizador compatible con BaseTokenizer.
            max_seq_len: Longitud de secuencia de salida (x e y tendrán esta forma).
            buffer_size: Tamaño del buffer para shuffling local.
            text_field: Campo de texto en archivos JSONL.
            file_format: Formato de archivo, "jsonl" o "txt".
        """
        super().__init__()
        self.file_paths = file_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size
        self.text_field = text_field
        self.file_format = file_format

    def _line_iterator(self) -> Iterator[str]:
        """
        Iterador infinito sobre las líneas de los archivos fuente (cíclico).
        Recorre todos los archivos en orden y al terminar vuelve a empezar.
        """
        while True:
            for file_path in self.file_paths:
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        yield line

    def _extract_text(self, line: str) -> str | None:
        """
        Extrae el texto de una línea según el formato configurado.
        Retorna None si la línea no es válida.
        """
        if self.file_format == "jsonl":
            try:
                data = json.loads(line)
                if self.text_field in data:
                    return data[self.text_field]
                return None
            except (json.JSONDecodeError, TypeError):
                return None
        else:
            # Formato "txt": la línea completa es el texto
            return line

    def _sequence_iterator(self) -> Iterator[list[int]]:
        """
        Acumulador de tokens: acumula tokens entre líneas y corta en
        secuencias de exactamente max_seq_len+1 tokens.
        """
        token_accumulator: list[int] = []
        seq_length = self.max_seq_len + 1  # +1 para poder separar x e y

        for line in self._line_iterator():
            text = self._extract_text(line)
            if text is None:
                continue

            # Tokenizar la línea y añadir al acumulador
            tokens = self.tokenizer.encode(text)
            token_accumulator.extend(tokens)

            # Cortar secuencias completas del acumulador
            while len(token_accumulator) >= seq_length:
                sequence = token_accumulator[:seq_length]
                token_accumulator = token_accumulator[seq_length:]
                yield sequence

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterador infinito que yield tuplas (x, y) de shape (max_seq_len,).

        Acumula secuencias en un buffer, cuando el buffer está lleno
        aplica shuffling y emite todas las secuencias del buffer.
        Cada elemento emitido es una tupla (x, y) donde:
          - x = tensor(seq[:-1]) con shape (max_seq_len,)
          - y = tensor(seq[1:])  con shape (max_seq_len,)
        """
        buffer: list[list[int]] = []

        for sequence in self._sequence_iterator():
            buffer.append(sequence)

            # Cuando el buffer está lleno, shufflear y emitir todo
            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                for seq in buffer:
                    seq_tensor = torch.tensor(seq, dtype=torch.long)
                    x = seq_tensor[:-1]  # (max_seq_len,)
                    y = seq_tensor[1:]   # (max_seq_len,)
                    yield (x, y)
                buffer.clear()
