"""
Dataset JSONL para HALO-S.

Lee archivos JSONL (una línea JSON por ejemplo) con soporte para
carga en memoria (eager) o lectura bajo demanda (lazy).

Uso básico:
    from halo.datasets.jsonl import JSONLDataset
    from halo.tokenizers.char import CharacterTokenizer

    tokenizer = CharacterTokenizer("abcdefghijklmnopqrstuvwxyz ")
    dataset = JSONLDataset("data.jsonl", tokenizer, max_seq_len=128)
    x, y = dataset[0]  # x: tokens[:-1], y: tokens[1:]
"""

import json
import torch
from torch.utils.data import Dataset
from typing import Optional

from halo.core.logging import get_logger
from halo.tokenizers.base import BaseTokenizer

# Logger del módulo
logger = get_logger("halo.datasets.jsonl")


class JSONLDataset(Dataset):
    """
    Dataset que lee archivos JSONL línea por línea.

    Cada línea del archivo debe ser un objeto JSON con un campo de texto
    configurable (por defecto "text"). Soporta dos modos de operación:

    - Eager (lazy=False): Carga y tokeniza todas las líneas durante __init__.
      Más rápido en acceso pero usa más RAM.
    - Lazy (lazy=True): Almacena solo offsets de bytes de cada línea válida.
      Lee y tokeniza bajo demanda en __getitem__. Ideal para archivos grandes.

    Las secuencias se truncan a max_seq_len+1 tokens para producir tuplas
    (x, y) donde x = tokens[:-1] e y = tokens[1:], cada uno de longitud max_seq_len.
    Líneas con menos de max_seq_len+1 tokens se omiten.
    """

    def __init__(
        self,
        file_path: str,
        tokenizer: BaseTokenizer,
        max_seq_len: int,
        text_field: str = "text",
        lazy: bool = False,
    ):
        """
        Inicializa el dataset JSONL.

        Args:
            file_path: Ruta al archivo .jsonl
            tokenizer: Tokenizador compatible con BaseTokenizer
            max_seq_len: Longitud máxima de secuencia (x e y tendrán este largo)
            text_field: Nombre del campo JSON que contiene el texto
            lazy: Si True, usa offsets de bytes en vez de cargar todo en RAM
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.text_field = text_field
        self.lazy = lazy

        # Longitud requerida para producir (x, y) de max_seq_len cada uno
        self._required_len = max_seq_len + 1

        if lazy:
            # Modo lazy: almacenar solo offsets de bytes de líneas válidas
            self._offsets: list[int] = []
            self._scan_offsets()
        else:
            # Modo eager: cargar y tokenizar todas las líneas válidas
            self._sequences: list[torch.Tensor] = []
            self._load_all()

    def _load_all(self) -> None:
        """
        Modo eager: lee y tokeniza todas las líneas del archivo.
        Almacena solo secuencias con suficientes tokens (>= max_seq_len+1).
        Líneas inválidas se omiten con un warning.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                text = self._parse_line(line, line_num)
                if text is None:
                    continue

                tokens = self.tokenizer.encode(text)

                # Omitir secuencias demasiado cortas
                if len(tokens) < self._required_len:
                    continue

                # Truncar a max_seq_len+1 tokens
                tokens = tokens[: self._required_len]
                self._sequences.append(torch.tensor(tokens, dtype=torch.long))

    def _scan_offsets(self) -> None:
        """
        Modo lazy: escanea el archivo para registrar offsets de bytes
        de las líneas válidas (JSON correcto, campo presente, tokens suficientes).
        """
        with open(self.file_path, "rb") as f:
            line_num = 0
            while True:
                offset = f.tell()
                raw_line = f.readline()
                if not raw_line:
                    break

                line_num += 1
                line = raw_line.decode("utf-8").strip()
                if not line:
                    continue

                text = self._parse_line(line, line_num)
                if text is None:
                    continue

                # Verificar que la línea produce suficientes tokens
                tokens = self.tokenizer.encode(text)
                if len(tokens) < self._required_len:
                    continue

                self._offsets.append(offset)

    def _parse_line(self, line: str, line_num: int) -> Optional[str]:
        """
        Parsea una línea JSON y extrae el campo de texto.

        Args:
            line: Contenido de la línea (ya sin whitespace)
            line_num: Número de línea para mensajes de warning

        Returns:
            El texto extraído, o None si la línea es inválida.
        """
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning(
                f"Línea {line_num}: JSON inválido, se omite."
            )
            return None

        if self.text_field not in data:
            logger.warning(
                f"Línea {line_num}: campo '{self.text_field}' no encontrado, se omite."
            )
            return None

        return data[self.text_field]

    def _read_line_at_offset(self, offset: int) -> str:
        """
        Lee una línea del archivo en la posición de bytes indicada.

        Args:
            offset: Posición de bytes en el archivo.

        Returns:
            Contenido de la línea decodificado.
        """
        with open(self.file_path, "rb") as f:
            f.seek(offset)
            raw_line = f.readline()
        return raw_line.decode("utf-8").strip()

    def __len__(self) -> int:
        """Retorna el número de secuencias válidas en el dataset."""
        if self.lazy:
            return len(self._offsets)
        return len(self._sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna una tupla (x, y) para el índice dado.

        x = tokens[:-1] (entrada del modelo)
        y = tokens[1:]  (target desplazado un paso)

        Ambos tensores tienen shape (max_seq_len,).

        Args:
            idx: Índice del ejemplo en el dataset.

        Returns:
            Tupla (x, y) de tensores long de longitud max_seq_len.
        """
        if self.lazy:
            # Leer y tokenizar bajo demanda
            offset = self._offsets[idx]
            line = self._read_line_at_offset(offset)
            data = json.loads(line)
            text = data[self.text_field]
            tokens = self.tokenizer.encode(text)
            tokens = tokens[: self._required_len]
            tokens_tensor = torch.tensor(tokens, dtype=torch.long)
        else:
            # Acceso directo desde memoria
            tokens_tensor = self._sequences[idx]

        # Desplazamiento: x = tokens[:-1], y = tokens[1:]
        x = tokens_tensor[:-1]
        y = tokens_tensor[1:]
        return x, y
