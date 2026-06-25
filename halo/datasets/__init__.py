"""
HALO-S Datasets — Datasets sintéticos y de texto para entrenamiento.
"""

from halo.datasets.synthetic import CopyDataset, NeedleDataset
from halo.datasets.jsonl import JSONLDataset

__all__ = ["CopyDataset", "NeedleDataset", "JSONLDataset"]
