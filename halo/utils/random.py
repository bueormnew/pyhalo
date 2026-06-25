import random
import torch
import numpy as np

def set_seed(seed: int = 42):
    """
    Fija todas las semillas necesarias para asegurar reproducibilidad
    entre experimentos y pasadas.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
