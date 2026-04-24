from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    """
    Fija semillas globales para reproducibilidad:

    - random
    - numpy
    - torch (CPU y GPU)

    Parameters
    ----------
    seed:
        Semilla a usar.
    deterministic:
        Si True, activa modos más deterministas en PyTorch
        (puede impactar rendimiento).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True