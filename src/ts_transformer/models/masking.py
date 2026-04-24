from __future__ import annotations

from typing import Optional

import torch


def create_causal_mask(seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Crea una máscara causal de shape [seq_len, seq_len], donde
    las posiciones (i, j) con j > i se ponen a -inf, y el resto a 0.0.

    Esta máscara está pensada para sumarse a los logits de atención
    antes del softmax (como un attn_mask adicional).

    Parameters
    ----------
    seq_len:
        Longitud de la secuencia.
    device:
        Dispositivo opcional.

    Returns
    -------
    attn_mask:
        Tensor float32 de shape [seq_len, seq_len].
    """
    # Matriz de ceros
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.float32)
    # True donde j > i
    upper = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    mask = mask.masked_fill(upper, float("-inf"))
    return mask