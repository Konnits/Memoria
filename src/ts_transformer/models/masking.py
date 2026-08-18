from __future__ import annotations

from typing import Optional

import torch


def validate_target_mask_structure(
    is_target_mask: torch.Tensor,
    batch_size: int,
    seq_len: int,
) -> int:
    """Valida el contrato estructural de los tokens objetivo.

    Todas las secuencias deben contener la misma cantidad positiva de targets y
    estos deben formar un bloque contiguo al final. Devuelve el número de tokens
    objetivo por secuencia para que los modelos puedan hacer slicing seguro.
    """
    expected_shape = (batch_size, seq_len)
    if is_target_mask.shape != expected_shape:
        raise ValueError(
            f"is_target_mask debe tener shape [B, L]={expected_shape}, "
            f"pero se obtuvo {tuple(is_target_mask.shape)}."
        )

    target_counts = is_target_mask.sum(dim=1)
    if not torch.all(target_counts > 0):
        raise ValueError(
            "Cada secuencia debe tener al menos un token target "
            f"(is_target_mask True). Se obtuvieron cuentas {target_counts.tolist()}."
        )
    if not torch.all(target_counts == target_counts[0]):
        raise ValueError(
            "Todas las secuencias del batch deben tener el mismo número de "
            f"target tokens. Se obtuvieron cuentas {target_counts.tolist()}."
        )

    num_target_tokens = int(target_counts[0].item())
    expected_mask = torch.zeros_like(is_target_mask, dtype=torch.bool)
    expected_mask[:, -num_target_tokens:] = True
    if not torch.equal(is_target_mask.to(torch.bool), expected_mask):
        raise ValueError(
            "is_target_mask debe indicar un bloque contiguo de tokens target "
            "estrictamente al final de cada secuencia."
        )

    return num_target_tokens


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
