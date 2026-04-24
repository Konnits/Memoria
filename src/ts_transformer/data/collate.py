from __future__ import annotations

from typing import Callable, Sequence, Mapping, Any, Dict, Optional

import torch

from .sequence_builder import SequenceBuilder


class _CollateFn:
    """Callable picklable para usar como collate_fn con num_workers > 0."""

    def __init__(
        self,
        pad_to_max_length: bool,
    ) -> None:
        self.pad_to_max_length = pad_to_max_length

    def __call__(self, samples: Sequence[Mapping[str, Any]]) -> Dict[str, torch.Tensor]:
        # En esta versión optimizada, asumimos que el dataset ya devolvió 
        # el dict procesado (posiblemente vía sequence_builder en __getitem__).
        
        # Longitudes de cada secuencia
        lengths = [p["input_values"].shape[0] for p in samples]
        max_len = max(lengths)

        # Dimensiones base
        first = samples[0]
        input_dim = first["input_values"].shape[1]
        output_dim = first["target_values"].shape[-1]
        targets_len = first["target_values"].shape[0] # M
        batch_size = len(samples)

        has_target_loss_mask = "target_loss_mask" in first

        # Validación de homogeneidad del batch
        for p in samples[1:]:
            if p["target_values"].shape[0] != targets_len:
                raise ValueError("Todas las muestras del batch deben tener el mismo número de targets.")
            if has_target_loss_mask:
                if "target_loss_mask" not in p or p["target_loss_mask"].shape != first["target_loss_mask"].shape:
                    raise ValueError("Todas las muestras del batch deben tener target_loss_mask con la misma shape.")

        if self.pad_to_max_length:
            # Pre-asignación de tensores en CPU (el DataLoader se encarga de pin_memory)
            input_values_batch = torch.zeros(batch_size, max_len, input_dim, dtype=torch.float32)
            input_timestamps_batch = torch.zeros(batch_size, max_len, dtype=torch.float32)
            is_target_mask_batch = torch.zeros(batch_size, max_len, dtype=torch.bool)
            padding_mask_batch = torch.ones(batch_size, max_len, dtype=torch.bool)  # True = padding

            has_sensor_ids = "input_sensor_ids" in first
            input_sensor_ids_batch = torch.zeros(batch_size, max_len, dtype=torch.long) if has_sensor_ids else None

            if has_target_loss_mask:
                tdim = first["target_loss_mask"].shape[-1]
                target_loss_mask_batch = torch.zeros(batch_size, targets_len, tdim, dtype=torch.float32)
            else:
                target_loss_mask_batch = None

            target_values_batch = torch.zeros(batch_size, targets_len, output_dim, dtype=torch.float32)
            target_timestamps_batch = torch.zeros(batch_size, targets_len, dtype=torch.float32)

            for i, p in enumerate(samples):
                L = lengths[i]
                start = max_len - L
                end = max_len

                # Left-padding: mantiene los tokens target al final global.
                input_values_batch[i, start:end] = p["input_values"]
                input_timestamps_batch[i, start:end] = p["input_timestamps"]
                is_target_mask_batch[i, start:end] = p["is_target_mask"]
                padding_mask_batch[i, start:end] = False
                
                if has_sensor_ids:
                    input_sensor_ids_batch[i, start:end] = p["input_sensor_ids"]

                target_values_batch[i] = p["target_values"]
                target_timestamps_batch[i] = p["target_timestamp"]
                if has_target_loss_mask:
                    target_loss_mask_batch[i] = p["target_loss_mask"]

            out = {
                "input_values": input_values_batch,
                "input_timestamps": input_timestamps_batch,
                "is_target_mask": is_target_mask_batch,
                "padding_mask": padding_mask_batch,
                "target_values": target_values_batch,
                "target_timestamps": target_timestamps_batch,
                "lengths": torch.as_tensor(lengths, dtype=torch.long),
            }
            if has_sensor_ids:
                out["input_sensor_ids"] = input_sensor_ids_batch
            if has_target_loss_mask:
                out["target_loss_mask"] = target_loss_mask_batch
            return out
        else:
            # Sin padding: devolvemos listas o stacks simples
            out = {
                "input_values": [p["input_values"] for p in samples],
                "input_timestamps": [p["input_timestamps"] for p in samples],
                "is_target_mask": [p["is_target_mask"] for p in samples],
                "target_values": torch.stack([p["target_values"] for p in samples], dim=0),
                "target_timestamps": torch.stack([p["target_timestamp"] for p in samples], dim=0),
                "lengths": torch.as_tensor(lengths, dtype=torch.long),
            }

            if "input_sensor_ids" in first:
                out["input_sensor_ids"] = [p["input_sensor_ids"] for p in samples]
            if "target_loss_mask" in first:
                out["target_loss_mask"] = torch.stack([p["target_loss_mask"] for p in samples], dim=0)

            return out


def build_collate_fn(
    pad_to_max_length: bool = True,
) -> _CollateFn:
    """
    Construye una función `collate_fn` para usar con un DataLoader de PyTorch.
    """
    return _CollateFn(pad_to_max_length)