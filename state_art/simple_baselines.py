"""
Baselines simples para comparación:
  - PersistenceModel: predice el último valor observado.
  - LinearBaselineModel: regresión lineal sobre la historia.

Ambos siguen la misma interfaz forward() que TimeSeriesTransformer,
y se pueden entrenar con el mismo Trainer (aunque Persistence no aprende nada).
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, cast


class PersistenceModel(nn.Module):
    """
    Baseline de persistencia: predice el último valor observado de la historia
    para todos los horizontes futuros.

    No tiene parámetros entrenables; se usa sólo para evaluación.
    Se le asigna un parámetro dummy para que el Trainer/optimizer no falle.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Parámetro dummy para que el optimizer no falle
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(
        self,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        is_target_mask: torch.Tensor,
        input_sensor_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.Tensor | Dict[str, Any]:
        B, L, D = input_values.shape

        # Número de tokens target
        target_counts = is_target_mask.sum(dim=1)
        num_targets = int(target_counts[0].item())

        # Máscara de historia válida: ni target ni padding
        if padding_mask is not None:
            hist_mask = ~is_target_mask & ~padding_mask  # [B, L], True=valid hist
        else:
            hist_mask = ~is_target_mask

        # Último índice válido de historia por muestra
        idx = hist_mask.long().cumsum(dim=1).argmax(dim=1)  # [B]
        last_values = input_values[
            torch.arange(B, device=input_values.device), idx, :
        ]  # [B, D]

        # Repetir para cada horizonte target
        # Salida: [B, M, output_dim]
        preds = last_values[:, : self.output_dim].unsqueeze(1).expand(
            -1, num_targets, -1
        )

        # Añadir dummy para que haya gradiente (no afecta el resultado)
        preds = preds + self._dummy * 0.0

        if preds.shape[1] == 1:
            preds = preds.squeeze(1)

        if not return_dict:
            return preds
        return {"preds": preds}


class LinearBaselineModel(nn.Module):
    """
    Baseline lineal: proyección lineal sobre los últimos N valores de la historia
    para predecir cada horizonte futuro.

    Tiene parámetros entrenables (una capa lineal por horizonte no es práctico
    con num_targets variable, así que usamos un MLP simple).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        d_model: int = 64,
        max_history: int = 50,
        time_scale: float = 900.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_history = max_history
        self.time_scale = time_scale  # normaliza timestamps relativos (evita overflow FP16)

        # Encoder: aplana los últimos max_history valores + tiempo relativo
        flat_dim = max_history * (input_dim + 1)  # valores + timestamps relativos
        self.encoder = nn.Sequential(
            nn.Linear(flat_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        # Decodificador: dado el embedding + tiempo target, predice output_dim
        self.decoder = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        is_target_mask: torch.Tensor,
        input_sensor_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.Tensor | Dict[str, Any]:
        B, L, D = input_values.shape

        # Máscara de historia = ni target ni padding
        if padding_mask is not None:
            hist_mask = ~is_target_mask & ~padding_mask  # [B, L] True=valid hist
        else:
            hist_mask = ~is_target_mask

        target_counts = is_target_mask.sum(dim=1)
        if not torch.all(target_counts == target_counts[0]):
            raise ValueError("Todos los elementos del batch deben tener el mismo número de targets.")
        num_targets = int(target_counts[0].item())
        if num_targets <= 0:
            raise ValueError("Se requiere al menos un target por muestra.")

        # Vectorización GPU-friendly: extrae historia/targets sin bucles Python.
        seq_idx = torch.arange(L, device=input_values.device).unsqueeze(0).expand(B, -1)

        hist_idx_masked = torch.where(hist_mask, seq_idx, torch.full_like(seq_idx, -1))
        last_hist_idx = hist_idx_masked.max(dim=1).values
        if torch.any(last_hist_idx < 0):
            raise ValueError("Cada muestra debe tener al menos un paso válido de historia.")

        k_hist = min(self.max_history, L)
        hist_idx = torch.topk(
            hist_idx_masked,
            k=k_hist,
            dim=1,
            largest=True,
            sorted=True,
        ).values
        hist_idx = torch.flip(hist_idx, dims=[1])  # más antiguo -> más reciente

        if k_hist < self.max_history:
            left_pad = torch.full(
                (B, self.max_history - k_hist),
                -1,
                dtype=torch.long,
                device=input_values.device,
            )
            hist_idx = torch.cat([left_pad, hist_idx], dim=1)

        hist_valid = hist_idx >= 0
        safe_hist_idx = hist_idx.clamp_min(0)

        hist_vals = input_values.gather(
            1,
            safe_hist_idx.unsqueeze(-1).expand(-1, -1, D),
        )
        hist_vals = hist_vals * hist_valid.unsqueeze(-1).to(hist_vals.dtype)

        hist_ts = input_timestamps.gather(1, safe_hist_idx)
        last_hist_ts = input_timestamps.gather(1, last_hist_idx.unsqueeze(1)).squeeze(1)
        hist_trel = (hist_ts - last_hist_ts.unsqueeze(1)) / self.time_scale
        hist_trel = hist_trel * hist_valid.to(hist_trel.dtype)

        tgt_idx_masked = torch.where(
            is_target_mask,
            seq_idx,
            torch.full_like(seq_idx, -1),
        )
        if torch.any(tgt_idx_masked.max(dim=1).values < 0):
            raise ValueError("Cada muestra debe tener al menos un target válido.")

        k_tgt = min(num_targets, L)
        tgt_idx = torch.topk(
            tgt_idx_masked,
            k=k_tgt,
            dim=1,
            largest=True,
            sorted=True,
        ).values
        tgt_idx = torch.flip(tgt_idx, dims=[1])

        if k_tgt < num_targets:
            left_pad_tgt = torch.full(
                (B, num_targets - k_tgt),
                -1,
                dtype=torch.long,
                device=input_values.device,
            )
            tgt_idx = torch.cat([left_pad_tgt, tgt_idx], dim=1)

        tgt_valid = tgt_idx >= 0
        safe_tgt_idx = tgt_idx.clamp_min(0)
        tgt_ts = input_timestamps.gather(1, safe_tgt_idx)
        tgt_trel = (tgt_ts - last_hist_ts.unsqueeze(1)) / self.time_scale
        tgt_trel = tgt_trel * tgt_valid.to(tgt_trel.dtype)

        # Concatenar valores + timestamps relativos y aplanar
        hist_combined = torch.cat(
            [hist_vals, hist_trel.unsqueeze(-1)], dim=-1
        )  # [B, max_history, D+1]
        flat = hist_combined.reshape(B, -1)  # [B, max_history*(D+1)]

        h = self.encoder(flat)  # [B, d_model]

        # Expandir h para cada target
        h_exp = h.unsqueeze(1).expand(-1, num_targets, -1)  # [B, M, d_model]
        t_exp = tgt_trel.unsqueeze(-1)  # [B, M, 1]

        decoder_input = torch.cat([h_exp, t_exp], dim=-1)  # [B, M, d_model+1]
        preds = self.decoder(decoder_input)  # [B, M, output_dim]

        if preds.shape[1] == 1:
            preds = preds.squeeze(1)

        if not return_dict:
            return preds
        return {"preds": preds}


class NoTimeEncodingTransformer(nn.Module):
    """
    Variante de ablación: Transformer idéntico al Custom pero SIN encoding
    temporal continuo. Usa positional encoding estándar (ordinal).

    Sirve para demostrar el aporte del time encoding continuo.
    """

    def __init__(self, config):
        super().__init__()
        from ts_transformer.models.time_series_transformer import TimeSeriesTransformer

        self.inner = TimeSeriesTransformer(config)
        # Reemplazar time_encoding por un positional encoding ordinal
        d_model = config.d_model
        self.inner.time_encoding = OrdinalPositionalEncoding(d_model)

    def forward(self, **kwargs):
        return self.inner(**kwargs)

    # Hacer accesible encoder para freeze schedule
    @property
    def encoder(self):
        return self.inner.encoder


class OrdinalPositionalEncoding(nn.Module):
    """Positional encoding estándar basado en posición ordinal (0,1,2,...),
    ignorando completamente los timestamps reales."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe_buffer", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(
        self,
        timestamps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L = timestamps.shape
        pe_buffer = cast(torch.Tensor, self.pe_buffer)
        return pe_buffer[:, :L, :].expand(B, -1, -1)


class NoTargetTokenTransformer(nn.Module):
    """
    Variante de ablación: Transformer SIN target token embedding.
    Se desactiva completamente la rama de flag embedding para evitar
    cómputo y asignaciones innecesarias.

    Sirve para demostrar el aporte del mecanismo de target token.
    """

    def __init__(self, config):
        super().__init__()
        from ts_transformer.models.time_series_transformer import TimeSeriesTransformer

        config.use_target_flag_embedding = False
        config.validate_inputs = False
        self.inner = TimeSeriesTransformer(config)

    def forward(self, **kwargs):
        return self.inner(**kwargs)

    @property
    def encoder(self):
        return self.inner.encoder
