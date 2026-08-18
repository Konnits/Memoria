from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from ..features.sensor_embedding import SensorEmbedding
from ..features.time_encoding import TimePositionalEncoding
from ..features.value_embedding import FeatureEmbedding
from .masking import validate_target_mask_structure
from .query_cross_attention import StableDiagonalContinuousState
from .time_series_transformer import TimeSeriesTransformerConfig
from .transformer_blocks import TransformerEncoder


@dataclass(frozen=True)
class ContinuousBasisDecoderConfig:
    """Opciones del decoder que representa el forecast como una función.

    Los centros y períodos se expresan en unidades de ``time_scale``. El
    decoder no asigna embeddings ordinales a los targets: cada salida depende
    de su horizonte físico respecto del último evento histórico válido.
    """

    trend_degree: int = 2
    num_rbf_bases: int = 8
    num_fourier_frequencies: int = 4
    min_basis_scale: float = 0.25
    max_basis_scale: float = 64.0
    rbf_width_multiplier: float = 0.75
    temporal_feature_dim: int = 0
    derive_gap_features: bool = True
    use_history_time_encoding: bool = True
    use_ctssm: bool = False
    use_last_value_residual: bool = True

    def __post_init__(self) -> None:
        if self.trend_degree < 0:
            raise ValueError("trend_degree debe ser >= 0.")
        if self.num_rbf_bases < 0:
            raise ValueError("num_rbf_bases debe ser >= 0.")
        if self.num_fourier_frequencies < 0:
            raise ValueError("num_fourier_frequencies debe ser >= 0.")
        if self.num_basis_functions < 1:
            raise ValueError("Se requiere al menos una función base.")
        if self.min_basis_scale <= 0.0 or self.max_basis_scale <= 0.0:
            raise ValueError("Las escalas de las bases deben ser positivas.")
        if self.min_basis_scale > self.max_basis_scale:
            raise ValueError("min_basis_scale no puede superar max_basis_scale.")
        if self.rbf_width_multiplier <= 0.0:
            raise ValueError("rbf_width_multiplier debe ser positivo.")
        if self.temporal_feature_dim < 0:
            raise ValueError("temporal_feature_dim debe ser >= 0.")

    @property
    def num_basis_functions(self) -> int:
        return (
            self.trend_degree
            + 1
            + self.num_rbf_bases
            + 2 * self.num_fourier_frequencies
        )


class ContinuousHorizonBasis(nn.Module):
    """Bases trend + RBF + Fourier evaluadas en tiempo físico continuo."""

    def __init__(self, config: ContinuousBasisDecoderConfig) -> None:
        super().__init__()
        self.config = config
        if config.num_rbf_bases:
            rbf_centers = torch.logspace(
                math.log10(config.min_basis_scale),
                math.log10(config.max_basis_scale),
                steps=config.num_rbf_bases,
                dtype=torch.float32,
            )
            rbf_widths = (rbf_centers * config.rbf_width_multiplier).clamp_min(
                config.min_basis_scale * 0.25
            )
        else:
            rbf_centers = torch.empty(0, dtype=torch.float32)
            rbf_widths = torch.empty(0, dtype=torch.float32)
        if config.num_fourier_frequencies:
            fourier_periods = torch.logspace(
                math.log10(config.min_basis_scale),
                math.log10(config.max_basis_scale),
                steps=config.num_fourier_frequencies,
                dtype=torch.float32,
            )
        else:
            fourier_periods = torch.empty(0, dtype=torch.float32)
        self.register_buffer("rbf_centers", rbf_centers, persistent=True)
        self.register_buffer("rbf_widths", rbf_widths, persistent=True)
        self.register_buffer("fourier_periods", fourier_periods, persistent=True)

    @property
    def output_dim(self) -> int:
        return self.config.num_basis_functions

    def forward(self, normalized_horizon: torch.Tensor) -> torch.Tensor:
        """Evalúa las bases; ``normalized_horizon`` puede tener shape arbitrario."""
        horizon = normalized_horizon.to(torch.float32)
        # signed-log conserva continuidad para auditorías que incluyan h < 0 y
        # evita que el trend polinomial explote en horizontes muy largos.
        trend_coordinate = torch.sign(horizon) * torch.log1p(horizon.abs())
        pieces = [torch.ones_like(horizon).unsqueeze(-1)]
        for degree in range(1, self.config.trend_degree + 1):
            pieces.append(trend_coordinate.pow(degree).unsqueeze(-1))

        if self.config.num_rbf_bases:
            centers = self.rbf_centers.to(device=horizon.device)
            widths = self.rbf_widths.to(device=horizon.device)
            distance = (horizon.unsqueeze(-1) - centers) / widths
            pieces.append(torch.exp(-0.5 * distance.square()))

        if self.config.num_fourier_frequencies:
            periods = self.fourier_periods.to(device=horizon.device)
            phase = 2.0 * math.pi * horizon.unsqueeze(-1) / periods
            pieces.extend((phase.sin(), phase.cos() - 1.0))
        return torch.cat(pieces, dim=-1)


class TimeSeriesContinuousBasisDecoder(nn.Module):
    """Encoder histórico + decoder eficiente de función temporal continua.

    La historia se codifica una sola vez. A partir de su estado global (y del
    estado por sensor en modo evento) se predicen coeficientes para bases
    multiescala. Evaluar ``K`` horizontes cuesta ``O(K * n_bases)`` y no hay
    interacción ni embedding de slot entre targets.
    """

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        basis_config: Optional[ContinuousBasisDecoderConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.basis_config = basis_config or ContinuousBasisDecoderConfig()
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.d_model = int(config.d_model)
        self.use_sensor_embedding = bool(config.use_sensor_embedding)
        self.prediction_head = str(config.prediction_head).lower()
        if self.prediction_head not in {"point", "gaussian"}:
            raise ValueError("prediction_head debe ser 'point' o 'gaussian'.")
        if self.use_sensor_embedding and int(config.num_sensors) < self.output_dim:
            raise ValueError("num_sensors debe ser >= output_dim en modo evento.")

        self.value_embedding = FeatureEmbedding(
            d_in=self.input_dim, d_model=self.d_model, use_layernorm=True
        )
        # Una configuración ordinal nunca llega al decoder. También se evita
        # que el encoder use el índice como proxy oculto del tiempo.
        history_time_mode = (
            "sinusoidal"
            if str(config.time_encoding_mode).lower() == "ordinal"
            else config.time_encoding_mode
        )
        self.time_encoding = TimePositionalEncoding(
            d_model=self.d_model,
            time_scale=config.time_scale,
            mode=history_time_mode,
            time_transform=config.time_transform,
            learnable_time_scale=config.learnable_time_scale,
        )
        self.time_emb_scale = nn.Parameter(torch.tensor(1.0))
        self.sensor_emb_scale = nn.Parameter(torch.tensor(1.0))
        if self.use_sensor_embedding:
            self.sensor_embedding: Optional[nn.Module] = SensorEmbedding(
                int(config.num_sensors),
                self.d_model,
                include_target_token=False,
            )
        else:
            self.sensor_embedding = None

        self.gap_projection = (
            nn.Linear(2, self.d_model)
            if self.basis_config.derive_gap_features
            else None
        )
        feature_dim = self.basis_config.temporal_feature_dim
        self.history_feature_projection = (
            nn.Linear(feature_dim, self.d_model) if feature_dim else None
        )
        self.target_feature_projection = (
            nn.Linear(feature_dim, self.d_model) if feature_dim else None
        )
        self.history_norm = nn.LayerNorm(self.d_model)
        self.continuous_state = (
            StableDiagonalContinuousState(self.d_model, config.time_scale)
            if self.basis_config.use_ctssm
            else None
        )
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )

        fusion_inputs = 3 if self.use_sensor_embedding else 2
        self.context_network = nn.Sequential(
            nn.Linear(fusion_inputs * self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.LayerNorm(self.d_model),
        )
        self.basis = ContinuousHorizonBasis(self.basis_config)
        parameter_multiplier = 2 if self.prediction_head == "gaussian" else 1
        coefficient_outputs = self.basis.output_dim * parameter_multiplier
        if not self.use_sensor_embedding:
            coefficient_outputs *= self.output_dim
        self.coefficient_projection = nn.Linear(self.d_model, coefficient_outputs)
        self.target_coefficient_projection = (
            nn.Linear(self.d_model, coefficient_outputs, bias=False)
            if feature_dim
            else None
        )
        # Arrancar desde la referencia de persistencia evita que la suma de
        # muchas bases aleatorias produzca forecasts (y, en particular,
        # log-scales gaussianos) extremos antes del primer update. La capa
        # sigue siendo completamente aprendible: su primer gradiente actualiza
        # directamente estos coeficientes a partir del contexto codificado.
        nn.init.zeros_(self.coefficient_projection.weight)
        nn.init.zeros_(self.coefficient_projection.bias)
        if self.target_coefficient_projection is not None:
            nn.init.zeros_(self.target_coefficient_projection.weight)

        self.dense_residual_projection: nn.Module = (
            nn.Identity()
            if self.input_dim == self.output_dim
            else nn.Linear(self.input_dim, self.output_dim)
        )
        self.event_residual_projection: nn.Module = (
            nn.Identity() if self.input_dim == 1 else nn.Linear(self.input_dim, 1)
        )
        if not self.basis_config.use_history_time_encoding:
            self.time_emb_scale.requires_grad_(False)
            # El decoder siempre usa ``current_time_scale`` para evaluar las
            # bases continuas. Sólo se congelan Time2Vec/MLP, que pertenecen
            # exclusivamente al encoding histórico desactivado.
            shared_scale = getattr(self.time_encoding, "log_time_scale", None)
            for parameter in self.time_encoding.parameters():
                if parameter is not shared_scale:
                    parameter.requires_grad_(False)
        self.sensor_emb_scale.requires_grad_(self.use_sensor_embedding)

    @staticmethod
    def _last_valid_index(valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(valid.shape[1], device=valid.device).view(1, -1)
        return positions.masked_fill(~valid, -1).amax(dim=1)

    def _relative_history_time(
        self,
        timestamps: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        first_idx = valid.to(torch.int64).argmax(dim=1, keepdim=True)
        origin = timestamps.gather(1, first_idx)
        scale = self.time_encoding.current_time_scale(
            device=timestamps.device, dtype=timestamps.dtype
        )
        return timestamps - origin, scale

    def _history_gap_features(
        self,
        relative_time: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        gaps = torch.zeros_like(relative_time)
        if relative_time.shape[1] > 1:
            adjacent = (relative_time[:, 1:] - relative_time[:, :-1]).clamp_min(0.0)
            pair_valid = valid[:, 1:] & valid[:, :-1]
            gaps[:, 1:] = torch.where(pair_valid, adjacent, 0.0)
        return torch.stack((torch.log1p(gaps), torch.exp(-gaps)), dim=-1)

    def _canonical_target_sensor_ids(
        self,
        input_sensor_ids: torch.Tensor,
        history_len: int,
        num_targets: int,
        target_timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_sensor_ids = input_sensor_ids[:, :history_len].to(torch.long)
        raw_targets = input_sensor_ids[:, history_len:].to(torch.long)
        if num_targets % self.output_dim:
            raise ValueError(
                f"num_target_tokens={num_targets} no es divisible por "
                f"output_dim={self.output_dim}."
            )
        blocks = target_timestamps.view(
            target_timestamps.shape[0], num_targets // self.output_dim, self.output_dim
        )
        if not torch.all(blocks == blocks[..., :1]):
            raise ValueError(
                "Cada bloque contiguo de output_dim targets debe compartir timestamp."
            )
        special_id = int(self.config.num_sensors)
        uses_special = raw_targets == special_id
        if torch.any(uses_special):
            if not torch.all(uses_special):
                raise ValueError(
                    "Los target ids deben ser reales o especiales de forma homogénea."
                )
            canonical = torch.arange(
                num_targets, device=raw_targets.device, dtype=torch.long
            ).remainder(self.output_dim)
            target_sensor_ids = canonical.unsqueeze(0).expand_as(raw_targets)
        else:
            target_sensor_ids = raw_targets
            expected = torch.arange(
                self.output_dim, device=raw_targets.device, dtype=torch.long
            ).repeat(num_targets // self.output_dim)
            if not torch.all(target_sensor_ids == expected.unsqueeze(0)):
                raise ValueError(
                    "Cada bloque target debe ordenar sensores como 0..output_dim-1."
                )
        return history_sensor_ids, target_sensor_ids

    def _event_context(
        self,
        encoded: torch.Tensor,
        valid: torch.Tensor,
        history_sensor_ids: torch.Tensor,
        target_sensor_ids: torch.Tensor,
    ) -> torch.Tensor:
        num_sensors = int(self.config.num_sensors)
        one_hot = F.one_hot(
            history_sensor_ids.clamp(0, num_sensors - 1), num_sensors
        ).to(encoded.dtype)
        one_hot = one_hot * valid.unsqueeze(-1).to(encoded.dtype)
        sensor_counts = one_hot.sum(dim=1).clamp_min(1.0)
        sensor_mean = torch.einsum("bls,bld->bsd", one_hot, encoded)
        sensor_mean = sensor_mean / sensor_counts.unsqueeze(-1)

        positions = torch.arange(encoded.shape[1], device=encoded.device).view(1, -1, 1)
        last_positions = positions.expand(encoded.shape[0], -1, num_sensors).masked_fill(
            one_hot.to(torch.bool).logical_not(), -1
        ).amax(dim=1)
        safe_positions = last_positions.clamp_min(0)
        gather_index = safe_positions.unsqueeze(-1).expand(-1, -1, self.d_model)
        sensor_last = encoded.gather(1, gather_index)
        sensor_last = torch.where(
            (last_positions >= 0).unsqueeze(-1),
            sensor_last,
            torch.zeros_like(sensor_last),
        )

        global_mean = (encoded * valid.unsqueeze(-1)).sum(dim=1)
        global_mean = global_mean / valid.sum(dim=1, keepdim=True).clamp_min(1)
        selected_mean = sensor_mean.gather(
            1, target_sensor_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        selected_last = sensor_last.gather(
            1, target_sensor_ids.unsqueeze(-1).expand(-1, -1, self.d_model)
        )
        global_targets = global_mean.unsqueeze(1).expand_as(selected_mean)
        return self.context_network(
            torch.cat((global_targets, selected_mean, selected_last), dim=-1)
        )

    def _last_value_baseline(
        self,
        history_values: torch.Tensor,
        valid: torch.Tensor,
        history_sensor_ids: Optional[torch.Tensor],
        target_sensor_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch = history_values.shape[0]
        if not self.use_sensor_embedding:
            last_idx = self._last_valid_index(valid)
            last = history_values[
                torch.arange(batch, device=history_values.device), last_idx
            ]
            return self.dense_residual_projection(last)
        if history_sensor_ids is None or target_sensor_ids is None:
            raise RuntimeError("Faltan ids de sensor para el residual event-based.")
        projected = self.event_residual_projection(history_values).squeeze(-1)
        positions = torch.arange(history_values.shape[1], device=history_values.device)
        positions = positions.view(1, -1, 1)
        matches = (
            history_sensor_ids.unsqueeze(-1) == target_sensor_ids.unsqueeze(1)
        ) & valid.unsqueeze(-1)
        last_position = positions.masked_fill(~matches, -1).amax(dim=1)
        safe_position = last_position.clamp_min(0)
        baseline = projected.gather(1, safe_position)
        return torch.where(last_position >= 0, baseline, torch.zeros_like(baseline))

    def forward(
        self,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        is_target_mask: torch.Tensor,
        input_sensor_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        return_all_layers: bool = False,
        temporal_features: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        del attn_mask, return_attention_weights
        if input_values.ndim != 3:
            raise ValueError("input_values debe tener shape [B,L,input_dim].")
        batch, total_len, input_dim = input_values.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Se esperaba input_dim={self.input_dim}, se recibió {input_dim}.")
        if input_timestamps.shape != (batch, total_len):
            raise ValueError("input_timestamps debe tener shape [B,L].")
        if padding_mask is not None and padding_mask.shape != (batch, total_len):
            raise ValueError("padding_mask debe tener shape [B,L].")
        if lengths is not None:
            lengths = torch.as_tensor(lengths, device=input_values.device)
            if lengths.shape != (batch,):
                raise ValueError("lengths debe tener shape [B].")
            if lengths.dtype.is_floating_point and not torch.all(lengths == lengths.round()):
                raise ValueError("lengths debe contener enteros.")
            lengths = lengths.to(torch.long)
            if torch.any(lengths < 1) or torch.any(lengths > total_len):
                raise ValueError("lengths debe estar en el rango [1,L].")
            positions = torch.arange(total_len, device=input_values.device).unsqueeze(0)
            lengths_padding = positions < (total_len - lengths).unsqueeze(1)
            if padding_mask is None:
                padding_mask = lengths_padding
            elif not torch.equal(padding_mask, lengths_padding):
                raise ValueError("padding_mask y lengths describen secuencias distintas.")

        num_targets = validate_target_mask_structure(
            is_target_mask, batch_size=batch, seq_len=total_len
        )
        history_len = total_len - num_targets
        if history_len < 1:
            raise ValueError("Se requiere al menos un token histórico.")
        history_padding = padding_mask[:, :history_len] if padding_mask is not None else None
        if padding_mask is not None and torch.any(padding_mask[:, history_len:]):
            raise ValueError("Los tokens target no pueden ser padding.")
        valid = (
            torch.ones(batch, history_len, dtype=torch.bool, device=input_values.device)
            if history_padding is None
            else ~history_padding
        )
        if not torch.all(valid.any(dim=1)):
            raise ValueError("Cada secuencia requiere al menos un evento histórico válido.")

        if temporal_features is not None:
            expected = (batch, total_len, self.basis_config.temporal_feature_dim)
            if temporal_features.shape != expected:
                raise ValueError(
                    f"temporal_features debe tener shape {expected}; "
                    f"se recibió {tuple(temporal_features.shape)}."
                )
        elif self.basis_config.temporal_feature_dim:
            temporal_features = input_values.new_zeros(
                batch, total_len, self.basis_config.temporal_feature_dim
            )

        history_values = input_values[:, :history_len]
        history_times = input_timestamps[:, :history_len]
        target_times = input_timestamps[:, history_len:]
        history = self.value_embedding(history_values)
        relative_history, time_scale = self._relative_history_time(history_times, valid)
        if self.basis_config.use_history_time_encoding:
            history = history + self.time_emb_scale.to(history.dtype) * self.time_encoding(
                relative_history.to(history.dtype),
                padding_mask=history_padding,
                lengths=valid.sum(dim=1) if history_padding is not None else None,
            ).to(history.dtype)
        if self.gap_projection is not None:
            history = history + self.gap_projection(
                self._history_gap_features(
                    relative_history / time_scale, valid
                ).to(history.dtype)
            )
        if self.history_feature_projection is not None and temporal_features is not None:
            history = history + self.history_feature_projection(
                temporal_features[:, :history_len].to(history.dtype)
            )

        history_sensor_ids: Optional[torch.Tensor] = None
        target_sensor_ids: Optional[torch.Tensor] = None
        if self.use_sensor_embedding:
            if input_sensor_ids is None or input_sensor_ids.shape != (batch, total_len):
                raise ValueError("input_sensor_ids [B,L] es requerido en modo evento.")
            history_sensor_ids, target_sensor_ids = self._canonical_target_sensor_ids(
                input_sensor_ids, history_len, num_targets, target_times
            )
            if torch.any(history_sensor_ids < 0) or torch.any(
                history_sensor_ids >= int(self.config.num_sensors)
            ):
                raise ValueError("Los ids históricos están fuera del rango de sensores.")
            if self.sensor_embedding is None:
                raise RuntimeError("sensor_embedding no inicializado.")
            history = history + self.sensor_emb_scale.to(history.dtype) * self.sensor_embedding(
                history_sensor_ids
            ).to(history.dtype)

        history = self.history_norm(history)
        if self.continuous_state is not None:
            history = self.continuous_state(
                history,
                history_times,
                history_padding,
                time_scale=time_scale,
            )
        if return_all_layers:
            encoded, encoder_layers = self.encoder(
                history, key_padding_mask=history_padding, return_all_layers=True
            )
        else:
            encoded = self.encoder(
                history, key_padding_mask=history_padding, return_all_layers=False
            )
            encoder_layers = None

        last_idx = self._last_valid_index(valid)
        last_time = history_times.gather(1, last_idx.unsqueeze(1))
        # La resta ocurre antes del cast a float32 para conservar gaps pequeños
        # cuando los timestamps absolutos son grandes.
        normalized_horizon = ((target_times - last_time) / time_scale).to(torch.float32)
        horizon_basis = self.basis(normalized_horizon).to(encoded.dtype)

        if self.use_sensor_embedding:
            if history_sensor_ids is None or target_sensor_ids is None:
                raise RuntimeError("Faltan ids canónicos de sensor.")
            target_context = self._event_context(
                encoded, valid, history_sensor_ids, target_sensor_ids
            )
            if self.sensor_embedding is not None:
                target_context = target_context + self.sensor_embedding(
                    target_sensor_ids
                ).to(target_context.dtype)
            parameters = self.coefficient_projection(target_context)
            if self.target_coefficient_projection is not None and temporal_features is not None:
                target_features = self.target_feature_projection(
                    temporal_features[:, history_len:].to(encoded.dtype)
                )
                parameters = parameters + self.target_coefficient_projection(target_features)
            parameters = parameters.view(
                batch,
                num_targets,
                2 if self.prediction_head == "gaussian" else 1,
                self.basis.output_dim,
            )
            evaluated = torch.einsum("btmp,btp->btm", parameters, horizon_basis)
            predictions_flat = evaluated[..., 0]
            if self.basis_config.use_last_value_residual:
                predictions_flat = predictions_flat + self._last_value_baseline(
                    history_values, valid, history_sensor_ids, target_sensor_ids
                )
            horizons = num_targets // self.output_dim
            predictions = predictions_flat.view(batch, horizons, self.output_dim)
            log_scale = (
                evaluated[..., 1].clamp(-7.0, 5.0).view(
                    batch, horizons, self.output_dim
                )
                if self.prediction_head == "gaussian"
                else None
            )
            coefficient_result = parameters
        else:
            global_mean = (encoded * valid.unsqueeze(-1)).sum(dim=1)
            global_mean = global_mean / valid.sum(dim=1, keepdim=True).clamp_min(1)
            last_state = encoded[
                torch.arange(batch, device=encoded.device), last_idx
            ]
            context = self.context_network(torch.cat((global_mean, last_state), dim=-1))
            parameters = self.coefficient_projection(context).view(
                batch,
                self.output_dim,
                2 if self.prediction_head == "gaussian" else 1,
                self.basis.output_dim,
            )
            if self.target_coefficient_projection is not None and temporal_features is not None:
                target_features = self.target_feature_projection(
                    temporal_features[:, history_len:].to(encoded.dtype)
                )
                delta = self.target_coefficient_projection(target_features).view(
                    batch,
                    num_targets,
                    self.output_dim,
                    2 if self.prediction_head == "gaussian" else 1,
                    self.basis.output_dim,
                )
                target_parameters = parameters.unsqueeze(1) + delta
                evaluated = torch.einsum(
                    "btdmp,btp->btdm", target_parameters, horizon_basis
                )
                coefficient_result = target_parameters
            else:
                evaluated = torch.einsum("bdmp,btp->btdm", parameters, horizon_basis)
                coefficient_result = parameters
            predictions = evaluated[..., 0]
            if self.basis_config.use_last_value_residual:
                predictions = predictions + self._last_value_baseline(
                    history_values, valid, None, None
                ).unsqueeze(1)
            log_scale = (
                evaluated[..., 1].clamp(-7.0, 5.0)
                if self.prediction_head == "gaussian"
                else None
            )
            target_context = context.unsqueeze(1).expand(-1, num_targets, -1)

        if predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
            if log_scale is not None:
                log_scale = log_scale.squeeze(1)
        if not return_dict:
            return predictions
        result: Dict[str, Any] = {
            "preds": predictions,
            "target_states": target_context,
            "encoder_output": encoded,
            "cross_attn_weights": None,
            "relative_lags": normalized_horizon,
            "relative_horizons": normalized_horizon,
            "horizon_basis": horizon_basis,
            "basis_coefficients": coefficient_result,
        }
        if log_scale is not None:
            result["log_scale"] = log_scale
        if return_all_layers:
            result["all_layers"] = {"encoder": encoder_layers, "queries": []}
        return result


CustomContinuousBasisDecoder = TimeSeriesContinuousBasisDecoder


__all__ = [
    "ContinuousBasisDecoderConfig",
    "ContinuousHorizonBasis",
    "CustomContinuousBasisDecoder",
    "TimeSeriesContinuousBasisDecoder",
]
