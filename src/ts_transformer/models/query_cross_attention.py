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
from .heads import GaussianRegressionHead, RegressionHead
from .masking import validate_target_mask_structure
from .time_series_transformer import TimeSeriesTransformerConfig
from .transformer_blocks import TransformerEncoder


@dataclass(frozen=True)
class QueryCrossAttentionConfig:
    """Opciones exclusivas del predictor con consultas temporales continuas.

    ``temporal_feature_dim`` describe sólo las features externas pasadas a
    :meth:`TimeSeriesQueryCrossAttention.forward`. Si
    ``derive_temporal_features`` está activo se agregan, además, tres features
    calculadas por el modelo: gap global, edad del sensor y densidad local.

    Los flags temporales son ortogonales para construir ablaciones limpias:
    ``use_relative_time_bias`` controla sólo el kernel query--observación,
    ``use_temporal_film`` la modulación por gaps/edad/densidad,
    ``use_query_horizon`` el horizonte físico explícito,
    ``use_history_time_encoding`` el encoding temporal de historia y
    ``use_ctssm`` la transición continua entre eventos. La relación de sensor
    tiene su propio flag y nunca usa el índice del token target.
    """

    num_cross_layers: Optional[int] = None
    temporal_feature_dim: int = 0
    derive_temporal_features: bool = True
    use_relative_time_bias: bool = True
    use_temporal_film: bool = True
    use_query_horizon: bool = True
    use_history_time_encoding: bool = True
    use_sensor_relation_bias: bool = True
    lag_num_frequencies: int = 4
    lag_min_scale: float = 0.25
    lag_max_scale: float = 64.0
    use_last_value_residual: bool = True
    use_ctssm: bool = False
    mask_history_after_query: bool = True

    def __post_init__(self) -> None:
        if self.num_cross_layers is not None and self.num_cross_layers < 1:
            raise ValueError("num_cross_layers debe ser >= 1.")
        if self.temporal_feature_dim < 0:
            raise ValueError("temporal_feature_dim debe ser >= 0.")
        if self.lag_num_frequencies < 1:
            raise ValueError("lag_num_frequencies debe ser >= 1.")
        if self.lag_min_scale <= 0.0 or self.lag_max_scale <= 0.0:
            raise ValueError("Las escalas de lag deben ser positivas.")
        if self.lag_min_scale > self.lag_max_scale:
            raise ValueError("lag_min_scale no puede superar lag_max_scale.")


def _inverse_softplus(value: torch.Tensor) -> torch.Tensor:
    return value + torch.log(-torch.expm1(-value))


class RelativeLagBias(nn.Module):
    """Bias por cabeza para ``query_time - observation_time``.

    Cada cabeza combina una penalización monótona estable con bases Fourier y
    RBF multiescala. Las partes flexibles parten en cero para que el modelo
    comience con una preferencia reciente interpretable, pero pueden aprender
    periodicidad o dependencias no monótonas.
    """

    def __init__(
        self,
        num_heads: int,
        num_frequencies: int = 4,
        min_scale: float = 0.25,
        max_scale: float = 64.0,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads debe ser >= 1.")
        if num_frequencies < 1:
            raise ValueError("num_frequencies debe ser >= 1.")
        if min_scale <= 0.0 or max_scale <= 0.0 or min_scale > max_scale:
            raise ValueError("Se requieren 0 < min_scale <= max_scale.")

        self.num_heads = int(num_heads)
        self.num_frequencies = int(num_frequencies)

        head_scales = torch.logspace(
            math.log10(min_scale),
            math.log10(max_scale),
            steps=num_heads,
            dtype=torch.float32,
        )
        initial_rates = head_scales.reciprocal().clamp_min(1e-4)
        self.raw_decay_rate = nn.Parameter(_inverse_softplus(initial_rates))

        basis_scales = torch.logspace(
            math.log10(min_scale),
            math.log10(max_scale),
            steps=num_frequencies,
            dtype=torch.float32,
        )
        self.log_fourier_frequency = nn.Parameter(
            (2.0 * math.pi / basis_scales)
            .log()
            .unsqueeze(0)
            .expand(num_heads, -1)
            .clone()
        )
        self.log_rbf_scale = nn.Parameter(
            basis_scales.log().unsqueeze(0).expand(num_heads, -1).clone()
        )
        self.fourier_sin_weight = nn.Parameter(
            torch.zeros(num_heads, num_frequencies)
        )
        self.fourier_cos_weight = nn.Parameter(
            torch.zeros(num_heads, num_frequencies)
        )
        self.rbf_weight = nn.Parameter(torch.zeros(num_heads, num_frequencies))

    def forward(
        self,
        query_timestamps: torch.Tensor,
        key_timestamps: torch.Tensor,
        *,
        time_scale: float | torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_timestamps.ndim != 2 or key_timestamps.ndim != 2:
            raise ValueError("query_timestamps y key_timestamps deben ser [B, L].")
        if query_timestamps.shape[0] != key_timestamps.shape[0]:
            raise ValueError("Queries y keys deben tener el mismo batch size.")

        # Restar antes de convertir a float32 conserva gaps pequeños cuando los
        # timestamps absolutos llegan en float64.
        lag = query_timestamps.unsqueeze(-1) - key_timestamps.unsqueeze(-2)
        if isinstance(time_scale, torch.Tensor):
            scale = time_scale.to(device=lag.device, dtype=lag.dtype)
        else:
            scale = torch.as_tensor(time_scale, device=lag.device, dtype=lag.dtype)
        if scale.numel() != 1 or bool((scale <= 0).detach().item()):
            raise ValueError("time_scale debe ser un escalar positivo.")
        lag = (lag / scale).to(torch.float32)
        nonnegative_lag = lag.clamp_min(0.0)

        decay_rate = F.softplus(self.raw_decay_rate).view(1, -1, 1, 1)
        bias = -decay_rate * nonnegative_lag.unsqueeze(1)

        expanded_lag = nonnegative_lag.unsqueeze(1).unsqueeze(-1)
        frequency = self.log_fourier_frequency.clamp(-9.0, 9.0).exp()
        phase = expanded_lag * frequency.view(1, self.num_heads, 1, 1, -1)
        sin_term = (
            phase.sin()
            * self.fourier_sin_weight.view(1, self.num_heads, 1, 1, -1)
        ).sum(dim=-1)
        # cos(x)-1 evita introducir un offset dependiente de la base en lag=0.
        cos_term = (
            (phase.cos() - 1.0)
            * self.fourier_cos_weight.view(1, self.num_heads, 1, 1, -1)
        ).sum(dim=-1)

        rbf_scale = self.log_rbf_scale.clamp(-9.0, 9.0).exp()
        rbf = torch.exp(
            -expanded_lag / rbf_scale.view(1, self.num_heads, 1, 1, -1)
        )
        rbf_term = (
            (rbf - 1.0)
            * self.rbf_weight.view(1, self.num_heads, 1, 1, -1)
        ).sum(dim=-1)

        normalization = math.sqrt(float(self.num_frequencies))
        bias = bias + (sin_term + cos_term + rbf_term) / normalization
        output_dtype = dtype if dtype is not None else query_timestamps.dtype
        return bias.clamp(-40.0, 40.0).to(output_dtype), lag


class RelativeTimeCrossAttention(nn.Module):
    """Cross-attention con un score temporal explícito por cabeza."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        *,
        num_lag_frequencies: int,
        lag_min_scale: float,
        lag_max_scale: float,
        mask_history_after_query: bool,
        use_relative_time_bias: bool,
        use_sensor_relation_bias: bool,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model debe ser divisible por num_heads.")
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = d_model // num_heads
        self.mask_history_after_query = bool(mask_history_after_query)
        self.use_relative_time_bias = bool(use_relative_time_bias)
        self.use_sensor_relation_bias = bool(use_sensor_relation_bias)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.lag_bias = RelativeLagBias(
            num_heads,
            num_frequencies=num_lag_frequencies,
            min_scale=lag_min_scale,
            max_scale=lag_max_scale,
        )
        # Sólo el contraste same-vs-cross es identificable: sumar un offset
        # uniforme a todos los keys de una cabeza se cancela en el softmax.
        if self.use_sensor_relation_bias:
            self.sensor_relation_bias = nn.Parameter(
                torch.full((num_heads,), 0.25)
            )
        else:
            self.register_parameter("sensor_relation_bias", None)
        if not self.use_relative_time_bias:
            self.lag_bias.requires_grad_(False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        return x.view(batch, length, self.num_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_timestamps: torch.Tensor,
        memory_timestamps: torch.Tensor,
        *,
        time_scale: float | torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        query_sensor_ids: Optional[torch.Tensor] = None,
        memory_sensor_ids: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        batch, query_len, _ = query.shape
        memory_len = memory.shape[1]
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(memory))
        v = self._split_heads(self.v_proj(memory))

        # Construir sólo el bias aditivo en la ruta SDPA. Antes se calculaba
        # QK^T aquí y ``scaled_dot_product_attention`` volvía a calcularlo,
        # duplicando el trabajo y la memoria temporal.
        additive_bias = q.new_zeros(batch, self.num_heads, query_len, memory_len)
        if self.use_relative_time_bias:
            temporal_bias, lag = self.lag_bias(
                query_timestamps,
                memory_timestamps,
                time_scale=time_scale,
                dtype=q.dtype,
            )
            additive_bias = additive_bias + temporal_bias
        else:
            raw_lag = query_timestamps.unsqueeze(-1) - memory_timestamps.unsqueeze(-2)
            if isinstance(time_scale, torch.Tensor):
                scale = time_scale.to(device=raw_lag.device, dtype=raw_lag.dtype)
            else:
                scale = torch.as_tensor(
                    time_scale, device=raw_lag.device, dtype=raw_lag.dtype
                )
            lag = (raw_lag / scale).to(torch.float32)

        if self.use_sensor_relation_bias and (
            query_sensor_ids is not None or memory_sensor_ids is not None
        ):
            if query_sensor_ids is None or memory_sensor_ids is None:
                raise ValueError(
                    "query_sensor_ids y memory_sensor_ids deben entregarse juntos."
                )
            if query_sensor_ids.shape != (batch, query_len):
                raise ValueError("query_sensor_ids debe tener shape [B,L_target].")
            if memory_sensor_ids.shape != (batch, memory_len):
                raise ValueError("memory_sensor_ids debe tener shape [B,L_history].")
            same_sensor = query_sensor_ids.unsqueeze(-1) == memory_sensor_ids.unsqueeze(-2)
            if self.sensor_relation_bias is None:
                raise RuntimeError("sensor_relation_bias no inicializado.")
            sensor_bias = (
                same_sensor.unsqueeze(1).to(additive_bias.dtype)
                * self.sensor_relation_bias.view(1, -1, 1, 1)
            )
            additive_bias = additive_bias + sensor_bias.to(additive_bias.dtype)

        valid = torch.ones(
            batch, 1, query_len, memory_len, dtype=torch.bool, device=query.device
        )
        if memory_padding_mask is not None:
            if memory_padding_mask.shape != (batch, memory_len):
                raise ValueError("memory_padding_mask debe tener shape [B, L_history].")
            valid = valid & (~memory_padding_mask).view(batch, 1, 1, memory_len)
        if self.mask_history_after_query:
            valid = valid & (lag >= 0.0).unsqueeze(1)

        if attn_mask is not None:
            if attn_mask.ndim == 2 and attn_mask.shape == (query_len, memory_len):
                if attn_mask.dtype == torch.bool:
                    valid = valid & attn_mask.view(1, 1, query_len, memory_len)
                else:
                    additive_bias = additive_bias + attn_mask.to(additive_bias.dtype).view(
                        1, 1, query_len, memory_len
                    )
            elif attn_mask.ndim == 3 and attn_mask.shape == (
                batch * self.num_heads,
                query_len,
                memory_len,
            ):
                reshaped_mask = attn_mask.view(
                    batch, self.num_heads, query_len, memory_len
                )
                if attn_mask.dtype == torch.bool:
                    valid = valid & reshaped_mask
                else:
                    additive_bias = additive_bias + reshaped_mask.to(
                        additive_bias.dtype
                    )
            else:
                raise ValueError(
                    "attn_mask debe ser [L_target, L_history] o "
                    "[B*num_heads, L_target, L_history]."
                )

        if not torch.all(valid.any(dim=-1)):
            raise ValueError(
                "Cada query debe tener al menos una observación histórica válida."
            )
        additive_bias = additive_bias.masked_fill(~valid, float("-inf"))
        if return_attention_weights:
            scores = torch.matmul(
                q / math.sqrt(self.d_head), k.transpose(-2, -1)
            ) + additive_bias
            weights = torch.softmax(scores, dim=-1)
            attended = torch.matmul(self.dropout(weights), v)
        else:
            # SDPA calcula QK^T una sola vez y recibe únicamente los kernels
            # temporales/sensoriales y máscaras como bias aditivo.
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=additive_bias,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            weights = None
        attended = (
            attended.transpose(1, 2).contiguous().view(batch, query_len, self.d_model)
        )
        return self.out_proj(attended), weights, lag


class TemporalFeatureFiLM(nn.Module):
    """Modulación gated/FiLM; preserva identidad al inicializar."""

    def __init__(self, feature_dim: int, d_model: int) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim debe ser >= 1.")
        hidden_dim = max(16, min(2 * d_model, 128))
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * d_model),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        gamma, beta, raw_gate = self.net(self.feature_norm(features)).chunk(3, dim=-1)
        modulated = x * (1.0 + torch.tanh(gamma)) + beta
        gate = torch.sigmoid(raw_gate)
        return x + gate * (modulated - x)


class StableDiagonalContinuousState(nn.Module):
    """Estado continuo solver-free con transición diagonal estable.

    Entre eventos aplica ``z^- = exp(-softplus(rate) * dt) * z`` y en cada
    observación realiza una actualización gated hacia un estado candidato.
    """

    def __init__(self, d_model: int, time_scale: float) -> None:
        super().__init__()
        if d_model < 1 or time_scale <= 0.0:
            raise ValueError("d_model y time_scale deben ser positivos.")
        rates = torch.logspace(-2.0, 1.0, d_model, dtype=torch.float32)
        self.raw_decay_rate = nn.Parameter(_inverse_softplus(rates))
        self.candidate = nn.Linear(d_model, d_model)
        # El gate depende de la observación actual. Esto convierte la
        # recurrencia en transformaciones afines asociativas y permite un scan
        # paralelo O(log L), evitando un kernel Python por evento.
        self.update_gate = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.time_scale = float(time_scale)

    def forward(
        self,
        inputs: torch.Tensor,
        timestamps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        time_scale: Optional[float | torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs.ndim != 3 or timestamps.shape != inputs.shape[:2]:
            raise ValueError("inputs debe ser [B,L,D] y timestamps [B,L].")
        batch, length, d_model = inputs.shape
        if padding_mask is None:
            valid = torch.ones(batch, length, dtype=torch.bool, device=inputs.device)
        else:
            if padding_mask.shape != (batch, length):
                raise ValueError("padding_mask debe tener shape [B,L].")
            valid = ~padding_mask
        if not torch.all(valid.any(dim=1)):
            raise ValueError("Cada secuencia requiere al menos un evento válido.")

        decay_rate = F.softplus(self.raw_decay_rate).to(inputs.dtype).view(1, -1)
        scale_value: float | torch.Tensor = (
            self.time_scale if time_scale is None else time_scale
        )
        scale = torch.as_tensor(
            scale_value,
            device=timestamps.device,
            dtype=timestamps.dtype,
        )
        if scale.numel() != 1 or bool((scale <= 0).detach().item()):
            raise ValueError("time_scale debe ser un escalar positivo.")
        if torch.any(valid[:, :-1] & ~valid[:, 1:]):
            raise ValueError(
                "StableDiagonalContinuousState requiere left-padding contiguo."
            )

        dt = torch.zeros(batch, length, device=timestamps.device, dtype=timestamps.dtype)
        if length > 1:
            adjacent = ((timestamps[:, 1:] - timestamps[:, :-1]) / scale).clamp_min(0.0)
            pair_valid = valid[:, 1:] & valid[:, :-1]
            dt[:, 1:] = torch.where(pair_valid, adjacent, torch.zeros_like(adjacent))

        transition = torch.exp(
            -dt.to(inputs.dtype).unsqueeze(-1) * decay_rate.unsqueeze(1)
        )
        candidate = torch.tanh(self.candidate(inputs))
        gate = torch.sigmoid(self.update_gate(inputs))
        affine_a = transition * (1.0 - gate)
        affine_b = gate * candidate
        valid_expanded = valid.unsqueeze(-1)
        affine_a = torch.where(valid_expanded, affine_a, torch.ones_like(affine_a))
        affine_b = torch.where(valid_expanded, affine_b, torch.zeros_like(affine_b))
        states = self._parallel_affine_scan(affine_a, affine_b)
        emitted = self.norm(inputs + self.output(states))
        return torch.where(valid_expanded, emitted, inputs)

    @staticmethod
    def _parallel_affine_scan(
        coefficients: torch.Tensor,
        innovations: torch.Tensor,
    ) -> torch.Tensor:
        """Prefijo de ``z_i = a_i*z_{i-1}+b_i`` por composición asociativa."""
        if coefficients.shape != innovations.shape or coefficients.ndim != 3:
            raise ValueError("coefficients e innovations deben compartir shape [B,L,D].")
        composed_a = coefficients
        composed_b = innovations
        offset = 1
        length = coefficients.shape[1]
        while offset < length:
            next_a = composed_a.clone()
            next_b = composed_b.clone()
            next_a[:, offset:] = (
                composed_a[:, offset:] * composed_a[:, :-offset]
            )
            next_b[:, offset:] = (
                composed_b[:, offset:]
                + composed_a[:, offset:] * composed_b[:, :-offset]
            )
            composed_a, composed_b = next_a, next_b
            offset *= 2
        return composed_b


class IndependentQueryCrossAttentionBlock(nn.Module):
    """Bloque sin self-attention entre targets: las queries son independientes."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        activation: str,
        options: QueryCrossAttentionConfig,
        *,
        sensor_relations_available: bool,
    ) -> None:
        super().__init__()
        self.norm_cross = nn.LayerNorm(d_model)
        self.norm_ff = nn.LayerNorm(d_model)
        self.cross_attention = RelativeTimeCrossAttention(
            d_model,
            num_heads,
            dropout,
            num_lag_frequencies=options.lag_num_frequencies,
            lag_min_scale=options.lag_min_scale,
            lag_max_scale=options.lag_max_scale,
            mask_history_after_query=options.mask_history_after_query,
            use_relative_time_bias=options.use_relative_time_bias,
            use_sensor_relation_bias=(
                options.use_sensor_relation_bias and sensor_relations_available
            ),
        )
        activation_layer: nn.Module
        if activation == "relu":
            activation_layer = nn.ReLU()
        elif activation == "gelu":
            activation_layer = nn.GELU()
        else:
            raise ValueError(f"Activación no soportada: {activation}")
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            activation_layer,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        query_timestamps: torch.Tensor,
        memory_timestamps: torch.Tensor,
        *,
        time_scale: float | torch.Tensor,
        memory_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
        query_sensor_ids: Optional[torch.Tensor],
        memory_sensor_ids: Optional[torch.Tensor],
        return_attention_weights: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        cross, weights, lag = self.cross_attention(
            self.norm_cross(query),
            memory,
            query_timestamps,
            memory_timestamps,
            time_scale=time_scale,
            memory_padding_mask=memory_padding_mask,
            attn_mask=attn_mask,
            query_sensor_ids=query_sensor_ids,
            memory_sensor_ids=memory_sensor_ids,
            return_attention_weights=return_attention_weights,
        )
        query = query + self.dropout(cross)
        query = query + self.dropout(self.feedforward(self.norm_ff(query)))
        return query, weights, lag


class TimeSeriesQueryCrossAttention(nn.Module):
    """Forecasting irregular mediante historia codificada y queries continuas.

    Mantiene el contrato de ``TimeSeriesTransformer.forward``. Los tokens con
    ``is_target_mask=True`` deben estar al final, pero no interactúan entre sí:
    permutar esos tokens sólo permuta las predicciones. Sus valores placeholder
    se ignoran deliberadamente.
    """

    def __init__(
        self,
        config: TimeSeriesTransformerConfig,
        query_config: Optional[QueryCrossAttentionConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.query_config = query_config or QueryCrossAttentionConfig()
        self.input_dim = int(config.input_dim)
        self.output_dim = int(config.output_dim)
        self.d_model = int(config.d_model)

        self.value_embedding = FeatureEmbedding(
            d_in=self.input_dim, d_model=self.d_model, use_layernorm=True
        )
        self.time_encoding = TimePositionalEncoding(
            d_model=self.d_model,
            time_scale=config.time_scale,
            mode=config.time_encoding_mode,
            time_transform=config.time_transform,
            learnable_time_scale=config.learnable_time_scale,
        )
        # El target nunca usa posición ordinal. Incluso en la ablación ordinal
        # de historia, una query temporal habilitada se codifica por su horizonte
        # físico respecto de la última observación válida.
        query_time_mode = (
            "sinusoidal" if config.time_encoding_mode == "ordinal"
            else config.time_encoding_mode
        )
        self.query_time_encoding = TimePositionalEncoding(
            d_model=self.d_model,
            time_scale=config.time_scale,
            mode=query_time_mode,
            time_transform=config.time_transform,
            learnable_time_scale=config.learnable_time_scale,
        )
        if config.learnable_time_scale:
            # Historia, horizonte, lag bias y CTSSM deben compartir unidad.
            self.query_time_encoding.log_time_scale = self.time_encoding.log_time_scale
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.query_token, std=0.02)
        self.time_emb_scale = nn.Parameter(torch.tensor(1.0))
        self.sensor_emb_scale = nn.Parameter(torch.tensor(1.0))
        self.use_sensor_embedding = bool(config.use_sensor_embedding)
        if self.use_sensor_embedding:
            if config.num_sensors < 1:
                raise ValueError("num_sensors debe ser >= 1 en modo evento.")
            self.sensor_embedding: Optional[nn.Module] = SensorEmbedding(
                int(config.num_sensors),
                self.d_model,
                include_target_token=False,
            )
        else:
            self.sensor_embedding = None

        derived_dim = 3 if self.query_config.derive_temporal_features else 0
        film_feature_dim = derived_dim + self.query_config.temporal_feature_dim
        self.temporal_film = (
            TemporalFeatureFiLM(film_feature_dim, self.d_model)
            if film_feature_dim > 0 and self.query_config.use_temporal_film
            else None
        )
        self.history_norm = nn.LayerNorm(self.d_model)
        self.query_norm = nn.LayerNorm(self.d_model)

        self.continuous_state = (
            StableDiagonalContinuousState(self.d_model, config.time_scale)
            if self.query_config.use_ctssm
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
        num_cross_layers = (
            self.query_config.num_cross_layers
            if self.query_config.num_cross_layers is not None
            else (config.decoder_num_layers or config.num_layers)
        )
        self.cross_layers = nn.ModuleList(
            [
                IndependentQueryCrossAttentionBlock(
                    self.d_model,
                    config.num_heads,
                    config.dim_feedforward,
                    config.dropout,
                    config.activation,
                    self.query_config,
                    sensor_relations_available=self.use_sensor_embedding,
                )
                for _ in range(int(num_cross_layers))
            ]
        )

        self.prediction_head = str(config.prediction_head).lower()
        if self.prediction_head not in {"point", "gaussian"}:
            raise ValueError("prediction_head debe ser 'point' o 'gaussian'.")
        if self.use_sensor_embedding:
            self.event_head = nn.Linear(
                self.d_model, 2 if self.prediction_head == "gaussian" else 1
            )
            self.head = None
        elif self.prediction_head == "gaussian":
            self.head = GaussianRegressionHead(self.d_model, self.output_dim)
            self.event_head = None
        else:
            self.head = RegressionHead(
                self.d_model,
                self.output_dim,
                dropout=config.dropout,
                activation=config.activation,
            )
            self.event_head = None

        if self.input_dim == self.output_dim:
            self.dense_residual_projection: nn.Module = nn.Identity()
        else:
            self.dense_residual_projection = nn.Linear(self.input_dim, self.output_dim)
        self.event_residual_projection: nn.Module = (
            nn.Identity() if self.input_dim == 1 else nn.Linear(self.input_dim, 1)
        )

        # No anunciar ni optimizar parámetros que una ablación no puede usar.
        # Esto hace que el conteo de capacidad y AdamW reflejen el grafo real.
        uses_time_embedding = bool(
            self.query_config.use_history_time_encoding
            or self.query_config.use_query_horizon
        )
        self.time_emb_scale.requires_grad_(uses_time_embedding)
        self.sensor_emb_scale.requires_grad_(self.use_sensor_embedding)
        uses_derived_timing = bool(
            self.temporal_film is not None
            and self.query_config.derive_temporal_features
        )
        uses_shared_time_scale = bool(
            (
                self.query_config.use_history_time_encoding
                and self.time_encoding.mode != "ordinal"
            )
            or self.query_config.use_query_horizon
            or self.query_config.use_relative_time_bias
            or uses_derived_timing
            or self.query_config.use_ctssm
        )
        shared_scale = getattr(self.time_encoding, "log_time_scale", None)

        def freeze_exclusive_encoding_parameters(
            encoding: TimePositionalEncoding,
        ) -> None:
            for parameter in encoding.parameters():
                if parameter is not shared_scale:
                    parameter.requires_grad_(False)

        if not self.query_config.use_history_time_encoding:
            freeze_exclusive_encoding_parameters(self.time_encoding)
        if not self.query_config.use_query_horizon:
            freeze_exclusive_encoding_parameters(self.query_time_encoding)
        if shared_scale is not None:
            shared_scale.requires_grad_(uses_shared_time_scale)

    @staticmethod
    def _last_valid_index(valid: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(valid.shape[1], device=valid.device).view(1, -1)
        return positions.masked_fill(~valid, -1).amax(dim=1)

    @staticmethod
    def _encode_time_preserving_gaps(
        encoding: TimePositionalEncoding,
        timestamps: torch.Tensor,
        *,
        dtype: torch.dtype,
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Resta el origen en alta precisión antes de entrar a capas float32."""
        if encoding.mode == "ordinal":
            safe_timestamps = timestamps.to(dtype)
        else:
            if lengths is not None:
                first_index = (
                    timestamps.shape[1]
                    - lengths.to(device=timestamps.device, dtype=torch.long)
                ).unsqueeze(1)
            elif padding_mask is not None:
                valid = ~padding_mask
                if not torch.all(valid.any(dim=1)):
                    raise ValueError("Cada secuencia requiere un timestamp válido.")
                first_index = valid.to(torch.int64).argmax(dim=1, keepdim=True)
            else:
                first_index = torch.zeros(
                    timestamps.shape[0],
                    1,
                    dtype=torch.long,
                    device=timestamps.device,
                )
            origin = timestamps.gather(1, first_index)
            safe_timestamps = (timestamps - origin).to(dtype)
        return encoding(
            safe_timestamps,
            padding_mask=padding_mask,
            lengths=lengths,
        ).to(dtype)

    def _derive_temporal_features(
        self,
        timestamps: torch.Tensor,
        sensor_ids: Optional[torch.Tensor],
        history_len: int,
        history_padding_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_times = timestamps[:, :history_len]
        query_times = timestamps[:, history_len:]
        if history_padding_mask is None:
            valid = torch.ones_like(history_times, dtype=torch.bool)
        else:
            valid = ~history_padding_mask

        first_idx = valid.to(torch.int64).argmax(dim=1, keepdim=True)
        origin = history_times.gather(1, first_idx)
        scale = self.time_encoding.current_time_scale(
            device=timestamps.device, dtype=timestamps.dtype
        )
        relative_history = (history_times - origin) / scale
        relative_query = (query_times - origin) / scale

        global_gap = torch.zeros_like(relative_history)
        pair_valid = valid[:, 1:] & valid[:, :-1]
        adjacent_gap = (relative_history[:, 1:] - relative_history[:, :-1]).clamp_min(0)
        global_gap[:, 1:] = torch.where(pair_valid, adjacent_gap, 0.0)

        if sensor_ids is None:
            sensor_age = global_gap
            last_history_index = self._last_valid_index(valid).unsqueeze(1)
            last_history_time = relative_history.gather(1, last_history_index)
            query_age = (relative_query - last_history_time).clamp_min(0)
            query_sensor_age = query_age
        else:
            history_sensor_ids = sensor_ids[:, :history_len].to(torch.long)
            query_sensor_ids = sensor_ids[:, history_len:].to(torch.long)
            num_sensors = int(self.config.num_sensors)
            one_hot = F.one_hot(history_sensor_ids.clamp(0, num_sensors - 1), num_sensors)
            observations = torch.where(
                one_hot.to(torch.bool) & valid.unsqueeze(-1),
                relative_history.unsqueeze(-1),
                torch.full_like(relative_history.unsqueeze(-1), float("-inf")),
            )
            previous = torch.full_like(observations, float("-inf"))
            if history_len > 1:
                previous[:, 1:] = observations[:, :-1]
            previous = previous.cummax(dim=1).values
            previous_for_token = previous.gather(
                2, history_sensor_ids.unsqueeze(-1)
            ).squeeze(-1)
            sensor_age = torch.where(
                previous_for_token.isfinite(),
                (relative_history - previous_for_token).clamp_min(0),
                torch.zeros_like(relative_history),
            )

            last_by_sensor = observations.amax(dim=1)
            last_for_query = last_by_sensor.gather(1, query_sensor_ids)
            query_sensor_age = torch.where(
                last_for_query.isfinite(),
                (relative_query - last_for_query).clamp_min(0),
                torch.zeros_like(relative_query),
            )
            last_history_index = self._last_valid_index(valid).unsqueeze(1)
            last_history_time = relative_history.gather(1, last_history_index)
            query_age = (relative_query - last_history_time).clamp_min(0)

        history_features = torch.stack(
            (
                torch.log1p(global_gap),
                torch.log1p(sensor_age),
                torch.exp(-global_gap),
            ),
            dim=-1,
        )
        query_features = torch.stack(
            (
                torch.log1p(query_age),
                torch.log1p(query_sensor_age),
                torch.exp(-query_age),
            ),
            dim=-1,
        )
        return history_features, query_features

    def _last_value_baseline(
        self,
        history_values: torch.Tensor,
        target_sensor_ids: Optional[torch.Tensor],
        history_sensor_ids: Optional[torch.Tensor],
        history_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, history_len, _ = history_values.shape
        valid = (
            torch.ones(batch, history_len, dtype=torch.bool, device=history_values.device)
            if history_padding_mask is None
            else ~history_padding_mask
        )
        if not self.use_sensor_embedding:
            last_idx = self._last_valid_index(valid)
            last_values = history_values[
                torch.arange(batch, device=history_values.device), last_idx
            ]
            return self.dense_residual_projection(last_values)

        if target_sensor_ids is None or history_sensor_ids is None:
            raise ValueError("Se requieren sensor_ids para el residual en modo evento.")
        projected = self.event_residual_projection(history_values).squeeze(-1)
        num_sensors = int(self.config.num_sensors)
        positions = torch.arange(history_len, device=history_values.device).view(1, -1)
        result = history_values.new_zeros(batch, target_sensor_ids.shape[1])
        for sensor in range(num_sensors):
            matches = valid & (history_sensor_ids == sensor)
            last_position = positions.masked_fill(~matches, -1).amax(dim=1)
            has_value = last_position >= 0
            safe_position = last_position.clamp_min(0)
            sensor_value = projected[
                torch.arange(batch, device=history_values.device), safe_position
            ]
            sensor_value = torch.where(has_value, sensor_value, torch.zeros_like(sensor_value))
            result = torch.where(
                target_sensor_ids == sensor, sensor_value.unsqueeze(1), result
            )
        return result

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
                raise ValueError(
                    "padding_mask y lengths describen secuencias distintas; "
                    "QueryCross requiere left-padding contiguo."
                )
        if self.use_sensor_embedding:
            if input_sensor_ids is None or input_sensor_ids.shape != (batch, total_len):
                raise ValueError("input_sensor_ids [B,L] es requerido en modo evento.")

        num_targets = validate_target_mask_structure(
            is_target_mask, batch_size=batch, seq_len=total_len
        )
        history_len = total_len - num_targets
        if history_len < 1:
            raise ValueError("Se requiere al menos un token histórico.")
        history_padding = padding_mask[:, :history_len] if padding_mask is not None else None
        if padding_mask is not None and torch.any(padding_mask[:, history_len:]):
            raise ValueError("Los tokens target no pueden ser padding.")

        if temporal_features is not None:
            expected = (batch, total_len, self.query_config.temporal_feature_dim)
            if temporal_features.shape != expected:
                raise ValueError(
                    f"temporal_features debe tener shape {expected}; "
                    f"se recibió {tuple(temporal_features.shape)}."
                )
        elif self.query_config.temporal_feature_dim > 0:
            temporal_features = input_values.new_zeros(
                batch, total_len, self.query_config.temporal_feature_dim
            )

        history_values = input_values[:, :history_len]
        history = self.value_embedding(history_values)
        query = self.query_token.to(history.dtype).expand(batch, num_targets, -1)
        history_timestamps = input_timestamps[:, :history_len]
        target_timestamps = input_timestamps[:, history_len:]
        history_valid = (
            torch.ones(
                batch,
                history_len,
                dtype=torch.bool,
                device=input_values.device,
            )
            if history_padding is None
            else ~history_padding
        )
        last_history_idx = self._last_valid_index(history_valid).unsqueeze(1)
        last_history_timestamp = history_timestamps.gather(1, last_history_idx)

        if self.query_config.use_history_time_encoding:
            history_lengths = (
                history_valid.sum(dim=1) if history_padding is not None else None
            )
            history_time_embedding = self._encode_time_preserving_gaps(
                self.time_encoding,
                history_timestamps,
                dtype=history.dtype,
                padding_mask=history_padding,
                lengths=history_lengths,
            )
            history = history + self.time_emb_scale.to(history.dtype) * history_time_embedding
        if self.query_config.use_query_horizon:
            query_time_input = torch.cat(
                (last_history_timestamp, target_timestamps), dim=1
            )
            query_time_embedding = self._encode_time_preserving_gaps(
                self.query_time_encoding,
                query_time_input,
                dtype=history.dtype,
            )[:, 1:]
            query = query + self.time_emb_scale.to(history.dtype) * query_time_embedding.to(
                history.dtype
            )

        history_sensor_ids: Optional[torch.Tensor] = None
        target_sensor_ids: Optional[torch.Tensor] = None
        temporal_sensor_ids = input_sensor_ids
        if self.sensor_embedding is not None and input_sensor_ids is not None:
            history_sensor_ids = input_sensor_ids[:, :history_len].to(torch.long)
            if torch.any(history_sensor_ids < 0) or torch.any(
                history_sensor_ids >= int(self.config.num_sensors)
            ):
                raise ValueError(
                    "Los ids históricos deben estar en el rango de sensores "
                    f"[0, {int(self.config.num_sensors) - 1}]."
                )
            raw_target_sensor_ids = input_sensor_ids[:, history_len:].to(torch.long)
            if num_targets % self.output_dim != 0:
                raise ValueError(
                    f"num_target_tokens={num_targets} no es divisible por "
                    f"output_dim={self.output_dim}."
                )
            timestamp_blocks = target_timestamps.view(
                batch, num_targets // self.output_dim, self.output_dim
            )
            if not torch.all(timestamp_blocks == timestamp_blocks[..., :1]):
                raise ValueError(
                    "Cada bloque contiguo de output_dim queries debe compartir "
                    "el mismo timestamp."
                )
            special_id = int(self.config.num_sensors)
            uses_special = raw_target_sensor_ids == special_id
            if torch.any(uses_special):
                if not torch.all(uses_special):
                    raise ValueError(
                        "Los targets en modo evento deben usar ids de sensor reales "
                        "o el id target especial de forma homogénea."
                    )
                if self.output_dim > int(self.config.num_sensors):
                    raise ValueError(
                        "No se pueden inferir canales target: output_dim supera "
                        "num_sensors."
                    )
                canonical = torch.arange(
                    num_targets, device=input_values.device, dtype=torch.long
                ).remainder(self.output_dim)
                target_sensor_ids = canonical.unsqueeze(0).expand(batch, -1)
                temporal_sensor_ids = torch.cat(
                    (history_sensor_ids, target_sensor_ids), dim=1
                )
            else:
                target_sensor_ids = raw_target_sensor_ids
                if torch.any(target_sensor_ids < 0) or torch.any(
                    target_sensor_ids >= self.output_dim
                ):
                    raise ValueError(
                        "Cada sensor target debe corresponder a un canal de salida."
                    )
                expected_channels = torch.arange(
                    self.output_dim,
                    device=input_values.device,
                    dtype=torch.long,
                ).repeat(num_targets // self.output_dim)
                if not torch.all(
                    target_sensor_ids == expected_channels.unsqueeze(0)
                ):
                    raise ValueError(
                        "Cada bloque target debe ordenar los sensores como "
                        "0..output_dim-1."
                    )

            # Las queries ya poseen un token de rol propio; usar aquí el canal
            # efectivo conserva identidad por sensor incluso cuando el builder
            # legacy entregó el id target genérico.
            embedding_sensor_ids = torch.cat(
                (history_sensor_ids, target_sensor_ids), dim=1
            )
            sensor_embedding = self.sensor_embedding(embedding_sensor_ids).to(
                history.dtype
            )
            history = history + self.sensor_emb_scale.to(history.dtype) * sensor_embedding[:, :history_len]
            query = query + self.sensor_emb_scale.to(history.dtype) * sensor_embedding[:, history_len:]

        feature_parts_history = []
        feature_parts_query = []
        if self.temporal_film is not None and self.query_config.derive_temporal_features:
            derived_history, derived_query = self._derive_temporal_features(
                input_timestamps,
                temporal_sensor_ids,
                history_len,
                history_padding,
            )
            if not self.query_config.use_query_horizon:
                derived_query = torch.zeros_like(derived_query)
            feature_parts_history.append(derived_history.to(history.dtype))
            feature_parts_query.append(derived_query.to(history.dtype))
        if self.temporal_film is not None and temporal_features is not None:
            feature_parts_history.append(temporal_features[:, :history_len].to(history.dtype))
            feature_parts_query.append(temporal_features[:, history_len:].to(history.dtype))
        if self.temporal_film is not None:
            history = self.temporal_film(history, torch.cat(feature_parts_history, dim=-1))
            query = self.temporal_film(query, torch.cat(feature_parts_query, dim=-1))

        history = self.history_norm(history)
        query = self.query_norm(query)
        time_scale = self.time_encoding.current_time_scale(
            device=input_timestamps.device, dtype=input_timestamps.dtype
        )
        if self.continuous_state is not None:
            history = self.continuous_state(
                history,
                input_timestamps[:, :history_len],
                history_padding,
                time_scale=time_scale,
            )
        if return_all_layers:
            encoder_output, encoder_layers = self.encoder(
                history,
                key_padding_mask=history_padding,
                return_all_layers=True,
            )
        else:
            encoder_output = self.encoder(
                history,
                key_padding_mask=history_padding,
                return_all_layers=False,
            )
            encoder_layers = None

        query_layers = []
        attention_weights = None
        relative_lags = None
        for layer_index, layer in enumerate(self.cross_layers):
            query, attention_weights, relative_lags = layer(
                query,
                encoder_output,
                input_timestamps[:, history_len:],
                input_timestamps[:, :history_len],
                time_scale=time_scale,
                memory_padding_mask=history_padding,
                attn_mask=attn_mask,
                query_sensor_ids=target_sensor_ids,
                memory_sensor_ids=history_sensor_ids,
                return_attention_weights=(
                    return_attention_weights
                    and layer_index == len(self.cross_layers) - 1
                ),
            )
            if return_all_layers:
                query_layers.append(query)

        log_scale = None
        if self.use_sensor_embedding:
            if self.event_head is None:
                raise RuntimeError("event_head no inicializado.")
            event_params = self.event_head(query)
            if self.prediction_head == "gaussian":
                predictions_flat = event_params[..., 0]
                log_scale_flat = event_params[..., 1].clamp(-7.0, 5.0)
            else:
                predictions_flat = event_params.squeeze(-1)
            if num_targets % self.output_dim != 0:
                raise ValueError(
                    f"num_target_tokens={num_targets} no es divisible por "
                    f"output_dim={self.output_dim}."
                )
            horizons = num_targets // self.output_dim
            if self.query_config.use_last_value_residual:
                predictions_flat = predictions_flat + self._last_value_baseline(
                    history_values,
                    target_sensor_ids,
                    history_sensor_ids,
                    history_padding,
                )
            predictions = predictions_flat.view(batch, horizons, self.output_dim)
            if self.prediction_head == "gaussian":
                log_scale = log_scale_flat.view(batch, horizons, self.output_dim)
        else:
            if self.head is None:
                raise RuntimeError("head no inicializado.")
            if self.prediction_head == "gaussian":
                predictions, log_scale = self.head(query)
            else:
                predictions = self.head(query)
            if self.query_config.use_last_value_residual:
                baseline = self._last_value_baseline(
                    history_values, None, None, history_padding
                ).unsqueeze(1)
                predictions = predictions + baseline

        if predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
            if log_scale is not None:
                log_scale = log_scale.squeeze(1)

        if not return_dict:
            return predictions
        result: Dict[str, Any] = {
            "preds": predictions,
            "target_states": query,
            "encoder_output": encoder_output,
            "cross_attn_weights": attention_weights,
            "relative_lags": relative_lags,
        }
        if log_scale is not None:
            result["log_scale"] = log_scale
        if return_all_layers:
            result["all_layers"] = {
                "encoder": encoder_layers,
                "queries": query_layers,
            }
        return result


# Nombre corto para recipes y notebooks.
CustomQueryCrossAttention = TimeSeriesQueryCrossAttention


__all__ = [
    "CustomQueryCrossAttention",
    "IndependentQueryCrossAttentionBlock",
    "QueryCrossAttentionConfig",
    "RelativeLagBias",
    "RelativeTimeCrossAttention",
    "StableDiagonalContinuousState",
    "TemporalFeatureFiLM",
    "TimeSeriesQueryCrossAttention",
]
