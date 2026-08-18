"""
Componentes de modelo para el TimeSeriesTransformer.

Incluye:
- Implementación de atención multi-cabeza.
- Bloques encoder estilo Transformer.
- Construcción de máscaras.
- Cabezas de salida para regresión.
- Modelo de alto nivel TimeSeriesTransformer.
"""

from .attention import MultiHeadSelfAttention
from .transformer_blocks import TransformerEncoderBlock, TransformerEncoder
from .masking import create_causal_mask
from .heads import RegressionHead
from .time_series_transformer import TimeSeriesTransformer
from .time_series_encoder_decoder import TimeSeriesEncoderDecoder
from .query_cross_attention import (
    CustomQueryCrossAttention,
    QueryCrossAttentionConfig,
    TimeSeriesQueryCrossAttention,
)
from .continuous_basis_decoder import (
    ContinuousBasisDecoderConfig,
    ContinuousHorizonBasis,
    CustomContinuousBasisDecoder,
    TimeSeriesContinuousBasisDecoder,
)

__all__ = [
    "MultiHeadSelfAttention",
    "TransformerEncoderBlock",
    "TransformerEncoder",
    "create_causal_mask",
    "RegressionHead",
    "TimeSeriesTransformer",
    "TimeSeriesEncoderDecoder",
    "CustomQueryCrossAttention",
    "QueryCrossAttentionConfig",
    "TimeSeriesQueryCrossAttention",
    "ContinuousBasisDecoderConfig",
    "ContinuousHorizonBasis",
    "CustomContinuousBasisDecoder",
    "TimeSeriesContinuousBasisDecoder",
]
