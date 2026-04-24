"""
Módulos de embeddings y encodings para el TimeSeriesTransformer.

Incluye:
- FeatureEmbedding: proyección de features continuas multivariadas.
- TimePositionalEncoding: encoding temporal continuo para timestamps no equiespaciados.
- Time2Vec: encoding temporal con componentes periódicos aprendidos.
- TargetFlagEmbedding: embedding para distinguir tokens de historia vs token target.
- SensorEmbedding: embedding por identificador de sensor para tokens de evento.
- TemporalAttentionBias: bias de atención basado en diferencias temporales.
"""

from .value_embedding import FeatureEmbedding
from .time_encoding import TimePositionalEncoding, Time2Vec
from .target_flag_embedding import TargetFlagEmbedding
from .sensor_embedding import SensorEmbedding
from .temporal_attention_bias import TemporalAttentionBias

__all__ = [
    "FeatureEmbedding",
    "TimePositionalEncoding",
    "Time2Vec",
    "TargetFlagEmbedding",
    "SensorEmbedding",
    "TemporalAttentionBias",
]