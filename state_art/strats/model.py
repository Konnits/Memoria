import torch
import torch.nn as nn
from typing import Tuple, Optional
from .embeddings import ContinuousValueEmbedding

class FusionSelfAttention(nn.Module):
    """
    Módulo de agregación para colapsar las secuencias codificadas por el Transformer a 
    un solo vector por paciente, aprendiendo los pesos de agregación (pooling temporal/variable).
    """
    def __init__(self, d_model: int, d_a: int = 64):
        super().__init__()
        self.W_a = nn.Linear(d_model, d_a)
        self.u_a = nn.Linear(d_a, 1, bias=False)

    def forward(self, c: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        -----------
        c: [B, L, d_model] Contextual embeddings from Transformer.
        valid_mask: [B, L] Máscara booleana (True si el token original es válido, False si es padding).
        
        Returns:
        --------
        e_t: [B, d_model] Embedding fusionado.
        alpha: [B, L] Pesos de atención calculados.
        """
        # Calcular los pesos de atencion "a"
        a = self.u_a(torch.tanh(self.W_a(c))).squeeze(-1)  # [B, L]
        
        if valid_mask is not None:
            # Enmascarar las posiciones de padding con aproximación a -inf
            # Usar -1e4 para compatibilidad con AMP (FP16 max ≈ 65504)
            a = a.masked_fill(~valid_mask, float('-1e4'))
            
        alpha = torch.softmax(a, dim=-1)  # [B, L]
        
        # Combinación lineal guiada por alpha
        e_T = torch.sum(alpha.unsqueeze(-1) * c, dim=1)  # [B, d_model]
        
        return e_T, alpha


class STraTSNetwork(nn.Module):
    """
    Arquitectura STraTS: Self-Supervised Transformer para Series de Tiempo Sparse irregulares.
    Elaborada a partir de los conceptos del paper base 2107.14293v2.
    """
    def __init__(
        self, 
        num_features: int, 
        d_model: int = 50, 
        n_heads: int = 4, 
        n_layers: int = 2, 
        d_ff: int = 128, 
        dropout: float = 0.1, 
        num_classes: int = 1
    ):
        super().__init__()
        self.d_model = d_model
        
        # 1. Componentes de Embeddings de Tripleta
        self.feature_emb = nn.Embedding(num_features, d_model)
        self.value_emb = ContinuousValueEmbedding(d_model)
        self.time_emb = ContinuousValueEmbedding(d_model)
        
        # 2. Transformer Contextual
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_ff, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 3. Fusion Attention Aggregation
        self.fusion_attention = FusionSelfAttention(d_model)
        
        # 4. Final Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(
        self, 
        times: torch.Tensor, 
        feature_ids: torch.Tensor, 
        values: torch.Tensor, 
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Parameters:
        -----------
        times: [B, L] Timestamps numéricos de los eventos.
        feature_ids: [B, L] Identificador entero que indica la variable de la tripleta.
        values: [B, L] Valores numéricos registrados.
        valid_mask: [B, L] Máscara booleana, donde True = válida, False = padding.
        
        Returns:
        --------
        out: [B, num_classes] Predicción cruda para el objetivo.
        """
        # 1. Embedding Inicial de la Tripleta
        f_emb = self.feature_emb(feature_ids.long())  # [B, L, d_model]
        v_emb = self.value_emb(values.float())        # [B, L, d_model]
        t_emb = self.time_emb(times.float())          # [B, L, d_model]
        
        # Sumar los 3 embeddings
        x = f_emb + v_emb + t_emb  # [B, L, d_model]
        
        # 2. Codificación con Transformer
        # PyTorch espera un src_key_padding_mask donde True = padding token para ignorar.
        src_key_padding_mask = ~valid_mask if valid_mask is not None else None
        
        c = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # [B, L, d_model]
        
        # 3. Agregación temporal-variable global
        e_T, attention_weights = self.fusion_attention(c, valid_mask=valid_mask)
        
        # 4. Predicción clasificador final
        out = self.classifier(e_T)
        return out
