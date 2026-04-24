import torch
import torch.nn as nn
from typing import Optional
from .encodings import MeasurementEmbedding, VariateTimeEncoding
from .attention import CoFormerAttentionLayer

class CompatibleTransformer(nn.Module):
    """
    CoFormer (Compatible Transformer) model.
    Construído utilizando los principios del paper 2310.11022v1.
    """
    def __init__(
        self, 
        num_variates: int, 
        d_model: int = 256, 
        d_var: int = 32, 
        d_time: int = 256, 
        n_heads: int = 8, 
        n_layers: int = 4, 
        dropout: float = 0.1, 
        k_neighbors: int = 30, 
        num_classes: int = 1
    ):
        super().__init__()
        self.num_variates = num_variates
        self.d_model = d_model
        
        # 1. Componentes de codificación (Sample-wise Initial feature learning)
        self.me = MeasurementEmbedding(d_model)
        self.vte = VariateTimeEncoding(num_variates, d_var=d_var, d_time=d_time)
        
        # Projection layer para igualar la dimensionalidad (agregación de variate code y tiempo)
        self.agg_linear = nn.Linear(d_model + d_time, d_model - d_var)
        
        # 2. Capas de Atención Sucesiva (Intra/Inter)
        self.layers = nn.ModuleList([
            CoFormerAttentionLayer(d_model, n_heads=n_heads, dropout=dropout, k_neighbors=k_neighbors)
            for _ in range(n_layers)
        ])
        
        # 3. Módulos de agregación Observation-wise
        self.agg_mha = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # 4. Clasificador
        self.classifier = nn.Sequential(
             nn.Linear(d_model, d_model),
             nn.ReLU(),
             nn.Linear(d_model, num_classes)
        )
        
    def forward(
        self, 
        times: torch.Tensor, 
        feature_ids: torch.Tensor, 
        values: torch.Tensor, 
        valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Los datos ingresan ya aplanados en points irregulares.
        times, feature_ids, values: [B, S]
        """
        B, S = feature_ids.shape
        device = feature_ids.device
        
        # 1. Feature Learning (Paso 1a, 1b, 1c)
        e_l = self.me(values.float())  # [B, S, d_model]
        v_l, u_l = self.vte(feature_ids, times)  # v_l: [B, S, d_var], u_l: [B, S, d_time]
        
        # h_l = v_l (+) Linear(e_l (+) u_l)
        eu = torch.cat([e_l, u_l], dim=-1)
        h_l = torch.cat([v_l, self.agg_linear(eu)], dim=-1)  # [B, S, d_model]
        
        # 2. Pasar por las capas CoFormer
        for layer in self.layers:
            h_l = layer(h_l, feature_ids, times, valid_mask)
            
        # 3. Observation-wise aggregation
        f_variates_attended = torch.zeros(B, self.num_variates, self.d_model, device=device)

        # Procesar por cada variable/sensor (variate)
        for i in range(self.num_variates):
            # Seleccionar los puntos que pertenecen a esta variable
            var_mask = (feature_ids == i)
            if valid_mask is not None:
                var_mask = var_mask & valid_mask
                
            # Promedio sobre la dimensión temporal
            sums = torch.sum(h_l * var_mask.unsqueeze(-1), dim=1)  # [B, d_model]
            counts = torch.sum(var_mask, dim=1, keepdim=True).clamp(min=1)  # [B, 1]
            f_mean = sums / counts  # [B, d_model]
            
            # Atención MHA para agregar importancia a las mediciones
            q = f_mean.unsqueeze(1)  # [B, 1, d_model]
            k_v = h_l                # [B, S, d_model]
            
            mha_mask = ~var_mask     # [B, S]
            
            # Evitar NaNs cuando un batch no tiene mediciones de la variable (todo enmascarado)
            all_masked = mha_mask.all(dim=1)
            if all_masked.any():
                mha_mask[all_masked, 0] = False
                
            attn_out, _ = self.agg_mha(q, k_v, k_v, key_padding_mask=mha_mask)
            f_variates_attended[:, i, :] = attn_out.squeeze(1)

        # 4. Global Average Pooling a través de las variables
        f = torch.mean(f_variates_attended, dim=1)  # [B, d_model]
        
        # Salida del clasificador
        out = self.classifier(f)
        return out
