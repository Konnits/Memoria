import math
import torch
import torch.nn as nn
from typing import Tuple

class MeasurementEmbedding(nn.Module):
    """
    Encoder de Mediciones para CoFormer.
    Mapea un valor escalar a un espacio latente de alta dimensión (ej. 256).
    Se usa un MLP de 3 capas con activaciones ReLU.
    """
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x shape: [B, S] -> Output: [B, S, d_model] """
        return self.mlp(x.unsqueeze(-1))


class VariateTimeEncoding(nn.Module):
    """
    Encoder de Variable y Tiempo (Variate-Time Encoding) para CoFormer.
    Combina:
      1. Diccionario (Embedding tabular) para el ID discreto de la variable.
      2. Encoding Trigonométrico sobre un valor de tiempo continuo.
    """
    def __init__(self, num_variates: int, d_var: int = 32, d_time: int = 256):
        super().__init__()
        self.var_emb = nn.Embedding(num_variates, d_var)
        self.d_time = d_time
        
        # d_time debe ser par para que sin/cos tengan igual dimensión
        if d_time % 2 != 0:
            raise ValueError("d_time debe ser divisible por 2.")

    def forward(self, variates: torch.Tensor, times: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        -----------
        variates: [B, S]
        times: [B, S]
        
        Returns:
        --------
        v: [B, S, d_var]
        t: [B, S, d_time]
        """
        B, S = times.shape
        device = times.device
        
        # 1. Variate Code
        v = self.var_emb(variates.long())  # [B, S, d_var]
        
        # 2. Time Code continuo
        half_dim = self.d_time // 2
        emb_scale = math.log(10000.0) / (half_dim - 1)
        emb_freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=device) * -emb_scale)
        
        # Proyectar el tiempo continuo a las frecuencias
        t_proj = times.unsqueeze(-1) * emb_freqs.unsqueeze(0).unsqueeze(0)  # [B, S, half_dim]
        
        # Intercalar senos y cosenos
        t = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)  # [B, S, d_time]
        
        return v, t
