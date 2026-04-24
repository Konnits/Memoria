import math
import torch
import torch.nn as nn

class ContinuousValueEmbedding(nn.Module):
    """
    Continuous Value Embedding (CVE) propuesto en STraTS.
    Utiliza una red Feed-Forward con una capa oculta de dimensión sqrt(d_model) y tanh.
    Mapea un escalar temporal o valor a un espacio d_model.
    """
    def __init__(self, d_model: int):
        super().__init__()
        hidden_dim = max(1, int(math.sqrt(d_model)))
        
        self.ffn = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x: Tensor de dimensiones [Batch, Seq_len] con los valores continuos.
        
        Returns:
        --------
        Tensor de dimensiones [Batch, Seq_len, d_model]
        """
        x = x.unsqueeze(-1)  # [Batch, Seq_len, 1]
        return self.ffn(x)
