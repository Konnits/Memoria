import torch
import torch.nn as nn
from typing import Optional

class CoFormerAttentionLayer(nn.Module):
    """
    Capa sucesiva del Compatible Transformer.
    Consta de Extraer Atencion Temporal (Intra-variate) y Atencion de Interacción (Inter-variate).
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1, k_neighbors: int = 30):
        super().__init__()
        self.n_heads = n_heads
        self.k_neighbors = k_neighbors
        
        self.intra_mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.inter_mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, variates: torch.Tensor, times: torch.Tensor, valid_mask: Optional[torch.Tensor] = None):
        """
        x: [B, S, d_model]
        variates: [B, S] IDs de variable
        times: [B, S] Float timestamps
        valid_mask: [B, S] Boolean, True = Válido, False = Padding.
        """
        B, S = x.shape[:2]
        
        # Para nn.MultiheadAttention en PyTorch, attn_mask como Booleano donde True significa "No mirar / Padding".
        # 1 ========================
        # Intra-Variate Attention
        # ========================
        # La máscara bloquea atencion si son de variables distintas
        same_var = (variates.unsqueeze(1) == variates.unsqueeze(2))  # [B, S, S]
        
        if valid_mask is not None:
            valid_2d = valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)
            same_var = same_var & valid_2d
            
        # Intra mask -> True si bloqueamos la atención (es decir, NO son misma variable o NO son validos)
        intra_mask = ~same_var
        
        # Prevención de NaNs: Si hay filas completamente bloqueadas, les permitimos mirarse a sí mismas
        all_blocked_intra = intra_mask.all(dim=-1)
        if all_blocked_intra.any():
            diag_idx = torch.arange(S, device=intra_mask.device)
            # Para los batches/elementos afectados, des-bloqueamos la diagonal
            # (Aunque sean padding, la salida se enmascarará igual después, mitigamos el NaN)
            intra_mask[:, diag_idx, diag_idx] = False

        # Ajuste para las múltiples cabezas: [B * num_heads, S, S]
        intra_mask_heads = intra_mask.repeat_interleave(self.n_heads, dim=0)

        # Self-attention Intra
        attn_out1, _ = self.intra_mha(x, x, x, attn_mask=intra_mask_heads)
        x = self.norm1(x + attn_out1)
        
        # 2 ========================
        # Inter-Variate Attention
        # ========================
        # Aquí permitimos atender a diferentes variables que se encuentren cerca en el tiempo
        diff_var = (variates.unsqueeze(1) != variates.unsqueeze(2))  # [B, S, S]
        
        # Matriz de distancias de tiempo absolutas
        time_dist = torch.abs(times.unsqueeze(1) - times.unsqueeze(2))  # [B, S, S]
        
        # Sólo consideramos variables distintas y válidas
        if valid_mask is not None:
            time_dist = time_dist.masked_fill(~valid_2d, float('inf'))
            
        time_dist = time_dist.masked_fill(~diff_var, float('inf'))
        
        # Seleccionar las K más cercanas en tiempo (vecinos K-NN temporal)
        k = min(self.k_neighbors, S)
        _, topk_idx = torch.topk(time_dist, k, dim=-1, largest=False)
        knn_mask = torch.zeros_like(time_dist, dtype=torch.bool).scatter_(-1, topk_idx, True)

        if valid_mask is not None:
            knn_mask = knn_mask & valid_2d
            
        inter_mask = ~knn_mask
        
        all_blocked_inter = inter_mask.all(dim=-1)
        if all_blocked_inter.any():
            diag_idx = torch.arange(S, device=inter_mask.device)
            inter_mask[:, diag_idx, diag_idx] = False
            
        inter_mask_heads = inter_mask.repeat_interleave(self.n_heads, dim=0)
        
        # Self-attention Inter
        attn_out2, _ = self.inter_mha(x, x, x, attn_mask=inter_mask_heads)
        x = self.norm2(x + attn_out2)
        
        return x
