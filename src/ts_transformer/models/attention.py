from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """
    Módulo de atención multi-cabeza para self-attention.

    Esta implementación trabaja con entradas de shape:
        x: [batch_size, seq_len, d_model]

    y permite recibir una máscara de padding opcional:
        key_padding_mask: [batch_size, seq_len] (bool)
            True indica posiciones de padding que no deben ser atendidas.

    Salida:
        out: [batch_size, seq_len, d_model]
        attn_weights: [batch_size, num_heads, seq_len, seq_len]
            (opcional; puede usarse para inspección/visualización).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.return_attn_weights = return_attn_weights

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        """
        Reorganiza el tensor de [B, L, d_model] a [B, num_heads, L, d_head].
        """
        return (
            x.view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        temporal_bias: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        x:
            Tensor de entrada, shape [batch_size, seq_len, d_model].
        key_padding_mask:
            Tensor bool opcional, shape [batch_size, seq_len].
            True indica posiciones de padding que no deben contribuir.
        attn_mask:
            Máscara de atención opcional, shape [seq_len, seq_len] o
            [batch_size * num_heads, seq_len, seq_len], típicamente usada
            para máscaras causales u otras restricciones estructurales.
        temporal_bias:
            Bias temporal aditivo opcional, shape [B, num_heads, L, L].
            Se suma a los scores de atención antes del softmax.

        Returns
        -------
        out:
            Tensor de salida, shape [batch_size, seq_len, d_model].
        attn_weights:
            Tensor con los pesos de atención si `return_attn_weights` es True,
            en caso contrario None. Shape [batch_size, num_heads, seq_len, seq_len].
        """
        import torch.nn.functional as F
        
        batch_size, seq_len, _ = x.shape

        # Proyecciones a Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reorganizar para multi-head
        q = self._shape(q, batch_size, seq_len)  # [B, H, L, d_head]
        k = self._shape(k, batch_size, seq_len)  # [B, H, L, d_head]
        v = self._shape(v, batch_size, seq_len)  # [B, H, L, d_head]

        if self.return_attn_weights:
            # --- Fallback Matemático para retornar attn_weights ---
            q_scaled = q / math.sqrt(self.d_head)
            scores = torch.matmul(q_scaled, k.transpose(-2, -1))

            if is_causal:
                causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
                scores = scores.masked_fill(~causal_mask.view(1, 1, seq_len, seq_len), float("-inf"))

            if key_padding_mask is not None:
                padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len)
                scores = scores.masked_fill(padding_mask, float("-inf"))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask.view(1, 1, seq_len, seq_len)
                elif attn_mask.dim() == 3:
                    scores = scores.view(batch_size * self.num_heads, seq_len, seq_len)
                    scores = scores + attn_mask
                    scores = scores.view(batch_size, self.num_heads, seq_len, seq_len)

            if temporal_bias is not None:
                scores = scores + temporal_bias

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        else:
            # --- Fast Path con FlashAttention (SDPA) ---
            merged_mask = None
            sdpa_is_causal = False
            
            has_padding = key_padding_mask is not None
            has_attn = attn_mask is not None
            has_temporal = temporal_bias is not None

            # Rutas eficientes sin materializar float mask
            if not has_attn and not has_temporal:
                if is_causal and not has_padding:
                    sdpa_is_causal = True
                elif not is_causal and has_padding:
                    # En SDPA, la máscara bool es True = participar, False = ignorar
                    merged_mask = (~key_padding_mask).view(batch_size, 1, 1, seq_len)
                elif is_causal and has_padding:
                    # SDPA no permite is_causal=True + attn_mask simultáneamente.
                    # Materializamos máscara booleana conjunta
                    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
                    merged_mask = causal.view(1, 1, seq_len, seq_len) & (~key_padding_mask).view(batch_size, 1, 1, seq_len)
            else:
                # Si hay máscaras aditivas o bias temporal, construimos una máscara
                # combinada evitando tensores densos de ceros cuando no son necesarios.
                if is_causal:
                    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril().view(1, 1, seq_len, seq_len)
                    merged_mask = causal

                if has_temporal:
                    temporal_component = temporal_bias.to(dtype=q.dtype)
                    merged_mask = (
                        temporal_component
                        if merged_mask is None
                        else merged_mask.to(dtype=q.dtype).masked_fill(~merged_mask, float("-inf")) + temporal_component
                        if merged_mask.dtype == torch.bool
                        else merged_mask + temporal_component
                    )

                if has_attn:
                    if attn_mask.dim() == 2:
                        attn_component = attn_mask.view(1, 1, seq_len, seq_len)
                    elif attn_mask.dim() == 3:
                        attn_component = attn_mask.view(batch_size, self.num_heads, seq_len, seq_len)
                    else:
                        raise ValueError(
                            "attn_mask debe ser 2D [L, L] o 3D [B*H, L, L]."
                        )

                    attn_component = attn_component.to(dtype=q.dtype)
                    if merged_mask is None:
                        merged_mask = attn_component
                    elif merged_mask.dtype == torch.bool:
                        merged_mask = merged_mask.to(dtype=q.dtype).masked_fill(~merged_mask, float("-inf")) + attn_component
                    else:
                        merged_mask = merged_mask + attn_component

                if has_padding:
                    key_is_valid = (~key_padding_mask).view(batch_size, 1, 1, seq_len)
                    if merged_mask is None:
                        merged_mask = key_is_valid
                    elif merged_mask.dtype == torch.bool:
                        merged_mask = merged_mask & key_is_valid
                    else:
                        merged_mask = merged_mask.masked_fill(~key_is_valid, float("-inf"))

            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=merged_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=sdpa_is_causal
            )
            attn_weights = None

        # Reorganizar de [B, H, L, d_head] a [B, L, d_model]
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_model)
        )

        # Proyección final
        out = self.out_proj(attn_output)

        return out, attn_weights


class MultiHeadCrossAttention(nn.Module):
    """
    Módulo de atención multi-cabeza para cross-attention.
    
    Esta implementación trabaja con entradas de shape:
        query: [batch_size, tgt_len, d_model]
        key_value: [batch_size, src_len, d_model]
        
    y permite recibir máscaras opcionales.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        return_attn_weights: bool = False,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) debe ser divisible por num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.return_attn_weights = return_attn_weights

        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)

        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _shape(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return (
            x.view(batch_size, seq_len, self.num_heads, self.d_head)
            .transpose(1, 2)
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parameters
        ----------
        query:
            [batch_size, tgt_len, d_model]
        key_value:
            [batch_size, src_len, d_model]
        key_padding_mask:
            Tensor bool opcional para el key_value (fuente), shape [batch_size, src_len].
        attn_mask:
            Máscara de atención opcional, shape [tgt_len, src_len] o
            [batch_size * num_heads, tgt_len, src_len].
        """
        import torch.nn.functional as F
        
        batch_size, tgt_len, _ = query.shape
        _, src_len, _ = key_value.shape

        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        q = self._shape(q, batch_size, tgt_len)  # [B, H, tgt_len, d_head]
        k = self._shape(k, batch_size, src_len)  # [B, H, src_len, d_head]
        v = self._shape(v, batch_size, src_len)  # [B, H, src_len, d_head]

        if self.return_attn_weights:
            q_scaled = q / math.sqrt(self.d_head)
            # [B, H, tgt_len, d_head] @ [B, H, d_head, src_len] -> [B, H, tgt_len, src_len]
            scores = torch.matmul(q_scaled, k.transpose(-2, -1))

            if key_padding_mask is not None:
                padding_mask = key_padding_mask.view(batch_size, 1, 1, src_len)
                scores = scores.masked_fill(padding_mask, float("-inf"))

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask.view(1, 1, tgt_len, src_len)
                elif attn_mask.dim() == 3:
                    scores = scores.view(batch_size * self.num_heads, tgt_len, src_len)
                    scores = scores + attn_mask
                    scores = scores.view(batch_size, self.num_heads, tgt_len, src_len)

            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # [B, H, tgt_len, src_len] @ [B, H, src_len, d_head] -> [B, H, tgt_len, d_head]
            attn_output = torch.matmul(attn_weights, v)
        else:
            merged_mask = None
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    merged_mask = attn_mask.view(1, 1, tgt_len, src_len).to(dtype=q.dtype)
                elif attn_mask.dim() == 3:
                    merged_mask = attn_mask.view(batch_size, self.num_heads, tgt_len, src_len).to(dtype=q.dtype)
                else:
                    raise ValueError(
                        "attn_mask debe ser 2D [L_tgt, L_src] o 3D [B*H, L_tgt, L_src]."
                    )

            if key_padding_mask is not None:
                key_is_valid = (~key_padding_mask).view(batch_size, 1, 1, src_len)
                if merged_mask is None:
                    merged_mask = key_is_valid
                elif merged_mask.dtype == torch.bool:
                    merged_mask = merged_mask & key_is_valid
                else:
                    merged_mask = merged_mask.masked_fill(~key_is_valid, float("-inf"))
                        
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=merged_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            attn_weights = None

        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, tgt_len, self.d_model)
        )

        out = self.out_proj(attn_output)

        return out, attn_weights