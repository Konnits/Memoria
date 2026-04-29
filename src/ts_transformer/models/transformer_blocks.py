from __future__ import annotations

from typing import Optional, Tuple, List

import torch
from torch import nn

from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention


class TransformerEncoderBlock(nn.Module):
    """
    Bloque encoder estilo Transformer:
      - Multi-Head Self-Attention
      - Residual + LayerNorm
      - Feed-Forward (MLP)
      - Residual + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activación no soportada: {activation}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        temporal_bias: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor de entrada, shape [B, L, d_model].
        key_padding_mask:
            Tensor bool opcional, shape [B, L].
        attn_mask:
            Máscara de atención opcional para restringir la atención
            (por ejemplo, máscara causal), shape [L, L] o [B*H, L, L].
        temporal_bias:
            Bias temporal aditivo opcional, shape [B, H, L, L].

        Returns
        -------
        out:
            Tensor de salida, shape [B, L, d_model].
        """
        # Pre-LN: normalizar antes de cada sub-capa para mejorar estabilidad.
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            temporal_bias=temporal_bias,
            is_causal=is_causal,
        )
        x = x + self.dropout(attn_output)

        x_norm = self.norm2(x)
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(x_norm))))
        x = x + self.dropout(ff)

        return x


class TransformerEncoder(nn.Module):
    """
    Pila de bloques TransformerEncoderBlock.

    Parameters
    ----------
    num_layers:
        Número de bloques apilados.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        temporal_bias: Optional[torch.Tensor] = None,
        temporal_bias_layers: Optional[int] = None,
        is_causal: bool = False,
        return_all_layers: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Parameters
        ----------
        x:
            Tensor de entrada, shape [B, L, d_model].
        key_padding_mask:
            Tensor bool opcional, shape [B, L].
        attn_mask:
            Máscara de atención opcional, shape [L, L] o [B*H, L, L].
        temporal_bias:
            Bias temporal aditivo opcional, shape [B, H, L, L].
        return_all_layers:
            Si True, devuelve una lista con las representaciones de todas
            las capas además de la representación final.

        Returns
        -------
        Si return_all_layers es False:
            out: [B, L, d_model]
        Si return_all_layers es True:
            out: [B, L, d_model]
            all_layers: lista de [B, L, d_model] por capa.
        """
        all_layers = []

        for layer_idx, layer in enumerate(self.layers):
            layer_temporal_bias = temporal_bias
            if temporal_bias is not None and temporal_bias_layers is not None:
                layer_temporal_bias = (
                    temporal_bias if layer_idx < max(0, int(temporal_bias_layers))
                    else None
                )
            x = layer(
                x,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
                temporal_bias=layer_temporal_bias,
                is_causal=is_causal,
            )
            if return_all_layers:
                all_layers.append(x)

        if return_all_layers:
            return x, all_layers
        else:
            return x


class TransformerDecoderBlock(nn.Module):
    """
    Bloque decoder estilo Transformer:
      - Masked Multi-Head Self-Attention (sobre el target)
      - Multi-Head Cross-Attention (sobre el encoder)
      - Residual + LayerNorm
      - Feed-Forward (MLP)
      - Residual + LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.self_attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.cross_attn = MultiHeadCrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Activación no soportada: {activation}")

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x:
            Tensor de entrada (target), shape [B, L_tgt, d_model].
        encoder_out:
            Salida del encoder (memory), shape [B, L_src, d_model].
        tgt_key_padding_mask:
            Máscara de padding para target, shape [B, L_tgt].
        memory_key_padding_mask:
            Máscara de padding para encoder_out, shape [B, L_src].
        tgt_attn_mask:
            Máscara de atención para self-attention (ej. causal).
        cross_attn_mask:
            Máscara de atención para cross-attention.
        """
        # Pre-LN: mantiene el decoder estable cuando se entrena con varios targets.
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm,
            key_padding_mask=tgt_key_padding_mask,
            attn_mask=tgt_attn_mask,
            is_causal=is_causal,
        )
        x = x + self.dropout(attn_output)

        x_norm = self.norm2(x)
        cross_output, _ = self.cross_attn(
            query=x_norm,
            key_value=encoder_out,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=cross_attn_mask,
        )
        x = x + self.dropout(cross_output)

        x_norm = self.norm3(x)
        ff = self.linear2(self.dropout_ff(self.activation(self.linear1(x_norm))))
        x = x + self.dropout(ff)

        return x


class TransformerDecoder(nn.Module):
    """
    Pila de bloques TransformerDecoderBlock.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        encoder_out: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
        return_all_layers: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        
        all_layers = []

        for layer in self.layers:
            x = layer(
                x,
                encoder_out=encoder_out,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_attn_mask=tgt_attn_mask,
                cross_attn_mask=cross_attn_mask,
                is_causal=is_causal,
            )
            if return_all_layers:
                all_layers.append(x)

        if return_all_layers:
            return x, all_layers
        else:
            return x
