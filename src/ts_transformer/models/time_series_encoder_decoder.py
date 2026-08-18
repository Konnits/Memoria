from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import nn

from .transformer_blocks import TransformerEncoder, TransformerDecoder
from .heads import RegressionHead
from .masking import validate_target_mask_structure
from .time_series_transformer import TimeSeriesTransformerConfig

from ..features.value_embedding import FeatureEmbedding
from ..features.time_encoding import TimePositionalEncoding
from ..features.target_flag_embedding import TargetFlagEmbedding
from ..features.sensor_embedding import SensorEmbedding


class TimeSeriesEncoderDecoder(nn.Module):
    """
    Modelo Encoder-Decoder para series de tiempo.
    
    Flujo:
      - Los tokens del historial (is_target_mask == False) van al Encoder.
      - Los tokens a predecir (is_target_mask == True) van al Decoder.
      - El Decoder atiende secuencialmente utilizando el estado del Encoder.
    """

    def __init__(self, config: TimeSeriesTransformerConfig) -> None:
        super().__init__()

        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.d_model = config.d_model

        unsupported_experiments = []
        if str(config.prediction_head).lower() != "point":
            unsupported_experiments.append("prediction_head")
        if config.learnable_time_scale:
            unsupported_experiments.append("learnable_time_scale")
        if config.use_continuous_rope:
            unsupported_experiments.append("use_continuous_rope")
        if config.temporal_attention_window is not None:
            unsupported_experiments.append("temporal_attention_window")
        if unsupported_experiments:
            raise ValueError(
                "Las variantes experimentales siguientes están implementadas "
                "para TimeSeriesTransformer, no para EncoderDecoder: "
                + ", ".join(unsupported_experiments)
            )

        self.value_embedding = FeatureEmbedding(
            d_in=self.input_dim,
            d_model=self.d_model,
            use_layernorm=True,
        )

        self.time_encoding: nn.Module = TimePositionalEncoding(
            d_model=self.d_model,
            time_scale=config.time_scale,
            mode=config.time_encoding_mode,
            time_transform=config.time_transform,
        )

        self.time_emb_scale = nn.Parameter(torch.tensor(1.0))
        self.flag_emb_scale = nn.Parameter(torch.tensor(1.0))
        self.sensor_emb_scale = nn.Parameter(torch.tensor(1.0))
        self.input_norm = nn.LayerNorm(self.d_model)

        self.use_target_flag_embedding = bool(config.use_target_flag_embedding)
        if self.use_target_flag_embedding:
            self.flag_embedding: Optional[nn.Module] = TargetFlagEmbedding(
                d_model=self.d_model,
            )
        else:
            self.flag_embedding = None

        self.use_sensor_embedding = bool(config.use_sensor_embedding)
        if self.use_sensor_embedding:
            self.sensor_embedding = SensorEmbedding(
                num_sensors=int(config.num_sensors),
                d_model=self.d_model,
            )
        else:
            self.sensor_embedding = None

        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )
        
        decoder_num_layers = (
            int(config.decoder_num_layers)
            if config.decoder_num_layers is not None
            else int(config.num_layers)
        )
        decoder_num_layers = max(1, decoder_num_layers)

        self.decoder = TransformerDecoder(
            d_model=self.d_model,
            num_heads=config.num_heads,
            num_layers=decoder_num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )

        self.head = RegressionHead(
            d_model=self.d_model,
            output_dim=self.output_dim,
            hidden_dim=None,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.per_target_head = nn.Linear(self.d_model, 1)

    def _embed_tokens(
        self,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        is_target_mask: torch.Tensor,
        input_sensor_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        value_emb = self.value_embedding(input_values)
        time_emb = self.time_encoding(
            input_timestamps,
            padding_mask=padding_mask,
            lengths=lengths,
        ).to(value_emb.dtype)

        x_all = value_emb + self.time_emb_scale.to(value_emb.dtype) * time_emb

        if self.flag_embedding is not None:
            flag_emb = self.flag_embedding(is_target_mask).to(value_emb.dtype)
            x_all = x_all + self.flag_emb_scale.to(value_emb.dtype) * flag_emb

        if self.use_sensor_embedding:
            if input_sensor_ids is None:
                raise ValueError("input_sensor_ids es requerido cuando use_sensor_embedding=True.")
            sensor_emb = self.sensor_embedding(input_sensor_ids.to(torch.long)).to(value_emb.dtype)
            x_all = x_all + self.sensor_emb_scale.to(value_emb.dtype) * sensor_emb

        return self.input_norm(x_all)

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
    ) -> torch.Tensor | Dict[str, Any]:
        """
        Calcula las representaciones separadas y envía la historia al encoder
        y los targets al decoder.
        """
        B, L, D_in = input_values.shape

        x_all = self._embed_tokens(
            input_values=input_values,
            input_timestamps=input_timestamps,
            is_target_mask=is_target_mask,
            input_sensor_ids=input_sensor_ids,
            padding_mask=padding_mask,
            lengths=lengths,
        )

        # Separar historial de target estructuralmente (Slicing)
        # Asumiendo contrato de SequenceBuilder: targets siempre al final
        if self.config.validate_inputs:
            num_target_tokens = validate_target_mask_structure(
                is_target_mask,
                batch_size=B,
                seq_len=L,
            )
        else:
            num_target_tokens = int(is_target_mask[0].sum().item())
        num_history_tokens = L - num_target_tokens

        # x_enc: historia
        x_enc = x_all[:, :num_history_tokens, :]
        
        # padding mask para encoder (historia)
        if padding_mask is not None:
            enc_padding_mask = padding_mask[:, :num_history_tokens]
        else:
            enc_padding_mask = None
            
        # x_dec: targets
        x_dec = x_all[:, -num_target_tokens:, :]
        
        if padding_mask is not None:
            dec_padding_mask = padding_mask[:, -num_target_tokens:]
        else:
            dec_padding_mask = None

        # --- ENCODER PASS ---
        encoder_output = self.encoder(
            x_enc,
            key_padding_mask=enc_padding_mask,
            attn_mask=None, 
            return_all_layers=False,
        )

        # Máscara causal opcional para Decoder delegada por flag booleano
        is_causal = False
        tgt_attn_mask = attn_mask
        if attn_mask is None and self.config.use_causal_mask:
            is_causal = True

        # --- DECODER PASS ---
        if return_all_layers:
            decoder_output, all_layers = self.decoder(
                x_dec,
                encoder_out=encoder_output,
                tgt_key_padding_mask=dec_padding_mask,
                memory_key_padding_mask=enc_padding_mask,
                tgt_attn_mask=tgt_attn_mask,
                cross_attn_mask=None,
                is_causal=is_causal,
                return_all_layers=True,
            )
        else:
            decoder_output = self.decoder(
                x_dec,
                encoder_out=encoder_output,
                tgt_key_padding_mask=dec_padding_mask,
                memory_key_padding_mask=enc_padding_mask,
                tgt_attn_mask=tgt_attn_mask,
                cross_attn_mask=None,
                is_causal=is_causal,
                return_all_layers=False,
            )
            all_layers = None

        # Regression
        target_states = decoder_output

        if self.use_sensor_embedding:
            preds_flat = self.per_target_head(target_states).squeeze(-1)
            M = num_target_tokens // self.output_dim
            preds = preds_flat.view(B, M, self.output_dim)
        else:
            preds = self.head(target_states)

        if preds.shape[1] == 1:
            preds = preds.squeeze(1)

        if not return_dict:
            return preds

        out: Dict[str, Any] = {
            "preds": preds,
            "target_states": target_states,
            "encoder_output": encoder_output,
            "decoder_output": decoder_output
        }
        if all_layers is not None:
            out["all_layers"] = all_layers
        return out

    @torch.no_grad()
    def generate(
        self,
        history_values: torch.Tensor,
        history_timestamps: torch.Tensor,
        target_timestamps: torch.Tensor,
        history_sensor_ids: Optional[torch.Tensor] = None,
        target_sensor_ids: Optional[torch.Tensor] = None,
        history_padding_mask: Optional[torch.Tensor] = None,
        history_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Genera predicciones autoregresivamente para un conjunto de timestamps futuros.
        
        Parameters
        ----------
        history_values : [B, L, input_dim]
        history_timestamps : [B, L]
        target_timestamps : [B, M]
        history_padding_mask : [B, L], opcional
            True indica padding en la historia. Permite evaluar batches con
            left-padding sin contaminar el encoder ni el origen temporal.
        ...
        
        Returns
        -------
        preds : [B, M, output_dim]
        """
        self.eval()
        B, L, D_in = history_values.shape
        M = target_timestamps.shape[1]
        
        device = history_values.device
        dtype = history_values.dtype
        if history_padding_mask is not None:
            history_padding_mask = history_padding_mask.to(device=device, dtype=torch.bool)
        if history_lengths is not None:
            history_lengths = history_lengths.to(device=device)

        if self.use_sensor_embedding:
            if history_sensor_ids is None or target_sensor_ids is None:
                raise ValueError(
                    "history_sensor_ids y target_sensor_ids son requeridos en modo evento."
                )
            target_sensor_ids = target_sensor_ids.to(device=device, dtype=torch.long)
            target_times = target_timestamps.repeat_interleave(self.output_dim, dim=1)
            target_token_count = M * self.output_dim
            if target_sensor_ids.shape != (B, target_token_count):
                raise ValueError(
                    "target_sensor_ids debe tener shape "
                    f"[B, M * output_dim]={B, target_token_count}; "
                    f"se obtuvo {tuple(target_sensor_ids.shape)}."
                )
            history_sensor_ids = history_sensor_ids.to(device=device, dtype=torch.long)
        else:
            target_times = target_timestamps
            target_token_count = M

        history_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        history_emb = self._embed_tokens(
            input_values=history_values,
            input_timestamps=history_timestamps,
            is_target_mask=history_mask,
            input_sensor_ids=history_sensor_ids,
            padding_mask=history_padding_mask,
            lengths=history_lengths,
        )
        encoder_output = self.encoder(
            history_emb,
            key_padding_mask=history_padding_mask,
            attn_mask=None,
            return_all_layers=False,
        )
        
        # Guardaremos las predicciones que generemos
        generations = []
        
        # El primer input al decoder es un token placeholder (ej. ceros)
        current_target_inputs = torch.zeros(B, 1, D_in, dtype=dtype, device=device)
        
        for k in range(target_token_count):
            # Construir embeddings con la referencia temporal de la historia,
            # pero reutilizar la salida del encoder ya calculada.
            input_values = torch.cat([history_values, current_target_inputs], dim=1)
            current_target_timestamps = target_times[:, :k + 1]
            input_timestamps = torch.cat([history_timestamps, current_target_timestamps], dim=1)
            
            is_target_mask = torch.zeros(B, L + k + 1, dtype=torch.bool, device=device)
            is_target_mask[:, L:] = True
            full_padding_mask = None
            if history_padding_mask is not None:
                target_padding = torch.zeros(B, k + 1, dtype=torch.bool, device=device)
                full_padding_mask = torch.cat([history_padding_mask, target_padding], dim=1)

            input_sensor_ids = None
            if self.use_sensor_embedding:
                input_sensor_ids = torch.cat(
                    [history_sensor_ids, target_sensor_ids[:, :k + 1]], dim=1
                )

            x_all = self._embed_tokens(
                input_values=input_values,
                input_timestamps=input_timestamps,
                is_target_mask=is_target_mask,
                input_sensor_ids=input_sensor_ids,
                padding_mask=full_padding_mask,
            )
            x_dec = x_all[:, L:, :]

            decoder_output = self.decoder(
                x_dec,
                encoder_out=encoder_output,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=history_padding_mask,
                tgt_attn_mask=None,
                cross_attn_mask=None,
                is_causal=bool(self.config.use_causal_mask),
                return_all_layers=False,
            )
            if self.use_sensor_embedding:
                latest_pred = self.per_target_head(decoder_output[:, -1:, :])
            else:
                preds = self.head(decoder_output)
                latest_pred = preds if preds.dim() == 3 else preds.unsqueeze(1)
                latest_pred = latest_pred[:, -1:, :]

            generations.append(latest_pred)
            
            # Convertimos predicción a input_dim si no es el último paso
            if k < target_token_count - 1:
                next_input = torch.zeros(B, 1, D_in, dtype=dtype, device=device)
                out_d = min(D_in, latest_pred.shape[-1])
                next_input[:, 0, :out_d] = latest_pred[:, 0, :out_d]
                current_target_inputs = torch.cat([current_target_inputs, next_input], dim=1)
                
        all_generations = torch.cat(generations, dim=1)
        if self.use_sensor_embedding:
            all_generations = all_generations.squeeze(-1).view(B, M, self.output_dim)
        return all_generations
