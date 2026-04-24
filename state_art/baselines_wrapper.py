import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .strats.embeddings import ContinuousValueEmbedding
from .strats.model import STraTSNetwork
from .coformer.model import CompatibleTransformer

class MultiHorizonBaselineWrapper(nn.Module):
    """
    Wrapper unificador tipo drop-in replacement para el TimeSeriesTransformer.

    Para STraTS en modo denso: incluye los tokens target como tripletas
    adicionales en el transformer, permitiendo cross-attention entre historia
    y futuro (comparación justa con el modelo Custom). Los targets se marcan
    con un feature_id especial = D (último embedding de feature_emb).

    Para CoFormer o modos event: mantiene el enfoque original de inyección
    post-hoc de timestamps a futuro.
    """
    def __init__(self, base_model: nn.Module, model_type: str, d_model: int, output_dim: int, use_sensor_embedding: bool = False, time_scale: float = 900.0):
        super().__init__()
        self.base_model = base_model
        self.model_type = model_type.lower()
        self.d_model = d_model
        self.output_dim = output_dim
        self.use_sensor_embedding = use_sensor_embedding
        self.time_scale = float(time_scale)
        
        # Feature embeddings adicionales para replicar inyección temporal del Transformer
        # (se usa en el fallback para event mode y CoFormer)
        self.target_time_emb = ContinuousValueEmbedding(d_model)
        
        # Reemplazamos la cabeza por una multi-objetivo
        self.target_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )
        # Cabeza para modalidad evento multi-variable multi-temporal
        self.per_target_head = nn.Linear(d_model, 1)

    def _extract_history_and_targets(self, input_values, input_timestamps, is_target_mask, padding_mask, input_sensor_ids):
        """
        Extrae tensores rectangulares para historia y targets.
        Aplica un único punto de sincronización CPU-GPU para extraer num_targets y max_H.
        Usa vectorización con argsort para evitar indexación booleana.
        """
        B, max_len, D = input_values.shape
        device = input_values.device
        
        hist_mask = ~is_target_mask
        if padding_mask is not None:
            hist_mask = hist_mask & ~padding_mask
            
        hist_lens = hist_mask.sum(dim=1)
        
        # --- ÚNICO PUNTO DE SINCRONIZACIÓN ---
        # Se acepta el costo de este .item() porque recortar al max_H real
        # es crítico para evitar explosión cuadrática de atención en STraTS denso.
        # No se cachea num_targets por seguridad (asume variabilidad entre batches).
        max_H = hist_lens.max().item() if hist_lens.max().item() > 0 else 1
        num_targets = int(is_target_mask[0].sum().item())
        
        # Vectorización asíncrona compartida
        pos = torch.arange(max_len, device=device).unsqueeze(0).expand(B, max_len)
        
        # --- Extracción de Targets ---
        # Empujar los elementos válidos de target a la izquierda usando argsort
        # Esto evita la indexación booleana `tensor[mask]` que causa un CPU-GPU sync oculto
        sort_keys_tgt = torch.where(is_target_mask, pos, pos + max_len)
        _, sorted_idx_tgt = torch.sort(sort_keys_tgt, dim=1)
        tgt_idx = sorted_idx_tgt[:, :num_targets]
        
        target_times = torch.gather(input_timestamps, 1, tgt_idx)
        target_s_ids = None
        if self.use_sensor_embedding and input_sensor_ids is not None:
            target_s_ids = torch.gather(input_sensor_ids, 1, tgt_idx)

        # --- Extracción de Historia ---
        # Empujar historia válida a la izquierda
        sort_keys_hist = torch.where(hist_mask, pos, pos + max_len)
        _, sorted_idx_hist = torch.sort(sort_keys_hist, dim=1)
        
        # Cortar a la cantidad máxima REAL de historia en el batch
        hist_idx = sorted_idx_hist[:, :max_H]
        
        hist_times = torch.gather(input_timestamps, 1, hist_idx)
        hist_values = torch.gather(input_values, 1, hist_idx.unsqueeze(-1).expand(-1, -1, D))
        hist_valid = torch.gather(hist_mask, 1, hist_idx)
        
        hist_s_ids = None
        if self.use_sensor_embedding and input_sensor_ids is not None:
            hist_s_ids = torch.gather(input_sensor_ids, 1, hist_idx)
            
        return hist_times, hist_values, hist_valid, hist_s_ids, target_times, target_s_ids, num_targets

    def _forward_strats_dense(
        self,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        is_target_mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor],
        return_dict: bool,
    ) -> torch.Tensor | Dict[str, Any]:
        
        B, max_len, D = input_values.shape
        device = input_values.device

        # Extraer historia y targets de manera segura
        hist_times, hist_values, hist_valid, _, target_times, _, num_targets = self._extract_history_and_targets(
            input_values, input_timestamps, is_target_mask, padding_mask, None
        )
        max_H = hist_times.shape[1]

        # --- Tripletas de historia: [B, max_H * D] ---
        hist_times_exp = hist_times.unsqueeze(2).expand(-1, -1, D).reshape(B, -1)
        hist_values_flat = hist_values.reshape(B, -1)
        hist_fids = torch.arange(D, device=device).view(1, 1, D).expand(B, max_H, -1).reshape(B, -1)
        hist_vmask = hist_valid.unsqueeze(2).expand(-1, -1, D).reshape(B, -1)

        # --- Tripletas target: [B, M] con feature_id = D (marcador target) ---
        target_values_flat = torch.zeros(B, num_targets, device=device)  # placeholder
        target_fids = torch.full((B, num_targets), D, device=device, dtype=torch.long)
        target_vmask = torch.ones(B, num_targets, dtype=torch.bool, device=device)

        # --- Concatenar historia + targets ---
        all_times = torch.cat([hist_times_exp, target_times], dim=1)
        all_values = torch.cat([hist_values_flat, target_values_flat], dim=1)
        all_fids = torch.cat([hist_fids, target_fids], dim=1)
        all_vmask = torch.cat([hist_vmask, target_vmask], dim=1)

        # Normalizar timestamps: (t - t₀) / time_scale
        # Usamos compresión logarítmica para evitar que deltas enormes 
        # (ej. meses/años en segundos) saturen la Tanh del ContinuousValueEmbedding.
        # El clamp previene NaNs causados por tokens de padding (que tienen valor 0.0).
        t0 = all_times[:, :1]
        all_times_norm = torch.log1p(torch.clamp((all_times - t0) / self.time_scale, min=0.0))

        # --- STraTS embeddings sobre la secuencia completa ---
        f_emb = self.base_model.feature_emb(all_fids.long())
        v_emb = self.base_model.value_emb(all_values.float())
        t_emb = self.base_model.time_emb(all_times_norm.float())
        x = f_emb + v_emb + t_emb  # [B, max_H*D + M, d_model]

        src_key_padding_mask = ~all_vmask
        c = self.base_model.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)

        # --- Extraer representaciones de los M tokens target (últimas M pos.) ---
        target_repr = c[:, -num_targets:, :]  # [B, M, d_model]

        # --- Predicción ---
        preds = self.target_head(target_repr)  # [B, M, output_dim]

        if preds.shape[1] == 1:
            preds = preds.squeeze(1)

        if not return_dict:
            return preds

        return {
            "preds": preds,
            "target_states": target_repr,
            "encoder_output": c[:, :-num_targets, :],
        }

    def forward(
        self,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        is_target_mask: torch.Tensor,
        input_sensor_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs
    ) -> torch.Tensor | Dict[str, Any]:

        if self.model_type == "strats" and not self.use_sensor_embedding:
            return self._forward_strats_dense(
                input_values, input_timestamps, is_target_mask,
                padding_mask, return_dict,
            )

        B = input_timestamps.shape[0]
        device = input_timestamps.device
        
        hist_times, hist_values_orig, valid_mask, hist_s_ids, target_timestamps, target_s_ids, num_targets = self._extract_history_and_targets(
            input_values, input_timestamps, is_target_mask, padding_mask, input_sensor_ids
        )
        max_H = hist_times.shape[1]
        
        if not self.use_sensor_embedding:
            D = hist_values_orig.shape[2]
            hist_times_exp = hist_times.unsqueeze(2).expand(-1, -1, D).reshape(B, -1)
            hist_values = hist_values_orig.reshape(B, -1)
            f_ids = torch.arange(D, device=device).unsqueeze(0).unsqueeze(0).expand(B, max_H, -1)
            hist_fids = f_ids.reshape(B, -1)
            v_mask = valid_mask.unsqueeze(2).expand(-1, -1, D).reshape(B, -1)
        else:
            hist_times_exp = hist_times
            hist_values = hist_values_orig.squeeze(-1)
            hist_fids = hist_s_ids
            v_mask = valid_mask
            
        if self.model_type == "strats":
            t0_hist = hist_times_exp[:, :1]
            hist_times_norm = torch.log1p(torch.clamp((hist_times_exp - t0_hist) / self.time_scale, min=0.0))
            f_emb = self.base_model.feature_emb(hist_fids.long())
            v_emb = self.base_model.value_emb(hist_values.float())
            t_emb = self.base_model.time_emb(hist_times_norm.float())
            x = f_emb + v_emb + t_emb
            
            src_key_padding_mask = ~v_mask
            c = self.base_model.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
            h_global, _ = self.base_model.fusion_attention(c, valid_mask=v_mask)
            
        elif self.model_type == "coformer":
            e_l = self.base_model.me(hist_values.float())
            v_l, u_l = self.base_model.vte(hist_fids, hist_times_exp)
            eu = torch.cat([e_l, u_l], dim=-1)
            h_l = torch.cat([v_l, self.base_model.agg_linear(eu)], dim=-1)
            for layer in self.base_model.layers:
                h_l = layer(h_l, hist_fids, hist_times_exp, v_mask)
                
            f_variates = torch.zeros(B, self.base_model.num_variates, self.d_model, device=device)
            for i in range(self.base_model.num_variates):
                var_mask = (hist_fids == i) & v_mask
                sums = torch.sum(h_l * var_mask.unsqueeze(-1), dim=1)
                counts = torch.sum(var_mask, dim=1, keepdim=True).clamp(min=1)
                f_mean = sums / counts
                q = f_mean.unsqueeze(1)
                mha_mask = ~var_mask
                
                # Vectorizado sin sync CPU-GPU:
                # mha_mask.all(dim=1) devuelve True donde la fila completa es True (ningún token válido).
                # Establecemos el índice 0 a False en esas filas mediante bitwise AND.
                all_masked = mha_mask.all(dim=1)
                mha_mask[:, 0] = mha_mask[:, 0] & (~all_masked)
                
                attn_out, _ = self.base_model.agg_mha(q, h_l, h_l, key_padding_mask=mha_mask)
                f_variates[:, i, :] = attn_out.squeeze(1)
            h_global = torch.mean(f_variates, dim=1)
        else:
            raise ValueError(f"Model type no soportado: {self.model_type}")
            
        # Normalizar timestamps target relativo al primer timestamp de historia
        t0_global = input_timestamps[:, :1]
        target_timestamps_norm = torch.log1p(torch.clamp((target_timestamps - t0_global) / self.time_scale, min=0.0))
        
        h_expanded = h_global.unsqueeze(1).expand(-1, num_targets, -1)
        t_tgt_emb = self.target_time_emb(target_timestamps_norm.float())
        h_combined = h_expanded + t_tgt_emb 
        
        if self.use_sensor_embedding:
            preds_flat = self.per_target_head(h_combined).squeeze(-1)
            
            if num_targets % self.output_dim == 0:
                M = num_targets // self.output_dim
                preds = preds_flat.view(B, M, self.output_dim)
            else:
                preds = preds_flat
        else:
            preds = self.target_head(h_combined)
            
            if preds.shape[1] == 1:
                preds = preds.squeeze(1)
                
        if not return_dict:
            return preds
            
        out = {
            "preds": preds,
            "target_states": h_combined,
            "encoder_output": h_global.unsqueeze(1),
        }
        return out

