from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
from torch import nn

from .transformer_blocks import TransformerEncoder
from .heads import RegressionHead

# Importamos los módulos de embeddings/encodings de tiempo y valores.
# Se asume que existen en ts_transformer.features con las firmas:
#   FeatureEmbedding(d_in: int, d_model: int)
#       forward(x: [B, L, d_in]) -> [B, L, d_model]
#   TimePositionalEncoding(d_model: int, time_scale: float = 1.0)
#       forward(timestamps: [B, L]) -> [B, L, d_model]
#   TargetFlagEmbedding(d_model: int)
#       forward(is_target_mask: [B, L]) -> [B, L, d_model]
from ..features.value_embedding import FeatureEmbedding
from ..features.time_encoding import TimePositionalEncoding
from ..features.target_flag_embedding import TargetFlagEmbedding
from ..features.sensor_embedding import SensorEmbedding
from ..features.temporal_attention_bias import TemporalAttentionBias


@dataclass
class TimeSeriesTransformerConfig:
    """
    Configuración del modelo TimeSeriesTransformer.
    """

    input_dim: int          # Dimensión de entrada (nº de variables: temp, presión, etc.)
    output_dim: int         # Dimensión de salida (nº de variables a predecir)
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    activation: str = "relu"
    time_scale: float = 900.0  # p.ej. 900 segundos = 15 minutos
    use_causal_mask: bool = False  # para experimentos autoregresivos (opcional)
    use_sensor_embedding: bool = False
    num_sensors: int = 0
    time_encoding_mode: str = "sinusoidal"  # "sinusoidal" | "mlp" | "time2vec"
    use_temporal_attn_bias: bool = False
    use_target_flag_embedding: bool = True
    validate_inputs: bool = True


class TimeSeriesTransformer(nn.Module):
    """
    Modelo Transformer para series de tiempo con:
      - Timestamps no equiespaciados.
      - Encoding temporal continuo.
      - Token "target" añadido a la secuencia con el timestamp objetivo.

    Flujo (por batch):
      input_values     : [B, L, input_dim]
      input_timestamps : [B, L]
      is_target_mask   : [B, L] (bool; True sólo en el token target)
      padding_mask     : [B, L] (bool; True = padding), opcional

    El modelo:
      1. Proyecta valores continuos a d_model (FeatureEmbedding).
      2. Calcula encoding temporal continuo (TimePositionalEncoding).
      3. Añade embedding de flag target/historia (TargetFlagEmbedding).
      4. Suma todo para obtener embeddings de tokens.
      5. Pasa por un encoder Transformer.
      6. Extrae el estado del token target y lo pasa por la RegressionHead.
    """

    def __init__(self, config: TimeSeriesTransformerConfig) -> None:
        super().__init__()

        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.d_model = config.d_model

        # Embedding de valores (features continuas multivariadas)
        self.value_embedding = FeatureEmbedding(
            d_in=self.input_dim,
            d_model=self.d_model,
        )

        # Encoding posicional temporal continuo
        self.time_encoding: nn.Module = TimePositionalEncoding(
            d_model=self.d_model,
            time_scale=config.time_scale,
            mode=config.time_encoding_mode,
        )

        # Embedding para distinguir tokens de historia vs token target
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

        # Temporal Attention Bias (opcional)
        self.use_temporal_attn_bias = bool(config.use_temporal_attn_bias)
        if self.use_temporal_attn_bias:
            self.temporal_attn_bias = TemporalAttentionBias(
                num_heads=config.num_heads,
            )
        else:
            self.temporal_attn_bias = None

        # Encoder Transformer
        self.encoder = TransformerEncoder(
            d_model=self.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
        )

        # Cabeza de regresión sobre el token target
        self.head = RegressionHead(
            d_model=self.d_model,
            output_dim=self.output_dim,
            hidden_dim=None,
            dropout=config.dropout,
            activation=config.activation,
        )
        # Head auxiliar para modo multi-target-token (K tokens target -> K salidas escalares).
        self.per_target_head = nn.Linear(self.d_model, 1)

    def forward(
        self,
        input_values: torch.Tensor,
        input_timestamps: torch.Tensor,
        is_target_mask: torch.Tensor,
        input_sensor_ids: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        return_all_layers: bool = False,
    ) -> torch.Tensor | Dict[str, Any]:
        """
        Parameters
        ----------
        input_values:
            Tensor [B, L, input_dim].
        input_timestamps:
            Tensor [B, L] con timestamps (numéricos).
        is_target_mask:
            Tensor bool [B, L], con exactamente un True por secuencia
            indicando la posición del token target.
        padding_mask:
            Tensor bool [B, L], True indica padding. Opcional.
        attn_mask:
            Máscara adicional de atención (por ejemplo causal), shape [L, L] o [B*H, L, L].
            Si es None y config.use_causal_mask es True, se construye una máscara causal
            compartida para todas las capas.
        return_dict:
            Si True, devuelve un diccionario con salidas y representaciones intermedias.
        return_all_layers:
            Si True, el encoder devuelve las representaciones de todas las capas.

        Returns
        -------
        Si return_dict es False:
            preds: [B, output_dim]
        Si return_dict es True:
            {
                "preds": [B, output_dim],
                "target_states": [B, d_model],
                "encoder_output": [B, L, d_model],
                "all_layers": list de [B, L, d_model] (si return_all_layers=True)
            }
        """
        B, L, D_in = input_values.shape

        if self.config.validate_inputs:
            if D_in != self.input_dim:
                raise ValueError(
                    f"input_values.shape[2] = {D_in}, pero se esperaba input_dim={self.input_dim}."
                )

            if input_timestamps.shape != (B, L):
                raise ValueError(
                    f"input_timestamps debe tener shape [B, L]={B, L}, "
                    f"pero se obtuvo {tuple(input_timestamps.shape)}."
                )

            if is_target_mask.shape != (B, L):
                raise ValueError(
                    f"is_target_mask debe tener shape [B, L]={B, L}, "
                    f"pero se obtuvo {tuple(is_target_mask.shape)}."
                )

            if padding_mask is not None and padding_mask.shape != (B, L):
                raise ValueError(
                    f"padding_mask debe tener shape [B, L]={B, L}, "
                    f"pero se obtuvo {tuple(padding_mask.shape)}."
                )

            # Comprobamos que haya al menos un token target por secuencia
            target_counts = is_target_mask.sum(dim=1)
            if not torch.all(target_counts > 0):
                raise ValueError(
                    "Cada secuencia debe tener al menos un token target (is_target_mask True). "
                    f"Se obtuvieron cuentas {target_counts.tolist()}."
                )
            if not torch.all(target_counts == target_counts[0]):
                raise ValueError(
                    "Todas las secuencias del batch deben tener el mismo número de target tokens. "
                    f"Se obtuvieron cuentas {target_counts.tolist()}."
                )
            num_target_tokens = int(target_counts[0].item())
        else:
            num_target_tokens = int(is_target_mask[0].sum().item())

        # Embeddings de valores y tiempo
        value_emb = self.value_embedding(input_values)         # [B, L, d_model]
        time_emb = self.time_encoding(input_timestamps).to(value_emb.dtype) # [B, L, d_model]
        x = value_emb + time_emb                               # [B, L, d_model]

        # Embedding de flag opcional (historia/target)
        if self.use_target_flag_embedding:
            if self.flag_embedding is None:
                raise RuntimeError("use_target_flag_embedding=True pero flag_embedding no está inicializado.")
            flag_emb = self.flag_embedding(is_target_mask).to(value_emb.dtype)  # [B, L, d_model]
            x = x + flag_emb

        if self.use_sensor_embedding:
            if input_sensor_ids is None:
                raise ValueError(
                    "El modelo requiere input_sensor_ids cuando use_sensor_embedding=True."
                )
            if input_sensor_ids.shape != (B, L):
                raise ValueError(
                    f"input_sensor_ids debe tener shape [B, L]={B, L}, "
                    f"pero se obtuvo {tuple(input_sensor_ids.shape)}."
                )
            sensor_emb = self.sensor_embedding(input_sensor_ids.to(torch.long))
            x = x + sensor_emb

        # Máscara causal opcional delegada por flag booleano
        is_causal = False
        if attn_mask is None and self.config.use_causal_mask:
            is_causal = True

        # Temporal Attention Bias (si está habilitado)
        temporal_bias = None
        if self.use_temporal_attn_bias and self.temporal_attn_bias is not None:
            # Usar timestamps normalizados para el bias
            t0 = input_timestamps[:, :1]
            tau = (input_timestamps - t0) / self.config.time_scale
            temporal_bias = self.temporal_attn_bias(tau)

        # Encoder Transformer
        if return_all_layers:
            encoder_output, all_layers = self.encoder(
                x,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
                temporal_bias=temporal_bias,
                is_causal=is_causal,
                return_all_layers=True,
            )
        else:
            encoder_output = self.encoder(
                x,
                key_padding_mask=padding_mask,
                attn_mask=attn_mask,
                temporal_bias=temporal_bias,
                is_causal=is_causal,
                return_all_layers=False,
            )
            all_layers = None

        # Extraer estados de tokens target
        # En esta arquitectura estructurada (dictada por SequenceBuilder),
        # los targets siempre se ubican al final de la secuencia.
        if __debug__ and self.config.validate_inputs:
            assert is_target_mask[:, -num_target_tokens:].all() and (~is_target_mask[:, :-num_target_tokens]).all(), \
                "Error: is_target_mask indica que los targets no están estrictamente al final."
        target_states = encoder_output[:, -num_target_tokens:, :]

        # Cabeza de regresión
        if self.use_sensor_embedding:
            # Event mode: cada token corresponde a un sensor a predecir.
            # Salida escalar por token [B, K_total, 1] -> [B, K_total]
            preds_flat = self.per_target_head(target_states).squeeze(-1)
            # Reconstruir [B, M, output_dim]. Asumimos que K_total = M * output_dim
            if num_target_tokens % self.output_dim != 0:
                raise ValueError(f"num_target_tokens={num_target_tokens} no es divisible por output_dim={self.output_dim}")
            M = num_target_tokens // self.output_dim
            preds = preds_flat.view(B, M, self.output_dim)
        else:
            # Dense mode: cada token futuro (M tokens) predice todas las variables.
            # head produce [B, M, output_dim] directamente (pues num_target_tokens == M)
            preds = self.head(target_states)

        # Retornar a compatibilidad con código univariado en tiempo
        if preds.shape[1] == 1:
            preds = preds.squeeze(1) # [B, output_dim]

        if not return_dict:
            return preds

        out: Dict[str, Any] = {
            "preds": preds,
            "target_states": target_states,
            "encoder_output": encoder_output,
        }
        if all_layers is not None:
            out["all_layers"] = all_layers
        return out
    
    def summary(
        self,
        seq_len: int,
        batch_size: int = 1,
        device: Optional[str] = None,
        print_fn=print,
    ) -> str:
        """
        Imprime un resumen tipo Keras del modelo.

        Parameters
        ----------
        seq_len:
            Longitud de la secuencia a usar para inferir las shapes.
        batch_size:
            Tamaño de batch ficticio para la pasada forward del resumen.
        device:
            Dispositivo a usar ("cpu", "cuda", "cuda:0", ...).
            Si es None, intenta usar el dispositivo actual de los parámetros.
        print_fn:
            Función a la que se le envía el texto (por defecto `print`).

        Returns
        -------
        summary_str:
            Cadena con el resumen completo.
        """
        # ------------------------------------------------------------------
        # 1) Elegir device
        # ------------------------------------------------------------------
        if device is None:
            try:
                first_param = next(self.parameters())
                device_obj = first_param.device
            except StopIteration:
                device_obj = torch.device("cpu")
        else:
            device_obj = torch.device(device)

        self.to(device_obj)

        # Guardar modo entrenamiento para restaurarlo luego
        was_training = self.training
        self.eval()

        # ------------------------------------------------------------------
        # 2) Construir batch dummy para inferir shapes
        # ------------------------------------------------------------------
        B = batch_size
        L = seq_len
        D_in = self.input_dim

        # Valores de entrada
        dummy_input_values = torch.zeros(B, L, D_in, dtype=torch.float32, device=device_obj)
        # Timestamps (cualquier cosa creciente sirve; aquí 0,1,2,...)
        dummy_timestamps = torch.arange(L, dtype=torch.float32, device=device_obj).unsqueeze(0)
        dummy_timestamps = dummy_timestamps.expand(B, -1)  # [B, L]
        # Token target: último elemento de la secuencia
        dummy_is_target = torch.zeros(B, L, dtype=torch.bool, device=device_obj)
        dummy_is_target[:, -1] = True
        # Sin padding (todo válido)
        dummy_padding = torch.zeros(B, L, dtype=torch.bool, device=device_obj)

        # ------------------------------------------------------------------
        # 3) Registrar hooks en submódulos para capturar output shapes
        # ------------------------------------------------------------------
        summary_data = []
        hooks = []

        # Mapa módulo -> nombre jerárquico
        module_to_name: Dict[nn.Module, str] = {}
        for name, module in self.named_modules():
            module_to_name[module] = name if name != "" else "model"

        def register_hook(module: nn.Module) -> None:
            # Evitamos registrar en contenedores "puros" y en el propio modelo raíz
            if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
                return
            if module is self:
                return

            def hook(module, inputs, output):
                class_name = module.__class__.__name__
                module_name = module_to_name.get(module, "")

                # Parámetros sólo de este módulo (no de hijos)
                params = sum(p.numel() for p in module.parameters(recurse=False))
                trainable = sum(
                    p.numel() for p in module.parameters(recurse=False) if p.requires_grad
                )

                # Determinar shape de salida
                out_shape_str = "?"
                out_tensor = None

                if isinstance(output, torch.Tensor):
                    out_tensor = output
                elif isinstance(output, (list, tuple)):
                    for o in output:
                        if isinstance(o, torch.Tensor):
                            out_tensor = o
                            break

                if out_tensor is not None:
                    out_shape_str = str(tuple(out_tensor.size()))

                summary_data.append(
                    {
                        "name": module_name,
                        "class_name": class_name,
                        "output_shape": out_shape_str,
                        "num_params": params,
                        "trainable": trainable,
                    }
                )

            hooks.append(module.register_forward_hook(hook))

        # Registrar en todos los submódulos
        self.apply(register_hook)

        # ------------------------------------------------------------------
        # 4) Forward pass dummy (sin gradientes)
        # ------------------------------------------------------------------
        with torch.no_grad():
            _ = self(
                input_values=dummy_input_values,
                input_timestamps=dummy_timestamps,
                is_target_mask=dummy_is_target,
                padding_mask=dummy_padding,
                attn_mask=None,
                return_dict=False,
            )

        # Quitar hooks
        for h in hooks:
            h.remove()

        # Restaurar modo entrenamiento original
        self.train(was_training)

        # ------------------------------------------------------------------
        # 5) Construir tabla tipo Keras
        # ------------------------------------------------------------------
        # Configuración de ancho de columnas
        line_length = 100
        name_width = 50
        shape_width = 30
        param_width = 15

        lines = []
        lines.append(f'Model: "{self.__class__.__name__}"')
        lines.append("=" * line_length)
        header = (
            f"{'Layer (name)':<{name_width}} "
            f"{'Output Shape':<{shape_width}} "
            f"{'Param #':>{param_width}}"
        )
        lines.append(header)
        lines.append("=" * line_length)

        for layer_info in summary_data:
            name = layer_info["name"]
            if name == "model":
                # Por si se coló el root (no debería), lo saltamos
                continue

            name_str = f"{name} ({layer_info['class_name']})"
            out_shape = layer_info["output_shape"]
            num_params = layer_info["num_params"]

            line = (
                f"{name_str:<{name_width}} "
                f"{out_shape:<{shape_width}} "
                f"{num_params:>{param_width}d}"
            )
            lines.append(line)

        lines.append("=" * line_length)

        # Totales
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        lines.append(f"Total params: {total_params:,}")
        lines.append(f"Trainable params: {trainable_params:,}")
        lines.append(f"Non-trainable params: {non_trainable_params:,}")
        lines.append("=" * line_length)

        summary_str = "\n".join(lines)
        print_fn(summary_str)
