from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
import importlib.util

import os
import time
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .dilate_loss import DILATELoss
from .losses import get_loss_fn
from .metrics import (
    compute_gaussian_metrics,
    compute_regression_metrics,
    compute_structured_regression_metrics,
)
from .optimizers import OptimizerConfig, build_optimizer, build_scheduler


@dataclass
class TrainingConfig:
    """
    Configuración de alto nivel para el entrenamiento.

    Atributos principales:
    - num_epochs:
        Número de épocas.
    - device:
        Dispositivo ("cpu", "cuda", "cuda:0", etc.).
    - loss_name:
        Nombre de la función de pérdida ("mse", "mae", "huber").
    - optimizer_config:
        Configuración del optimizador y scheduler.
    - grad_clip_norm:
        Si > 0, aplica clipping de gradiente por norma L2.
    - log_every_n_steps:
        Frecuencia de logging (en batches).
    - checkpoint_dir:
        Carpeta donde guardar checkpoints (si no es None).
    - save_best_on:
        Métrica a usar para guardar el mejor modelo.
        Opciones típicas:
            * "val_loss"
            * "val_rmse"
            * etc. (debe coincidir con la llave en el dict de métricas).
    """

    num_epochs: int = 20
    device: str = "cpu"
    loss_name: str = "mse"
    optimizer_config: OptimizerConfig = field(default_factory=OptimizerConfig)
    grad_clip_norm: float = 0.0
    log_every_n_steps: int = 50
    checkpoint_dir: Optional[str] = None
    save_best_on: str = "val_loss"
    freeze_encoder_epochs: int = 0
    unfreeze_lr: Optional[float] = None
    input_noise_std: float = 0.0
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    restore_best_weights: bool = True
    use_amp: bool = False
    enable_cuda_runtime_optimizations: bool = True
    use_torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    torch_compile_fullgraph: bool = False


class Trainer:
    """
    Bucle de entrenamiento para TimeSeriesTransformer (o modelos compatibles).

    Supone que los batches del DataLoader tienen al menos:
        - "input_values": [B, L, input_dim]
        - "input_timestamps": [B, L]
        - "is_target_mask": [B, L]
        - "target_values": [B, output_dim]
      y opcionalmente:
        - "padding_mask": [B, L]
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """
        Parameters
        ----------
        model:
            Modelo a entrenar (por ejemplo, TimeSeriesTransformer).
        train_loader:
            DataLoader para el conjunto de entrenamiento.
        val_loader:
            DataLoader para validación (opcional).
        config:
            Configuración de entrenamiento.
        loss_fn:
            Función de pérdida. Si es None, se construye con get_loss_fn().
        optimizer:
            Optimizador. Si es None, se construye con build_optimizer().
        scheduler:
            Scheduler de LR opcional. Si es None, se construye con build_scheduler().
        """
        if config is None:
            config = TrainingConfig()

        self.config = config
        self.device = torch.device(config.device)

        self._configure_cuda_runtime()

        self.model = model.to(self.device)
        self._compile_model_if_requested()
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Pérdida
        if loss_fn is None:
            self.loss_fn = get_loss_fn(config.loss_name)
        else:
            self.loss_fn = loss_fn

        # Optimizador
        if optimizer is None:
            self.optimizer = build_optimizer(self.model, config.optimizer_config)
        else:
            self.optimizer = optimizer

        # Scheduler
        if scheduler is None:
            self.scheduler = build_scheduler(self.optimizer, config.optimizer_config)
        else:
            self.scheduler = scheduler

        # Estado interno
        self.best_metric_value: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self._encoder_is_frozen: Optional[bool] = None
        self._epochs_without_improvement: int = 0
        self._best_model_state_in_memory: Optional[Dict[str, torch.Tensor]] = None

        # AMP (Automatic Mixed Precision)
        self.use_amp = config.use_amp and self.device.type == "cuda"
        self.amp_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        # Crear carpeta de checkpoints si aplica
        if self.config.checkpoint_dir is not None:
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)

    def _configure_cuda_runtime(self) -> None:
        """Activa rutas rápidas de CUDA/SDPA cuando se entrena en GPU."""
        if self.device.type != "cuda" or not self.config.enable_cuda_runtime_optimizations:
            return

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True

        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

        if hasattr(torch.backends, "cuda"):
            if hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(True)
            if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            if hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)

    def _compile_model_if_requested(self) -> None:
        """Compila el modelo con torch.compile de forma segura y opcional."""
        if not self.config.use_torch_compile:
            return
        if self.device.type != "cuda":
            return
        if not hasattr(torch, "compile"):
            return

        if importlib.util.find_spec("triton") is None:
            print(
                "[Trainer] torch.compile omitido: Triton no está disponible en este entorno."
            )
            return

        try:
            import torch._dynamo as torch_dynamo  # type: ignore

            torch_dynamo.config.suppress_errors = True
        except Exception:
            pass

        try:
            self.model = torch.compile(
                self.model,
                mode=self.config.torch_compile_mode,
                fullgraph=bool(self.config.torch_compile_fullgraph),
            )
            print(
                "[Trainer] torch.compile activado "
                f"(mode={self.config.torch_compile_mode}, "
                f"fullgraph={self.config.torch_compile_fullgraph})."
            )
        except Exception as exc:
            print(
                "[Trainer] torch.compile no disponible para este modelo/runtime; "
                f"se continúa en eager. Motivo: {exc}"
            )

    def fit(self) -> Dict[str, Any]:
        """
        Ejecuta el bucle de entrenamiento completo.

        Returns
        -------
        history:
            Diccionario con listas de métricas por época:
                {
                    "train_loss": [...],
                    "val_loss": [...],
                    "val_rmse": [...],
                    ...
                }
        """
        history: Dict[str, list] = {}
        stopped_early = False

        for epoch in range(1, self.config.num_epochs + 1):
            start_time = time.time()

            # Los datasets con muestreo temporal reproducible (horizontes,
            # subsampling de historia) deben avanzar su stream en cada época.
            self._set_loader_epoch(self.train_loader, epoch - 1)

            # Política opcional para fine-tuning en dos fases:
            # 1) congelar encoder durante N épocas
            # 2) descongelar y opcionalmente bajar LR
            self._apply_finetune_schedule(epoch)

            train_loss = self._train_one_epoch(epoch)
            self._append_history(history, "train_loss", train_loss)

            metrics_val = {}
            if self.val_loader is not None:
                metrics_val = self._evaluate(epoch)
                # Copiar métricas de val a history
                for k, v in metrics_val.items():
                    self._append_history(history, k, v)

                # Elegir métrica para guardar mejor modelo
                if self.config.save_best_on in metrics_val:
                    current_value = metrics_val[self.config.save_best_on]
                else:
                    # Fallback: val_loss si existe, sino nada
                    current_value = metrics_val.get("val_loss", None)

                if current_value is not None:
                    improved = self._maybe_save_best(epoch, current_value)
                    if self.config.early_stopping_patience > 0:
                        if improved:
                            self._epochs_without_improvement = 0
                        else:
                            self._epochs_without_improvement += 1

            # Scheduler step (después de cada época)
            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - start_time
            self._log_epoch_summary(epoch, train_loss, metrics_val, elapsed)

            # Early stopping (solo aplica si hay validación)
            if (
                self.val_loader is not None
                and self.config.early_stopping_patience > 0
                and self._epochs_without_improvement >= self.config.early_stopping_patience
            ):
                print(
                    "[Trainer] Early stopping activado: "
                    f"sin mejora en {self._epochs_without_improvement} épocas."
                )
                stopped_early = True
                break

        if (
            stopped_early
            and self.config.restore_best_weights
            and self._best_model_state_in_memory is not None
        ):
            self.model.load_state_dict(self._best_model_state_in_memory)
            print("[Trainer] Se restauraron los mejores pesos en memoria tras early stopping.")

        return history

    @classmethod
    def _set_dataset_epoch(
        cls,
        dataset: Any,
        epoch: int,
        visited: Optional[set[int]] = None,
    ) -> bool:
        """Propaga ``set_epoch`` a wrappers como Subset/ConcatDataset."""
        if visited is None:
            visited = set()
        identity = id(dataset)
        if identity in visited:
            return False
        visited.add(identity)

        changed = False
        setter = getattr(dataset, "set_epoch", None)
        epoch_dependent = bool(
            getattr(dataset, "epoch_dependent_sampling", True)
        )
        if callable(setter) and epoch_dependent:
            setter(int(epoch))
            changed = True

        nested = getattr(dataset, "dataset", None)
        if nested is not None:
            changed = cls._set_dataset_epoch(nested, epoch, visited) or changed
        children = getattr(dataset, "datasets", None)
        if children is not None:
            for child in children:
                changed = cls._set_dataset_epoch(child, epoch, visited) or changed
        return changed

    @classmethod
    def _set_loader_epoch(cls, loader: DataLoader, epoch: int) -> None:
        sampler = getattr(loader, "sampler", None)
        sampler_setter = getattr(sampler, "set_epoch", None)
        if callable(sampler_setter):
            sampler_setter(int(epoch))

        dataset_changed = cls._set_dataset_epoch(loader.dataset, epoch)
        if (
            dataset_changed
            and bool(getattr(loader, "persistent_workers", False))
            and int(getattr(loader, "num_workers", 0)) > 0
        ):
            raise RuntimeError(
                "set_epoch no se propaga a copias ya persistentes del dataset. "
                "Use persistent_workers=False para muestreo temporal por época."
            )

    # ------------------------------------------------------------------
    # Entrenamiento y evaluación
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        running_loss = torch.zeros((), device=self.device)
        num_batches = 0
        total = len(self.train_loader)

        for step, batch in enumerate(self.train_loader, start=1):
            loss_value = self._train_step(batch, synchronize=False)
            if not isinstance(loss_value, torch.Tensor):
                raise TypeError("_train_step sin sincronización debe retornar un tensor.")
            running_loss.add_(loss_value)
            num_batches += 1

            should_log = (
                self.config.log_every_n_steps > 0
                and (
                    step % self.config.log_every_n_steps == 0
                    or step == total
                )
            )
            if should_log:
                avg_loss = float((running_loss / num_batches).item())
                progress = int(30 * step / total) if total > 0 else 0
                bar = "=" * progress + "." * max(0, 30 - progress)
                # ponytail: progreso en-lugar (sin dependencias extra) como TensorFlow para reducir spam
                print(
                    f"\r[Epoch {epoch:03d}] {step:05d}/{total:05d} [{bar}] - loss: {avg_loss:.4f}",
                    end="",
                    flush=True,
                )

        if self.config.log_every_n_steps > 0 and num_batches > 0:
            print()

        if num_batches == 0:
            return 0.0
        return float((running_loss / num_batches).item())

    def _train_step(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        synchronize: bool = True,
    ) -> float | torch.Tensor:
        self.optimizer.zero_grad(set_to_none=True)

        # Mover batch al device (non_blocking requiere pin_memory en DataLoader)
        input_values = batch["input_values"].to(self.device, non_blocking=True)           # [B, L, D_in]
        input_timestamps = batch["input_timestamps"].to(self.device, non_blocking=True)   # [B, L]
        is_target_mask = batch["is_target_mask"].to(self.device, non_blocking=True)       # [B, L]
        input_sensor_ids = batch.get("input_sensor_ids", None)
        if input_sensor_ids is not None:
            input_sensor_ids = input_sensor_ids.to(self.device, non_blocking=True)
        target_values = batch["target_values"].to(self.device, non_blocking=True)         # [B, D_out]
        target_loss_mask = batch.get("target_loss_mask", None)
        if target_loss_mask is not None:
            target_loss_mask = target_loss_mask.to(self.device, non_blocking=True)
        padding_mask = batch.get("padding_mask", None)
        if padding_mask is not None:
            padding_mask = padding_mask.to(self.device, non_blocking=True)
        lengths = batch.get("lengths", None)
        if lengths is not None:
            lengths = lengths.to(self.device, non_blocking=True)
        temporal_features = batch.get("temporal_features", None)
        if temporal_features is not None:
            temporal_features = temporal_features.to(self.device, non_blocking=True)

        # Denoising opcional: añade ruido gaussiano sólo en posiciones no-padding.
        # Útil para robustecer la tarea proxy en pretraining.
        if self.config.input_noise_std > 0.0:
            noise = torch.randn_like(input_values) * float(self.config.input_noise_std)
            if padding_mask is not None:
                valid_mask = (~padding_mask).unsqueeze(-1).to(input_values.dtype)
                noise = noise * valid_mask
            input_values = input_values + noise

        # Forward con AMP autocast
        with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
            return_distribution = (
                str(getattr(getattr(self.model, "config", None), "prediction_head", "point"))
                .lower()
                == "gaussian"
            )
            forward_kwargs = dict(
                input_values=input_values,
                input_timestamps=input_timestamps,
                is_target_mask=is_target_mask,
                input_sensor_ids=input_sensor_ids,
                padding_mask=padding_mask,
                lengths=lengths,
                attn_mask=None,
                return_dict=return_distribution,
            )
            if temporal_features is not None:
                forward_kwargs["temporal_features"] = temporal_features
            model_output = self.model(**forward_kwargs)
            if return_distribution:
                preds = model_output["preds"]
                log_scale = model_output["log_scale"]
            else:
                preds = model_output
                log_scale = None

            preds, target_values, target_loss_mask, log_scale = (
                self._align_prediction_target_shapes(
                    preds,
                    target_values,
                    target_loss_mask,
                    log_scale=log_scale,
                )
            )

            loss = self._compute_loss(
                preds,
                target_values,
                target_loss_mask,
                log_scale=log_scale,
            )

        # Backward con GradScaler
        self.scaler.scale(loss).backward()

        # Clip de gradiente opcional
        if self.config.grad_clip_norm > 0.0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        self.scaler.step(self.optimizer)
        self.scaler.update()

        detached_loss = loss.detach()
        return float(detached_loss.item()) if synchronize else detached_loss

    def _apply_finetune_schedule(self, epoch: int) -> None:
        """
        Aplica congelado/descongelado del encoder según configuración.

        - Si freeze_encoder_epochs <= 0: no hace nada.
        - Si el modelo no tiene atributo `encoder`: no hace nada.
        """
        freeze_epochs = int(self.config.freeze_encoder_epochs)
        if freeze_epochs <= 0:
            return

        if not hasattr(self.model, "encoder"):
            if self._encoder_is_frozen is None:
                print(
                    "[Trainer] freeze_encoder_epochs > 0 pero el modelo no tiene 'encoder'; "
                    "se ignora el schedule."
                )
                self._encoder_is_frozen = False
            return

        should_freeze = epoch <= freeze_epochs
        if self._encoder_is_frozen is None or self._encoder_is_frozen != should_freeze:
            self._set_encoder_trainable(trainable=not should_freeze)
            self._encoder_is_frozen = should_freeze

            if should_freeze:
                print(
                    f"[Trainer] Epoch {epoch:03d}: encoder congelado "
                    f"(fase 1/{freeze_epochs} épocas)."
                )
            else:
                print(f"[Trainer] Epoch {epoch:03d}: encoder descongelado.")
                if self.config.unfreeze_lr is not None:
                    self._set_optimizer_lr(float(self.config.unfreeze_lr))
                    print(
                        f"[Trainer] LR ajustado a unfreeze_lr={self.config.unfreeze_lr:.6g} "
                        "tras descongelar encoder."
                    )

    def _set_encoder_trainable(self, trainable: bool) -> None:
        encoder = getattr(self.model, "encoder")
        for p in encoder.parameters():
            p.requires_grad = trainable

    def _set_optimizer_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        # Mantener consistencia con schedulers que usan base_lrs.
        if self.scheduler is not None and hasattr(self.scheduler, "base_lrs"):
            self.scheduler.base_lrs = [lr for _ in self.scheduler.base_lrs]

    @torch.inference_mode()
    def evaluate_on_loader(
        self, loader: DataLoader, prefix: str = "test_"
    ) -> Dict[str, float]:
        """
        Evalúa el modelo en un DataLoader arbitrario (test, val, etc.).

        Parameters
        ----------
        loader : DataLoader
            DataLoader sobre el cual evaluar.
        prefix : str
            Prefijo para las llaves de métricas (e.g. "test_", "val_").

        Returns
        -------
        metrics : dict
            Diccionario con métricas prefijadas.
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_target_masks = []
        all_log_scales = []

        for batch in loader:
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            input_timestamps = batch["input_timestamps"].to(self.device, non_blocking=True)
            is_target_mask = batch["is_target_mask"].to(self.device, non_blocking=True)
            input_sensor_ids = batch.get("input_sensor_ids", None)
            if input_sensor_ids is not None:
                input_sensor_ids = input_sensor_ids.to(self.device, non_blocking=True)
            target_values = batch["target_values"].to(self.device, non_blocking=True)
            target_loss_mask = batch.get("target_loss_mask", None)
            if target_loss_mask is not None:
                target_loss_mask = target_loss_mask.to(self.device, non_blocking=True)
            padding_mask = batch.get("padding_mask", None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device, non_blocking=True)
            lengths = batch.get("lengths", None)
            if lengths is not None:
                lengths = lengths.to(self.device, non_blocking=True)
            temporal_features = batch.get("temporal_features", None)
            if temporal_features is not None:
                temporal_features = temporal_features.to(
                    self.device, non_blocking=True
                )

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                return_distribution = (
                    str(getattr(getattr(self.model, "config", None), "prediction_head", "point"))
                    .lower()
                    == "gaussian"
                )
                forward_kwargs = dict(
                    input_values=input_values,
                    input_timestamps=input_timestamps,
                    is_target_mask=is_target_mask,
                    input_sensor_ids=input_sensor_ids,
                    padding_mask=padding_mask,
                    lengths=lengths,
                    attn_mask=None,
                    return_dict=return_distribution,
                )
                if temporal_features is not None:
                    forward_kwargs["temporal_features"] = temporal_features
                model_output = self.model(**forward_kwargs)
                if return_distribution:
                    preds = model_output["preds"]
                    log_scale = model_output["log_scale"]
                else:
                    preds = model_output
                    log_scale = None

            preds, target_values, target_loss_mask, log_scale = (
                self._align_prediction_target_shapes(
                    preds,
                    target_values,
                    target_loss_mask,
                    log_scale=log_scale,
                )
            )

            all_preds.append(preds.detach().cpu())
            all_targets.append(target_values.detach().cpu())
            if log_scale is not None:
                all_log_scales.append(log_scale.detach().cpu())
            if target_loss_mask is not None:
                all_target_masks.append(target_loss_mask.detach().cpu())

        if not all_preds:
            return {}

        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        target_mask_cat = (
            torch.cat(all_target_masks, dim=0) if all_target_masks else None
        )

        log_scale_cat = torch.cat(all_log_scales, dim=0) if all_log_scales else None
        val_loss = self._compute_loss(
            preds_cat,
            targets_cat,
            target_mask_cat,
            log_scale=log_scale_cat,
        ).item()

        if target_mask_cat is not None:
            valid = target_mask_cat > 0.0
            if torch.any(valid):
                preds_for_metrics = preds_cat[valid].view(-1, 1)
                targets_for_metrics = targets_cat[valid].view(-1, 1)
            else:
                raise ValueError(
                    "El loader de evaluación no contiene ningún target válido; "
                    "no se pueden calcular métricas ni seleccionar checkpoint."
                )
        else:
            preds_for_metrics = preds_cat
            targets_for_metrics = targets_cat

        metrics = compute_regression_metrics(
            preds_for_metrics, targets_for_metrics, prefix=prefix
        )
        metrics.update(
            compute_structured_regression_metrics(
                preds_cat, targets_cat, target_mask_cat, prefix=prefix
            )
        )
        if log_scale_cat is not None:
            metrics.update(
                compute_gaussian_metrics(
                    preds_cat,
                    targets_cat,
                    log_scale_cat,
                    target_mask_cat,
                    prefix=prefix,
                )
            )
        metrics[f"{prefix}loss"] = val_loss
        return metrics

    @torch.inference_mode()
    def _evaluate(self, epoch: int) -> Dict[str, float]:
        self.model.eval()

        all_preds = []
        all_targets = []
        all_target_masks = []
        all_log_scales = []

        for batch in self.val_loader:
            input_values = batch["input_values"].to(self.device, non_blocking=True)
            input_timestamps = batch["input_timestamps"].to(self.device, non_blocking=True)
            is_target_mask = batch["is_target_mask"].to(self.device, non_blocking=True)
            input_sensor_ids = batch.get("input_sensor_ids", None)
            if input_sensor_ids is not None:
                input_sensor_ids = input_sensor_ids.to(self.device, non_blocking=True)
            target_values = batch["target_values"].to(self.device, non_blocking=True)
            target_loss_mask = batch.get("target_loss_mask", None)
            if target_loss_mask is not None:
                target_loss_mask = target_loss_mask.to(self.device, non_blocking=True)
            padding_mask = batch.get("padding_mask", None)
            if padding_mask is not None:
                padding_mask = padding_mask.to(self.device, non_blocking=True)
            lengths = batch.get("lengths", None)
            if lengths is not None:
                lengths = lengths.to(self.device, non_blocking=True)
            temporal_features = batch.get("temporal_features", None)
            if temporal_features is not None:
                temporal_features = temporal_features.to(
                    self.device, non_blocking=True
                )

            with torch.amp.autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                return_distribution = (
                    str(getattr(getattr(self.model, "config", None), "prediction_head", "point"))
                    .lower()
                    == "gaussian"
                )
                forward_kwargs = dict(
                    input_values=input_values,
                    input_timestamps=input_timestamps,
                    is_target_mask=is_target_mask,
                    input_sensor_ids=input_sensor_ids,
                    padding_mask=padding_mask,
                    lengths=lengths,
                    attn_mask=None,
                    return_dict=return_distribution,
                )
                if temporal_features is not None:
                    forward_kwargs["temporal_features"] = temporal_features
                model_output = self.model(**forward_kwargs)
                if return_distribution:
                    preds = model_output["preds"]
                    log_scale = model_output["log_scale"]
                else:
                    preds = model_output
                    log_scale = None

            preds, target_values, target_loss_mask, log_scale = (
                self._align_prediction_target_shapes(
                    preds,
                    target_values,
                    target_loss_mask,
                    log_scale=log_scale,
                )
            )

            all_preds.append(preds.detach().cpu())
            all_targets.append(target_values.detach().cpu())
            if log_scale is not None:
                all_log_scales.append(log_scale.detach().cpu())
            if target_loss_mask is not None:
                all_target_masks.append(target_loss_mask.detach().cpu())

        if not all_preds:
            return {}

        preds_cat = torch.cat(all_preds, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        target_mask_cat = torch.cat(all_target_masks, dim=0) if all_target_masks else None

        # Pérdida de validación
        log_scale_cat = torch.cat(all_log_scales, dim=0) if all_log_scales else None
        val_loss = self._compute_loss(
            preds_cat,
            targets_cat,
            target_mask_cat,
            log_scale=log_scale_cat,
        ).item()

        # Métricas adicionales
        if target_mask_cat is not None:
            valid = target_mask_cat > 0.0
            if torch.any(valid):
                preds_for_metrics = preds_cat[valid].view(-1, 1)
                targets_for_metrics = targets_cat[valid].view(-1, 1)
            else:
                raise ValueError(
                    "El loader de validación no contiene ningún target válido; "
                    "no se pueden calcular métricas ni seleccionar checkpoint."
                )
        else:
            preds_for_metrics = preds_cat
            targets_for_metrics = targets_cat

        metrics = compute_regression_metrics(preds_for_metrics, targets_for_metrics, prefix="val_")
        metrics.update(
            compute_structured_regression_metrics(
                preds_cat, targets_cat, target_mask_cat, prefix="val_"
            )
        )
        if log_scale_cat is not None:
            metrics.update(
                compute_gaussian_metrics(
                    preds_cat,
                    targets_cat,
                    log_scale_cat,
                    target_mask_cat,
                    prefix="val_",
                )
            )
        metrics["val_loss"] = val_loss

        return metrics

    @staticmethod
    def _align_prediction_target_shapes(
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_loss_mask: Optional[torch.Tensor],
        *,
        log_scale: Optional[torch.Tensor] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Alinea el horizonte unitario sin permitir broadcasting ambiguo.

        Los heads históricos devuelven ``[B, D]`` cuando hay una sola query,
        mientras el dataset físico conserva ``[B, 1, D]``. PyTorch podría
        broadcast esas formas a ``[B, B, D]`` y mezclar ejemplos. Adaptamos
        exclusivamente ese eje unitario y rechazamos cualquier otra diferencia.
        """

        if preds.shape != targets.shape:
            if (
                targets.ndim == preds.ndim + 1
                and targets.shape[1] == 1
                and preds.shape[0] == targets.shape[0]
                and preds.shape[1:] == targets.shape[2:]
            ):
                preds = preds.unsqueeze(1)
                if log_scale is not None:
                    log_scale = log_scale.unsqueeze(1)
            elif (
                preds.ndim == targets.ndim + 1
                and preds.shape[1] == 1
                and preds.shape[0] == targets.shape[0]
                and preds.shape[2:] == targets.shape[1:]
            ):
                preds = preds.squeeze(1)
                if log_scale is not None:
                    log_scale = log_scale.squeeze(1)
            else:
                raise ValueError(
                    "La shape de la predicción no coincide con el target y no "
                    "corresponde a un horizonte unitario: "
                    f"preds={tuple(preds.shape)}, targets={tuple(targets.shape)}."
                )

        if preds.shape != targets.shape:
            raise ValueError(
                f"preds={tuple(preds.shape)} y targets={tuple(targets.shape)} "
                "deben coincidir exactamente."
            )
        if log_scale is not None and log_scale.shape != preds.shape:
            raise ValueError(
                f"log_scale={tuple(log_scale.shape)} debe coincidir con "
                f"preds={tuple(preds.shape)}."
            )
        if target_loss_mask is not None and target_loss_mask.shape != targets.shape:
            raise ValueError(
                f"target_loss_mask={tuple(target_loss_mask.shape)} debe coincidir "
                f"con targets={tuple(targets.shape)}."
            )
        return preds, targets, target_loss_mask, log_scale

    def _compute_loss(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        target_loss_mask: Optional[torch.Tensor],
        log_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calcula pérdida normal o enmascarada (masked loss) si se provee mask.
        """
        preds, targets, target_loss_mask, log_scale = (
            self._align_prediction_target_shapes(
                preds,
                targets,
                target_loss_mask,
                log_scale=log_scale,
            )
        )
        if log_scale is not None:
            inverse_variance = torch.exp(-2.0 * log_scale)
            per = (
                log_scale
                + 0.5 * (targets - preds).pow(2) * inverse_variance
                + 0.5 * math.log(2.0 * math.pi)
            )
            if target_loss_mask is None:
                return per.mean()
            mask = target_loss_mask.to(device=per.device, dtype=per.dtype)
            return (per * mask).sum() / mask.sum().clamp_min(1.0)

        if target_loss_mask is None:
            return self.loss_fn(preds, targets)

        if isinstance(self.loss_fn, DILATELoss):
            return self.loss_fn(preds, targets, target_mask=target_loss_mask)

        mask = target_loss_mask.float()
        mask_sum = mask.sum().clamp(min=1.0)

        if isinstance(self.loss_fn, nn.MSELoss):
            per = (preds - targets) ** 2
        elif isinstance(self.loss_fn, nn.L1Loss):
            per = torch.abs(preds - targets)
        elif isinstance(self.loss_fn, nn.SmoothL1Loss):
            per = F.smooth_l1_loss(preds, targets, reduction="none", beta=self.loss_fn.beta)
        else:
            # Fallback seguro: si la pérdida no soporta masking explícito,
            # usamos pérdida estándar para no romper compatibilidad.
            return self.loss_fn(preds, targets)

        return (per * mask).sum() / mask_sum

    # ------------------------------------------------------------------
    # Utilidades varias
    # ------------------------------------------------------------------
    def _maybe_save_best(self, epoch: int, current_value: float) -> bool:
        """
        Si checkpoint_dir está definido, guarda el mejor modelo en función
        de la métrica `save_best_on`.
        """
        threshold = float(self.config.early_stopping_min_delta)

        # Minimizar la métrica (asumimos que menor es mejor)
        improved = (
            self.best_metric_value is None
            or current_value < (self.best_metric_value - threshold)
        )

        if improved:
            self.best_metric_value = current_value
            self.best_epoch = epoch

            if self.config.restore_best_weights:
                self._best_model_state_in_memory = {
                    k: v.detach().cpu().clone()
                    for k, v in self.model.state_dict().items()
                }

            if self.config.checkpoint_dir is None:
                return True

            ckpt_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
            state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config_training": asdict(self.config),
            }
            torch.save(state, ckpt_path)
            print(
                f"  -> Nuevo mejor modelo en epoch {epoch} "
                f"({self.config.save_best_on} = {current_value:.6f}), "
                f"guardado en {ckpt_path}"
            )

        return improved

    @staticmethod
    def _append_history(history: Dict[str, list], key: str, value: float) -> None:
        if key not in history:
            history[key] = []
        history[key].append(float(value))

    @staticmethod
    def _log_epoch_summary(
        epoch: int,
        train_loss: float,
        metrics_val: Dict[str, float],
        elapsed: float,
    ) -> None:
        msg = f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}"
        if metrics_val:
            summary_keys = (
                "val_loss",
                "val_rmse",
                "val_mae",
                "val_nll",
                "val_crps",
            )
            metrics_str = ", ".join(
                f"{key}={metrics_val[key]:.6f}"
                for key in summary_keys
                if key in metrics_val
            )
            msg += f" | {metrics_str}"
        msg += f" | tiempo={elapsed:.1f}s"
        print(msg)


# ----------------------------------------------------------------------
# Helper de alto nivel
# ----------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Helper de conveniencia para entrenar un modelo con Trainer
    usando la configuración por defecto (o la que pases).

    Parameters
    ----------
    model:
        Modelo a entrenar.
    train_loader:
        DataLoader de entrenamiento.
    val_loader:
        DataLoader de validación (opcional).
    config:
        TrainingConfig. Si es None, se usa la configuración por defecto.

    Returns
    -------
    history:
        Diccionario con la historia de métricas por época.
    """
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )
    return trainer.fit()
