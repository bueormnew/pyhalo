import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ImportError:
    # Fallback simple si tqdm no está instalado
    def tqdm(iterable, *args, **kwargs):
        return iterable


class Trainer:
    """
    Entrenador profesional para HALO-S con soporte completo.
    Incluye mixed precision (AMP), gradient accumulation, gradient clipping,
    checkpointing automático y tracking de métricas.
    
    Mantiene compatibilidad con la interfaz original:
        Trainer(model, learning_rate, device)
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        device: str = None,
        mixed_precision: bool = False,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = None,
        log_every: int = 10
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = checkpoint_dir
        self.log_every = log_every

        # Mixed Precision - GradScaler para CUDA AMP
        self.scaler = torch.amp.GradScaler('cuda') if mixed_precision else None

        # Estado de entrenamiento
        self.global_step = 0
        self.current_epoch = 0
        self.training_history: list = []

    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Ejecuta un paso de entrenamiento individual con soporte AMP.
        
        Args:
            x: Tensor de entrada (batch, seq_len)
            y: Tensor de targets (batch, seq_len)
            
        Returns:
            Loss escalar del paso (antes de dividir por accumulation_steps)
        """
        x, y = x.to(self.device), y.to(self.device)

        # Determinar dispositivo para autocast
        amp_device = 'cuda' if self.device == 'cuda' else 'cpu'
        amp_dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16

        if self.mixed_precision:
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                logits, loss = self.model(x, targets=y)
        else:
            logits, loss = self.model(x, targets=y)

        # Dividir loss por pasos de acumulación para promediar correctamente
        scaled_loss = loss / self.gradient_accumulation_steps

        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        return loss.item()

    @torch.no_grad()
    def validation_step(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """
        Ejecuta un paso de validación sin gradientes.
        
        Args:
            x: Tensor de entrada (batch, seq_len)
            y: Tensor de targets (batch, seq_len)
            
        Returns:
            Loss escalar del paso
        """
        x, y = x.to(self.device), y.to(self.device)

        # Usar autocast también en validación para consistencia
        amp_device = 'cuda' if self.device == 'cuda' else 'cpu'
        amp_dtype = torch.float16 if self.device == 'cuda' else torch.bfloat16

        if self.mixed_precision:
            with torch.amp.autocast(device_type=amp_device, dtype=amp_dtype):
                logits, loss = self.model(x, targets=y)
        else:
            logits, loss = self.model(x, targets=y)

        return loss.item()

    def fit(
        self,
        dataset,
        epochs: int = 1,
        batch_size: int = 4,
        validation_dataset=None,
        save_every: int = None
    ) -> list:
        """
        Bucle de entrenamiento completo con gradient accumulation y AMP.
        
        Args:
            dataset: Dataset de entrenamiento (compatible con DataLoader)
            epochs: Número de épocas
            batch_size: Tamaño de batch
            validation_dataset: Dataset de validación opcional
            save_every: Guardar checkpoint cada N épocas (None = no guardar)
            
        Returns:
            Lista de dicts con métricas por época:
            [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.3}, ...]
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        history = []

        for epoch in range(epochs):
            self.current_epoch = epoch + 1
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            # Asegurar que gradientes estén limpios al inicio de la época
            self.optimizer.zero_grad()

            pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch}/{epochs}")
            for step, (x, y) in enumerate(pbar):
                # Paso de entrenamiento (acumula gradientes)
                loss_value = self.train_step(x, y)
                total_loss += loss_value
                num_batches += 1

                # Aplicar gradientes cada gradient_accumulation_steps pasos
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping antes del step del optimizer
                    if self.mixed_precision and self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.max_grad_norm
                    )

                    # Step del optimizer (con o sin scaler)
                    if self.mixed_precision and self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.global_step += 1

                # Actualizar barra de progreso
                if hasattr(pbar, "set_postfix"):
                    pbar.set_postfix({"loss": loss_value})

            # Manejar gradientes residuales si el número de batches no es divisible
            remaining = len(dataloader) % self.gradient_accumulation_steps
            if remaining != 0:
                if self.mixed_precision and self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.max_grad_norm
                )

                if self.mixed_precision and self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.global_step += 1

            # Calcular loss promedio de la época
            avg_train_loss = total_loss / num_batches if num_batches > 0 else 0.0

            # Métricas de la época
            epoch_metrics = {
                "epoch": self.current_epoch,
                "train_loss": avg_train_loss,
            }

            print(f"Epoch {self.current_epoch} | Train Loss: {avg_train_loss:.4f}")

            # Evaluación en dataset de validación
            if validation_dataset is not None:
                val_loss = self.evaluate(validation_dataset, batch_size)
                epoch_metrics["val_loss"] = val_loss
            else:
                epoch_metrics["val_loss"] = None

            history.append(epoch_metrics)
            self.training_history.append(epoch_metrics)

            # Auto-checkpointing
            if save_every is not None and self.current_epoch % save_every == 0:
                self.save_checkpoint(metrics=epoch_metrics)

        return history

    @torch.no_grad()
    def evaluate(self, dataset, batch_size: int = 4) -> float:
        """
        Evaluación completa sobre un dataset.
        
        Args:
            dataset: Dataset de evaluación
            batch_size: Tamaño de batch
            
        Returns:
            Loss promedio (float)
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=batch_size)
        total_loss = 0.0
        num_batches = 0

        for x, y in dataloader:
            loss_value = self.validation_step(x, y)
            total_loss += loss_value
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Validation Loss: {avg_loss:.4f}")
        return avg_loss

    def save_checkpoint(self, path: str = None, metrics: dict = None) -> str:
        """
        Guarda checkpoint completo del estado de entrenamiento.
        
        Args:
            path: Ruta del archivo. Si None, usa checkpoint_dir/checkpoint_epoch{N}.pt
            metrics: Métricas adicionales para incluir en el checkpoint
            
        Returns:
            Ruta del archivo guardado
        """
        import os
        from halo import __version__ as halo_version

        if path is None:
            if self.checkpoint_dir is None:
                self.checkpoint_dir = "."
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            path = os.path.join(
                self.checkpoint_dir,
                f"checkpoint_epoch{self.current_epoch}.pt"
            )

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": metrics,
            "training_history": self.training_history,
            "halo_version": halo_version,
        }

        # Incluir config si el modelo la tiene
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'to_dict'):
            checkpoint["config"] = self.model.config.to_dict()

        torch.save(checkpoint, path)
        print(f"Checkpoint guardado en: {path}")
        return path

    def load_checkpoint(self, path: str) -> dict:
        """
        Restaura estado completo desde un checkpoint.
        
        Args:
            path: Ruta al archivo de checkpoint
            
        Returns:
            Dict con metadatos del checkpoint (epoch, global_step, metrics)
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler is not None and checkpoint.get("scaler_state_dict") is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)
        self.training_history = checkpoint.get("training_history", [])

        print(f"Checkpoint cargado desde: {path} (epoch {self.current_epoch}, step {self.global_step})")

        return {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "metrics": checkpoint.get("metrics"),
            "config": checkpoint.get("config"),
        }
