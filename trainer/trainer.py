import os
import torch
from torch import amp
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from typing import Optional
from utils.logger import setup_logger
from utils.augmentations import mixup_data, cutmix_data
from utils.ema import ModelEMA


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.OneCycleLR | None,
        save_dir: Path,
        use_amp: bool = True,
        accumulation_steps: int = 1,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        ema_decay: float = 0.999,
        early_stopping_patience: Optional[int] = 5,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.amp = use_amp and (device.type == "cuda" or device.type == "cpu")
        self.scaler = amp.GradScaler(device=device.type, enabled=self.amp)
        self.accumulation_steps = accumulation_steps
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.logger = setup_logger("trainer", save_dir / "training.log")
        self.best_val_acc = 0.0
        self.epoch = 0
        self.early_stopping_patience = early_stopping_patience
        self.no_improve = 0
        self.ema = ModelEMA(self.model, decay=ema_decay)

    def _compute_accuracy(self, outputs, targets):
        preds = outputs.argmax(dim=1)
        correct = (preds == targets).sum().item()
        total = targets.size(0)
        return correct / total

    def _mixup_cutmix(self, images, labels):
        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            if torch.rand(1).item() < 0.5:
                return mixup_data(images, labels, self.mixup_alpha), "mixup"
            else:
                return cutmix_data(images, labels, self.cutmix_alpha), "cutmix"
        elif self.mixup_alpha > 0:
            return mixup_data(images, labels, self.mixup_alpha), "mixup"
        elif self.cutmix_alpha > 0:
            return cutmix_data(images, labels, self.cutmix_alpha), "cutmix"
        else:
            return (images, labels, labels, 1.0), None

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {self.epoch+1}")
        self.optimizer.zero_grad()
        for step, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            (mixed_images, y_a, y_b, lam), mode = self._mixup_cutmix(images, labels)

            with amp.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(mixed_images)
                if mode in ("mixup", "cutmix"):
                    loss = lam * self.criterion(outputs, y_a) + (1 - lam) * self.criterion(outputs, y_b)
                else:
                    loss = self.criterion(outputs, labels)
                loss = loss / self.accumulation_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler and not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.ema.update(self.model)

            with torch.no_grad():
                if mode in ("mixup", "cutmix"):
                    acc = lam * self._compute_accuracy(outputs, y_a) + (1 - lam) * self._compute_accuracy(outputs, y_b)
                else:
                    acc = self._compute_accuracy(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * self.accumulation_steps * batch_size
            running_acc += acc * batch_size
            total += batch_size
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "acc": f"{running_acc/total:.4f}"})

        epoch_loss = running_loss / total
        epoch_acc = running_acc / total
        self.logger.info(f"Train Epoch {self.epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate_epoch(self, use_ema: bool = True):
        self.model.eval()
        if use_ema:
            self.ema.apply_shadow(self.model)
        running_loss = 0.0
        running_acc = 0.0
        total = 0
        for images, labels in tqdm(self.val_loader, desc=f"Validate Epoch {self.epoch+1}"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            with amp.autocast(device_type=self.device.type, enabled=self.amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            acc = self._compute_accuracy(outputs, labels)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            running_acc += acc * batch_size
            total += batch_size

        if use_ema:
            self.ema.restore(self.model)

        epoch_loss = running_loss / total
        epoch_acc = running_acc / total
        self.logger.info(f"Validation Epoch {self.epoch+1}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
        return epoch_loss, epoch_acc

    def save_checkpoint(self, is_best: bool = False):
        ckpt = {
            "epoch": self.epoch + 1,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_acc": self.best_val_acc,
            "scaler_state": self.scaler.state_dict() if hasattr(self, "scaler") else None,
        }

        out_dir = Path(self.save_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint directory {out_dir}: {e}")
            raise

        out_path = out_dir / f"checkpoint_epoch_{self.epoch + 1}.pth"
        try:
            torch.save(ckpt, out_path)
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint to {out_path}: {e}")
            raise

        if is_best:
            best_path = out_dir / "best.pth"
            try:
                # atomic replace of best.pth
                tmp_path = out_dir / f".best_tmp_{os.getpid()}.pth"
                torch.save(ckpt, tmp_path)
                tmp_path.replace(best_path)
            except Exception as e:
                self.logger.error(f"Failed to update best checkpoint at {best_path}: {e}")
                raise

        self.logger.info(f"Saved checkpoint to {out_path} (best={is_best}); best_val_acc={self.best_val_acc}")

    def fit(self, num_epochs: int):
        for epoch in range(num_epochs):
            self.epoch = epoch
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate_epoch(use_ema=True)

            if self.scheduler and isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)

            improved = val_acc > self.best_val_acc
            if improved:
                self.best_val_acc = val_acc
                self.no_improve = 0
                self.save_checkpoint(is_best=True)
            else:
                self.no_improve += 1
                self.save_checkpoint(is_best=False)

            if self.early_stopping_patience and self.no_improve >= self.early_stopping_patience:
                self.logger.info(f"Early stopping after {epoch+1} epochs. Best acc: {self.best_val_acc:.4f}")
                break