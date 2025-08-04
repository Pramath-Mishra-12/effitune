import optuna
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.efficientnet_model import EfficientNetClassifier
from data.stanford_dogs import ImageFolderDataset
from trainer.trainer import Trainer


def objective(trial):
    lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
    backbone_lr_mult = trial.suggest_float("backbone_lr_mult", 0.05, 1.0)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    drop_path_rate = trial.suggest_float("drop_path_rate", 0.0, 0.3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.4)
    cutmix_alpha = trial.suggest_float("cutmix_alpha", 0.0, 0.4)
    label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)

    data_root = Path("data/stanford_dogs_prepared")
    img_size = 224
    batch_size = 64

    train_dataset = ImageFolderDataset(data_root, split="train", img_size=img_size, augment=True)
    val_dataset = ImageFolderDataset(data_root, split="val", img_size=img_size, augment=False)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetClassifier(
        num_classes=120,
        backbone_name="efficientnet_b0",
        pretrained=True,
        drop_rate=dropout,
        drop_path_rate=drop_path_rate,
    )

    backbone_lr = lr * backbone_lr_mult
    head_lr = lr

    def is_norm_or_bias(name):
        return any(k in name.lower() for k in ["bn", "batchnorm", "bias"])

    backbone_non_decay = []
    backbone_decay = []
    head_non_decay = []
    head_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            if is_norm_or_bias(name):
                backbone_non_decay.append(param)
            else:
                backbone_decay.append(param)
        elif name.startswith("classifier."):
            if is_norm_or_bias(name):
                head_non_decay.append(param)
            else:
                head_decay.append(param)
        else:
            # fallback: put in backbone_decay to avoid dropping anything unknowingly
            backbone_decay.append(param)

    param_groups = [
        {"params": backbone_decay, "weight_decay": weight_decay, "lr": backbone_lr},
        {"params": backbone_non_decay, "weight_decay": 0.0, "lr": backbone_lr},
        {"params": head_decay, "weight_decay": weight_decay, "lr": head_lr},
        {"params": head_non_decay, "weight_decay": 0.0, "lr": head_lr},
    ]

    optimizer = optim.AdamW(param_groups)
    total_steps = 10 * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        save_dir=Path("sweep_runs") / str(trial.number),
        use_amp=True,
        accumulation_steps=1,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        ema_decay=0.999,
        early_stopping_patience=3,
    )

    trainer.fit(num_epochs=10)
    return trainer.best_val_acc


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)
    print("Best trial:", study.best_trial.params)