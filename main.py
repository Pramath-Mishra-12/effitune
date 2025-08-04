import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

from models.efficientnet_model import EfficientNetClassifier
from data.stanford_dogs import ImageFolderDataset
from trainer.trainer import Trainer
from utils.logger import setup_logger


def load_config(path: Path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="EfficientNet fine-tuning with advanced features")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("main", output_dir / "run.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_dataset = ImageFolderDataset(Path(cfg["data_root"]), split="train", img_size=cfg["img_size"], augment=True)
    val_dataset = ImageFolderDataset(Path(cfg["data_root"]), split="val", img_size=cfg["img_size"], augment=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = EfficientNetClassifier(
        num_classes=cfg["num_classes"],
        backbone_name="efficientnet_b0",
        pretrained=cfg.get("pretrained", True),
        drop_rate=cfg.get("drop_rate", 0.2),
        drop_path_rate=cfg.get("drop_path_rate", 0.2),
    )

    if cfg.get("freeze_backbone", False):
        model.freeze_backbone()
        model.set_batchnorm_eval()

    # Layer-wise LR
    base_lr = cfg["lr"]
    backbone_lr = base_lr * cfg.get("backbone_lr_mult", 0.1)
    head_lr = base_lr

    def is_norm_or_bias(name):
        return any(k in name.lower() for k in ["bn", "batchnorm", "bias"])

    param_groups = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad and not is_norm_or_bias(n)],
            "weight_decay": cfg["weight_decay"],
            "lr": backbone_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad and is_norm_or_bias(n)],
            "weight_decay": 0.0,
            "lr": backbone_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad and not is_norm_or_bias(n)],
            "weight_decay": cfg["weight_decay"],
            "lr": head_lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n and p.requires_grad and is_norm_or_bias(n)],
            "weight_decay": 0.0,
            "lr": head_lr,
        },
    ]
    optimizer = optim.AdamW(param_groups)

    steps_per_epoch = len(train_loader)
    total_steps = cfg["epochs"] * steps_per_epoch
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cfg["lr"],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=25.0,
        final_div_factor=1e4,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        save_dir=output_dir,
        use_amp=cfg.get("use_amp", True),
        accumulation_steps=cfg.get("accumulation_steps", 1),
        mixup_alpha=cfg.get("mixup_alpha", 0.0),
        cutmix_alpha=cfg.get("cutmix_alpha", 0.0),
        ema_decay=cfg.get("ema_decay", 0.999),
        early_stopping_patience=5,
    )

    trainer.fit(num_epochs=cfg["epochs"])


if __name__ == "__main__":
    main()