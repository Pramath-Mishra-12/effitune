import timm
import torch
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F


class EfficientNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        drop_rate: float = 0.2,
        drop_path_rate: Optional[float] = None,
    ):
        super().__init__()
        create_kwargs = {
            "pretrained": pretrained,
            "drop_rate": drop_rate,
        }
        if drop_path_rate is not None:
            create_kwargs["drop_path_rate"] = drop_path_rate

        self.backbone = timm.create_model(
            backbone_name,
            features_only=False,
            **create_kwargs,
        )

        # Remove internal classifier if present; we use our own head.
        if hasattr(self.backbone, "classifier"):
            self.backbone.classifier = nn.Identity()

        in_features = self.backbone.num_features  # e.g., 1280 for efficientnet_b0
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(drop_rate),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        x = self.backbone.forward_features(x)  # returns (B, C, H, W)
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
            x = torch.flatten(x, 1)  # (B, C)
        else:
            x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def set_batchnorm_eval(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False