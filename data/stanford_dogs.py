from pathlib import Path
from typing import Optional
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class ImageFolderDataset(Dataset):
    def __init__(self, root: Path, split: str, img_size: int, augment: bool = False):
        self.root = root / split
        self.img_size = img_size
        self.augment = augment
        self.samples = []
        self.class_to_idx = {}
        self._build_index()
        self.transform = self._build_transform()

    def _build_index(self):
        classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for cls in classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.glob("*"):
                if img_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def _build_transform(self):
        if self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.RandomRotation(15)], p=0.5),
                transforms.RandomApply([transforms.ColorJitter(contrast=0.1)], p=0.5),
                transforms.ToTensor(),  # <-- convert to tensor before tensor-only ops
                transforms.RandomErasing(p=0.2),
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label