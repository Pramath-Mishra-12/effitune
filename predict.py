import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path
import argparse

from models.efficientnet_model import EfficientNetClassifier


def load_image(path: Path, img_size: int):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # batch dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=120)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetClassifier(num_classes=args.num_classes, backbone_name="efficientnet_b0", pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    img_tensor = load_image(args.image, args.img_size).to(device)
    with torch.no_grad():
        logits = model(img_tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)
        topk = torch.topk(probs, k=5)
    print("Top predictions (index, prob):")
    for idx, p in zip(topk.indices[0].cpu().tolist(), topk.values[0].cpu().tolist()):
        print(f"{idx}: {p:.4f}")


if __name__ == "__main__":
    main()