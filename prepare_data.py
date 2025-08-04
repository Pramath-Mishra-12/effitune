import tensorflow_datasets as tfds
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import os
from PIL import Image
import numpy as np


def save_split(split_name, ds, output_root: Path, img_size: int):
    split_root = output_root / split_name
    for image, label in tqdm(tfds.as_numpy(ds), desc=f"Saving {split_name}"):
        breed = tfds.features.ClassLabel(num_classes=120).int2str(label).decode() if isinstance(label, bytes) else None
        # In TFDS, stanford_dogs label is integer; need label_info to get string
        # We'll retrieve label string from ds_info outside and pass mapping
        pass  # placeholder; we will reimplement below


def prepare_stanford_dogs(output_root: Path, img_size: int = 224):
    (ds_train, ds_test), ds_info = tfds.load(
        "stanford_dogs",
        split=["train", "test"],
        with_info=True,
        as_supervised=True,
    )
    label_info = ds_info.features["label"]
    def process_and_dump(ds, split_name):
        for image, label in tqdm(tfds.as_numpy(ds), desc=f"Processing {split_name}"):
            breed_full = label_info.int2str(label)  # string like 'n02085620-Chihuahua'
            breed = breed_full.split("-")[1] if "-" in breed_full else breed_full
            out_dir = output_root / split_name / breed_full
            out_dir.mkdir(parents=True, exist_ok=True)
            # image is uint8 array
            img = Image.fromarray(image.astype("uint8"))
            img = img.resize((img_size, img_size))
            # use a unique filename
            fname = f"{label}_{np.random.randint(1e9)}.jpg"
            img.save(out_dir / fname)

    output_root.mkdir(parents=True, exist_ok=True)
    process_and_dump(ds_train, "train")
    process_and_dump(ds_test, "val")  # treat test as val for compatibility


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Prepare Stanford Dogs into folder layout.")
    parser.add_argument("--output", type=Path, default=Path("data/stanford_dogs_prepared"))
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()
    prepare_stanford_dogs(args.output, img_size=args.img_size)