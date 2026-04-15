import os
import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def main(
    base="datasets/inventory_v1",
    images_all="datasets/inventory_v1/images/all",
    labels_all="datasets/inventory_v1/labels/all",
    test_ratio=0.10,
    val_ratio=0.15,
    seed=42,
):
    random.seed(seed)
    base = Path(base)
    images_all = Path(images_all)
    labels_all = Path(labels_all)

    # Target dirs
    for split in ["train", "val", "test"]:
        ensure_dir(base / "images" / split)
        ensure_dir(base / "labels" / split)

    # Find image/label pairs
    pairs = []
    for img_path in images_all.iterdir():
        if not img_path.is_file() or img_path.suffix.lower() not in IMG_EXTS:
            continue
        label_path = labels_all / (img_path.stem + ".txt")
        if label_path.exists():
            pairs.append((img_path, label_path))

    if not pairs:
        raise RuntimeError("No image/label pairs found. Did you label and save YOLO .txt files?")

    random.shuffle(pairs)

    n = len(pairs)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_pairs = pairs[:n_test]
    val_pairs = pairs[n_test:n_test + n_val]
    train_pairs = pairs[n_test + n_val:]

    splits = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    # Copy
    for split, spairs in splits.items():
        for img_path, label_path in spairs:
            shutil.copy2(img_path, base / "images" / split / img_path.name)
            shutil.copy2(label_path, base / "labels" / split / label_path.name)

    print("Split complete:")
    for split, spairs in splits.items():
        print(f"  {split}: {len(spairs)}")

if __name__ == "__main__":
    main()