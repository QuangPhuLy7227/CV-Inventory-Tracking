import os
import shutil
from sklearn.model_selection import train_test_split

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def main(
    base="datasets/inventory_v1",
    images_all="datasets/inventory_v1/images/all",
    labels_all="datasets/inventory_v1/labels/all",
    test_size=0.10,
    val_size=0.15,
    seed=42
):
    # target dirs
    for split in ["train", "val", "test"]:
        ensure_dir(os.path.join(base, "images", split))
        ensure_dir(os.path.join(base, "labels", split))

    imgs = sorted([f for f in os.listdir(images_all) if f.lower().endswith((".jpg",".jpeg",".png"))])
    if not imgs:
        raise RuntimeError(f"No images found in {images_all}")
    
    # keep only those with matching label files
    pairs = []
    for img in imgs:
        stem = os.path.splitext(img)[0]
        lab = stem + ".txt"
        if os.path.exists(os.path.join(labels_all, lab)):
            pairs.append((img, lab))

    if not pairs:
        raise RuntimeError("No image/label pairs found. Make sure labels exist in labels/all.")
    
    X = [p[0] for p in pairs]
    y = [p[1] for p in pairs]

    # split
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    # val is fraction of trainval
    val_frac = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=seed
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    for split, (xs, ys) in splits.items():
        for img, lab in zip(xs, ys):
            shutil.copy2(os.path.join(images_all, img), os.path.join(base, "images", split, img))
            shutil.copy2(os.path.join(labels_all, lab), os.path.join(base, "labels", split, lab))

    print("Split complete:")
    for split in splits:
        print(split, "=", len(splits[split][0]))

if __name__ == "__main__":
    main()