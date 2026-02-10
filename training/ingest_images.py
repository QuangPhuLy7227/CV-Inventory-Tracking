import os
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main(
    src_dir: str,
    out_dir: str = "datasets/inventory_v1/raw_images",
    prefix: str = "src"
):
    src = Path(src_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if not src.exists():
        raise RuntimeError(f"Source dir not found: {src}")

    n = 0
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            dst = out / f"{prefix}_{p.stem}{p.suffix.lower()}"
            # avoid overwrite
            i = 1
            while dst.exists():
                dst = out / f"{prefix}_{p.stem}_{i}{p.suffix.lower()}"
                i += 1
            shutil.copy2(p, dst)
            n += 1

    print(f"Copied {n} images to {out_dir}")

if __name__ == "__main__":
    # Example:
    # python training/ingest_images.py "/path/to/my/photos"
    import sys
    if len(sys.argv) < 2:
        print("Usage: python training/ingest_images.py <src_dir>")
        raise SystemExit(1)
    main(sys.argv[1])