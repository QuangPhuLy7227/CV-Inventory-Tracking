import os
import time
import urllib.request
from pathlib import Path

def main(
    urls_txt: str,
    out_dir: str = "datasets/inventory_v1/raw_images",
    prefix: str = "web",
    delay_sec: float = 0.2
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(urls_txt, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

    n_ok = 0
    for i, url in enumerate(urls):
        try:
            ext = ".jpg"
            # try to infer extension
            for e in [".jpg",".jpeg",".png",".webp"]:
                if e in url.lower():
                    ext = e
                    break

            dst = out / f"{prefix}_{i:05d}{ext}"
            urllib.request.urlretrieve(url, dst)
            n_ok += 1
            time.sleep(delay_sec)
        except Exception as e:
            print("Failed:", url, "|", e)

    print(f"Downloaded {n_ok}/{len(urls)} images into {out_dir}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python training/download_images.py <urls.txt>")
        raise SystemExit(1)
    main(sys.argv[1])