"""
Generate print-ready QR codes optimized for long-distance scanning.

Usage:
    python cv/tools/gen_qr.py --ids PRUSA-01 PRUSA-02 PRUSA-03
    python cv/tools/gen_qr.py --ids PRUSA-01 --size 10  # 10cm square

Tips for 1-meter scanning:
  - Print at 8–10 cm square minimum
  - Use glossy paper (better contrast)
  - Keep payload short — fewer modules = larger cells at same size
"""
import argparse
import json
import os
import qrcode
from qrcode.constants import ERROR_CORRECT_L
from PIL import Image, ImageDraw, ImageFont


def make_qr(spool_id: str, size_cm: float = 10.0, dpi: int = 300) -> Image.Image:
    payload = json.dumps({"id": spool_id}, separators=(",", ":"))

    qr = qrcode.QRCode(
        version=None,           # auto-select smallest version that fits
        error_correction=ERROR_CORRECT_L,  # L = 7% redundancy → largest cells
        box_size=10,
        border=4,
    )
    qr.add_data(payload)
    qr.make(fit=True)

    # Calculate pixel size from cm + dpi (1 inch = 2.54 cm)
    px = int((size_cm / 2.54) * dpi)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    img = img.resize((px, px), Image.NEAREST)

    # Add label below
    label_h = max(40, int(px * 0.08))
    labeled = Image.new("RGB", (px, px + label_h), "white")
    labeled.paste(img, (0, 0))
    draw = ImageDraw.Draw(labeled)
    try:
        font = ImageFont.truetype("arial.ttf", size=label_h - 8)
    except Exception:
        font = ImageFont.load_default()
    draw.text((px // 2, px + label_h // 2), spool_id, fill="black", anchor="mm", font=font)

    return labeled


def main():
    parser = argparse.ArgumentParser(description="Generate optimized QR labels for spools")
    parser.add_argument("--ids", nargs="+", required=True, help="Spool IDs, e.g. PRUSA-01 PRUSA-02")
    parser.add_argument("--size", type=float, default=10.0, help="Print size in cm (default: 10)")
    parser.add_argument("--dpi", type=int, default=300, help="Print DPI (default: 300)")
    parser.add_argument("--out", default="qr_labels", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    for spool_id in args.ids:
        img = make_qr(spool_id, size_cm=args.size, dpi=args.dpi)
        path = os.path.join(args.out, f"{spool_id}.png")
        img.save(path, dpi=(args.dpi, args.dpi))
        payload = json.dumps({"id": spool_id}, separators=(",", ":"))
        modules = img.size[0]
        print(f"  {spool_id} -> {path}  ({args.size}cm, payload='{payload}')")

    print(f"\nPrint at {args.size}cm x {args.size}cm ({args.dpi} DPI) for reliable 1m scanning.")
    print("Tip: if still marginal, run again with --size 12")


if __name__ == "__main__":
    main()
