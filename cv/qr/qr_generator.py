import os
import json
import re
import qrcode

def prompt(msg: str, default: str = "") -> str:
    s = input(f"{msg}{' ['+default+']' if default else ''}: ").strip()
    return s if s else default

def safe_filename(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:180]

def build_payload():
    obj_type = prompt("Type (printer|filament)", "printer")
    obj_id = prompt("ID (unique, e.g. PRUSA-01 or FIL-PLA-RED-001)", "")
    name = prompt("Name/Label (human-friendly)", "")
    location = prompt("Location/zone (optional)", "")
    extra = prompt("Extra JSON (optional, leave blank)", "")

    payload = {
        "type": obj_type,
        "id": obj_id,
        "name": name,
        "location": location,
    }

    if extra:
        try:
            payload["extra"] = json.loads(extra)
        except Exception:
            payload["extra_raw"] = extra

    return payload

def main():
    out_dir = prompt("Output folder", "qr_codes")
    os.makedirs(out_dir, exist_ok=True)

    payload = build_payload()
    if not payload.get("id"):
        raise SystemExit("ID is required.")

    data_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    img = qrcode.make(data_str)

    fname = f"{payload['type']}_{payload['id']}.png"
    fname = safe_filename(fname)
    path = os.path.join(out_dir, fname)
    img.save(path)

    print("\n✅ QR created:")
    print("File:", path)
    print("Encoded JSON:", data_str)

if __name__ == "__main__":
    main()