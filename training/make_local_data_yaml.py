from pathlib import Path
import yaml

def main():
    root = Path(__file__).resolve().parent.parent  # repo root
    ds = root / "datasets" / "inventory_v1"

    data = {
        "path": str(ds),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "filament_spool"},
    }

    out = ds / "data.local.yaml"
    out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print("Wrote:", out)

if __name__ == "__main__":
    main()