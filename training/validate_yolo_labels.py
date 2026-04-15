from pathlib import Path

def parse_line(line: str):
    parts = line.strip().split()
    if len(parts) != 5:
        return None
    cls = int(float(parts[0]))
    x, y, w, h = map(float, parts[1:])
    return cls, x, y, w, h

def main(base="datasets/inventory_v1"):
    base = Path(base)
    problems = 0
    total = 0

    for split in ["train", "val", "test"]:
        label_dir = base / "labels" / split
        if not label_dir.exists():
            continue

        for txt in label_dir.glob("*.txt"):
            total += 1
            lines = txt.read_text(encoding="utf-8").strip().splitlines()
            for i, line in enumerate(lines):
                parsed = parse_line(line)
                if not parsed:
                    print(f"[BAD FORMAT] {txt} line {i+1}: {line}")
                    problems += 1
                    continue
                cls, x, y, w, h = parsed
                if cls != 0:
                    print(f"[BAD CLASS] {txt} line {i+1}: cls={cls} (expected 0)")
                    problems += 1
                if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 <= w <= 1.0 and 0.0 <= h <= 1.0):
                    print(f"[BAD RANGE] {txt} line {i+1}: {parsed}")
                    problems += 1
                if w <= 0.0 or h <= 0.0:
                    print(f"[BAD SIZE] {txt} line {i+1}: {parsed}")
                    problems += 1

    print(f"\nChecked {total} label files.")
    if problems == 0:
        print("Labels look good.")
    else:
        print(f"Found {problems} issues.")

if __name__ == "__main__":
    main()