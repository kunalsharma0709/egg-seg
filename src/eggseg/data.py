import re, shutil, csv
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/images")
EXTS = {".jpg", ".jpeg", ".png"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(RAW.rglob("*")):
        if p.suffix.lower() not in EXTS:
            continue
        split = next((s for s in ("train", "valid", "test") if s in p.parts), "unknown")
        new_name = f"{split}_{p.name}"
        shutil.copy2(p, OUT / new_name)
        group = re.split(r"\.rf\.", p.stem)[0]
        rows.append([new_name, split, group])

    with open("data/manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "roboflow_split", "group_id"])
        w.writerows(rows)

    groups = {r[2] for r in rows}
    print(f"Total images: {len(rows)} | Unique groups: {len(groups)}")


if __name__ == "__main__":
    main()