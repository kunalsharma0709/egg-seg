import csv
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path("outputs/task1")
rows = []
for img in sorted(Path("data/images").glob("*")):
    mp = OUT / "masks" / f"{img.stem}.png"
    n = len(np.unique(np.array(Image.open(mp)))) - 1 if mp.exists() else -1
    rows.append([img.name, n])
with open(OUT / "summary_final.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["filename", "n_eggs"])
    w.writerows(rows)
from collections import Counter
print(sorted(Counter(r[1] for r in rows).items()))