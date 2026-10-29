from pathlib import Path
import numpy as np
from PIL import Image

SRC = Path("outputs/task1/masks")
DST = Path("outputs/task1/masks_vis")
DST.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(1)
palette = np.vstack([[0, 0, 0], rng.integers(60, 255, (50, 3))]).astype(np.uint8)

for p in sorted(SRC.glob("*.png")):
    m = np.array(Image.open(p))
    Image.fromarray(palette[m]).save(DST / p.name)
print("saved to", DST)