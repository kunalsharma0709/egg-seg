from pathlib import Path
import random
from PIL import Image

files = sorted(Path("data/images").glob("*"))
print("count:", len(files))
random.seed(0)
pick = random.sample(files, 12)
tiles = []
for p in pick:
    im = Image.open(p).convert("RGB")
    print(p.name, im.size)
    tiles.append(im.resize((320, 320)))

sheet = Image.new("RGB", (320 * 4, 320 * 3))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % 4) * 320, (i // 4) * 320))
Path("outputs").mkdir(exist_ok=True)
sheet.save("outputs/peek.jpg")