import sys
from pathlib import Path
from PIL import Image

names = Path(sys.argv[1]).read_text().split()
out = sys.argv[2]
src = Path("outputs/task1/overlays")
files = [src / (Path(n).stem + ".jpg") for n in names]
cols, size = 3, 400
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * size, rows * size))
for i, p in enumerate(files):
    sheet.paste(Image.open(p).convert("RGB").resize((size, size)), ((i % cols) * size, (i // cols) * size))
sheet.save(out)
print("saved", len(files), "->", out)