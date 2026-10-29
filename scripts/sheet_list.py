import sys
from pathlib import Path
from PIL import Image

lst, out = sys.argv[1], sys.argv[2]
names = Path(lst).read_text().split()
src = Path("data/images")
files = [src / n for n in names]
cols, size = 3, 400
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * size, rows * size))
for i, p in enumerate(files):
    sheet.paste(Image.open(p).convert("RGB").resize((size, size)), ((i % cols) * size, (i // cols) * size))
sheet.save(out)
print("saved", len(files), "->", out)