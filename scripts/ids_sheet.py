import argparse
import cv2
import numpy as np
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("video")
ap.add_argument("--step", type=float, default=1.0)    # har kitne second pe frame
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=0.0)     # 0 = video ke end tak
ap.add_argument("--cols", type=int, default=4)
ap.add_argument("--per_sheet", type=int, default=12)
ap.add_argument("--out", default="outputs/task4/ids")
a = ap.parse_args()

cap = cv2.VideoCapture(a.video)
fps = cap.get(cv2.CAP_PROP_FPS)
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
end = a.end or dur
times = np.arange(a.start, end, a.step)

tiles = []
for t in times:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, fr = cap.read()
    if not ok:
        print(f"{t:.1f}s not readable, skipped")
        continue
    cv2.putText(fr, f"{t:.0f}s", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tiles.append(fr)
if not tiles:
    raise SystemExit("no frames read, check video path / timestamps")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
for s, i in enumerate(range(0, len(tiles), a.per_sheet), 1):
    chunk = tiles[i:i + a.per_sheet]
    while len(chunk) % a.cols:
        chunk.append(np.zeros_like(chunk[0]))
    rows = [np.hstack(chunk[j:j + a.cols]) for j in range(0, len(chunk), a.cols)]
    p = f"{a.out}_{s}.jpg"
    cv2.imwrite(p, np.vstack(rows))
    print("saved", p)