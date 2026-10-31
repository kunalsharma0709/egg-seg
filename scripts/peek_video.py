import sys
from pathlib import Path
import cv2

video = sys.argv[1] if len(sys.argv) > 1 else "data/video/egg.mp4"
cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit(f"Video nahi khula: {video}")

n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"frames {n} | fps {fps:.1f} | size {int(cap.get(3))}x{int(cap.get(4))} | duration {n / fps:.1f}s")

out = Path("outputs/task4/peek")
out.mkdir(parents=True, exist_ok=True)
for i, frac in enumerate((0.05, 0.25, 0.5, 0.75, 0.95)):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * frac))
    ok, fr = cap.read()
    if ok:
        cv2.imwrite(str(out / f"frame_{i}.jpg"), fr)
print(f"5 frames saved in {out}")