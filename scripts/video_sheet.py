import sys
import cv2
import numpy as np

video = sys.argv[1] if len(sys.argv) > 1 else "data/video/egg.mp4"
step = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0     # har kitne second pe frame
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
tiles, t = [], 0.0
while t < dur:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, fr = cap.read()
    if ok:
        fr = cv2.resize(fr, (320, 180))
        cv2.putText(fr, f"{t:.0f}s", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        tiles.append(fr)
    t += step
while len(tiles) % 4:
    tiles.append(np.zeros_like(tiles[0]))
rows = [np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
cv2.imwrite("outputs/task4/sheet.jpg", np.vstack(rows))
print(f"{len(tiles)} frames, saved outputs/task4/sheet.jpg")