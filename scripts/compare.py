import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

run = Path(sys.argv[1])
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
r = pd.read_csv(run / "dev_metrics_diag.csv")
if len(sys.argv) > 3 and sys.argv[3] == "over":
    r = r[r.area_ratio > 1.3]
r = r.sort_values("iou").head(n)
S = 240


def ov(img, lab):
    o = img.copy()
    fg = lab > 0
    o[fg] = (0.5 * o[fg] + 0.5 * np.array([255, 200, 0])).astype(np.uint8)
    cnts, _ = cv2.findContours(fg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(o, cnts, -1, (0, 255, 255), 2)
    return o


rows = []
for fn, iou in zip(r.filename, r.iou):
    stem = Path(fn).stem
    img = cv2.imread(f"data/images/{fn}")
    gt = cv2.imread(f"outputs/task1/masks/{stem}.png", cv2.IMREAD_UNCHANGED)
    pr = cv2.imread(str(run / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
    tiles = [cv2.resize(t, (S, S)) for t in (img, ov(img, gt), ov(img, pr))]
    row = np.hstack(tiles)
    cv2.putText(row, f"{stem[:30]} IoU={iou:.2f}", (5, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    rows.append(row)

cv2.imwrite(str(run / "worst.jpg"), np.vstack(rows))