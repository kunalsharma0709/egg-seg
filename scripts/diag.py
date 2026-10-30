import sys
from pathlib import Path
import cv2
import pandas as pd

run = Path(sys.argv[1])
r = pd.read_csv(run / "dev_metrics.csv")
ratios = []
for fn in r.filename:
    stem = Path(fn).stem
    gt = cv2.imread(f"outputs/task1/masks/{stem}.png", cv2.IMREAD_UNCHANGED) > 0
    pr = cv2.imread(str(run / "masks" / f"{stem}.png"), cv2.IMREAD_UNCHANGED) > 0
    ratios.append(pr.sum() / max(gt.sum(), 1))
r["area_ratio"] = ratios            # pred area / GT area
r.to_csv(run / "dev_metrics_diag.csv", index=False)

mid = r[(r.iou >= 0.1) & (r.iou < 0.8)]
over = (mid.area_ratio > 1.3).sum()
under = (mid.area_ratio < 0.77).sum()
print(f"mid images: {len(mid)} | over-segmented (ratio>1.3): {over} "
      f"| under-segmented (ratio<0.77): {under} | close: {len(mid) - over - under}")