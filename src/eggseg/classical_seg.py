import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def solidity(c):
    ha = cv2.contourArea(cv2.convexHull(c))
    return cv2.contourArea(c) / ha if ha > 0 else 0.0


def get_blobs(m, k, min_frac, max_frac):
    h, w = m.shape
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, ker)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, ker)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if min_frac * h * w <= cv2.contourArea(c) <= max_frac * h * w]


def postprocess(m, k, min_frac, max_eggs, rel_area, min_sol=0.0,
                complement=False, margin=0.0, fallback=False):
    h, w = m.shape
    all_cnts = get_blobs(m, k, min_frac, 1.0)
    cnts = all_cnts
    if min_sol > 0:
        cnts = [c for c in all_cnts if solidity(c) >= min_sol]
        if not cnts and complement:
            inv = 255 - m
            r = int(margin * min(h, w)) + 2
            inv[:r], inv[-r:], inv[:, :r], inv[:, -r:] = 0, 0, 0, 0
            cnts = [c for c in get_blobs(inv, k, min_frac, 0.9) if solidity(c) >= min_sol]
        if not cnts and fallback:
            cnts = all_cnts          # filter ne sab hata diya to T4 wale blobs wapas
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:max_eggs]
    if cnts and rel_area > 0:
        big = cv2.contourArea(cnts[0])
        cnts = [c for c in cnts if cv2.contourArea(c) >= rel_area * big]
    lab = np.zeros((h, w), np.uint16)
    for i, c in enumerate(cnts, 1):
        cv2.drawContours(lab, [c], -1, i, thickness=-1)
    return lab


def seg_otsu(bgr, blur=5):
    g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (blur, blur), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
    if (border > 0).mean() > 0.5:
        m = 255 - m
    return m


def seg_grabcut(bgr, margin=0.1, iters=5):
    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    rect = (int(margin * w), int(margin * h), int((1 - 2 * margin) * w), int((1 - 2 * margin) * h))
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    return np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)


def raw_mask(bgr, stem, a):
    """Raw (post-process se pehle) mask; disk pe cache hota hai."""
    if a.method == "otsu":
        return seg_otsu(bgr, a.blur)
    cache = Path(f"outputs/task2/raw_grabcut_m{a.margin}_i{a.iters}")
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{stem}.png"
    if p.exists():
        return cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    m = seg_grabcut(bgr, a.margin, a.iters)
    cv2.imwrite(str(p), m)
    return m


def iou_dice(gt, pr):
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    s = gt.sum() + pr.sum()
    return (inter / union if union else 1.0), (2 * inter / s if s else 1.0)


def main(a):
    from tqdm import tqdm
    df = pd.read_csv("data/split.csv")
    df = df[df.split == a.split]
    out = Path(a.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for fn in tqdm(df.filename):
        stem = Path(fn).stem
        img = cv2.imread(f"data/images/{fn}")
        gt = cv2.imread(f"outputs/task1/masks/{stem}.png", cv2.IMREAD_UNCHANGED)
        pr = postprocess(raw_mask(img, stem, a), a.k, a.min_frac, a.max_eggs,
                         a.rel_area, a.min_sol, a.complement, a.margin, a.fallback)
        cv2.imwrite(str(out / "masks" / f"{stem}.png"), pr)
        iou, dice = iou_dice(gt > 0, pr > 0)
        rows.append((fn, iou, dice, int(gt.max()), int(pr.max())))
    r = pd.DataFrame(rows, columns=["filename", "iou", "dice", "gt_n", "pred_n"])
    r.to_csv(out / f"{a.split}_metrics.csv", index=False)
    print(f"{a.split} [{a.method}]: mean IoU {r.iou.mean():.3f} | mean Dice {r.dice.mean():.3f} "
          f"| count acc {(r.gt_n == r.pred_n).mean():.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")        
    ap.add_argument("--out", default="outputs/task2/tmp")
    ap.add_argument("--method", default="otsu", choices=["otsu", "grabcut"])
    ap.add_argument("--blur", type=int, default=5)
    ap.add_argument("--k", type=int, default=7)
    ap.add_argument("--min_frac", type=float, default=0.005)
    ap.add_argument("--max_eggs", type=int, default=3)
    ap.add_argument("--rel_area", type=float, default=0.0)   # 0 = filter off
    ap.add_argument("--min_sol", type=float, default=0.0)    # 0 = filter off
    ap.add_argument("--complement", action="store_true")
    ap.add_argument("--fallback", action="store_true")
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=5)
    main(ap.parse_args())