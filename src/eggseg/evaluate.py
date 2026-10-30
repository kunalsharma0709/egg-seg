import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


def boundary(m):
    m = m.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return (m - cv2.erode(m, k)) > 0


def boundary_f1(gt, pr, tol=3):
    bg, bp = boundary(gt), boundary(pr)
    if bg.sum() == 0 and bp.sum() == 0:
        return 1.0
    if bg.sum() == 0 or bp.sum() == 0:
        return 0.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol + 1, 2 * tol + 1))
    dg = cv2.dilate(bg.astype(np.uint8), k) > 0
    dp = cv2.dilate(bp.astype(np.uint8), k) > 0
    p = (bp & dg).sum() / bp.sum()
    r = (bg & dp).sum() / bg.sum()
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def instance_stats(gl, pl, thr=0.5):
    g = [gl == i for i in np.unique(gl) if i != 0]
    p = [pl == i for i in np.unique(pl) if i != 0]
    if not g or not p:
        return 0, len(p), len(g)
    M = np.zeros((len(g), len(p)))
    for i, a in enumerate(g):
        for j, b in enumerate(p):
            u = (a | b).sum()
            M[i, j] = (a & b).sum() / u if u else 0
    r, c = linear_sum_assignment(-M)          # Hungarian matching
    tp = int((M[r, c] > thr).sum())
    return tp, len(p) - tp, len(g) - tp       # tp, fp, fn


def error_map(img, gt, pr):
    o = (img * 0.5).astype(np.uint8)
    o[gt & pr] = (0, 200, 0)          # sahi (green)
    o[~gt & pr] = (0, 0, 255)         # false positive (red)
    o[gt & ~pr] = (255, 120, 0)       # false negative (blue)
    return o


def main(a):
    df = pd.read_csv("data/split.csv")
    df = df[df.split == a.split]
    pred_dir = Path(a.pred) / "masks"
    out = Path(f"outputs/task3/{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for fname in df.filename:
        stem = Path(fname).stem
        gl = cv2.imread(f"outputs/task1/masks/{stem}.png", cv2.IMREAD_UNCHANGED)
        pl = cv2.imread(str(pred_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED)
        g, p = gl > 0, pl > 0
        inter = (g & p).sum()
        union = (g | p).sum()
        tp, fp, fn_ = instance_stats(gl, pl)
        rows.append(dict(
            filename=fname, gt_n=len(np.unique(gl)) - 1, pred_n=len(np.unique(pl)) - 1,
            iou=inter / union if union else 1.0,
            dice=2 * inter / (g.sum() + p.sum()) if (g.sum() + p.sum()) else 1.0,
            precision=inter / p.sum() if p.sum() else 0.0,
            recall=inter / g.sum() if g.sum() else 0.0,
            bf1=boundary_f1(g, p), tp=tp, fp=fp, fn=fn_))
    r = pd.DataFrame(rows)
    r.to_csv(out / "metrics.csv", index=False)

    TP, FP, FN = r.tp.sum(), r.fp.sum(), r.fn.sum()
    ip, ir = TP / max(TP + FP, 1), TP / max(TP + FN, 1)
    rng = np.random.default_rng(0)
    boots = [r.iou.values[rng.integers(0, len(r), len(r))].mean() for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])

    print(f"[{a.split}] n={len(r)}")
    print(f" pixel: IoU {r.iou.mean():.3f} (95% CI {lo:.3f}-{hi:.3f}) | Dice {r.dice.mean():.3f} | "
          f"precision {r.precision.mean():.3f} | recall {r.recall.mean():.3f} | boundary F1 {r.bf1.mean():.3f}")
    print(f" instance (IoU>0.5): P {ip:.3f} | R {ir:.3f} | F1 {2*ip*ir/max(ip+ir,1e-9):.3f} "
          f"| count acc {(r.gt_n == r.pred_n).mean():.3f}")
    print(f" IoU<0.1: {(r.iou < 0.1).sum()} | IoU>=0.8: {(r.iou >= 0.8).sum()}")
    grp = r.gt_n.clip(upper=2).map({1: "1 egg", 2: "2-3 eggs"})
    print(r.groupby(grp)[["iou", "dice"]].agg(["mean", "count"]).round(3))

    plt.figure(figsize=(5, 3.5))
    plt.hist(r.iou, bins=10, range=(0, 1), edgecolor="k")
    plt.xlabel("per-image IoU"); plt.ylabel("images"); plt.title(f"IoU distribution ({a.split})")
    plt.tight_layout(); plt.savefig(out / "iou_hist.png", dpi=150); plt.close()

    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(r.recall, r.precision, s=12)
    plt.xlabel("recall (pixel)"); plt.ylabel("precision (pixel)")
    plt.title("Under-seg (left) vs over-seg (bottom)")
    plt.tight_layout(); plt.savefig(out / "prec_recall.png", dpi=150); plt.close()

    S, rowsimg = 240, []
    for _, x in r.sort_values("iou").head(a.n_worst).iterrows():
        stem = Path(x.filename).stem
        img = cv2.imread(f"data/images/{x.filename}")
        gl = cv2.imread(f"outputs/task1/masks/{stem}.png", cv2.IMREAD_UNCHANGED) > 0
        pl = cv2.imread(str(pred_dir / f"{stem}.png"), cv2.IMREAD_UNCHANGED) > 0
        tiles = [cv2.resize(t, (S, S)) for t in (img, error_map(img, gl, pl))]
        row = np.hstack(tiles)
        cv2.putText(row, f"{stem[:28]} IoU={x.iou:.2f} P={x.precision:.2f} R={x.recall:.2f}",
                    (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        rowsimg.append(row)
    cv2.imwrite(str(out / "failures.jpg"), np.vstack(rowsimg))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="eval")
    ap.add_argument("--pred", default="outputs/task2/final_eval")
    ap.add_argument("--n_worst", type=int, default=8)
    main(ap.parse_args())