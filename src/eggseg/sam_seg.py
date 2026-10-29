import argparse, csv
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import pipeline

IMG_DIR = Path("data/images")
OUT = Path("outputs/task1")


def iou(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(np.logical_or(a, b).sum(), 1)


def largest_contour(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea)


def solidity(m):
    c = largest_contour(m)
    hull = cv2.contourArea(cv2.convexHull(c))
    return cv2.contourArea(c) / max(hull, 1)


def aspect(m):
    c = largest_contour(m)
    (_, _), (w, h), _ = cv2.minAreaRect(c)
    return max(w, h) / max(min(w, h), 1)


def keep_eggs(masks, scores, min_frac, max_frac, min_solidity, nms_iou, max_aspect, max_eggs):
    h, w = masks[0].shape
    cand = []
    for m, s in zip(masks, scores):
        if m.sum() == 0:
            continue
        frac = m.sum() / (h * w)
        if not (min_frac <= frac <= max_frac):
            continue
        if solidity(m) < min_solidity or aspect(m) > max_aspect:
            continue
        cand.append((float(s), m))
    cand.sort(key=lambda x: -x[0])
    kept = []
    for s, m in cand:
        if all(iou(m, k) < nms_iou for _, k in kept):
            kept.append((s, m))
        if len(kept) == max_eggs:
            break
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--model", default="facebook/sam-vit-base")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--points_per_batch", type=int, default=64)
    ap.add_argument("--min_frac", type=float, default=0.01)
    ap.add_argument("--max_frac", type=float, default=0.5)
    ap.add_argument("--min_solidity", type=float, default=0.9)
    ap.add_argument("--nms_iou", type=float, default=0.5)
    ap.add_argument("--max_aspect", type=float, default=2.0)
    ap.add_argument("--max_eggs", type=int, default=3)
    a = ap.parse_args()

    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    (OUT / "overlays").mkdir(parents=True, exist_ok=True)

    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline("mask-generation", model=a.model, device=device)
    print("Device:", gen.device)

    files = sorted(IMG_DIR.glob("*"))
    if a.limit:
        files = files[: a.limit]

    rows = []
    for p in files:
        img = Image.open(p).convert("RGB")
        out = gen(img, points_per_batch=a.points_per_batch)
        masks = [np.array(m).astype(bool) for m in out["masks"]]

        if a.debug:
            hh, ww = masks[0].shape
            print(p.name)
            for j, (m, s) in enumerate(zip(masks, out["scores"])):
                if m.sum() == 0:
                    continue
                print(f"   mask{j}: score={float(s):.2f} frac={m.sum()/(hh*ww):.3f} "
                      f"solidity={solidity(m):.2f} aspect={aspect(m):.2f}")

        kept = keep_eggs(masks, out["scores"], a.min_frac, a.max_frac,
                         a.min_solidity, a.nms_iou, a.max_aspect, a.max_eggs)

        label = np.zeros(masks[0].shape, np.uint16)
        overlay = np.array(img).copy()
        rng = np.random.default_rng(0)
        for i, (s, m) in enumerate(kept, 1):
            label[m] = i
            color = rng.integers(0, 255, 3)
            overlay[m] = (0.5 * overlay[m] + 0.5 * color).astype(np.uint8)
            cv2.drawContours(overlay, [largest_contour(m)], -1, (255, 255, 0), 2)
        Image.fromarray(label).save(OUT / "masks" / f"{p.stem}.png")
        Image.fromarray(overlay).save(OUT / "overlays" / f"{p.stem}.jpg")
        rows.append([p.name, len(masks), len(kept)])
        print(p.name, "raw:", len(masks), "kept:", len(kept))

    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "raw_masks", "kept_masks"])
        w.writerows(rows)


if __name__ == "__main__":
    main()