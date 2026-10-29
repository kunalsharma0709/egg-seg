import argparse, csv
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image
from transformers import (AutoProcessor, AutoModelForZeroShotObjectDetection,
                          SamModel, SamProcessor)

IMG_DIR = Path("data/images")
OUT = Path("outputs/task1")
PALETTE = [(0, 200, 255), (255, 80, 80), (80, 255, 80)]


def mask_iou(a, b):
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def containment(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(min(a.sum(), b.sum()), 1)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def largest_contour(m):
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max(cnts, key=cv2.contourArea)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dino", default="IDEA-Research/grounding-dino-tiny")
    ap.add_argument("--sam", default="facebook/sam-vit-base")
    ap.add_argument("--prompt", default="egg.")
    ap.add_argument("--box_thr", type=float, default=0.3)
    ap.add_argument("--fallback_thr", type=float, default=0.15)
    ap.add_argument("--text_thr", type=float, default=0.25)
    ap.add_argument("--max_eggs", type=int, default=3)
    ap.add_argument("--max_box_frac", type=float, default=0.8)
    ap.add_argument("--box_nms_iou", type=float, default=0.5)
    ap.add_argument("--dedupe_iou", type=float, default=0.7)
    ap.add_argument("--contain_thr", type=float, default=0.8)
    ap.add_argument("--min_mask_frac", type=float, default=0.005)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--names", default="", help="text file with image filenames to (re)run")
    a = ap.parse_args()

    for sub in ("masks", "masks_vis", "overlays"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", dev)

    dproc = AutoProcessor.from_pretrained(a.dino)
    dmodel = AutoModelForZeroShotObjectDetection.from_pretrained(a.dino).to(dev).eval()
    sproc = SamProcessor.from_pretrained(a.sam)
    smodel = SamModel.from_pretrained(a.sam).to(dev).eval()

    def detect(img, thr):
        W, H = img.size
        din = dproc(images=img, text=a.prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            dout = dmodel(**din)
        res = dproc.post_process_grounded_object_detection(
            dout, din.input_ids, threshold=thr,
            text_threshold=a.text_thr, target_sizes=[(H, W)])[0]
        boxes = [[float(v) for v in b] for b in res["boxes"].cpu()]
        scores = [float(s) for s in res["scores"].cpu()]
        n_raw = len(boxes)

        # drop huge boxes (tray / whole image)
        idx = [i for i, b in enumerate(boxes)
               if (b[2] - b[0]) * (b[3] - b[1]) <= a.max_box_frac * W * H]
        idx.sort(key=lambda i: -scores[i])

        # box NMS: one egg should not get two boxes
        keep = []
        for i in idx:
            if all(box_iou(boxes[i], boxes[j]) < a.box_nms_iou for j in keep):
                keep.append(i)
        keep = keep[: a.max_eggs]
        return [boxes[i] for i in keep], n_raw

    files = sorted(IMG_DIR.glob("*"))
    if a.names:
        wanted = set(Path(a.names).read_text().split())
        files = [f for f in files if f.name in wanted]
    if a.limit:
        files = files[: a.limit]

    rows = []
    for p in files:
        img = Image.open(p).convert("RGB")
        W, H = img.size

        boxes, n_raw = detect(img, a.box_thr)
        used_fallback = 0
        if not boxes and a.fallback_thr < a.box_thr:
            boxes, n_raw = detect(img, a.fallback_thr)
            used_fallback = 1

        label = np.zeros((H, W), np.uint16)
        color_mask = np.zeros((H, W, 3), np.uint8)
        overlay = np.array(img).copy()
        kept = 0

        if boxes:
            sin = sproc(img, input_boxes=[boxes], return_tensors="pt").to(dev)
            with torch.no_grad():
                sout = smodel(**sin, multimask_output=False)
            masks = sproc.image_processor.post_process_masks(
                sout.pred_masks.cpu(), sin["original_sizes"].cpu(),
                sin["reshaped_input_sizes"].cpu())[0]
            masks = masks[:, 0].numpy().astype(bool)

            final = []
            for m in masks:
                if m.sum() < a.min_mask_frac * W * H:
                    continue
                dup = any(mask_iou(m, k) >= a.dedupe_iou or
                          containment(m, k) >= a.contain_thr for k in final)
                if not dup:
                    final.append(m)

            for i, m in enumerate(final, 1):
                color = np.array(PALETTE[(i - 1) % len(PALETTE)], np.uint8)
                label[m] = i
                color_mask[m] = color
                overlay[m] = (0.5 * overlay[m] + 0.5 * color).astype(np.uint8)
                cv2.drawContours(overlay, [largest_contour(m)], -1, (255, 255, 0), 2)
            kept = len(final)

        Image.fromarray(label).save(OUT / "masks" / f"{p.stem}.png")
        Image.fromarray(color_mask).save(OUT / "masks_vis" / f"{p.stem}.png")
        Image.fromarray(overlay).save(OUT / "overlays" / f"{p.stem}.jpg")
        rows.append([p.name, n_raw, kept, used_fallback])
        print(p.name, "boxes:", n_raw, "kept:", kept, "fallback:", used_fallback)

    summ = "summary.csv" if not a.names else f"summary_{Path(a.names).stem}.csv"
    with open(OUT / summ, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "dino_boxes", "kept_masks", "used_fallback"])
        w.writerows(rows)
    print("summary ->", OUT / summ)


if __name__ == "__main__":
    main()