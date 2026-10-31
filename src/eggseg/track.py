import argparse
import csv
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from eggseg.classical_seg import seg_grabcut, postprocess


def white_candidates(fr, v_thr, s_thr):
    H, W = fr.shape[:2]
    hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 2] > v_thr) & (hsv[..., 1] < s_thr)).astype(np.uint8) * 255
    m[int(0.83 * H):, int(0.72 * W):] = 0                 # Mocomi logo corner
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)


def split_watershed(m, frac=0.55):
    """Touching eggs ko distance-transform watershed se alag karo."""
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    n_all, lab_all = cv2.connectedComponents(m)
    out, nid = np.zeros(m.shape, np.int32), 0
    for i in range(1, n_all):
        comp = (lab_all == i).astype(np.uint8)
        d = dist * comp
        peaks = (d > frac * d.max()).astype(np.uint8)
        npk, mk = cv2.connectedComponents(peaks)
        if npk <= 2:                                       # ek hi peak: split nahi
            nid += 1
            out[comp > 0] = nid
            continue
        markers = mk.astype(np.int32)                      # 0 unknown, 1..npk-1 peaks
        markers[comp == 0] = npk                           # background marker
        inv = (255 * (1 - d / d.max())).astype(np.uint8)
        ws = cv2.watershed(cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR), markers)
        for p in range(1, npk):
            nid += 1
            out[(ws == p) & (comp > 0)] = nid
    return out


def segment_frame(fr, a):
    if a.method == "white":
        return split_watershed(white_candidates(fr, a.v_thr, a.s_thr))
    raw = seg_grabcut(fr, a.margin, a.iters)               # Task 2 final algorithm
    return postprocess(raw, 7, 0.005, a.max_eggs, 0.3, 0.8, False, a.margin, True)


def mask_iou(x, y):
    u = (x | y).sum()
    return (x & y).sum() / u if u else 0.0


def get_dets(lab, W, H):
    dets = []
    for i in np.unique(lab):
        if i == 0:
            continue
        m = lab == i
        ys, xs = np.nonzero(m)
        cn, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cn, key=cv2.contourArea)
        hull = cv2.contourArea(cv2.convexHull(c))
        dets.append(dict(mask=m, c=np.array([xs.mean(), ys.mean()]), area=m.sum() / (W * H),
                         sol=cv2.contourArea(c) / hull if hull > 0 else 0.0))
    return dets


def is_egg(d, a, H):
    return (a.min_area <= d["area"] <= a.max_area
            and d["c"][1] >= a.y_min * H and d["sol"] >= a.min_sol)


class Tracker:
    def __init__(self, diag, iou_thr=0.2, dist_frac=0.15, max_age=100):
        self.tracks, self.next_id = {}, 1
        self.diag, self.iou_thr, self.dist_frac, self.max_age = diag, iou_thr, dist_frac, max_age

    def update(self, dets):
        ids, match = list(self.tracks), {}
        if ids and dets:
            IOU = np.zeros((len(ids), len(dets)))
            DIST = np.zeros_like(IOU)
            for i, t in enumerate(ids):
                for j, d in enumerate(dets):
                    IOU[i, j] = mask_iou(self.tracks[t]["mask"], d["mask"])
                    DIST[i, j] = np.linalg.norm(self.tracks[t]["c"] - d["c"]) / self.diag
            rows, cols = linear_sum_assignment((1 - IOU) + DIST)       # Hungarian
            for i, j in zip(rows, cols):
                if IOU[i, j] >= self.iou_thr or DIST[i, j] <= self.dist_frac:
                    match[j] = ids[i]
        out, seen = [], set()
        for j, d in enumerate(dets):
            tid = match.get(j)
            if tid is None:
                tid, self.next_id = self.next_id, self.next_id + 1
            self.tracks[tid] = dict(mask=d["mask"], c=d["c"], age=0)
            seen.add(tid)
            out.append((tid, d))
        for t in list(self.tracks):
            if t not in seen:
                self.tracks[t]["age"] += 1
                if self.tracks[t]["age"] > self.max_age:
                    del self.tracks[t]
        return out


def color(i):
    rng = np.random.default_rng(i * 7919)
    return tuple(int(x) for x in rng.integers(60, 255, 3))


def main(a):
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    W0, H0 = int(cap.get(3)), int(cap.get(4))
    W, H = a.width, int(H0 * a.width / W0)
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / a.stride, (W, H))
    tr = Tracker(diag=(W ** 2 + H ** 2) ** 0.5, iou_thr=a.iou_thr,
                 dist_frac=a.dist_frac, max_age=a.max_age)
    end = a.end if a.end > 0 else 1e9
    log, idx, done = [], 0, 0
    while True:
        ok, fr = cap.read()
        if not ok or (a.max_frames and done >= a.max_frames):
            break
        if idx % a.stride == 0:
            t = idx / fps
            fr = cv2.resize(fr, (W, H), interpolation=cv2.INTER_AREA)
            vis = fr.copy()
            if a.start <= t <= end:
                dets = [d for d in get_dets(segment_frame(fr, a), W, H) if is_egg(d, a, H)]
                for tid, d in tr.update(dets):
                    col = color(tid)
                    vis[d["mask"]] = (0.5 * vis[d["mask"]] + 0.5 * np.array(col)).astype(np.uint8)
                    cn, _ = cv2.findContours(d["mask"].astype(np.uint8), cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis, cn, -1, col, 2)
                    cx, cy = int(d["c"][0]), int(d["c"][1])
                    cv2.putText(vis, f"ID {tid}", (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 255, 255), 2)
                    log.append((idx, round(t, 2), tid, cx, cy, int(d["mask"].sum())))
                done += 1
                if done % 50 == 0:
                    print(f"processed {done} frames (t={t:.1f}s)")
            vw.write(vis)
        idx += 1
    cap.release()
    vw.release()
    with open(out_path.with_suffix(".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s", "id", "cx", "cy", "area_px"])
        w.writerows(log)
    print(f"done. {done} segmented frames, {len({r[2] for r in log})} unique IDs -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/video/egg.mp4")
    ap.add_argument("--out", default="outputs/task4/tracked.mp4")
    ap.add_argument("--method", default="white", choices=["white", "grabcut"])
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=0.0)          # 0 = end tak
    ap.add_argument("--max_frames", type=int, default=0)
    ap.add_argument("--v_thr", type=int, default=190)
    ap.add_argument("--s_thr", type=int, default=45)
    ap.add_argument("--y_min", type=float, default=0.38)
    ap.add_argument("--min_area", type=float, default=0.004)
    ap.add_argument("--max_area", type=float, default=0.35)
    ap.add_argument("--min_sol", type=float, default=0.75)
    ap.add_argument("--max_eggs", type=int, default=2)         # sirf grabcut method me
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--iou_thr", type=float, default=0.2)
    ap.add_argument("--dist_frac", type=float, default=0.15)
    ap.add_argument("--max_age", type=int, default=100)        # v2 final (v1 = 5)
    main(ap.parse_args())