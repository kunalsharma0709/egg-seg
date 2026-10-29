import csv, random
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np

IMG_DIR = Path("data/images")
MASK_DIR = Path("outputs/task1/masks")
EXCLUDE = Path("outputs/exclude.txt")
HAMMING = 6        # <= is distance pe near-duplicate maanenge
EVAL_FRAC = 0.2     # 4:1 split
SEED = 42

MANUAL_GROUPS = [
    ["train_not_damaged_94", "valid_not_damaged_75", "valid_not_damaged_9"],
]

def dhash(path):
    g = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (9, 8), interpolation=cv2.INTER_AREA)
    return (g[:, 1:] > g[:, :-1]).flatten()          # 64 bits


def main():
    excluded = set()
    if EXCLUDE.exists():
        excluded = {Path(l.strip()).stem for l in EXCLUDE.read_text().splitlines() if l.strip()}

    files = sorted(p for p in IMG_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = [p for p in files if p.stem not in excluded]
    print(f"images: {len(files)} (excluded: {len(excluded)})")

    hashes = [dhash(p) for p in files]
    n_eggs = []
    for p in files:
        m = cv2.imread(str(MASK_DIR / f"{p.stem}.png"), cv2.IMREAD_UNCHANGED)
        n_eggs.append(len(np.unique(m)) - (1 if (m == 0).any() else 0))

    # union-find: near-duplicates ek group me
    parent = list(range(len(files)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i in range(len(files)):
        for j in range(i + 1, len(files)):
            if np.count_nonzero(hashes[i] != hashes[j]) <= HAMMING:
                parent[find(i)] = find(j)

        idx_of = {p.name.split("_jpg")[0]: i for i, p in enumerate(files)}
    for grp in MANUAL_GROUPS:
        ids = [idx_of[n] for n in grp if n in idx_of]
        for a in ids[1:]:
            parent[find(a)] = find(ids[0])            

    groups = defaultdict(list)
    for i in range(len(files)):
        groups[find(i)].append(i)
    multi = [g for g in groups.values() if len(g) > 1]
    print(f"groups: {len(groups)} | groups with >1 image: {len(multi)}")
    for g in multi:
        print("  group:", [files[i].name for i in g])

    # stratified: group ke max egg count ke bucket me se ~20% groups eval me
    buckets = defaultdict(list)
    for gid, idx in groups.items():
        buckets[max(n_eggs[i] for i in idx)].append(gid)
    rng = random.Random(SEED)
    eval_groups = set()
    for k, gids in sorted(buckets.items()):
        gids = sorted(gids)
        rng.shuffle(gids)
        n_eval = max(1, round(EVAL_FRAC * len(gids))) if len(gids) >= 2 else 0
        eval_groups.update(gids[:n_eval])

    rows = []
    for gid, idx in groups.items():
        for i in idx:
            rows.append((files[i].name, n_eggs[i], gid, "eval" if gid in eval_groups else "dev"))
    rows.sort()
    with open("data/split.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "n_eggs", "group", "split"])
        w.writerows(rows)

    for s in ("dev", "eval"):
        sub = [r for r in rows if r[3] == s]
        cnt = {k: sum(1 for r in sub if r[1] == k) for k in (1, 2, 3)}
        print(f"{s}: {len(sub)} images | eggs distribution {cnt}")


if __name__ == "__main__":
    main()