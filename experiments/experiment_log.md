# Experiment log (Task 2 classical segmentation + Task 4 tracking)

Ground truth = Task 1 masks (Grounding DINO + SAM). All Task 2 numbers are on the **dev split
(126 images)**; the eval split (32 images) was run once, after the algorithm was frozen.
Defaults unless stated: Gaussian blur 5, morphology open+close with 7x7 ellipse, minimum blob area
0.5% of the image, at most 3 blobs, filled contours as instance masks. Metrics: mean per-image
pixel IoU / Dice, and count accuracy (predicted egg count == reference egg count).

## Task 2

| Trial | What changed | Why | Parameters | Dev IoU / Dice / count acc | Decision |
|---|---|---|---|---|---|
| T1 | Baseline: gray + blur + Otsu, polarity flip if >50% of the border is foreground | Simplest starting point | blur 5, k 7 | 0.510 / 0.600 / 0.548 | Replaced. Worst cases: white egg on white background, polarity flip on close-ups, textured backgrounds (cloth, leaves) picked instead of the egg |
| T2 | Otsu replaced by GrabCut initialised with a rectangle | Remove the polarity heuristic, use a colour model | margin 0.10, 5 iterations | 0.536 / 0.649 / 0.802 | Kept. 13 images IoU < 0.1, 24 images IoU >= 0.8, 89 in between. Of the 89: 38 under-segmented, 18 over-segmented, 33 close (area ratio thresholds 0.77 / 1.3) |
| T3b | Margin 0.10 -> 0.05 | Eggs touching the frame edge were cut, because everything outside the rectangle is sure background | margin 0.05 | 0.572 / 0.676 / 0.778 | Discarded: dominated by T3a (under 15, over 27, close 35) |
| T3a | Margin 0.10 -> 0.02 | Same reason, stronger | margin 0.02 | 0.605 / 0.706 / 0.770 | Kept. Under-segmentation 38 -> 8, over-segmentation 18 -> 32 (a bigger rectangle lets more background into the colour model). Re-run after refactoring with the raw-mask cache reproduced the same numbers |
| T4 | Drop blobs smaller than 30% of the largest blob | Count accuracy was low; extra small blobs suspected | margin 0.02, rel_area 0.3 | 0.605 / 0.706 / 0.817 | Kept. Count accuracy +0.047, IoU unchanged because the removed blobs were small |
| T5 | Drop blobs with solidity (area / convex hull area) < 0.9 | Failure blobs looked irregular, eggs are convex | + min_sol 0.9 | 0.410 / 0.459 / 0.540 | Discarded. 49 of 126 predictions became empty (all blobs removed), checked via the count of images with no predicted egg |
| T6 | T5 + fallback: if nothing remains, look for convex blobs in the inverted GrabCut mask | Hypothesis: GrabCut swapped foreground and background in some images | + complement | 0.428 / 0.487 / 0.651 | Discarded. 30 predictions still empty, IoU stayed low: the hypothesis was not supported |
| T7a | T5 + fallback: if the solidity filter removes everything, keep the blobs from before the filter | Fix the empty predictions of T5 | min_sol 0.9, fallback | 0.594 / 0.689 / 0.841 | Discarded: lower IoU than T7b |
| T7b | T7a with min_sol 0.8 | Less strict filter | min_sol 0.8, fallback | 0.600 / 0.697 / 0.841 | **Final.** Count accuracy +0.024 vs T4, IoU -0.005 (within noise on 126 images). Chosen because Task 4 needs the right number of eggs; T4 would also be defensible |

Not run: a GrabCut variant with a certain-foreground seed in the image centre was considered and
dropped before running, because the worst-case images have the egg at the frame edge, not the centre.

Final configuration: `--method grabcut --margin 0.02 --rel_area 0.3 --min_sol 0.8 --fallback`.
Final results (Task 3 evaluation): dev IoU 0.600 / Dice 0.697 / count acc 0.841; eval (run once)
IoU 0.560 / Dice 0.677 / count acc 0.875.

## Task 4 (video tracking)

| Trial | What changed | Why | Parameters | Result | Decision |
|---|---|---|---|---|---|
| V0 | Run the Task 2 algorithm (GrabCut) per frame, simple IoU tracker | Assignment asks to run the final algorithm | margin 0.02, stride 3, 320 px wide | On the title card it created IDs on the green background and on the logo. In the egg scene (about 10 to 41 s) the only blobs found had 38 to 63% of the frame area and were rejected by an area limit; no egg was output | Replaced. Likely cause (not verified): the near-full-frame rectangle and the dark table make GrabCut return a large dark region |
| V1 | Video-specific detector: HSV gate (V > 190, S < 45) for white eggs, logo corner masked, open/close, distance-transform watershed to split touching eggs, blob filter (area 0.4-35% of frame, centroid below 38% of the height, solidity >= 0.75). Tracker: Hungarian matching on (1 - mask IoU) + normalised centroid distance, match gate IoU >= 0.2 or distance <= 0.15 x diagonal | The egg scene has two white eggs on a dark table; thresholds were chosen by looking at 10 frames of this video (no ground truth available) | max_age 5, stride 2, 320 px wide | 9 unique IDs for 2 eggs | Replaced: every hand interaction broke the track |
| V2 | max_age 5 -> 100 | Occlusions by the hand last several seconds | max_age 100 | 3 unique IDs for 2 eggs | **Final** (`outputs/task4/tracked_final.csv`). Remaining failure: ID 3 starts at 22.5 s, when an egg is put down about 170 px away from where it was lifted, outside the matching gate |