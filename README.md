# Egg segmentation and tracking

Nash Robotics take-home. Task 1: segmentation with a pretrained model (Grounding DINO + SAM).
Task 2: classical image-processing segmentation (GrabCut + shape filters). Task 3: evaluation and
analysis. Task 4: video tracking of individual eggs.

- Report: `report/report.pdf`
- Experiment log (Task 2 and Task 4 trials): `experiments/experiment_log.md`
- Tracked video output (Task 4): <https://drive.google.com/file/d/1Le9tQuM2JbNpxv25Q5JKKlrfJSdOELYF/view?usp=sharing> 

## Setup

Tested on Windows (PowerShell), Python 3.10, torch 2.14.1+cu126, transformers 5.20.2, NVIDIA RTX 4050.

    python -m venv .venv
    .venv\Scripts\Activate.ps1              # Linux/macOS: source .venv/bin/activate
    # Install PyTorch first, using the command for your CUDA version from pytorch.org. Example (CUDA 12.6):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
    pip install -r requirements.txt
    pip install -e .

The Task 1 models (Grounding DINO tiny, SAM ViT-B) are downloaded from Hugging Face on first use.

## Data
Dataset: Roboflow Universe, `fire-bf7qe/egg-vzhtr`, version 3: <https://universe.roboflow.com/fire-bf7qe/egg-vzhtr/dataset/3>
Download the Roboflow dataset in "Folder Structure" format and extract it so that `data/raw/`
contains `train/`, `valid/` and `test/`. The Roboflow split is not used. All images are merged into
one folder and the dev/eval split is made by us (Task 2).

    python -m eggseg.data                   # -> data/images/, data/manifest.csv

Raw images are not committed. `data/split.csv` (the dev/eval split) is committed.

## Task 1: Grounding DINO + SAM masks

    python -m eggseg.gsam_seg               # -> outputs/task1/masks, overlays, summary

Text prompt "egg." gives boxes (Grounding DINO), the boxes prompt SAM (one mask per box).
`outputs/task1/masks/<image>.png` is a uint16 label image: 0 = background, 1..N = egg id (so it looks
black in an image viewer). Some images were re-run with other thresholds, see the report.
`outputs/exclude.txt` lists 2 images whose masks were judged unusable. Their masks stay in
`outputs/task1/masks/`, but the images are left out of the dev/eval split.

## Task 2: split and classical algorithm

    python -m eggseg.split                  # group-wise 4:1 split -> data/split.csv (seed 42)

Near-duplicate images are grouped with a 64-bit dHash (Hamming distance <= 6) plus one manual merge
of three images from the same scene, and groups are stratified by egg count. Do not re-run the split
after experiments have started, because it changes the dev/eval assignment.

Final algorithm (config T7b: GrabCut, 2% margin, relative-area filter 0.3, solidity filter 0.8 with
fallback). Dev split:

    python -m eggseg.classical_seg --method grabcut --margin 0.02 --rel_area 0.3 --min_sol 0.8 --fallback --split dev --out outputs/task2/final_dev

Eval split (run once, after the algorithm was frozen):

    python -m eggseg.classical_seg --method grabcut --margin 0.02 --rel_area 0.3 --min_sol 0.8 --fallback --split eval --out outputs/task2/final_eval

The first GrabCut run takes about 6 s per image. Raw GrabCut masks are cached in
`outputs/task2/raw_grabcut_m<margin>_i<iters>/`, so later runs only repeat the post-processing.
The commands for all trials are in `experiments/experiment_log.md`.

## Task 3: evaluation (Task 1 masks = ground truth, Task 2 masks = prediction)

    python -m eggseg.evaluate --split eval --pred outputs/task2/final_eval
    python -m eggseg.evaluate --split dev  --pred outputs/task2/final_dev

Writes metrics, an IoU histogram, a precision/recall plot and an error-map sheet of the worst images
to `outputs/task3/<split>/`.

## Task 4: video tracking

The test video is from YouTube (https://www.youtube.com/watch?v=Y4JyV94y2_A). I downloaded it
manually with the VidSave web tool and saved it as `data/video/egg.mp4`.

Alternatively, you can fetch an equivalent copy (up to 720p) with yt-dlp:

    yt-dlp -f "bv*[height<=720]+ba/b[height<=720]" --merge-output-format mp4 -o "data/video/egg.%(ext)s" "https://www.youtube.com/watch?v=Y4JyV94y2_A"

Then run the tracker:

    python -m eggseg.track --out outputs/task4/tracked_final.mp4

Per-frame results are written next to the video as a CSV (`outputs/task4/tracked_final.csv` is
committed, the video is not). The detector is a video-specific modification of the Task 2 algorithm,
see the report. `--method grabcut` runs the unmodified Task 2 algorithm on the video instead.

## Repository layout

    src/eggseg/     data.py, gsam_seg.py, split.py, classical_seg.py, evaluate.py, track.py
                    (sam_seg.py is the first attempt, see Notes)
    scripts/        helper scripts for contact sheets and diagnostics
    data/           split.csv, manifest.csv (raw images and video are not committed)
    outputs/        task1 masks (ground truth), task2 final masks, task3 figures, task4 tracking log
    experiments/    experiment_log.md
    report/         report.pdf and figures


## Notes

- `src/eggseg/sam_seg.py` is the first attempt (SAM automatic mask generation + filters). It was
  replaced by `gsam_seg.py`, because the automatic mode returns every segment and cannot tell which
  one is an egg. The final Task 1 masks come from `gsam_seg.py`.
- Helper scripts: `peek.py` (contact sheet of random images), `sheet_list.py` / `sheet_overlays.py`
  (contact sheets of a list of images / overlays), `viewmasks.py` (writes coloured copies of the
  label masks to `outputs/task1/masks_vis/`, not committed), `rebuild_summary.py` (egg counts from
  the mask files), `compare.py` and `diag.py` (worst-case sheet and over/under-segmentation counts
  for a Task 2 run), `peek_video.py`, `video_sheet.py`, `ids_sheet.py` (Task 4 contact sheets).
- `outputs/task2/*/dev_metrics.csv` and `eval_metrics.csv` are quick metrics written by
  `classical_seg.py`. The full Task 3 metrics are in `outputs/task3/`.

## AI tools

Claude (Anthropic) was used for code drafting, debugging, and discussing methods. Parameter choices,
visual checks of masks and failure cases, and the decisions in the experiment log were reviewed and
made by the author.