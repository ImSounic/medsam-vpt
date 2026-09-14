# ACCV TrustFMI experiments: status and handoff (for Carlo)

Updated 14 September 2026. Branch `accv-trustfmi` of the private repo
`ImSounic/medsam-vpt`. All result CSVs and figures live on the branch, so `git pull`
is enough to follow progress; nothing below requires cluster access unless a job fails.

## Done (results committed)

| Item | Where |
|---|---|
| CKA late_l10 pm20, seeds 0 to 2: tight-box table, paired Wilcoxon, six-curve robustness figure | `results/accv/cka_multiseed/`, `figures/accv/robustness_six_curves.png` |
| Hook-placement ablation, seed 0 (no CKA / decoder / encoder / both) | `figures/accv/hook_ablation.{png,csv}` |
| Adaptation-budget sweep, 3 methods x {50, 250, 1000, 2595} images | `figures/accv/budget_curves.{png,csv}`, `results/accv/runs_t2.csv` |
| Failure detection (drift vs IoU head), 7 methods, tight boxes and 50 px jitter | `results/accv/failure_detection_pm0/`, `results/accv/failure_detection_pm50/`, `figures/accv/failure_detection_pm*.png` |
| DMID loader and eval config | `src/data/dmid.py`, `configs/accv_dmid_eval.yaml` |

Headline numbers so far (tight boxes, Dice): CKA pm20 beats no-CKA LoRA by +0.05 on
BUSI and +0.10 on CBIS-DDSM over three seeds (Holm-corrected p < 1e-4 at 20 px jitter);
hooks on encoder and decoder together reach 0.861 / 0.725 on BUSI / CBIS at seed 0;
decoder drift flags LoRA failures with AUROC 0.95 / 0.82 while the IoU head is at or
below chance for PEFT methods.

## Running now (Sounic's account, all in parallel since the QOS upgrade)

| Job | What | Expected end |
|---|---|---|
| 587519 (array of 4) + 587520 | T5: seeds 1 and 2 of the "both" and "encoder" hook arms, then tight eval, bbox sweep, and a tight eval of the retrained pm=0 CKA checkpoint | about 8 h after start, eval 2 to 3 h more |
| 587523 | DMID: zero-shot plus 11 checkpoints, tight boxes, with drift | 1 to 2 h |

## Left after those finish

1. Pull results to the laptop, rerun `scripts/plot_hook_ablation.py` (extend to mean and
   std over seeds), `scripts/accv_cka_multiseed.py`, and `scripts/failure_detection.py`
   on `results/accv/raw_dmid`.
2. DMID decision (kill date 16 September): keep it if zero-shot Dice is sane (about
   0.55 to 0.85) and the method ranking matches CBIS-DDSM; otherwise drop it.
3. Co-author check of the result set, then writing from 19 September.

## If you need to run something yourself

Clone and environment on the head node:

```bash
git clone -b accv-trustfmi https://github.com/ImSounic/medsam-vpt.git ~/medsam-vpt
cd ~/medsam-vpt
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda create -n medsam-vpt python=3.10 -y && conda activate medsam-vpt
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Data and checkpoints are only in Sounic's home (`/home/s3702111/medsam-vpt/data`,
`.../checkpoints`), which is not group-readable, so ask him for a transfer before
submitting anything. Jobs are `cka/slurm/accv_*.sbatch`; each header lists what it
needs. Results are committed and pushed on this branch:

```bash
git add results/accv cka/results figures/accv && git commit -m "results: <what>" && git push
```

## Notes on DMID

- `TIFF Images/TIFF Images/IMG###.tif`: 8-bit RGBA, about 4750 x 6000; alpha dropped.
- `ROI Masks/ROI Masks/IMG###.tif`: binary 0/255 RGBA, LZW-compressed (needs
  `tifffile` and `imagecodecs`); multiple lesions already merged. 269 of 511 images have a
  mask; normals are skipped.
- `Pixel-level annotation/` holds contour drawings on the mammograms, not masks; unused.
- `Info.txt` is MIAS-style (class, benign/malignant, x, y, radius); unused.
