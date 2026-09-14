# DMID evaluation handoff (for Carlo)

Goal: evaluate zero-shot MedSAM and the adapted checkpoints on the DMID mammography
set (269 annotated images, tight boxes, plus per-image decoder drift). Evaluation
only, no training. Kill date for including DMID in the ACCV paper: 16 September.

## What you receive

- Collaborator access to the private GitHub repo `ImSounic/medsam-vpt`, branch
  `accv-trustfmi`. It contains the DMID loader (`src/data/dmid.py`), the eval config
  (`configs/accv_dmid_eval.yaml`), the SLURM job (`cka/slurm/accv_dmid.sbatch`) and all
  result CSVs so far. Results you produce go back on the same branch (see step 5).
- The data and checkpoints by one of the two routes in step 2.

## 1. Clone and environment (head node hpc-head2)

```bash
git clone -b accv-trustfmi https://github.com/ImSounic/medsam-vpt.git ~/medsam-vpt
cd ~/medsam-vpt
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda create -n medsam-vpt python=3.10 -y
conda activate medsam-vpt
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` includes `tifffile` and `imagecodecs`; the DMID masks are
LZW-compressed TIFFs that PIL cannot decode.

## 2. Data and checkpoints

DMID evaluation needs `data/dmid` (8.3 GB), `checkpoints/medsam_vit_b.pth` (375 MB)
and the checkpoints listed in `cka/slurm/accv_dmid.sbatch` (about 1 GB). Sounic will
tell you which route applies.

**Route A: shared group directory on the cluster** (if `/projects/<dmb group>` exists).
Sounic places `data/` and `checkpoints/` there; you symlink:

```bash
cd ~/medsam-vpt
SHARED=/projects/<path Sounic gives you>/medsam-vpt
ln -s "$SHARED/data" data
ln -s "$SHARED/checkpoints" checkpoints
ls data/dmid/ && ls checkpoints/runs/
```

**Route B: file transfer.** Sounic shares an archive (SURFdrive or Google Drive link);
download it on the head node and unpack into the checkout:

```bash
cd ~/medsam-vpt
curl -L -o dmid_bundle.tar "<link>"
tar xf dmid_bundle.tar      # creates data/dmid and checkpoints/...
ls data/dmid/ && ls checkpoints/runs/
```

## 3. Nothing to do here

(Kept so the section numbers match earlier messages.)

## 4. Smoke test, then submit

Quick check on the head node without a GPU (about a minute, 8 images, CPU):

```bash
python -m src.eval --config configs/accv_dmid_eval.yaml --quick --device cpu --results-csv /tmp/dmid_smoke.csv --per-image-dir /tmp/dmid_smoke
```

Expected: one `[eval] dmid: dice=...` line. Then submit:

```bash
mkdir -p logs && sbatch cka/slurm/accv_dmid.sbatch
```

Runtime about 1 to 2 hours (12 evaluations of 269 images, two model passes each
because of drift). Progress: `tail -f logs/accv_dmid_<jobid>.out`.

## 5. Sanity criterion and outputs

Zero-shot is the first evaluation in the log. CBIS-DDSM zero-shot Dice is 0.69; a
DMID zero-shot between about 0.55 and 0.85 is sane. Below 0.4 means a mask or
orientation problem; stop and report.

Outputs: `results/accv/runs_dmid.csv` (one row per checkpoint) and the directory
`results/accv/raw_dmid/` (per-image Dice, IoU, HD95, iou_pred, drift). Commit and push
them on the branch:

```bash
cd ~/medsam-vpt && git add results/accv/runs_dmid.csv results/accv/raw_dmid && git commit -m "results: DMID evaluation" && git push
```

## Notes on the dataset

- `TIFF Images/TIFF Images/IMG###.tif`: 8-bit RGBA, about 4750 x 6000; alpha dropped.
- `ROI Masks/ROI Masks/IMG###.tif`: binary 0/255 RGBA, LZW; multiple lesions are
  already merged in one mask. 269 of 511 images have a mask; the rest are normals and
  are skipped.
- `Pixel-level annotation/` holds the mammograms with drawn contours, not masks; unused.
- `Info.txt` is MIAS-style (class, benign/malignant, x, y, radius); unused by the loader.

## 6. Optional: run the T5 seeds while Sounic's account is upgraded

T5 trains seeds 1 and 2 of the two CKA hook-ablation arms ("both" and "enc"), about
7 to 8 hours each on an L40S, followed by their evaluation. Only if you have GPU
hours to spare; the "both" seeds (tasks 0 and 1) matter most.

```bash
cd ~/medsam-vpt && mkdir -p logs
T5=$(sbatch --parsable cka/slurm/accv_t5.sbatch)
sbatch --dependency=afterany:$T5 cka/slurm/accv_t5_eval.sbatch
squeue -u $USER --array
```

T5 also needs the ISIC, BUSI and CBIS-DDSM data (route A gives them automatically).
Checkpoints land in `checkpoints/runs_accv_t5/` inside your checkout (gitignored; Sounic
will fetch them if needed). Commit and push the results:

```bash
cd ~/medsam-vpt && git add cka/results/runs_accv_t5.csv bbox_robustness/results_accv_t5/runs.csv results/accv/raw_t5 results/accv/runs_cka_pm0_retrained.csv results/accv/raw_cka_pm0_retrained && git commit -m "results: T5 hook-ablation seeds" && git push
```

