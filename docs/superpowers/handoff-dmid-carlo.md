# DMID evaluation handoff (for Carlo)

Goal: evaluate zero-shot MedSAM and the adapted checkpoints on the DMID mammography
set (269 annotated images, tight boxes, plus per-image decoder drift). Evaluation
only, no training. Kill date for including DMID in the ACCV paper: 16 September.

## What you receive

- `medsam-vpt-accv-trustfmi.bundle`: the repo, branch `accv-trustfmi`, with the DMID
  loader (`src/data/dmid.py`), the eval config (`configs/accv_dmid_eval.yaml`) and the
  SLURM job (`cka/slurm/accv_dmid.sbatch`).
- Read access to Sounic's data and checkpoints on the cluster (see step 3), so nothing
  large has to be copied.

## 1. Clone and environment (head node hpc-head2)

```bash
git clone -b accv-trustfmi ~/medsam-vpt-accv-trustfmi.bundle ~/medsam-vpt
cd ~/medsam-vpt
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda create -n medsam-vpt python=3.10 -y
conda activate medsam-vpt
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt` includes `tifffile` and `imagecodecs`; the DMID masks are
LZW-compressed TIFFs that PIL cannot decode.

## 2. Data and checkpoints (symlinks into Sounic's home)

```bash
cd ~/medsam-vpt
mkdir -p data checkpoints
ln -s /home/s3702111/medsam-vpt/data/dmid data/dmid
ln -s /home/s3702111/medsam-vpt/checkpoints/medsam_vit_b.pth checkpoints/medsam_vit_b.pth
for d in runs runs_perfect_bboxes runs_cka_oodonly_late runs_accv_t1; do
  ln -s /home/s3702111/medsam-vpt/checkpoints/$d checkpoints/$d
done
ls data/dmid/ && ls checkpoints/runs/
```

If the `ls` fails with permission denied, Sounic still has to run step 3.

## 3. (Sounic) open read access

```bash
chmod o+x ~ ~/medsam-vpt ~/medsam-vpt/data ~/medsam-vpt/checkpoints
chmod -R o+rX ~/medsam-vpt/data/dmid ~/medsam-vpt/checkpoints
```

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

Outputs to send back (small): `results/accv/runs_dmid.csv` (one row per checkpoint)
and the directory `results/accv/raw_dmid/` (per-image Dice, IoU, HD95, iou_pred, drift).

```bash
tar czf ~/dmid_results.tar.gz -C ~/medsam-vpt results/accv/runs_dmid.csv results/accv/raw_dmid
```

## Notes on the dataset

- `TIFF Images/TIFF Images/IMG###.tif`: 8-bit RGBA, about 4750 x 6000; alpha dropped.
- `ROI Masks/ROI Masks/IMG###.tif`: binary 0/255 RGBA, LZW; multiple lesions are
  already merged in one mask. 269 of 511 images have a mask; the rest are normals and
  are skipped.
- `Pixel-level annotation/` holds the mammograms with drawn contours, not masks; unused.
- `Info.txt` is MIAS-style (class, benign/malignant, x, y, radius); unused by the loader.
