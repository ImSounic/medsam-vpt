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
3. Carlo: budget sweep seeds 1 and 2 (T6, section below).
4. Co-author read of the draft (`paper/main_llncs_preview.pdf`), template transfer, supplementary tables.

## Your compute task: budget sweep seeds 1 and 2 (T6)

The adaptation-budget result (Sect. 6 of the draft) is single-seed. Eighteen short
runs (three methods x three budgets x seeds 1 and 2, 15 to 35 minutes each) turn it
into a three-seed figure. Configs are `configs/budget/*_seed{1,2}.yaml`; the image
subsets are identical to seed 0, only the model initialisation and data order change.

### Data you need in your own checkout (copy-paste, run on the head node)

Sounic's home is not group-readable, so the datasets are downloaded fresh. Total
about 20 GB; every command below was checked on 14 September 2026. Run them from
`~/medsam-vpt`.

**ISIC 2018 Task 1** (six public archives on the challenge's S3 bucket; the loader
expects `data/{train,val,test}_{images,masks}/`):

```bash
mkdir -p data && cd data
B=https://isic-challenge-data.s3.amazonaws.com/2018
for f in ISIC2018_Task1-2_Training_Input ISIC2018_Task1_Training_GroundTruth \
         ISIC2018_Task1-2_Validation_Input ISIC2018_Task1_Validation_GroundTruth \
         ISIC2018_Task1-2_Test_Input ISIC2018_Task1_Test_GroundTruth; do
  curl -L --retry 5 -C - -o $f.zip $B/$f.zip && unzip -q $f.zip && rm $f.zip
done
mv ISIC2018_Task1-2_Training_Input train_images;    mv ISIC2018_Task1_Training_GroundTruth train_masks
mv ISIC2018_Task1-2_Validation_Input val_images;    mv ISIC2018_Task1_Validation_GroundTruth val_masks
mv ISIC2018_Task1-2_Test_Input test_images;         mv ISIC2018_Task1_Test_GroundTruth test_masks
cd ..
```

**BUSI and CBIS-DDSM** (Kaggle; needs `pip install kaggle` and an API token in
`~/.kaggle/kaggle.json` from your Kaggle account page):

```bash
kaggle datasets download -d aryashah2k/breast-ultrasound-images-dataset -p data/tmp_busi --unzip
mv data/tmp_busi/Dataset_BUSI_with_GT data/busi && rm -rf data/tmp_busi
kaggle datasets download -d awsaf49/cbis-ddsm-breast-cancer-image-dataset -p data/cbis-ddsm --unzip
ls data/cbis-ddsm/csv/dicom_info.csv data/cbis-ddsm/jpeg | head -3
```

**PH2** (200 images, 265 MB): request it from the official ADDI page
(https://www.fc.up.pt/addi/ph2%20database.html, RAR archive; the loader accepts the
original per-image folder layout under `data/ph2/`), or use the copy Sounic sends you
and unpack it to `data/ph2/`.

**Base weights:** `python scripts/download_medsam.py` writes
`checkpoints/medsam_vit_b.pth` (375 MB). No trained checkpoints are needed for T6;
it trains its own.

Expected file counts: `train_images` 2595, `train_masks` 2595, `val_images` 101,
`val_masks` 101, `test_images` 1001, `test_masks` 1001 (each image folder holds one
`LICENSE.txt`), `ph2` 400, `busi` 1578, `cbis-ddsm` 10243:

```bash
for d in train_images train_masks val_images val_masks test_images test_masks ph2 busi cbis-ddsm; do printf "%-12s %6d\n" $d "$(find data/$d -type f | wc -l)"; done
```

Loader sanity check (CPU, about a minute), expected `2594 647 362 2458 200`:

```bash
python -c "from src.data.isic import ISIC2018; from src.data.busi import BUSI; from src.data.cbis_ddsm import CBISDDSM; from src.data.ph2 import PH2; print(len(ISIC2018('data','train')), len(BUSI('data/busi')), len(CBISDDSM('data/cbis-ddsm','test')), len(CBISDDSM('data/cbis-ddsm','train')), len(PH2('data/ph2')))"
```

### Submit

```bash
cd ~/medsam-vpt && mkdir -p logs
T6=$(sbatch --parsable cka/slurm/accv_t6.sbatch)
sbatch --dependency=afterany:$T6 cka/slurm/accv_t6_eval.sbatch
squeue -u $USER --array
```

About 7 GPU-hours of training (two at a time under the student QOS, so roughly 4
hours of wall clock) plus about 2.5 hours of evaluation. Results to commit and push:

```bash
git add results/accv/runs_t6.csv results/accv/raw_t6 && git commit -m "results: budget sweep seeds 1-2" && git push
```

Checkpoints in `checkpoints/runs_accv_t6/` stay in your checkout (gitignored).

## If you need to run something else

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
