# ACCV TrustFMI experiments: status and handoff (for Carlo)

Updated 14 September 2026. Branch `accv-trustfmi` of the private GitHub repo
`ImSounic/medsam-vpt`.

Everything you run lives in your own home directory on your own cluster account:
your own clone of the repo, your own conda environment, your own copy of the
datasets, your own job submissions and checkpoints. Nothing is shared between
accounts and nothing here touches Sounic's directories. Results travel back through
the git branch only.

## Status

Done and committed on the branch: the three-seed CKA table with significance and the
six-curve robustness figure (`results/accv/cka_multiseed/`, `figures/accv/`); the
hook-placement ablation at seed 0 (`figures/accv/hook_ablation.*`); the adaptation
budget sweep at seed 0 (`figures/accv/budget_curves.*`, `results/accv/runs_t2.csv`);
failure detection for seven methods at tight boxes and 50 px jitter
(`results/accv/failure_detection_pm*/`); the DMID loader; and a full 8-page draft
(`paper/`, preview at `paper/main_llncs_preview.pdf`).

Running on Sounic's account: hook-ablation seeds 1 and 2 (T5) and the DMID
evaluation. Not needed from you.

Your part: the budget sweep for seeds 1 and 2 (T6), 18 runs of 15 to 35 minutes,
which turns the single-seed budget figure into a three-seed one. After that, a
reviewer-style read of the draft, transferring it into the official ACCV 2026
template to confirm the page count with line numbers, and the supplementary tables.

## 1. Clone into your home (head node)

```bash
cd ~ && git clone -b accv-trustfmi https://github.com/ImSounic/medsam-vpt.git medsam-vpt
cd ~/medsam-vpt
```

## 2. Your own environment

```bash
source /software/anaconda3/2025.06/etc/profile.d/conda.sh
conda create -n medsam-vpt python=3.10 -y
conda activate medsam-vpt
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts/download_medsam.py      # base weights -> checkpoints/medsam_vit_b.pth (375 MB)
```

The job scripts activate this environment by name (`conda activate medsam-vpt`), so
keep the name. Install everything on the head node: compute nodes have no internet,
so a missing package fails the job rather than being installed on the fly.

## 3. Download the datasets into your own checkout (copy-paste)

Total about 20 GB; every command was checked on 14 September 2026. Run them from
`~/medsam-vpt` on the head node. Downloads run on the head node without a GPU job.

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


## 4. Submit the T6 runs (your account)

```bash
cd ~/medsam-vpt && mkdir -p logs
T6=$(sbatch --parsable cka/slurm/accv_t6.sbatch)
sbatch --dependency=afterany:$T6 cka/slurm/accv_t6_eval.sbatch
squeue -u $USER --array
```

About 7 GPU-hours of training (two runs at a time under the student QOS, so roughly
4 hours of wall clock) plus about 2.5 hours of evaluation. Progress:
`tail -n 3 logs/accv_t6_*.out`. If a task fails, its `.err` file has the traceback;
resubmit one task with `sbatch --array=<n> cka/slurm/accv_t6.sbatch`.

## 5. Send results back

Results are small and go through git; checkpoints stay in your checkout.

```bash
cd ~/medsam-vpt && git add results/accv/runs_t6.csv results/accv/raw_t6 && git commit -m "results: budget sweep seeds 1-2" && git push
```

## 6. Afterwards: draft, template, supplement

- Read `paper/main_llncs_preview.pdf` as a reviewer would; anything that reads as
  overlap with the SAFER paper, or as too strong for one seed, is worth a comment.
- Move `paper/main.tex` and `paper/sections/*.tex` into the official ACCV 2026 LNCS
  template and report the page count with review line numbers on. Check the
  workshop's rules (page limit, whether references count, supplementary policy,
  OpenReview deadline time zone, any AI-use or data statement).
- Draft a supplement from the committed CSVs: full detector tables at both jitter
  levels, calibration and reliability files, the per-level robustness numbers
  (`results/accv/cka_multiseed/robustness_curves.csv`) and the budget table.

## Notes on DMID

- `TIFF Images/TIFF Images/IMG###.tif`: 8-bit RGBA, about 4750 x 6000; alpha dropped.
- `ROI Masks/ROI Masks/IMG###.tif`: binary 0/255 RGBA, LZW-compressed (needs
  `tifffile` and `imagecodecs`); multiple lesions already merged. 269 of 511 images have a
  mask; normals are skipped.
- `Pixel-level annotation/` holds contour drawings on the mammograms, not masks; unused.
- `Info.txt` is MIAS-style (class, benign/malignant, x, y, radius); unused.
