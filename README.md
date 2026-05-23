# MedSAM-VPT: Prompt Tuning vs Fine-Tuning for Medical Segmentation

**University project — Advanced Computer Vision and Pattern Recognition**

We compare six adaptation strategies for the **MedSAM** foundation model on
skin lesion segmentation, evaluating both in-distribution performance and
robustness under increasing distribution shift. The headline finding:
**parameter-efficient methods that aggressively modify the encoder (LoRA,
VPT) win in-distribution but collapse under extreme modality shift —
falling below zero-shot performance on mammography (CBIS-DDSM).** Methods
that leave the encoder mostly intact (Decoder-only FT, Full FT with gentle
LR) sacrifice a fraction of a Dice point on ID to retain robustness across
the drift ladder.

---

## Headline result

![Rank reversal across the drift ladder](results/figures/7_rank_reversal.png)

| Method | Trainable | ISIC (ID) Dice | PH² (near-OOD) Dice | BUSI (far-OOD: ultrasound) | CBIS-DDSM (far-OOD: mammography) |
|---|---:|---:|---:|---:|---:|
| Zero-shot | 0 | 0.9072 | 0.9054 | 0.8234 | 0.6924 |
| Decoder-only FT | 4.06 M | 0.9487 | 0.9466 | 0.8944 | **0.8273** |
| VPT-shallow | 4.07 M | 0.9457 | 0.9436 | 0.8112 | 0.6388 |
| VPT-deep | 4.15 M | 0.9472 | 0.9458 | 0.7995 | 0.5744 |
| LoRA (r=8) | 4.35 M | 0.9545 | 0.9557 | 0.7775 | **0.4979** |
| **Full FT** | 93.73 M | **0.9609** | **0.9583** | **0.8991** | **0.8280** |

**On ISIC (ID):** Full FT > LoRA > Decoder-only ≈ VPT-deep ≈ VPT-shallow > Zero-shot.
**On BUSI (ultrasound):** Full FT ≈ Decoder-only > Zero-shot > VPT-shallow > VPT-deep > LoRA.
**On CBIS-DDSM (mammography):** Full FT ≈ Decoder-only > **Zero-shot** > VPT-shallow > VPT-deep > LoRA.

The rank reversal is most dramatic on CBIS-DDSM mammography — the most
extreme modality shift in our drift ladder (visible-light dermoscopy →
X-ray). All three encoder-modifying methods (LoRA, VPT-deep, VPT-shallow)
fall **below zero-shot** performance there, with LoRA collapsing from
second-best on ID (0.9545) to worst overall (0.4979 — half of zero-shot).
On the milder ultrasound shift (BUSI), the same ordering holds but no
method falls below zero-shot. The explanation boils down to *where* and
*how aggressively* each method adapts the encoder.

See `results/figures/` for all eight plots and `results/summary_table.csv`
for the complete numeric table.

---

## Bbox prompt robustness — a second, harder finding

The standard eval above uses pixel-perfect bounding boxes derived from the
ground-truth mask. That's unrealistic — a clinician draws an approximate,
generous bbox, not a tight one. To test how each method holds up under
realistic prompt imprecision, we re-evaluate every model with bboxes whose
sides are independently expanded outward by 0–N px (random per image).
N ∈ {20, 50, 100, 200} px. Full details in `bbox_robustness/`.

The headline picture is in `bbox_robustness/results/figures/degradation_curves.png`.
A few of the numbers it summarises:

| | ISIC pm=0 → pm=200 | BUSI pm=0 → pm=200 | CBIS-DDSM pm=0 → pm=200 |
|---|:--:|:--:|:--:|
| Zero-shot | 0.907 → 0.767 (−0.140) | 0.823 → **0.602** | 0.692 → 0.189 |
| Full FT | 0.961 → 0.710 (−0.251) | 0.899 → 0.501 | 0.828 → 0.170 |
| Decoder-only | 0.949 → 0.708 | 0.894 → 0.489 | 0.827 → 0.157 |
| LoRA | 0.954 → **0.798** | 0.778 → 0.487 | 0.498 → **0.143** |
| VPT-deep | 0.947 → 0.726 | 0.800 → 0.470 | 0.574 → 0.134 |
| VPT-shallow | 0.946 → 0.731 | 0.811 → 0.489 | 0.639 → 0.146 |

Two patterns matter for the report:

- **Zero-shot is the most robust method at extreme bbox imprecision on
  ultrasound (BUSI).** All five trained methods fall harder than the
  untrained baseline. MedSAM's pretraining used a wider prompt distribution
  than our tight-bbox finetuning, so finetuning *removes* prompt-robustness
  capacity it already had.
- **On CBIS-DDSM mammography, every method collapses below 0.20 Dice at
  pm=200.** Far-OOD modality + heavily-perturbed prompts is essentially
  unsolvable with bbox prompting alone — that's the regime ceiling.

### Robustness fix attempt: train with bbox jitter

If the encoder learned to over-trust exact bbox boundaries, training with
jittered bboxes should fix it. We tested two training conditions on top
of the original `pm=0` (tight-bbox) training:

| Training | What it does | Configs | Checkpoints |
|---|---|---|---|
| `pm=0` (baseline) | every step uses the exact tight bbox | `configs/*.yaml` | `checkpoints/runs/` |
| `pm=20` (fixed jitter) | every step jitters each bbox corner uniformly in ±20 px | `configs/*_pm20.yaml` | `checkpoints/runs_pm20/` |
| `rand100` (random jitter) | every step samples `actual ~ U[0, 100]`, then jitters corners in ±actual | `configs/*_rand100.yaml` | `checkpoints/runs_rand100/` |

`rand100` mirrors how SAM/MedSAM were actually pretrained — a model
exposed to the full prompt-quality spectrum should ideally become
prompt-invariant. The eval results above repeat for both new trainings,
giving a 6 methods × 4 datasets × 5 perturb levels × 3 trainings =
360-cell experiment matrix.

The finding is a **clean trade-off curve**, not a uniform improvement:

> **Training on a wider bbox-jitter distribution trades far-OOD modality
> transfer for in-modality prompt robustness.** The wider the training
> jitter, the better the model becomes at handling sloppy prompts on
> dermoscopy, but the worse it becomes at handling far-OOD modalities
> at *any* prompt quality. Zero-shot MedSAM achieves both kinds of
> robustness simultaneously only because its pretraining used a much more
> diverse modality + prompt distribution than ISIC alone.

#### The headline numbers

**ISIC pm=200 (prompt robustness on the training modality):** rand100 is
a dramatic win across the board.

| Method | pm=0 train | pm=20 train | rand100 train |
|---|---:|---:|---:|
| Full FT | 0.710 | 0.725 | **0.839** (+0.129) |
| LoRA | 0.798 | 0.781 | **0.849** (+0.052) |
| Decoder-only | 0.708 | 0.741 | 0.778 (+0.070) |
| VPT-deep | 0.726 | 0.728 | 0.789 (+0.063) |
| VPT-shallow | 0.731 | 0.746 | 0.795 (+0.065) |
| *(Zero-shot reference)* | 0.767 | 0.767 | 0.767 |

With rand100 training, **all five trained methods beat zero-shot on ISIC
at pm=200**, which none of them could do under pm=0 or pm=20 training.

**CBIS-DDSM tight bbox (far-OOD modality transfer):** rand100 is a
catastrophe.

| Method | pm=0 train | pm=20 train | rand100 train |
|---|---:|---:|---:|
| Decoder-only | **0.827** | 0.787 | 0.653 (−0.174) |
| Full FT | **0.828** | 0.802 | 0.706 (−0.122) |
| VPT-deep | 0.574 | 0.577 | 0.405 (−0.169) |
| VPT-shallow | 0.639 | 0.417 | 0.250 (−0.389) |
| LoRA | 0.498 | 0.542 | 0.179 (−0.319) |

LoRA goes from 0.498 → 0.179 — a 64% relative drop just from changing
the training-time jitter distribution. The 5 trained models now all
score *below* zero-shot's CBIS-DDSM number (0.692) at tight bbox, where
under pm=0 training half of them were comfortably above it.

#### Why this happens (proposed mechanism)

Training pushes the encoder along a one-dimensional axis: as the
training prompt distribution widens, the encoder specializes harder
on the **single modality** it was finetuned on while gaining robustness
**within** that modality. Far-OOD modalities (ultrasound, mammography)
have no nearby points in the training distribution, so the further the
encoder moves along the prompt-robustness axis, the further it gets
from any signal that would help with the modality shift.

The original MedSAM zero-shot avoids this because its pretraining axis
was multi-dimensional: many modalities × many prompt qualities, with
the encoder forced to find features that generalize across both axes
simultaneously. Single-modality finetuning, no matter what prompt
distribution we use, cannot recover that.

#### Where to look

- `bbox_robustness/comparison/comparison_curves_3way.png` — overlay
  degradation curves: solid = pm=0 trained, dashed = pm=20 trained,
  dotted = rand100 trained, one panel per dataset.
- `bbox_robustness/comparison/delta_heatmap_3way.png` — per-cell ΔDice
  vs the pm=0 baseline. Top row = pm=20 effect, bottom row = rand100
  effect. Red = improvement, blue = regression.
- `bbox_robustness/comparison/tradeoff_id_vs_perturbation.png` — scatter
  of ISIC tight-bbox Dice vs ISIC pm=200 Dice. Each method's three
  trainings trace a trajectory; rand100 sits highest on the y-axis
  but moves slightly left on x.
- `bbox_robustness/comparison/tradeoff_isic_vs_cbis.png` — the modality
  trade-off: ISIC tight Dice vs CBIS-DDSM tight Dice. rand100 collapses
  the y-axis (CBIS) for every method.
- `summary_full.csv` at repo root — every metric across every (method,
  training, dataset, perturb level), 60 rows × 18 columns.

---

## Methods

All six methods share MedSAM ViT-B as the foundation model, the same
training data (ISIC 2018 Task 1, 2,594 dermoscopy images), the same loss
(½·BCE + ½·Dice), the same optimiser (AdamW with cosine LR), and the same
ground-truth-derived bounding-box prompts during forward passes. They
differ only in *which parameters are trainable*.

| Method | Trainable parameters | What's modified | Source |
|---|---|---|---|
| `zero_shot` | 0 | Nothing — pure inference | `configs/zero_shot.yaml` |
| `decoder_only` | 4,058,340 (4.33%) | Mask decoder only | `src/models/decoder_only.py` |
| `vpt_shallow` | 4,066,020 (4.34%) | 10 prompt tokens at encoder input + decoder | `src/models/vpt.py` |
| `vpt_deep` | 4,150,500 (4.43%) | 10 prompts at every transformer block + decoder | `src/models/vpt.py` |
| `lora` (r=8) | 4,353,252 (4.65%) | Low-rank residuals on encoder Q/V + decoder | `src/models/lora.py` |
| `full_ft` | 93,729,252 (99.99%) | Image encoder + mask decoder (LR=1e-5) | `src/models/full_ft.py` |

Implementation notes:

- **VPT** uses an additive-perturbation adaptation of Jia et al. (ECCV 2022)
  suitable for SAM's 2D-arranged ViT-B with window attention and relative
  positional embeddings. Token-prepending would require modifying SAM's
  attention pathway; additive perturbation matches the parameter count and
  per-layer modulation structure without architectural surgery.
- **LoRA** is implemented from scratch (no `peft` dependency) so the cluster
  environment doesn't have to drag in `transformers`/`tensorflow`. See
  `src/models/lora.py`.
- **Full FT** uses LR=1e-5 (50× lower than the PEFT methods) to keep
  encoder drift gentle.

---

## Datasets

| Dataset | Role | Size | Modality | Shift type |
|---|---|---|---|---|
| ISIC 2018 Task 1 | Train + ID test | 2,594 / 1,000 | Dermoscopy (RGB) | None |
| PH² | Near-OOD test | 200 | Dermoscopy (RGB) | Acquisition (different hospital, camera, cohort) |
| BUSI | Far-OOD test | 647 (487 benign + 210 malignant) | Breast ultrasound (greyscale) | Modality |
| CBIS-DDSM | Far-OOD test | 362 (test split, mass + calcification) | Mammography (X-ray) | Modality (most extreme) |

ISIC is downloaded from the ISIC Challenge archive. PH² is downloaded from
Kaggle (`athina123/ph2dataset`) — the original ADDI FTP is unreliable.
BUSI is from the Kaggle mirror (`sabahesaraki/breast-ultrasound-images-dataset`)
of the Cairo University release (Al-Dhabyani et al. 2020). CBIS-DDSM is
from Kaggle (`awsaf49/cbis-ddsm-breast-cancer-image-dataset`), the Curated
Breast Imaging Subset of DDSM (Lee et al. 2017). Loader filters to the
official test split via PatientID and OR-s multi-lesion masks together.

---

## Reproducing the results

### 1. Environment

```bash
# Activate your Python environment (the project uses a venv called mlenv)
mlenv

# Install dependencies (segment_anything, monai for HD95, etc.)
pip install -r requirements.txt
```

### 2. Download MedSAM weights

```bash
python scripts/download_medsam.py
# Pulls medsam_vit_b.pth (~358 MB) from Zenodo
```

### 3. Place datasets

```
data/
├── train_images/, train_masks/    # ISIC 2018 train split
├── val_images/,   val_masks/      # ISIC 2018 validation split
├── test_images/,  test_masks/     # ISIC 2018 test split
├── ph2/
│   ├── trainx/                    # 200 dermoscopy images (.bmp)
│   └── trainy/                    # 200 lesion masks (.bmp)
└── busi/
    ├── benign/, malignant/, normal/   # Image + mask pairs (.png)
```

### 4. Train

Each method has its own config. Training writes `checkpoints/runs/<name>/best.pth`
and `train_log.csv`.

```bash
python -m src.train --config configs/decoder_only.yaml
python -m src.train --config configs/vpt_shallow.yaml
python -m src.train --config configs/vpt_deep.yaml
python -m src.train --config configs/lora.yaml
python -m src.train --config configs/full_ft.yaml      # needs ≥16 GB GPU
```

Resume support is built in:

```bash
python -m src.train --config configs/lora.yaml --resume
```

### 5. Evaluate

```bash
# Zero-shot (no checkpoint required)
python -m src.eval --config configs/zero_shot.yaml

# Trained checkpoints — appends a row to results/runs.csv per dataset
python -m src.eval --config configs/zero_shot.yaml \
    --checkpoint checkpoints/runs/lora_seed0/best.pth
```

Eval runs on every dataset listed in `configs/zero_shot.yaml`'s `test_sets:`
block. Toggle datasets there to skip BUSI, PH², or ISIC test.

### 6. Generate plots

```bash
python scripts/plots.py
# Produces 8 figures in results/figures/ + results/summary_table.csv
```

---

## Repository layout

```
medsam-vpt/
├── configs/                  # One YAML per training method
│   ├── decoder_only.yaml
│   ├── vpt_shallow.yaml
│   ├── vpt_deep.yaml
│   ├── lora.yaml
│   ├── full_ft.yaml
│   └── zero_shot.yaml        # Also used as the eval config
├── src/
│   ├── data/                 # ISIC, PH², BUSI dataset classes
│   ├── models/               # Method wrappers (one file per adaptation strategy)
│   ├── train.py              # Training entry point (resume-capable)
│   ├── eval.py               # Evaluation entry point
│   ├── losses.py             # DiceBCELoss
│   ├── metrics.py            # Dice, IoU, HD95
│   └── ...
├── scripts/
│   ├── download_medsam.py    # Pulls MedSAM weights from Zenodo
│   └── plots.py              # Builds all report figures
├── colab/
│   ├── train.ipynb           # Bootstrap notebook for Colab training
│   └── README.md             # Colab workflow notes
├── results/
│   ├── runs.csv              # Master results log (committed)
│   ├── summary_table.csv     # Pivoted summary table
│   ├── figures/              # 8 PNG figures for the report
│   └── raw/                  # Per-image CSVs (gitignored)
├── checkpoints/              # Trained model checkpoints (gitignored)
├── data/                     # Datasets (gitignored)
├── requirements.txt
└── README.md
```

---

## Figures

All eight figures are committed under `results/figures/`:

| File | What it shows |
|---|---|
| `1_dice_per_dataset.png` | Per-panel Dice bars with value labels for each dataset |
| `2_pareto_id_vs_ood.png` | Dice vs trainable parameters, separately for ID and far-OOD |
| `3_drift_gap.png` | ID−OOD Dice gap per method |
| `4_hd95_split.png` | Boundary error (HD95) for skin domain vs ultrasound, native scales |
| `5_id_vs_ood_scatter.png` | ID Dice vs OOD Dice with y=x diagonal showing drift cost |
| `6_drift_curves.png` | One line per method across the drift ladder |
| `7_rank_reversal.png` | Bumps chart: method ranks change as drift increases |
| `8_summary_heatmap.png` | Heatmap of Dice for methods × datasets |

---

## Hardware notes

Training was performed across three environments:

- **Local laptop** (RTX 1000 Blackwell, 8 GB VRAM) — Decoder-only FT.
  All evaluation runs (~8 min each).
- **Google Colab T4** (16 GB VRAM) — VPT-shallow training.
- **University JupyterLab A16** (16 GB virtual GPU) — VPT-deep, LoRA, Full FT.

The training loop supports gradient checkpointing (VPT) and gentle
learning rates (Full FT) to fit ViT-B encoder gradients into 16 GB. See
`configs/*.yaml` for memory-aware settings.

---

## License

Code: educational / research use. Datasets retain their original licenses:
- ISIC 2018 (CC-BY-NC)
- PH² (research-use, Univ. of Porto / ADDI)
- BUSI (CC0, Cairo University)

MedSAM weights from Ma et al. (2024), distributed via Zenodo.

---

## Acknowledgements

- **MedSAM** — Ma et al., 2024. https://github.com/bowang-lab/MedSAM
- **SAM** — Kirillov et al., Meta AI, 2023.
- **VPT** — Jia et al., ECCV 2022.
- **LoRA** — Hu et al., 2021.
- **ISIC** — Codella et al., 2018.
- **PH²** — Mendonça et al., 2013.
- **BUSI** — Al-Dhabyani et al., 2020.
