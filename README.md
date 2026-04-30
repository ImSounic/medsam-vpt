# MedSAM-VPT: Visual Prompt Tuning vs Fine-Tuning for Medical Segmentation

University project — Advanced Computer Vision and Pattern Recognition.
Comparing parameter-efficient adaptation of MedSAM against full fine-tuning
on skin lesion segmentation, with a focus on out-of-distribution robustness.

## Quick start

```bash
# Activate your existing environment
conda activate mlenv  # or however you activate mlenv

# Install project-specific dependencies
pip install -r requirements.txt

# Download MedSAM weights (places file at checkpoints/medsam_vit_b.pth)
python scripts/download_medsam.py

# Sanity check: run zero-shot MedSAM on a single ISIC sample
python -m src.eval --config configs/zero_shot.yaml --quick

# (Once data is downloaded) Run zero-shot baseline on all four test sets
python -m src.eval --config configs/zero_shot.yaml
```

## Repository layout

```
medsam-vpt/
├── configs/             # YAML experiment configs (one per method)
├── src/
│   ├── data/            # Dataset loaders (ISIC, PH2, BUSI, Kvasir)
│   ├── models/          # MedSAM wrappers: VPT, LoRA, full FT
│   ├── train.py         # Training entry point
│   ├── eval.py          # Evaluation entry point
│   └── metrics.py       # Dice, IoU, HD95
├── scripts/             # One-off helpers (downloads, batch runs)
├── notebooks/           # Exploratory notebooks
├── checkpoints/         # MedSAM weights and trained model checkpoints (gitignored)
├── data/                # Datasets (gitignored)
├── results/             # Run logs and final tables
└── report/              # Final report drafts
```

## Methods compared

| Method | Trainable params | What changes |
|---|---|---|
| Zero-shot | 0 | Nothing — pure baseline |
| Decoder-only FT | ~4M | Mask decoder only |
| VPT-shallow | ~7.7K + decoder | Prompts at encoder input + decoder |
| VPT-deep | ~92K + decoder | Prompts at every encoder layer + decoder |
| LoRA (r=8) | ~1M + decoder | Low-rank attention adapters + decoder |
| Full FT | ~93M | Everything |

## Datasets

- **ISIC 2018 Task 1** — training + ID test (dermoscopy, lesion segmentation).
- **PH²** — near-OOD (different camera/site, same modality).
- **BUSI** — far-OOD (different modality: breast ultrasound).

## Hardware

Eval and the lightest method (decoder-only FT) run on an RTX 1000 Blackwell
(8 GB VRAM, mobile). VPT, LoRA, and full fine-tuning run on Colab T4 — see
[`colab/README.md`](colab/README.md) for the workflow.

## License / use

Educational project. Datasets retain their original licenses (ISIC CC-BY-NC, PH²
research-use, BUSI CC0).
