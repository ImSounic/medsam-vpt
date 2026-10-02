# Handoff

## ⚠️ Machine switch — 2026-09-15 (read this first)

**The Linux laptop (CachyOS, `/home/imsounic/Projects/`) is out for screen repair. All work happens on a Mac
for the next 1–2 weeks (roughly until early October 2026).** Update this section when the laptop is back.

- **Where the code is:** a full copy of `~/Projects` and `~/University` was made on 2026-09-15 onto the external
  1 TB exFAT drive (volume `5497-ED62`) at `<drive>/Projects/<project>/` and `<drive>/University/`. The originals
  on the Linux laptop were left untouched. When the laptop returns, reconcile via git (preferred) or rsync back.
- **Before working on the Mac:** copy the project from the drive onto the Mac's internal disk. Do not run git or
  build tools directly on exFAT — it is slow, case-insensitive, and stores no symlinks or executable bits, so
  `git status` may show spurious mode changes (`git config core.fileMode false` silences that).
- **Not copied (rebuild on the Mac):** `node_modules/`, `.next/`, `.venv/`, `venv/`, `__pycache__/`,
  `.mypy_cache/`, `.pytest_cache/`, `.cache/`. Linux-built binaries would not run on macOS anyway.
- **No CUDA on the Mac:** anything GPU-bound goes to the HPC/Colab or runs on MPS/CPU.
- **Git state at the time of the copy:** branch `accv-trustfmi`, **diverged from origin**: 5 local commits (T7 drift dumps, DMID held-out set, hook-ablation seeds, paper text) vs 2 remote commits pushed from elsewhere (budget sweep seeds 1–2, 3-seed budget graphs with mean/std bands). The 5 local commits are safe on GitHub as branch `accv-trustfmi-laptop-2026-09-15` (pushed 2026-09-15). **Merging is still TODO**: `scripts/plot_budget.py` conflicts in 3 hunks (both sides rewrote the plotting body) and `figures/accv/budget_curves.png` conflicts (binary — just regenerate it after merging). On the Mac: `git fetch`, then merge or rebase `accv-trustfmi-laptop-2026-09-15` onto `origin/accv-trustfmi`, resolve `plot_budget.py` by hand, rerun it, push. 33 uncommitted (`results/summary_full.csv` modified; `PROJECT_STATE.md` and `bbox_robustness/results*/` untracked).
- **Project-specific:** `data/` (28 GB: isic train/val/test, ph2, busi, cbis-ddsm, dmid) and `checkpoints/` (3.3 GB, incl. `runs/full_ft_seed0.zip`) are on the drive. Do not copy them onto the Mac's disk unless needed — configs use relative `data/<set>` roots, so either symlink `data` → the drive or run with cwd on the drive. No CUDA on the Mac: `device_utils.py` falls back to MPS/CPU (AMP off), which is fine for eval/CKA; full FT needs ≥16 GB VRAM → HPC/Colab. Related files copied to the drive root: `Projects/accv-trustfmi-overleaf.zip` (Overleaf export, 2026-09-15) and `Projects/ph2_for_carlo.tgz` (212 MB, 2026-09-14).

## Project Goal

Compare fine-tuning strategies for MedSAM (a ViT-B foundation model for bbox-prompted medical image segmentation) under joint domain shift and bounding-box prompt perturbation. The project trains on ISIC 2018 dermoscopy and evaluates on a four-dataset drift ladder: ISIC (in-domain), PH2 (close-OOD dermoscopy), BUSI (far-OOD ultrasound), CBIS-DDSM (far-OOD mammography X-ray). It also investigates CKA-based regularization to preserve decoder representations during adaptation. Built for a university Advanced Computer Vision course with a MICCAI 2026 paper extension.

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: PyTorch (assumes CUDA-capable torch pre-installed in a conda env called `mlenv`)
- **Foundation model**: [segment-anything](https://github.com/facebookresearch/segment-anything) (Meta SAM ViT-B architecture) with MedSAM weights from Zenodo (~358 MB)
- **Key libraries**: monai (HD95/Dice metrics), numpy, pandas, pillow, opencv-python-headless, scikit-image, matplotlib, seaborn, scipy, pyyaml, tqdm, wandb (optional)
- **No peft dependency**: LoRA is hand-rolled in `src/models/lora.py`
- **Config format**: one YAML per (method, jitter, seed) combo in `configs/`
- **Hardware**: trained on an HP laptop with RTX 1000 Ada (8 GB VRAM) and Google Colab T4/A16. Full FT needs >=16 GB VRAM. All code also runs on MPS and CPU (AMP/pin_memory disabled automatically).

## How to Run

### Environment setup

```bash
conda activate mlenv
pip install -r requirements.txt
python scripts/download_medsam.py   # downloads medsam_vit_b.pth to checkpoints/
```

Datasets go under `data/` (gitignored). Expected layout:
```
data/
  train_images/   train_masks/    # ISIC 2018 Task 1
  val_images/     val_masks/
  test_images/    test_masks/
  ph2/                            # PH2
  busi/                           # BUSI
  cbis-ddsm/                      # CBIS-DDSM
```

### Train

```bash
python -m src.train --config configs/lora.yaml
python -m src.train --config configs/full_ft.yaml
python -m src.train --config configs/lora_rand100.yaml   # jitter-trained
python -m src.train --config configs/lora.yaml --resume  # resume from latest.pth
```

### Evaluate

```bash
python -m src.eval --config configs/zero_shot.yaml
python -m src.eval --config configs/lora.yaml --checkpoint checkpoints/runs/lora_seed0/best.pth
python -m src.eval --config configs/lora.yaml --checkpoint checkpoints/runs/lora_seed0/best.pth --quick  # first 8 images only
```

Eval appends rows to `results/runs.csv`. Per-image CSVs land in `results/raw/`.

### Multi-seed pipeline

```bash
python scripts/generate_seed_configs.py
bash scripts/run_seed_pipeline.sh      # train+eval all seeds (long)
python scripts/seed_significance.py    # paired t-tests on headline claims
python scripts/plots.py               # results/figures/*.png
python bbox_robustness/compare_trainings.py  # comparison curves
```

### CKA sweep

```bash
python cka/generate_cka_configs.py
bash cka/run_cka_sweep.sh
python cka/analysis/aggregate_cka_sweep.py
```

## Architecture Overview

### Data flow

1. Config YAML specifies method, jitter, seed, hyperparams.
2. `src/train.py` loads base MedSAM via `src/models/medsam.py`, calls `setup_method()` which dispatches to the method-specific `apply_*` function, freezes/unfreezes the right parameters in place, and returns param counts.
3. Training loop: `forward_with_prompt()` runs image_encoder (batched), then per-sample prompt_encoder + mask_decoder. Loss is DiceBCE. Optimizer is AdamW with cosine annealing.
4. If CKA is enabled: a second forward pass on a fixed probe batch computes CKA similarity against cached base activations; the CKA loss backward runs separately from the task loss backward to halve peak memory.
5. Checkpoints store only trainable params (not the full 93M model). `best.pth` tracks best val Dice; `latest.pth` includes optimizer/scheduler state for resume.
6. `src/eval.py` loads base MedSAM + overlays trained weights. Evaluates on each test set in the config, writes per-image CSVs and appends to `runs.csv`.

### Key directories and files

```
src/
  train.py           Training entry point. Method-agnostic, resume-capable, CKA-aware.
  eval.py            Evaluation entry point. Auto-detects method from checkpoint.
  losses.py          DiceBCELoss: (1-w)*BCE + w*Dice.
  metrics.py         dice_score, iou_score, hd95, aggregate_metrics.
  device_utils.py    CUDA > MPS > CPU auto-selection, AMP/pin_memory guards.
  models/
    medsam.py        load_medsam() via sam_model_registry["vit_b"], strict=False.
    methods.py       Dispatcher: method string -> apply_* function. encoder_in_grad_path().
    decoder_only.py  Freeze everything, unfreeze mask decoder.
    vpt.py           VPTSAMEncoder: additive perturbation at spatial positions (not token prepending).
    lora.py          LoRALinear wraps fused QKV (768->2304) with rank-r adapter. No peft.
    full_ft.py       Unfreeze encoder + decoder, freeze prompt encoder.
  data/
    isic.py          ISIC 2018 dataset. _bbox_from_mask() with jitter. ImageNet normalization.
    ph2.py           PH2 dataset.
    busi.py          BUSI dataset (breast ultrasound).
    cbis_ddsm.py     CBIS-DDSM dataset (mammography). Filters by PatientID, OR-merges multi-masks.
  cka.py             linear_cka() and flatten_for_cka(). Gram vs feature-product route based on n vs d.

cka/
  hooks.py           HookHandle: forward hooks with accumulate mode for per-sample decoder loop.
  probe.py           build_probe_batch() (deterministic multi-modal), cache_base_activations().
  generate_cka_configs.py   Generates the 9 CKA YAML configs (3 positions x 3 lambdas).
  run_cka_sweep.sh          Runs the full CKA sweep.
  analysis/                 aggregate_cka_sweep.py, compare_probes.py, compare_cka_jitter.py.

configs/              56 YAMLs: 5 methods x 3 jitter x 3 seeds + 9 CKA + zero_shot + busi_eval.
scripts/              download_medsam.py, generate_seed_configs.py, run_seed_pipeline.sh,
                      seed_significance.py, plots.py, bbox_robustness.py, bbox_robustness_viz.py,
                      aggregate_seeds.py, eval_all_checkpoints.py, visualize_predictions.py.
bbox_robustness/      compare_trainings.py and generated comparison figures.
results/              runs.csv, summary CSVs, results/figures/*.png (gitignored except paper figures).
```

## Current State

### Fully working
- All 6 on-branch methods train and evaluate: zero_shot, decoder_only, vpt_shallow, vpt_deep, lora, full_ft.
- CKA-aware LoRA training (9 configs: early/mid/late hooks x lambda 0.1/1.0/10.0).
- Multi-seed pipeline (3 seeds per method x jitter) with significance tests.
- Bbox robustness evaluation at 5 perturbation levels (0/20/50/100/200 px) across all 4 datasets.
- Resume from checkpoint (`--resume` flag, restores optimizer moments + scheduler LR).
- Mixed-precision training (CUDA only; auto-disabled on MPS/CPU).
- Thermal cooldown between train/val (configurable, for laptop GPU throttling).
- All analysis scripts, comparison plots, and paper figures are generated and committed.

### Not on this branch
- **Encoder-only LoRA**: reported in the paper as the best parameter-efficient method but its training config is not on `main`. The method freezes the mask decoder and applies LoRA only to the encoder QKV. Results were produced on a separate branch or environment.

### Local branch
- `my-final-eval` branch exists locally but is not relevant to current work.

## Active Work

The last meaningful commits were repo cleanup and paper alignment:
- `569464a` Made bbox-jitter robustness the README headline.
- `b19c7a1` Slimmed the repo from 508 MB to 74 MB by purging intermediate artifacts from git history via git-filter-repo.
- `41ec431` De-slopped all comments and docstrings (removed AI-generated style, em/en dashes, collapsed multi-line comments).

The paper is submitted. The coding work is done. This handoff exists as a safety net in case anything needs revisiting.

## Known Issues / TODOs

1. **Bbox jitter RNG is unseeded**: `_bbox_from_mask()` in `src/data/isic.py:30` uses `np.random.default_rng()` without a seed argument. This means bbox perturbation is non-deterministic across runs even with `--seed` set. Training-time augmentation varies per epoch (which is fine for training diversity) but eval reproducibility at non-zero jitter depends on this. For exact reproducibility, pass the global seed to the rng.

2. **Decoder-only trained for 10 epochs, others for 6**: `configs/decoder_only.yaml:20` sets `epochs: 10` while all other methods use 5-6 epochs. This was intentional (decoder-only converges slower with only 4M params and no encoder signal) but means wall-clock and epoch counts are not directly comparable across methods.

3. **Encoder-only LoRA missing from main**: The paper reports 7 methods but the code only implements 6. Encoder-only LoRA (freeze decoder, LoRA on encoder QKV only) is the strongest parameter-efficient result in the paper. To implement it: add an `encoder_only_lora` path in `src/models/methods.py` that calls `apply_lora()` then re-freezes `sam.mask_decoder`.

4. **CKA math must run in fp32**: `src/train.py:152` explicitly disables autocast for CKA computation. The Frobenius norms in `||X^T X||_F^2` reach ~1e10, which overflows fp16's 65504 max. This was a production bug that caused NaN losses before the fix (commit `b1acb87`).

5. **peft listed in requirements.txt but unused**: `requirements.txt:11` lists `peft>=0.10.0` but LoRA is implemented from scratch. It was likely needed during early development. Removing it won't break anything.

6. **Repo is private on GitHub**: After the git-filter-repo history rewrite, the repo was force-pushed. The GitHub remote is private. Any external URL references (e.g., for Canva asset uploads) won't resolve without making it public or using local files.

## Context for Next Session

This is a completed university research project comparing MedSAM fine-tuning strategies, with a MICCAI 2026 paper submitted. The codebase implements 6 of the 7 methods discussed in the paper (encoder-only LoRA is the missing one). All training, evaluation, multi-seed statistical analysis, and figure generation are done. The repo was recently cleaned: git history was rewritten to remove 434 MB of intermediate artifacts, all comments were de-slopped to read like terse human-written code, and the README was rewritten to match the paper. The only substantive gap between paper and code is encoder-only LoRA on main. If resuming work, start by reading `README.md` for the full experimental setup and results, then `src/models/methods.py` to understand the method dispatch, and `src/train.py` for the training loop. Configs are in `configs/` with a 1:1 mapping to runs.
