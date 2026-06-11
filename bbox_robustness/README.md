# Bbox prompt robustness study

How does each adaptation method degrade as the input bounding box becomes
less precise, i.e. as a doctor's drawn box gets sloppier?

Each method/dataset is evaluated at four perturbation levels:
20, 50, 100, 200 px of maximum expansion per side. For every image,
each of the four sides of the ground-truth-derived bbox is pushed outward
by an independent random offset drawn from `[0, perturb_max_px]`. We
draw five such random bboxes per image and report mean ± std across
samples (per-image), then mean ± std across images (per dataset).

The original 0-px (tight) baseline lives in `../results/runs.csv`; we
reference those numbers rather than re-evaluating.

## Why

The standard eval uses `bbox_perturb_pixels: 0`, which means the prompt
is the tightest possible bbox around the ground-truth mask. That's
unrealistic: in a clinical workflow the doctor draws an approximate
bounding box around the lesion, which is generally:

- Loose (encloses the lesion with margin) rather than tight
- Asymmetric (more margin on one side than another)
- Variable (different doctors / different days)

This study measures how each method's segmentation quality holds up as
the bbox quality degrades. Methods that overfit the tight-bbox training
distribution should crumble fastest; methods that generalize well should
stay stable.

## Design

- **Direction**: expansion-only, bbox always contains the lesion (each
  side is pushed outward; never pulled inward). Models the realistic
  doctor scenario where the box might be too generous but not too tight.
- **Per-side independence**: each of the four sides gets an independent
  random expansion in `[0, perturb_max]`. The box centre and aspect
  ratio drift naturally as the four sides expand by different amounts.
- **Samples per image**: 5 random bboxes per image per perturbation
  level. RNG is seeded deterministically by
  `(image_id, perturb_max, sample_idx)` so reruns produce identical
  results.
- **Decoder batching**: the 5 random bboxes for a single image are
  passed to SAM's mask decoder as one batched call (K=5 prompts on one
  image embedding). This is exactly the multi-prompt segmentation
  pattern SAM was designed for: bit-identical to running 5 separate
  decoder calls but ~3x faster.

## Run

```bash
# Quick smoke test (8 images per dataset, full grid)
python bbox_robustness/eval_bbox_robust.py \
    --config configs/zero_shot.yaml \
    --quick

# Full eval (≈4-5 hours on A10, leave it running)
python bbox_robustness/eval_bbox_robust.py \
    --config configs/zero_shot.yaml \
    2>&1 | tee bbox_robustness/results/eval.log

# Quantitative plots (degradation curves, heatmap, relative drop)
python bbox_robustness/plot_bbox_robust.py

# Qualitative figures (one PNG per method × dataset × perturb level)
python bbox_robustness/visualize_bbox_robust.py
```

Optional knobs:
- `--perturb-levels 20 50 100 200`: change the perturbation grid
- `--n-samples 5`: change the per-image sample count
- `--checkpoint-glob 'checkpoints/runs/*/best.pth'`: restrict to a
  subset of trained methods

## Output

```
bbox_robustness/results/
├── runs.csv                                # main summary table (one row per method/dataset/level)
├── summary.csv                             # pivoted method × (dataset, perturb) table
├── per_image/                              # one row per image with mean+std across N samples
│   ├── decoder_only_seed0_isic2018_test_pm20.csv
│   ├── ...
│   └── zero_shot_cbis_ddsm_pm200.csv
└── figures/
    ├── degradation_curves.png              # 4 panels (one per dataset), dice vs perturb
    ├── degradation_heatmap.png             # method × perturb, faceted by dataset
    ├── relative_drop.png                   # % drop from 0-px baseline at pm=200
    └── qualitative/                        # per (method, dataset, sample image) figures
        ├── lora__cbis_ddsm__Mass-Test_P_00016_LEFT_CC.png
        ├── lora__cbis_ddsm__Mass-Test_P_00020_LEFT_MLO.png
        ├── ...                             # 6 methods × 4 datasets × N images
```

Each qualitative PNG shows ONE image's degradation across all four perturbation levels, same image throughout, with one row per perturb level so you can directly compare how the prediction collapses:

| | Col 0 | Col 1 | Col 2 | Col 3 |
|---|---|---|---|---|
| Row 0 (0 to 20 px) | Input + bbox (cyan = perturbed, yellow dashed = tight reference) | Ground truth overlay (green) | Prediction overlay (red) with Dice/IoU | TP green / FP red / FN blue |
| Row 1 (0 to 50 px) | same image, looser bbox | same GT | prediction at this level | diff at this level |
| Row 2 (0 to 100 px) | (same) | (same) | (same) | (same) |
| Row 3 (0 to 200 px) | (same) | (same) | (same) | (same) |

This makes it visually obvious which methods hold up vs collapse as the bbox becomes less precise on the *same* underlying image.

## runs.csv schema

| Column | Meaning |
|---|---|
| run_name | e.g. `lora_seed0`, `zero_shot` |
| method | e.g. `lora`, `zero_shot` |
| dataset | `isic2018_test`, `ph2`, `busi`, `cbis_ddsm` |
| seed | training seed (0 for all our runs) |
| perturb_max_px | 20, 50, 100, or 200 |
| n_samples | 5 (samples per image at this level) |
| dice_mean, dice_std | aggregated across image-level means |
| iou_mean, hd95_mean | likewise |
| n_images | number of test images contributing |
| trainable_params | parameter count for this method |
| peak_mem_mb | peak GPU memory during this method's eval |
| wall_clock_s | total seconds for this method on this dataset |
| timestamp | ISO 8601 |
