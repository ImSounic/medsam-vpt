# CKA probe comparison: best (position, λ) per dataset per probe

Baseline LoRA dice (seed 0, no CKA): isic2018_test=0.9545, ph2=0.9557, busi=0.7775, cbis_ddsm=0.4979

Zero-shot dice: isic2018_test=0.9072, ph2=0.9054, busi=0.8234, cbis_ddsm=0.6924


## Best CKA config per dataset per probe

| Dataset | Probe | Best position | Best λ | Dice | Δ vs baseline |
|---|---|---|---:|---:|---:|
| ISIC 2018 (ID) | original | early | 1 | 0.9572 | +0.0027 |
| ISIC 2018 (ID) | oodonly | mid | 1 | 0.9578 | +0.0033 |
| PH² (near-OOD) | original | early | 1 | 0.9597 | +0.0040 |
| PH² (near-OOD) | oodonly | late | 10 | 0.9594 | +0.0037 |
| BUSI (far-OOD US) | original | late | 10 | 0.8289 | +0.0514 |
| BUSI (far-OOD US) | oodonly | late | 10 | 0.8127 | +0.0352 |
| CBIS-DDSM (far-OOD X-ray) | original | late | 1 | 0.5498 | +0.0519 |
| CBIS-DDSM (far-OOD X-ray) | oodonly | late | 10 | 0.5360 | +0.0381 |

## Robustness: how many of 36 (config × dataset) cells fall in each bucket?

| Bucket | Original probe | OOD-only probe |
|---|---:|---:|
| Beats baseline (Δ ≥ 0) | 21 | 22 |
| Mild loss (0 > Δ ≥ -0.1) | 12 | 10 |
| Moderate loss (-0.1 > Δ ≥ -0.3) | 2 | 2 |
| Catastrophic (Δ < -0.3) | 1 | 2 |