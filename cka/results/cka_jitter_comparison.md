# CKA late_l10 effect across bbox-jitter trainings

Single seed for CKA, multi-seed avg for no-CKA baseline.


| Training | Dataset | No-CKA Dice | CKA Dice | Δ |
|---|---|---:|---:|---:|
| pm=0 | ISIC 2018 (ID) | 0.9556 | 0.9559 | +0.0003 |
| pm=0 | PH² (near-OOD) | 0.957 | 0.9594 | +0.0024 |
| pm=0 | BUSI (far-OOD ultrasound) | 0.7801 | 0.8127 | +0.0326 |
| pm=0 | CBIS-DDSM (far-OOD X-ray) | 0.5094 | 0.536 | +0.0266 |
| pm=20 | ISIC 2018 (ID) | 0.9492 | 0.9518 | +0.0026 |
| pm=20 | PH² (near-OOD) | 0.9511 | 0.955 | +0.0039 |
| pm=20 | BUSI (far-OOD ultrasound) | 0.7745 | 0.8192 | +0.0447 |
| pm=20 | CBIS-DDSM (far-OOD X-ray) | 0.437 | 0.4825 | +0.0455 |
| rand100 | ISIC 2018 (ID) | 0.9411 | 0.9407 | -0.0004 |
| rand100 | PH² (near-OOD) | 0.9496 | 0.9473 | -0.0023 |
| rand100 | BUSI (far-OOD ultrasound) | 0.7467 | 0.6877 | -0.0590 |
| rand100 | CBIS-DDSM (far-OOD X-ray) | 0.2143 | 0.1827 | -0.0316 |

## Per-training summary

| Training | CKA helps far-OOD? | Best at far-OOD overall? |
|---|---|---|
| pm=0 | BUSI +0.0326, CBIS +0.0266 -> yes |  |
| pm=20 | BUSI +0.0447, CBIS +0.0455 -> yes |  |
| rand100 | BUSI -0.0590, CBIS -0.0316 -> no | CKA + heavy random jitter overfits |