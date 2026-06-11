# Statistical reliability of headline claims

All claims are tested **two ways**:

  * **(A) Across-seed paired t-test** on per-seed dice means (n = number of (method, seed) pairs contributing to the claim). Tests one-sided H₁: mean (a − b) > 0 (or < 0 where the claim is a negative). Reported as `t_stat`, `p_one_sided`, and the empirical mean ± std of the diffs.

  * **(B) Per-image paired Wilcoxon signed-rank** on dice values averaged across 3 seeds. n = number of unique (image, method) pairs. This test has high power; the seed-level test is the conservative n=3 sanity check.

A claim **passes** when the relevant test rejects the null at p < 0.05. For claim C3 (a negative claim, rand100 does NOT beat zero-shot on CBIS-DDSM), it passes when the test in the opposite direction fails to reject (p ≥ 0.05).

| Claim | seed n | mean±std diff | seed t | seed p (1-sided) | seed pass | img n | Wilcoxon W | img p | img median diff | img pass |
|---|---:|---|---:|---:|:--:|---:|---:|---:|---:|:--:|
| **C1a-isic** | 15 | +0.0785 ± 0.0295 | +10.316 | 0.0000 | yes | 5000 | 11879946 | 0.00e+00 | +0.0636 | yes |
| **C1b-ph2** | 15 | +0.0683 ± 0.0310 | +8.543 | 0.0000 | yes | 1000 | 484123 | 7.57e-145 | +0.0594 | yes |
| **C2-cbis** | 3 | +0.3191 ± 0.0096 | +57.798 | 0.0001 | yes | n/a | n/a | n/a | n/a | n/a |
| **C3-cbis-zeroshot-ge-rand100** | 15 | -0.0667 ± 0.0257 | -10.043 | 1.0000 | yes | 1810 | 245304 | 1.00e+00 | -0.0481 | yes |

### Claim descriptions

- **C1a-isic**: Across all 5 PEFT methods, **rand100 training > pm=0 training on ISIC at pm=200** (rand100 produces better tight-bbox-trained method robustness on ISIC).
- **C1b-ph2**: Across all 5 PEFT methods, **rand100 training > pm=0 training on PH² at pm=200**.
- **C2-cbis**: **Full FT > LoRA on CBIS-DDSM tight bbox (pm=0)**, Full FT keeps far-OOD modality transfer that LoRA loses.
- **C3-cbis-zeroshot-ge-rand100**: **Zero-shot ≥ rand100-trained on CBIS-DDSM at pm=200** (averaged across 5 PEFT methods). Tests that rand100-trained methods do NOT beat zero-shot here.

### Reading the table

- For C1a/C1b/C2: we expect both `seed pass` and `img pass` to be yes.
- For C3: we expect both to be yes, meaning the data does NOT support 'rand100 > zero_shot on CBIS-DDSM at pm=200', consistent with the claim 'zero_shot ≥ rand100'.
