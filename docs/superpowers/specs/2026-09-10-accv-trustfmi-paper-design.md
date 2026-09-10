# ACCV 2026 TrustFMI paper: design spec

Date: 2026-09-10
Status: approved scope, pre-implementation
Owner: Sounic Akkaraju

## 1. Goal

Submit a full paper (6 to 8 pages, LNCS) to the ACCV 2026 Workshop on
Trustworthy Foundation Models in Medical Imaging (TrustFMI, W01) by
25 September 2026, 23:59. Notification 16 October, camera-ready 23 October,
workshop 14 December 2026 in Osaka. Submission via OpenReview, double-blind.

Working title: "Protect the decoder: representation-preserving adaptation and
failure detection for far-OOD MedSAM".

One-sentence claim: far-OOD failure in adapted MedSAM is decoder representational
drift; a CKA loss reduces it, encoder-only LoRA and decoder freezing confirm the
locus, the collapse scales with adaptation budget, and per-image drift flags the
failures that remain.

## 2. Prior publication and overlap constraint

The comparison paper "When Adaptation Hurts" is accepted (oral) at the MICCAI
2026 SAFER workshop, LNCS, published around 4 to 8 October 2026. ACCV author
guidelines forbid substantial overlap with published work and run plagiarism
checks. Rules for this paper:

- The SAFER paper is cited in the third person as prior work. No text reuse.
- Already published and therefore only summarised in one paragraph: the seven
  method comparison, the four dataset drift ladder, bbox jitter robustness
  curves, HD95 and significance tables, and the CKA correlation analysis
  (decoder CKA rho = 0.76 vs encoder CKA rho = 0.09).
- Everything in sections 4 to 6 below is new. The shared element is the
  testbed (datasets, methods, metrics), which is acceptable.
- Confirm with co-authors (Marko Haralovic, Carlo Baretta, Vasil Zapryanov,
  Alexia Briassouli) before launching runs, since the CKA line originated with
  Marko.

## 3. Research questions and contributions

RQ1. Does regularising decoder representations toward base MedSAM during
adaptation reduce far-OOD collapse, and is the decoder specifically the locus?

RQ2. Does far-OOD collapse scale with adaptation budget (number of training
images at fixed optimisation steps), and does the ranking of adapters change
across budgets?

RQ3. Can an adapted model flag its own far-OOD failures at inference, using
either the decoder's IoU prediction head or per-image decoder drift?

Contributions as they will be listed in the paper:

1. CKA-regularised LoRA with an OOD-only probe, evaluated across three seeds
   and three prompt-jitter regimes, with a hook-placement ablation (decoder,
   encoder, both) that isolates the decoder as the locus.
2. An adaptation-budget study showing how far-OOD loss and in-domain gain trade
   off with training-set size for LoRA, decoder-only FT, and encoder-only LoRA.
3. A per-image failure detector based on decoder drift, compared with the
   decoder's native IoU estimate, with AUROC on the drift ladder.
4. (Stretch) A fifth held-out mammography set, DMID, confirming that the
   ranking transfers beyond CBIS-DDSM.

## 4. Topic 1: CKA-regularised adaptation

### 4.1 Existing result (single seed, on HPC)

Config `lora_cka_oodonly_late_l10_pm20_seed0`: LoRA r=8 plus CKA loss on the
late decoder position, lambda 10, OOD-only 32-image probe (16 BUSI + 16
CBIS-DDSM), bbox jitter pm=20. Beats no-CKA LoRA pm=20 by +0.045 Dice on BUSI
and +0.046 on CBIS-DDSM at tight boxes. CKA + rand100 hurts (-0.059 BUSI,
-0.032 CBIS-DDSM). Both facts go in the paper.

### 4.2 New runs

| Run name | Change vs 4.1 | Purpose |
|---|---|---|
| `lora_cka_oodonly_late_l10_pm20_seed1` | seed 1 | multi-seed |
| `lora_cka_oodonly_late_l10_pm20_seed2` | seed 2 | multi-seed |
| `lora_cka_oodonly_enc_l10_pm20_seed0` | hooks on encoder only (neck output and last two block outputs), no decoder hooks | ablation |
| `lora_cka_oodonly_both_l10_pm20_seed0` | union of decoder late hooks and the encoder hooks above | ablation |

Each run: 6 epochs on ISIC 2018 train, same hyperparameters as 4.1, best
checkpoint by val Dice. About 6 hours each on an L40S.

Evaluations, all at tight boxes plus jitter 20/50/100/200 px:

- The four new checkpoints above on ISIC test, PH2, BUSI, CBIS-DDSM.
- The existing `lora_cka_oodonly_late_l10_seed0` (pm=0 training) bbox
  robustness sweep, which was never run. Completes the six-curve figure
  {CKA, no-CKA} x {pm0, pm20, rand100}.

### 4.3 Analysis

- Tight-box Dice, IoU, HD95: mean +/- std over three seeds for the winning
  config, against the existing three-seed no-CKA LoRA pm=20 baseline.
- Paired Wilcoxon over per-image Dice, Holm-corrected, CKA vs no-CKA per
  dataset.
- Hook ablation bar chart: decoder vs encoder vs both, far-OOD Dice, seed 0.
  Prediction: decoder-only hooks recover far-OOD; encoder-only hooks do not;
  both is no better than decoder-only.
- Robustness curves with shaded seed bands.

## 5. Topic 2: adaptation budget vs far-OOD collapse

### 5.1 Design

Budgets: 50, 250, 1000 training images, sampled once from ISIC 2018 train with
a fixed subset seed (0) so all methods see identical images; smaller budgets
are prefixes of larger ones. The full 2595-image point is the existing seed-0
pm=0 checkpoints.

Methods: `lora` (r=8), `decoder_only`, `lora_encoder_only` (r=28, all blocks,
matching the checkpoint reported in the README). Three methods x three budgets
= nine runs.

Training: fixed 6000 optimiser steps at batch size 1 for every budget (1000
images see 6 epochs, 250 see 24, 50 see 120). Cosine schedule over steps.
Tight boxes (pm=0). Validation on the full 101-image val set every 500 steps;
keep best val Dice checkpoint. Learning rates as in the existing configs for
each method. About 1 hour per LoRA run, less for decoder-only.

Evaluation: tight boxes only, on ISIC test, PH2, BUSI, CBIS-DDSM (plus DMID if
ready). Dice, IoU, HD95.

### 5.2 Analysis

One figure, two panels: in-domain Dice vs budget (log x) and far-OOD Dice
(mean of BUSI and CBIS-DDSM) vs budget, one line per method, the zero-shot
value as a horizontal reference. Report the budget at which each method first
drops below zero-shot on far-OOD, if any.

Optional if time allows: decoder CKA to base at each budget, to link the curve
to topic 1's mechanism.

## 6. Topic 3: failure self-detection

### 6.1 Signals

For each image, two candidate detectors:

- `iou_pred`: the scalar IoU estimate returned by SAM's mask decoder
  alongside the mask logits. Currently discarded in `src/eval.py`.
- `drift`: 1 - linear CKA between the base MedSAM and the adapted model's
  decoder upscaled mask embedding for that image, treating the 256 x 256
  spatial positions as samples and the 32 channels as features. Computed from
  one extra base-model decoder forward per image (the image embedding from the
  base encoder is also needed when the encoder was adapted, so base-model
  encoder forward too; cost is roughly 2x eval).

### 6.2 Design

Checkpoints: seed 0, pm=0 training, for zero-shot, decoder-only, VPT-shallow,
VPT-deep, LoRA, encoder-only LoRA, full FT (seven). Zero-shot has drift = 0 by
construction and serves as the IoU-head calibration reference.

Datasets: ISIC test, PH2, BUSI, CBIS-DDSM. Prompts: tight boxes and 50 px
jitter.

Failure label: per-image Dice < 0.5 (primary), Dice < 0.7 (secondary).

Metrics: AUROC and AUPRC of each detector per (method, dataset); calibration of
`iou_pred` against true IoU as a reliability diagram and expected calibration
error over 10 bins.

### 6.3 Analysis

One figure: AUROC bars per method on the two far-OOD sets, drift vs iou_pred,
plus a scatter of iou_pred against true Dice for LoRA on CBIS-DDSM to show
overconfidence if present. One compact table of AUROC values.

Day-one check: run iou_pred on the five local seed-0 checkpoints on a 100-image
CBIS-DDSM subset. If AUROC < 0.6 for every method, iou_pred is reported as a
negative result in one sentence and drift becomes the sole detector.

## 7. Stretch: DMID held-out set

DMID mammography (data/dmid: TIFF images, "Pixel-level annotation" masks,
Metadata.xlsx). Needs a loader in `src/data/dmid.py` following the CBIS-DDSM
loader: pair images with masks by ID, convert 16-bit TIFF to 8-bit RGB, derive
tight bbox from mask, skip empty masks. Evaluation only. Kill date 16
September: if the loader does not produce sane zero-shot Dice by then, DMID
is dropped and nothing else changes.

## 8. Code changes

All in the `medsam-vpt` repo, on a new branch `accv-trustfmi`.

1. Restore the CKA package: `git checkout ec000e0 -- cka/` and verify
   `src/cka.py` and `src/train.py` still import it. Regenerate configs with
   `cka/generate_cka_configs.py`, extended to emit the four runs in 4.2.
   Encoder hook targets are added to `cka/hooks.py` as a new position name.
2. Add `lora_encoder_only` to `src/models/methods.py`: call `apply_lora`,
   then re-freeze `sam.mask_decoder`. Add `configs/lora_encoder_only.yaml`
   with r=28 if not present.
3. ISIC loader: add `max_train_samples` and `subset_seed` to the dataset
   config; deterministic prefix subset.
4. Trainer: add `train.max_steps` and `train.val_every_steps`; when set, the
   step budget overrides epochs and the scheduler runs over steps.
5. Eval: save `iou_pred` per image in the per-image CSV; add a
   `--drift` flag that loads base MedSAM alongside the adapted model and
   writes per-image drift.
6. New scripts: `scripts/budget_sweep_configs.py` (emits the nine configs),
   `scripts/plot_budget.py`, `scripts/failure_detection.py` (AUROC, AUPRC,
   calibration, figure), `scripts/plot_hook_ablation.py`.
7. Restore `bbox_robustness/eval_bbox_robust.py` from history if it is
   missing from the working tree.
8. SLURM: `cka/slurm/accv_t1.sbatch` (array of 4), `cka/slurm/accv_t2.sbatch`
   (array of 9), `cka/slurm/accv_t3.sbatch`, eval jobs with `afterany`
   dependencies. QOS limit is 2 concurrent GPU jobs; T1 goes first.

Smoke test each change locally on the laptop GPU (8 GB) with `--quick` before
submitting to HPC.

## 9. Compute plan and schedule

| Part | GPU hours | Notes |
|---|---|---|
| T1 four trainings | 24 | 6 h each on L40S |
| T1 evals incl. pm=0 robustness sweep | 3 | |
| T2 nine trainings | 10 | |
| T2 evals | 5 | |
| T3 drift and iou_pred dumps | 4 | 7 methods x 4 datasets x 2 prompt levels |
| Total | ~46 | ~24 h wall at 2 concurrent, plan 3 to 4 days |

Dates:

- 11 to 13 Sep: code changes 1 to 8, local smoke tests, day-one iou_pred
  check, launch T1 array, queue T2 behind it. Confirm plan with co-authors.
- 14 to 18 Sep: runs finish, T3 dumps, DMID loader attempt, draft intro,
  related work, method, setup sections.
- 16 Sep: DMID kill date.
- 19 Sep: general kill date. Any run not finished moves to supplementary or is
  cut. Topic 2 is the first to be cut, topic 3 second; topic 1 is never cut.
- 20 to 23 Sep: all figures, results and discussion, full draft to co-authors
  by 21 Sep, revisions.
- 24 Sep: freeze, anonymity check, OpenReview upload.
- 25 Sep: deadline.

## 10. Paper structure and page budget (8 pages, LNCS)

| Section | Pages | Content |
|---|---|---|
| 1 Introduction | 1.0 | Far-OOD collapse of PEFT on MedSAM; decoder drift hypothesis; three contributions |
| 2 Related work | 0.5 | SAM and MedSAM adaptation; PEFT and feature distortion; representation similarity; failure detection for segmentation |
| 3 Method | 1.25 | CKA loss and probe; hook placement; budget protocol; drift detector definition |
| 4 Setup | 0.75 | Datasets and drift ladder; methods; metrics; prior results summarised in one paragraph |
| 5.1 Results: CKA regularisation | 2.0 | Multi-seed table, robustness curves, hook ablation, rand100 negative result |
| 5.2 Results: adaptation budget | 0.75 | Budget figure and one paragraph |
| 5.3 Results: failure detection | 0.75 | AUROC table and figure |
| 6 Limitations and conclusion | 0.75 | Asymmetric ladder; single training domain; CKA cost; DMID if included |
| References | extra | Unlimited |

Figures (target six): overview schematic; CKA delta heatmap with seed bars;
six-curve robustness plot; hook ablation bars; budget curves; failure
detection AUROC plus scatter. Tables (two): multi-seed CKA results with
significance; detector AUROC.

Anonymity: no author names, no acknowledgements, no repository link, the
SAFER paper cited as third-party work, no HPC or institution identifiers in
figures.

## 11. Related-work anchors to verify before writing

To be verified for exact venue and year during drafting; do not cite from
memory.

- Kirillov et al., Segment Anything (ICCV 2023).
- Ma et al., Segment Anything in Medical Images, MedSAM (Nature
  Communications 2024).
- Hu et al., LoRA (ICLR 2022).
- Jia et al., Visual Prompt Tuning (ECCV 2022).
- Kornblith et al., Similarity of Neural Network Representations Revisited,
  CKA (ICML 2019).
- Kumar et al., Fine-Tuning Can Distort Pretrained Features and Underperform
  Out-of-Distribution (ICLR 2022).
- Lee et al., Surgical Fine-Tuning Improves Adaptation to Distribution Shifts
  (ICLR 2023).
- Wortsman et al., Robust Fine-Tuning of Zero-Shot Models, WiSE-FT (CVPR 2022).
- Cheng et al., SAM-Med2D (arXiv 2023) and other SAM medical adaptation
  surveys.
- Segmentation failure and quality-estimation work (for example reverse
  classification accuracy, and IoU-head calibration studies of SAM).
- The SAFER 2026 paper itself, cited anonymously as prior work.

## 12. Risks and responses

| Risk | Response |
|---|---|
| Multi-seed shrinks the CKA gain | Report honestly; lean on the ablation and topic 3; framing becomes "consistent, modest, mechanistically grounded" |
| Encoder-hook ablation also helps | Reframe as "any representational anchor helps, decoder most"; still a result |
| iou_pred uninformative | One-sentence negative result; drift is the detector |
| Drift detector no better than chance | Report per dataset; failure detection section shrinks to half a page and topic 2 gets the space |
| HPC queue or node failures | T1 launched first and alone if needed; T2 is the first cut |
| Co-author disagreement on scope | Resolve before 13 Sep; topics 2 and 3 are detachable |
| DMID mask format unusable | Dropped on 16 Sep |
| Overlap complaint from reviewers | Prior-work paragraph is explicit; no shared figures or text |

## 13. Out of scope

Symmetric drift ladder (adapting on BUSI or CBIS-DDSM), jitter scheduling,
subspace analysis of adapter weights, CKA loss on VPT or full FT, and any new
model architecture. These are follow-ups for a spring 2027 venue.

## 14. Writing workflow: Academic Research Skills (ARS) pipeline

The draft is written with the `academic-pipeline` skill from
Imbad0202/academic-research-skills (v3.21.x, CC BY-NC 4.0), installed as a
plugin from an interactive session:

```
/plugin marketplace add Imbad0202/academic-research-skills
/plugin install academic-research-skills
```

ARS never runs experiments. Sections 4 to 7 of this spec are executed outside
it on the HPC; their result CSVs and figures are declared to ARS as external
experiment provenance at intake.

Entry point: Stage 2 (WRITE), "has research data", on 19 September once the
kill date has fixed the result set. This spec is the paper plan. Stages used:
2 write, 2.5 integrity check (citation existence against Semantic Scholar,
OpenAlex, Crossref, arXiv; claim to result alignment), 3 review (treated as a
mock TrustFMI review before the co-author round), 4 revise, 3' re-review, 4.5
final integrity, 5 finalize as LaTeX plus BibTeX. Stage 6 process record is
skipped. Expected cost is on the order of 300K tokens.

Output handling: ARS emits generic LaTeX. The text is moved into the official
ACCV 2026 LNCS template (`ACCV_2026_template.zip` from the author guidelines)
with `splncs04` bibliography style; figures come from the repo scripts, not
from ARS's visualization agent. Page count is checked in the LNCS template,
not in ARS output.
