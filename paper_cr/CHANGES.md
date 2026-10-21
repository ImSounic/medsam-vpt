# Camera-ready changes against the submitted version (TrustFMI #9)

`submitted.tex` is a word-for-word reconstruction of the OpenReview PDF (checked with a word diff).
`main.tex` is the camera-ready. Everything not listed here is unchanged.

## Format
- ACCV template in final mode (no line numbers or page back-references), authors with ORCID, affiliation.
- Fig. 1 and Fig. 2 scaled to 0.9 and 0.85 of the text width; Table 1 widened (two columns added, scaled to text width).

## Reviewer points
| Reviewer | Point | Change |
|---|---|---|
| rrHd | OOD not defined | Intro: one sentence defining OOD as shift from the adaptation data |
| SNjd 3 | modalities are in MedSAM's pretraining | Same sentence: far-OOD loss is closer to forgetting |
| SNjd 2c | full FT also adapts both halves | Contribution 2 and budget paragraph: "at PEFT learning rates"; full FT at 1e-5 stays robust |
| SNjd 3 | where the BUSI probe images come from | Method: BUSI has no split, probe images are among the 647 evaluated; excluding them changes Dice by <= 0.003 (`scripts/probe_overlap.py`). Replaces the incorrect "disjoint from all evaluation images" |
| SNjd 3 | in-domain probe claim untested | Method: "(untested)" |
| rrHd | lambda choice | Method: lambda = 10 is the largest value tested; Conclusion: wider sweeps open |
| SNjd 2a | learning-rate confound | Setup: one LR for all weights, so LoRA/VPT train the decoder 5x faster than decoder-only |
| kTc3 W1, SNjd 2b | rank/target confound | Setup: encoder-only LoRA has a similar parameter count (4.1 M vs 4.4 M) but other target layers |
| kTc3 W3, SNjd 1 | compare with simpler methods | Table 1: decoder-only and full FT columns (same regime, 3 seeds); Conclusion states they remain more robust |
| rrHd | where/how to use CKA | Conclusion: recommendation (decoder or both, lambda = 10, unlabeled probe of protected domains, never encoder alone) |
| kTc3 W4 | duplicated "0.12 to 0.36" sentence | Budget paragraph: repeated sentence (and the "In contrast" sentence restating it) removed |
| SNjd 5 | AUPRC and ECE announced, not reported | Detection: AUPRC for drift vs IoU estimate, IoU ECE range |
| kTc3 W2, SNjd 4, SNjd 5 | per-term ablation, baselines (WiSE-FT, feature protection, L2, VPT), other detectors | Conclusion: listed as open |
| kTc3 W5 | references without venue | SAM-Med2D, ISIC 2018 and Tan et al. now carry their arXiv identifiers |

Also: DMID sentence now states which training regime each number comes from (0.697 tight-box, 0.366 20 px).

## Edits to untouched text (typos, and space for the additions)
- Typos: "methods.Experiments", "expansion.The", "tight-box ,", "boxes,)", missing period after "Adaptation-budget protocol", lowercase "for activation matrices" and the missing "linear CKA" subject.
- Removed to fit eight pages: the two sentences that only announce what follows ("CKA Regularization results are presented, ..." and "The reliability of the per-image decoder drift ... systematic manner"), the Method sentence on forward-pass cost (repeated in Sect. 5), "This is reflected in the drift panel ..." (now "(Fig. 2, right)"), the last rand100 sentence of the prompt-robustness paragraph (repeats the multi-seed paragraph), the SAM-Med2D statistics shortened to a parenthesis, and "Decoder drift identifies predictions ..." folded into the next sentence.
- The abstract changed only by the missing space; OpenReview must get the same fix.

## Pending
- T10 controls (frozen-decoder rank-8 LoRA; LoRA with decoder at 1e-4): drafted as a comment in the budget paragraph.
