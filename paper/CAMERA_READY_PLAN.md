# TrustFMI camera-ready: reviewer concerns → changes

Deadline **23 Oct 2026** (TrustFMI site). Full papers 6–8 LNCS pages, references extra.
ACCV rules: official template in final mode, title and abstract identical to OpenReview
(plain-text abstract), author list unchanged, fonts embedded, PDF ≤ 10 MB, supplement optional
(one PDF ≤ 100 MB, cannot extend the page limit). No copyright form now (Springer, ~Jan 2027).

Canonical title (OpenReview): *Protect the Decoder: Understanding and Mitigating Representation
Drift in MedSAM*. The PDF abstract is revised here; the plain-text version is pasted into OpenReview.

Draft state (21 Oct): all text items done in `paper/` on top of the submitted version; body ends on
page 8 with ~6 lines of slack (tectonic; re-check on Overleaf with pdflatex). Fig. 1 (six robustness
curves) moved to the supplement to make room. Remaining: T10 numbers in Sect. 6 (`\pending`),
acknowledgements, repo URL, SAFER volume/pages, merge with the Overleaf export.

Status key: [x] done, [~] in progress, [ ] todo, [E] needs T10 results.

## Mechanical
- [x] Switch to `accv.sty` (final) + `accvabbrv`, `orcidlink`, plain `hyperref`; drop review-only bits.
- [x] Authors: Sounic Akkaraju, Carlo Baretta, Alexia Briassouli (University of Twente), ORCIDs.
- [~] Declarations shrunk to one acknowledgements paragraph (ACCV needs no AI statement): fill funding + repo URL.
- [x] `prior2026safer` → real citation: Haralović, Akkaraju, Baretta, Zapryanov, Briassouli,
      *When Adaptation Hurts: Connecting Representational Drift to OOD Failures in MedSAM
      Fine-Tuning*, MICCAI 2026 SAFER workshop. Rewrite "prior study" sentences as our earlier work.
- [x] Remove float-spacing overrides in the preamble if the page budget allows (spacing rule).

## Reviewer kTc3
- [E] W1 encoder-only LoRA differs in rank/targets → T10 rank-8 QKV LoRA, frozen decoder (3 seeds).
      Fallback text: state the confound in Setup and Limitations.
- [x] W2 which decoder hook term matters → not run; state as limitation (per-term ablation is future work).
- [x] W3 / SNjd W1 compare with simpler methods → add decoder-only and full FT rows (pm20, 3 seeds,
      already exist) to Table 1; say plainly they beat CKA-LoRA far-OOD (and full FT also in-domain);
      position the CKA loss as a repair for jointly adapted PEFT, not as the best adapter.
- [x] W4 duplicated "0.12 to 0.36" sentence → check Overleaf version, keep once.
- [x] W5 references → arXiv ids for SAM-Med2D, ISIC 2018 (Codella), Zenk et al.; venue for every entry.

## Reviewer rrHd
- [x] Define OOD: shift between the adaptation data and the test data (type b); far-OOD datasets are
      inside MedSAM's pretraining modalities (SNjd), so the effect is partly forgetting of base competence.
- [x] Each result paragraph opens with its question and closes with the takeaway; add a short
      "Recommendations" paragraph (where to put CKA, which λ, when not to use it).
- [x] λ: 10 is the edge of {0.1, 1, 10}; say so, no claim that it is optimal; per-domain λ is future work.

## Reviewer SNjd
- [x] 1 repaired LoRA below simple methods → see kTc3 W3.
- [E] 2a learning-rate confound (LoRA decoder at 5e-4, one AdamW group) → T10 LoRA with decoder at 1e-4.
      State the LR of every method's decoder in Setup.
- [E] 2b rank/target confound → T10 (kTc3 W1).
- [x] 2c full FT adapts both halves yet is robust → soften "joint, unanchored adaptation" to
      "joint adaptation at PEFT learning rates"; full FT runs at 1e-5.
- [x] 3d BUSI probe images come from the evaluated set (BUSI has no split; `cka/probe.py`) and one
      CBIS probe patient has test images → `scripts/probe_overlap.py`: excluding them moves any run's
      Dice by ≤ 0.003 (≤ 0.002 for regularised runs). Correct the Method sentence, add one line.
- [x] 3a-c far-OOD domains are in the probe; DMID is same modality as CBIS; in-domain probe untested →
      say the probe makes the far-OOD domains seen-unlabelled, DMID tests a new source not a new modality;
      remove the untested "in-domain anchor would compete" claim or mark it as untested.
- [x] 4 missing baselines (WiSE-FT, L2 distance, feature protection, CKA on VPT) → limitation.
- [x] 5 detector scope: tight boxes only, fails under 50 px jitter, on DMID for full/decoder-only FT and
      for the CKA model, needs the base model; mask-agreement and benchmark detectors not compared →
      Limitations; point to AUPRC and ECE in the supplement (Method promises them).
