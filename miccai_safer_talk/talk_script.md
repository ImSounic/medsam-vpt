# Talk script: "When Adaptation Hurts"

MICCAI 2026 SAFER workshop, Oral Presentation 1, Sunday 27 Sep 2026, 08:15–09:15 CEST, Room Curie B, Strasbourg.
Four talks share the hour, so plan on **12 minutes speaking + 3 minutes questions**. Confirm the exact slot with the organisers.

Deck: `talk_16x9.pptx` (12 slides, 16:9). Every slide carries this script in its speaker notes, so Presenter View shows it. Pace of about 130 words per minute; the script is about 1,600 words, roughly 12 minutes. Times in brackets are cumulative.

---

## Slide 1 — Title  [0:00–0:30]

Good morning. I'm Sounic Akkaraju from the University of Twente. This is joint work with Marko Haralović, who shares first authorship, together with Carlo Baretta, Vasil Zapryanov and Alexia Briassouli.

The title is "When Adaptation Hurts". The one-line version: fine-tuning MedSAM can make it worse on exactly the data you did not have, and we can point to the part of the network that is responsible.

## Slide 2 — Three challenges  [0:30–1:45]

MedSAM is the medical version of Segment Anything. You give it an image and a bounding box, and it returns a mask. Deploying it runs into three problems that the literature tends to study one at a time.

First, ROI sensitivity. The box matters. In practice a clinician draws a loose, off-centre box in two seconds, and Dice follows the box quality.

Second, distribution shift. Change the scanner, the organ or the modality and accuracy falls.

Third, adaptation cost. Fine-tuning helps in-domain but is expensive. LoRA and visual prompt tuning are cheap, but they are sensitive to the data and the prompts you adapt with.

Prior work refines prompts, or adapts to a target domain, or detects out-of-distribution inputs, each separately. We evaluate all three jointly: adaptation strategy, prompt noise and domain shift in one grid. And then we ask which internal representations explain who survives.

## Slide 3 — Setup  [1:45–3:15]

Here is the testbed. Six adaptation strategies plus the frozen zero-shot model. Decoder-only fine-tuning, about four million parameters. VPT shallow and deep, ten prompt tokens at the encoder input or at every block, with the decoder trained. LoRA rank 8 on the encoder and decoder, four point four million. Encoder-only LoRA, the same adapter with the decoder frozen. And full fine-tuning of ninety-four million parameters at a ten-times lower learning rate.

Every model is trained once, on ISIC 2018 dermoscopy, two and a half thousand images, five epochs, three seeds. The prompt encoder is always frozen so boxes are encoded identically.

Then we evaluate along a drift ladder. In-domain: the ISIC test set. Close-OOD: PH², another dermoscopy dataset. Far-OOD: BUSI, breast ultrasound, and CBIS-DDSM, mammography. Same task, segment the lesion from a box, but the imaging physics changes completely.

The prompt protocol has two axes. Training: clean boxes, a fixed 20-pixel jitter, or a random jitter between 0 and 100 pixels. Evaluation: 0, 20, 50, 100 and 200 pixels on a 1024 image. Two hundred pixels sounds small, but it almost triples the box area on average, and on the small mammography lesions it is a fourteen-fold increase.

## Slide 4 — Result 1: adaptation helps in-domain, hurts far-OOD  [3:15–4:45]

First result, clean boxes everywhere. Read the table column by column. In-domain, every adapter beats zero-shot, and full fine-tuning and encoder-only LoRA are on top at 0.96 Dice. That is the result everyone expects.

Now walk right. On ultrasound and mammography, LoRA and both VPT variants drop below zero-shot. LoRA, the best in-domain adapter after full fine-tuning, gets 0.51 on mammography. Zero-shot MedSAM, with no training at all, gets 0.69. So LoRA is significantly worse than doing nothing.

Two adapters keep their gains. Full fine-tuning at 0.83, which is the best overall trade-off. And encoder-only LoRA at 0.79. Same adapter as LoRA, same rank, same encoder weights, but the decoder is frozen. That one change is worth 0.28 Dice on mammography. Hold on to that contrast, it comes back in the mechanism.

## Slide 5 — Boundary error  [4:45–5:45]

Dice understates how bad the collapse is. Here is the 95th-percentile Hausdorff distance, median and interquartile range, in pixels.

On ultrasound, encoder-only LoRA's median is three pixels, full fine-tuning four. LoRA's median is 26, which already sounds worse, but look at the upper quartile: 402 pixels. On mammography it is 438. That is not a slightly wrong contour. That is a quarter of far-OOD cases where the model segments the wrong structure entirely. The Dice gap between LoRA and full fine-tuning on ultrasound is about 0.12. The boundary gap is more than tenfold. So "adaptation hurts" means catastrophic failures on a subset of images, not a uniform small loss.

## Slide 6 — Result 2: prompt brittleness  [5:45–7:00]

Second result, the prompt axis. All models here were trained on clean boxes. Now we jitter the box at test time.

Look at the zero-shot row first. With 20 pixels of slack, zero-shot on mammography goes up, from 0.69 to 0.75. MedSAM was pretrained on imperfect boxes and it likes a little room. Every adapted model goes down.

At 100 pixels, zero-shot is the best model on both far-OOD datasets. Full fine-tuning, decoder-only, LoRA, even encoder-only LoRA, all below it. And the last row is the surprise: in-domain at 200 pixels, encoder-only LoRA, our strongest adapter, becomes the most jitter-sensitive model of all, 0.65 against zero-shot's 0.77. Clean-box training unlearns the prompt tolerance that MedSAM came with.

## Slide 7 — Result 3: training with jitter  [7:00–8:30]

So we train with jitter. The table shows the mean Dice gain over zero-shot, first trained with clean boxes, then trained with random 0-to-100-pixel jitter.

In-domain, every method improves. Full fine-tuning goes from slightly below zero-shot to plus 0.034. Encoder-only LoRA from minus 0.027 to plus 0.053. At 200 pixels of jitter, full fine-tuning on ISIC goes from 0.71 to 0.84. The brittleness is fixed. And fixed 20-pixel training does not do this, it overfits one noise level.

Now the far-OOD column. Jitter training costs far-OOD Dice for every method. But the size of the cost is the story. For full fine-tuning it is one and a half points. For encoder-only LoRA, three. For LoRA, decoder-only and VPT it is eight to ten points; LoRA on mammography goes from 0.51 to 0.21. Jitter training makes the adapter lean harder on image content, and for adapters that have already rewritten the decoder, that content-dependence does not transfer.

The overall column sums it up. After jitter training, only full fine-tuning and encoder-only LoRA are above zero-shot on average across all regimes. Those are the two adapters we recommend, and random jitter is the regime we recommend.

## Slide 8 — Mechanism: correlations  [8:30–9:30]

Why does LoRA collapse and encoder-only LoRA survive, when they change the same encoder weights? We compute linear CKA between each adapted model and base MedSAM, component by component, and correlate similarity with the Dice gain over zero-shot across all methods, seeds and jitter levels.

In-domain and close-OOD, nothing correlates. Representation drift is free there. Far-OOD, the decoder-side features light up: IoU token layer 0.76, output layer 0.76, decoder layers 0.73, upscaled embedding 0.71. And the encoder layers: 0.09. Nothing. In fact encoder similarity is weakly negative in-domain, because the adapters that gain most in-domain are the ones that move the encoder most.

The ranking makes it concrete. Decoder CKA on mammography: full fine-tuning 0.95, decoder-only 0.94, encoder-only LoRA 0.91, VPT-shallow 0.83, VPT-deep 0.80, LoRA 0.76. That is exactly the far-OOD Dice ranking. One caveat: the three thousand pooled observations share images and backbones, so treat the p-values as descriptive. The 0.7 versus 0.1 contrast is the evidence.

## Slide 9 — Mechanism: where the drift happens  [9:30–10:15]

Same story, component by component, left to right through the network. Encoder blocks: LoRA, VPT and encoder-only LoRA all drift, down to 0.4 to 0.6 at the image embedding. Encoder-only LoRA drifts as much as LoRA and is still robust, so encoder drift is not the problem.

Decoder: decoder-only, full fine-tuning and encoder-only LoRA stay at 0.7 to 0.9. LoRA and VPT collapse to 0.2 or 0.3 at the final attention queries and at the IoU and mask tokens, the components that turn a prompt into a mask. It is not how much the representation drifts, it is where. That mapping is the transferable part of MedSAM. Encoder-only LoRA works because it adapts the encoder to the visual shift and leaves that mapping alone.

## Slide 10 — Practical guidance  [10:15–11:00]

What should a site actually do? If compute is available and far-OOD data is expected, full fine-tuning at a low learning rate is the best trade-off in every regime. If you need a parameter-efficient adapter, use encoder-only LoRA and leave the decoder frozen; it is the best PEFT method far-OOD and competitive with full fine-tuning at four million parameters. If you only ever deploy on the training domain with clean prompts, standard LoRA still gives the largest in-domain gain. Whatever you pick, train with random 0-to-100-pixel jitter. We would not recommend VPT on MedSAM in any regime. And under severe prompt noise on far-OOD data, zero-shot MedSAM is still the model to beat, so keep it as a fallback.

## Slide 11 — Limitations  [11:00–11:30]

Three honest caveats. The ladder is asymmetric: one training domain and two far-OOD targets, so modality shift is confounded with lesion-type mismatch. Read the ranking, not the absolute Dice. The same pixel jitter is a much harder task far-OOD, plus 150 percent box area on ISIC against plus 1,300 percent on mammography. And the CKA correlations are descriptive, with a fixed recipe of five epochs, rank 8 and ten prompt tokens.

## Slide 12 — Take-home  [11:30–12:00]

Three things to take away. Adaptation can hurt: LoRA and VPT beat zero-shot in-domain and fall below it far-OOD, with boundary errors in the hundreds of pixels. Prompt noise must be trained for: random jitter gives the most reliable adapters, and only full fine-tuning and encoder-only LoRA end up above zero-shot overall. And the decoder is the locus: preserving decoder and output representations predicts far-OOD robustness, encoder similarity does not, and encoder-only LoRA is the adapter that follows from that.

Code and configs are on GitHub. Thank you. I'm happy to take questions, and I will be at the poster over lunch.

---

## Likely questions and short answers

**"Isn't this just overfitting to dermoscopy?"** Parameter count would then predict collapse, and it predicts the opposite: full fine-tuning with 94 million parameters collapses least, LoRA with 4 million collapses most. Which component moves predicts collapse; how many parameters move does not.

**"Why does zero-shot improve with 20 px of jitter?"** MedSAM was trained with perturbed boxes, so a tight box is slightly off-distribution for it. Clean-box adapters remove that slack, which is why jitter training brings it back.

**"Decoder-only fine-tuning also preserves the decoder CKA at 0.94 and does well on clean far-OOD boxes. Why not recommend it?"** It is the best 4M-parameter method on clean far-OOD boxes, but it pays the largest far-OOD price under jitter training, minus 0.088, and never beats zero-shot overall. Encoder-only LoRA adapts to the visual shift, which decoder-only cannot, and keeps its far-OOD gain under jitter.

**"Would a higher LoRA rank or decoder-only LoRA change the picture?"** The CKA result predicts decoder-only LoRA would behave like standard LoRA, because it is the decoder change that hurts. We have not run it. Rank was fixed at 8.

**"Can decoder CKA be used as a training signal or a monitor?"** In principle yes: a penalty toward base decoder activations is a straightforward regulariser, and per-image decoder drift is computable at inference with one extra frozen forward pass. We present the correlation here; turning it into a control is future work.

**"Is the ladder fair to ultrasound and mammography?"** No, and we say so. It is a stress test of transfer, not an estimate of clinical performance on those modalities. A symmetric protocol adapting on each modality in turn would separate shift from task mismatch.

**"Why are the CKA p-values not to be trusted?"** The 3,000 observations pool methods, seeds, datasets and jitter levels that share images and a common backbone, so they are not independent. The correlation sizes, 0.7 versus 0.1, are the evidence, not the stars.

**"What about the VPT implementation?"** SAM's windowed attention prevents canonical token prepending, so the ten prompt tokens are inserted at fixed positions at the encoder input or at every block, with the decoder trained. In both settings VPT shows negligible in-domain gain and a large far-OOD loss.

---

## Figures in the deck

All figures are embedded in `talk_16x9.pptx`: paper Fig. 1 (slides 1 and 2), the HD95 curves (slide 5), paper Fig. 2 (slide 6), the training-regime degradation curves (slide 7), the far-OOD CKA scatter (slide 8) and the component-wise CKA plot (slide 9). Rebuild after any edit with `node build_deck.js` from a directory that has `pptxgenjs` installed.
