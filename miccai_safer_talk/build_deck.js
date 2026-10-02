// 16:9 talk deck for "When Adaptation Hurts" (MICCAI 2026 SAFER, Oral 1).
// Run from a directory with pptxgenjs installed:  node build_deck.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const REPO = "/Users/imsounic/medsam-vpt";
const IMG = path.join(REPO, "miccai_safer_talk/pdf_images");
const FIG = path.join(REPO, "figures");
const OUT = path.join(REPO, "miccai_safer_talk/talk_16x9.pptx");

const C = {
  ink: "12262E", deep: "0F2F3C", teal: "0E7C86", tealSoft: "DDEDEF", card: "F2F6F7",
  line: "D5DEE2", muted: "5B6B72", bad: "C8452C", badSoft: "FBE4DE", good: "1E8E5A", goodSoft: "E3F2EA", white: "FFFFFF",
};
const FONT = "Calibri";
const W = 13.333, H = 7.5, M = 0.5;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Haralović, Akkaraju, Baretta, Zapryanov, Briassouli";
pres.title = "When Adaptation Hurts — MICCAI 2026 SAFER";

let slideNo = 0;
const TOTAL = 12;

// ---------- helpers ----------
function newSlide(dark = false) {
  const s = pres.addSlide();
  slideNo += 1;
  s.background = { color: dark ? C.deep : C.white };
  if (!dark) {
    s.addText("When Adaptation Hurts · MICCAI 2026 SAFER Workshop", { x: M, y: H - 0.42, w: 7, h: 0.3, fontFace: FONT, fontSize: 9, color: C.muted, isTextBox: true, margin: 0 });
    s.addText(`${slideNo} / ${TOTAL}`, { x: W - M - 1.2, y: H - 0.42, w: 1.2, h: 0.3, fontFace: FONT, fontSize: 9, color: C.muted, align: "right", isTextBox: true, margin: 0 });
  }
  return s;
}
function title(s, kicker, text, dark = false) {
  let y = 0.38;
  if (kicker) {
    s.addText(kicker.toUpperCase(), { x: M, y, w: 8, h: 0.28, fontFace: FONT, fontSize: 11, bold: true, color: dark ? "9FD1D6" : C.teal, charSpacing: 2, isTextBox: true, margin: 0 });
    y += 0.3;
  }
  s.addText(text, { x: M, y, w: W - 2 * M, h: 0.62, fontFace: FONT, fontSize: 27, bold: true, color: dark ? C.white : C.deep, isTextBox: true, margin: 0, valign: "top" });
  return y + 0.8;
}
function card(s, x, y, w, h, fill = C.card) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.1 });
}
function runs(arr, base) {
  return arr.map(r => ({ text: r.text, options: Object.assign({}, base, r.options || {}) }));
}
const B = (t) => ({ text: t, options: { bold: true } });
const BAD = (t) => ({ text: t, options: { bold: true, color: C.bad } });
const GOOD = (t) => ({ text: t, options: { bold: true, color: C.good } });

function para(s, x, y, w, arr, opts = {}) {
  const size = opts.size || 14;
  const cpl = Math.floor(w / (size / 72 * 0.47));
  const txt = arr.map(r => r.text).join("");
  const lines = Math.max(1, Math.ceil(txt.length / cpl));
  const h = opts.h || lines * size / 72 * 1.25 + 0.05;
  s.addText(runs(arr, { fontFace: FONT, fontSize: size, color: opts.color || C.ink }), { x, y, w, h, isTextBox: true, margin: 0, valign: opts.valign || "top", align: opts.align || "left" });
  return y + h;
}
function bullets(s, x, y, w, items, opts = {}) {
  const size = opts.size || 14;
  const ind = size / 72 * 1.1, tw = w - ind;
  const cpl = Math.floor(tw / (size / 72 * 0.47));
  let yy = y;
  items.forEach((it, i) => {
    const arr = Array.isArray(it) ? it : [{ text: it }];
    const txt = arr.map(r => r.text).join("");
    const lines = Math.max(1, Math.ceil(txt.length / cpl));
    const h = lines * size / 72 * 1.25 + 0.04;
    s.addText(opts.numbered ? `${i + 1}.` : "•", { x, y: yy, w: ind, h: size / 72 * 1.25 + 0.04, fontFace: FONT, fontSize: size, bold: !!opts.numbered, color: opts.glyph || C.teal, isTextBox: true, margin: 0 });
    s.addText(runs(arr, { fontFace: FONT, fontSize: size, color: opts.color || C.ink }), { x: x + ind, y: yy, w: tw, h, isTextBox: true, margin: 0, valign: "top" });
    yy += h + (opts.gap == null ? 0.1 : opts.gap);
  });
  return yy;
}
function table(s, x, y, w, rows, colW, opts = {}) {
  const size = opts.size || 12;
  const base = size / 72 * 1.9;
  const rowH = rows.map(r => {
    let lines = 1;
    r.forEach((c, ci) => {
      const t = typeof c === "string" ? c : c.text;
      const cpl = Math.floor((colW[ci] - 0.12) / (size / 72 * 0.5));
      lines = Math.max(lines, Math.ceil(t.length / cpl));
    });
    return base + (lines - 1) * size / 72 * 1.15;
  });
  const data = rows.map((r, ri) => r.map((c, ci) => {
    const cell = typeof c === "string" ? { text: c } : c;
    const head = ri === 0;
    return {
      text: cell.text,
      options: {
        fontFace: FONT, fontSize: size, bold: head || !!cell.bold,
        color: head ? C.white : (cell.color || C.ink),
        fill: { color: head ? C.teal : (cell.fill || (ri % 2 === 0 ? C.white : "EEF3F5")) },
        align: ci === 0 ? "left" : "center", valign: "middle", margin: [1, 4, 1, 4],
      },
    };
  }));
  s.addTable(data, { x, y, w, colW, rowH, border: { type: "solid", pt: 0.5, color: C.line } });
  return y + rowH.reduce((a, b) => a + b, 0);
}
function image(s, x, y, w, file, aspect, cap) {
  const h = w / aspect;
  s.addImage({ path: file, x, y, w, h });
  let yy = y + h + 0.05;
  if (cap) yy = para(s, x, yy, w, [{ text: cap }], { size: 10, color: C.muted });
  return yy;
}
// big number callout inside a soft card
function stat(s, x, y, w, h, big, label, tone = "teal", bigSize) {
  const fill = tone === "bad" ? C.badSoft : tone === "good" ? C.goodSoft : C.tealSoft;
  const col = tone === "bad" ? C.bad : tone === "good" ? C.good : C.teal;
  card(s, x, y, w, h, fill);
  s.addText(big, { x: x + 0.15, y: y + 0.08, w: w - 0.3, h: h * 0.55, fontFace: FONT, fontSize: bigSize || Math.min(34, h * 30), bold: true, color: col, isTextBox: true, margin: 0, valign: "middle" });
  s.addText(label, { x: x + 0.15, y: y + h * 0.58, w: w - 0.3, h: h * 0.4, fontFace: FONT, fontSize: 11, color: C.ink, isTextBox: true, margin: 0, valign: "top" });
}
function numCircle(s, x, y, n, d = 0.42) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: C.teal }, line: { color: C.teal, width: 0 } });
  s.addText(String(n), { x, y, w: d, h: d, fontFace: FONT, fontSize: 14, bold: true, color: C.white, align: "center", valign: "middle", isTextBox: true, margin: 0 });
}

// =====================================================================
// 1. Title
// =====================================================================
{
  const s = newSlide(true);
  s.addText("MICCAI 2026 · SAFER WORKSHOP · STRASBOURG · 27 SEPTEMBER 2026", { x: M, y: 0.55, w: 10, h: 0.3, fontFace: FONT, fontSize: 11, bold: true, color: "9FD1D6", charSpacing: 2, isTextBox: true, margin: 0 });
  s.addText("When Adaptation Hurts", { x: M, y: 0.95, w: 9.6, h: 1.0, fontFace: FONT, fontSize: 52, bold: true, color: C.white, isTextBox: true, margin: 0, valign: "top" });
  s.addText("Connecting Representational Drift to OOD Failures in MedSAM Fine-Tuning", { x: M, y: 2.0, w: 9.8, h: 0.5, fontFace: FONT, fontSize: 21, color: "DDEDEF", isTextBox: true, margin: 0, valign: "top" });
  s.addText([
    { text: "Marko Haralović", options: { bold: true } }, { text: "1,2,*", options: { superscript: true } }, { text: "    " },
    { text: "Sounic Akkaraju", options: { bold: true } }, { text: "2,*", options: { superscript: true } }, { text: "    " },
    { text: "Carlo Baretta", options: { bold: true } }, { text: "2", options: { superscript: true } }, { text: "    " },
    { text: "Vasil Zapryanov", options: { bold: true } }, { text: "2", options: { superscript: true } }, { text: "    " },
    { text: "Alexia Briassouli", options: { bold: true } }, { text: "2", options: { superscript: true } },
  ], { x: M, y: 2.75, w: 10.5, h: 0.4, fontFace: FONT, fontSize: 16, color: C.white, isTextBox: true, margin: 0 });
  s.addText("1 University of Zagreb, FER   ·   2 University of Twente   ·   * equal contribution", { x: M, y: 3.15, w: 10.5, h: 0.3, fontFace: FONT, fontSize: 12, color: "9FD1D6", isTextBox: true, margin: 0 });
  // figure strip
  const fh = 2.85, fw = fh * (4023 / 1142);
  s.addImage({ path: path.join(FIG, "paper_visuals/prompt_perturbation_examples.png"), x: M, y: H - fh - 0.45, w: fw, h: fh });
  s.addText("Same task, four domains, one loose box.", { x: M + fw + 0.3, y: H - fh - 0.45, w: W - M - fw - M - 0.3, h: fh, fontFace: FONT, fontSize: 13, italic: true, color: "9FD1D6", isTextBox: true, margin: 0, valign: "bottom" });
  // QR
  card(s, W - M - 1.5, 0.5, 1.5, 1.5, C.white);
  s.addImage({ path: path.join(IMG, "qr_repo.png"), x: W - M - 1.4, y: 0.6, w: 1.3, h: 1.3 });
  s.addText("github.com/ImSounic/medsam-vpt", { x: W - M - 2.6, y: 2.05, w: 2.6, h: 0.3, fontFace: FONT, fontSize: 10, color: "9FD1D6", align: "right", isTextBox: true, margin: 0 });
  s.addNotes("Good morning. I'm Sounic Akkaraju from the University of Twente. Joint work with Marko Haralović, who shares first authorship, with Carlo Baretta, Vasil Zapryanov and Alexia Briassouli. One-line version: fine-tuning MedSAM can make it worse on exactly the data you did not have, and we can point to the part of the network that is responsible. [0:30]");
}

// =====================================================================
// 2. Three challenges
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Motivation", "Three challenges that are usually studied one at a time");
  const cw = (W - 2 * M - 2 * 0.3) / 3, ch = 2.05;
  const items = [
    ["ROI sensitivity", "MedSAM needs a bounding box. Clinical boxes are loose, inconsistent or missing, and Dice follows box quality."],
    ["Distribution shift", "Change the scanner, organ or imaging modality and accuracy falls. Far-OOD transfer is where foundation models are supposed to earn their keep."],
    ["Adaptation cost", "Fine-tuning helps in-domain but is expensive. LoRA and VPT are cheap, yet sensitive to the data and prompts they are adapted with."],
  ];
  items.forEach((it, i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, y, cw, ch);
    numCircle(s, x + 0.25, y + 0.25, i + 1);
    s.addText(it[0], { x: x + 0.8, y: y + 0.22, w: cw - 1.0, h: 0.5, fontFace: FONT, fontSize: 18, bold: true, color: C.deep, isTextBox: true, margin: 0, valign: "middle" });
    para(s, x + 0.25, y + 0.9, cw - 0.5, [{ text: it[1] }], { size: 14 });
  });
  y += ch + 0.25;
  card(s, M, y, W - 2 * M, 0.95, C.tealSoft);
  para(s, M + 0.3, y + 0.12, W - 2 * M - 0.6, [
    B("Prior work "), { text: "refines prompts, adapts to a target domain, or detects OOD inputs, each in isolation.  " },
    B("This work "), { text: "evaluates adaptation strategy × prompt noise × domain shift jointly, then asks which internal representations explain who survives the shift." },
  ], { size: 14 });
  y += 0.95 + 0.25;
  const fh = 6.9 - y, fw = fh * (4023 / 1142);
  s.addImage({ path: path.join(FIG, "paper_visuals/prompt_perturbation_examples.png"), x: M, y, w: fw, h: fh });
  para(s, M + fw + 0.3, y + 0.1, W - M - fw - M - 0.3, [B("Fig. 1. "), { text: "Tight box (white, dashed), perturbed prompt (red), predicted mask (cyan). The same pixel jitter is a far larger relative change on the small far-OOD lesions in ultrasound and mammography." }], { size: 11.5, color: C.muted });
  s.addNotes("MedSAM is the medical Segment Anything: image plus bounding box in, mask out. Three problems in deployment, usually studied separately. ROI sensitivity: the box matters and clinicians draw loose boxes. Distribution shift: change scanner, organ or modality and accuracy falls. Adaptation cost: fine-tuning helps in-domain but is expensive; LoRA and VPT are cheap but sensitive. We evaluate all three jointly, then ask which internal representations explain who survives. [1:45]");
}

// =====================================================================
// 3. Setup
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Setup", "Six adapters, one training domain, a drift ladder");
  const lw = 6.6;
  let yy = table(s, M, y, lw, [
    ["Method", "Params", "What is updated"],
    ["Zero-shot MedSAM", "0", "reference"],
    ["Decoder-only FT", "4.1M", "mask decoder"],
    ["VPT-shallow", "4.1M", "10 prompt tokens at encoder input + decoder"],
    ["VPT-deep", "4.2M", "10 prompt tokens at every block + decoder"],
    ["LoRA (r = 8, α = 16)", "4.4M", "encoder + decoder adapters"],
    ["Encoder-only LoRA", "4.1M", "encoder adapters, decoder frozen"],
    ["Full FT", "93.7M", "image encoder + mask decoder"],
  ], [2.1, 0.9, lw - 3.0], { size: 13 });
  yy += 0.2;
  card(s, M, yy, lw, 6.85 - yy);
  para(s, M + 0.2, yy + 0.15, lw - 0.4, [B("Train "), { text: "on ISIC 2018 dermoscopy: 2,594 images, 80/10/10 split, 5 epochs, AdamW, Dice + CE, prompt encoder frozen, 3 seeds. Full FT: LR 1e-5, batch 1. PEFT: LR 1e-4, batch 4. Hyperparameters fixed across datasets; 8 GPUs with 16–22 GB, about 8 hours per training." }], { size: 12.5 });

  // right: ladder
  const rx = M + lw + 0.4, rw = W - M - rx;
  para(s, rx, y, rw, [B("Evaluate along a drift ladder")], { size: 15 });
  const chips = [["ISIC 2018", "in-domain · dermoscopy", "1,000 test images", C.tealSoft], ["PH²", "close-OOD · dermoscopy", "200 images", C.tealSoft], ["BUSI", "far-OOD · breast ultrasound", "647 images", C.badSoft], ["CBIS-DDSM", "far-OOD · mammography", "362 images", C.badSoft]];
  let cy = y + 0.45;
  chips.forEach((c, i) => {
    card(s, rx, cy, rw, 0.62, c[3]);
    s.addText([{ text: c[0], options: { bold: true, fontSize: 14 } }, { text: "   " + c[1], options: { fontSize: 12 } }], { x: rx + 0.2, y: cy, w: rw - 2.0, h: 0.62, fontFace: FONT, color: C.ink, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(c[2], { x: rx + rw - 1.9, y: cy, w: 1.7, h: 0.62, fontFace: FONT, fontSize: 11, color: C.muted, align: "right", isTextBox: true, margin: 0, valign: "middle" });
    cy += 0.62 + 0.1;
    if (i < 3) { s.addText("▼", { x: rx + rw / 2 - 0.2, y: cy - 0.16, w: 0.4, h: 0.2, fontFace: FONT, fontSize: 9, color: C.muted, align: "center", isTextBox: true, margin: 0 }); }
  });
  cy += 0.1;
  card(s, rx, cy, rw, 6.85 - cy);
  let py = para(s, rx + 0.2, cy + 0.15, rw - 0.4, [B("Prompt protocol. "), { text: "Train with clean boxes, fixed 20 px, or random 0–100 px jitter. Evaluate at 0 / 20 / 50 / 100 / 200 px on 1024² images." }], { size: 12.5 });
  stat(s, rx + 0.2, py + 0.15, (rw - 0.6) / 2, 6.7 - py - 0.15, "+280%", "mean box area at 200 px jitter, all datasets", "teal", 26);
  stat(s, rx + 0.4 + (rw - 0.6) / 2, py + 0.15, (rw - 0.6) / 2, 6.7 - py - 0.15, "+1,328%", "on small CBIS-DDSM lesions: far-OOD is harder by construction", "bad", 26);
  s.addNotes("Six strategies plus frozen zero-shot. Decoder-only, VPT shallow and deep with the decoder trained, LoRA rank 8 on encoder and decoder, encoder-only LoRA with the decoder frozen, and full fine-tuning at a lower learning rate. Trained once on ISIC 2018, five epochs, three seeds, prompt encoder frozen. Drift ladder: ISIC test in-domain, PH2 close-OOD, BUSI ultrasound and CBIS-DDSM mammography far-OOD. Prompt protocol: train clean, fixed 20 px or random 0-100 px; evaluate 0 to 200 px. 200 px almost triples box area, fourteen-fold on small mammography lesions. [3:15]");
}

// =====================================================================
// 4. Result 1
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Result 1", "Adaptation helps in-domain and often hurts far-OOD");
  const lw = 7.9;
  para(s, M, y, lw, [{ text: "Dice with clean boxes, models trained with clean boxes, mean of 3 seeds" }], { size: 11, color: C.muted });
  let yy = table(s, M, y + 0.3, lw, [
    ["Method", "ISIC (ID)", "PH² (close-OOD)", "BUSI (far-OOD)", "CBIS-DDSM (far-OOD)"],
    ["Zero-shot", "0.907", "0.905", "0.823", "0.692"],
    ["Decoder-only", "0.949", "0.947", "0.894", "0.828"],
    ["VPT-shallow", "0.945", "0.943", { text: "0.792", bold: true, color: C.bad }, { text: "0.566", bold: true, color: C.bad }],
    ["VPT-deep", "0.947", "0.946", { text: "0.802", bold: true, color: C.bad }, { text: "0.573", bold: true, color: C.bad }],
    ["LoRA", "0.956", "0.957", { text: "0.780", bold: true, color: C.bad }, { text: "0.509", bold: true, color: C.bad }],
    ["Encoder-only LoRA", "0.958", { text: "0.960", bold: true, color: C.good }, { text: "0.908", bold: true, color: C.good }, "0.793"],
    ["Full FT", { text: "0.961", bold: true, color: C.good }, "0.958", "0.901", { text: "0.829", bold: true, color: C.good }],
  ], [2.0, 1.3, 1.5, 1.5, lw - 6.3], { size: 13 });
  yy += 0.2;
  bullets(s, M, yy, lw, [
    [{ text: "Every adapter beats zero-shot in-domain. LoRA and both VPT variants " }, BAD("fall below zero-shot"), { text: " on ultrasound and mammography." }],
    [B("Full FT"), { text: " is the best overall trade-off. " }, B("Encoder-only LoRA"), { text: " is the strongest parameter-efficient method: same adapter as LoRA, decoder left alone." }],
  ], { size: 13 });
  const rx = M + lw + 0.4, rw = W - M - rx;
  stat(s, rx, y, rw, 1.55, "0.51 vs 0.69", "LoRA vs zero-shot Dice on mammography. The best in-domain PEFT adapter is worse than doing nothing far-OOD.", "bad");
  stat(s, rx, y + 1.7, rw, 1.55, "+0.28 Dice", "Encoder-only LoRA over standard LoRA on CBIS-DDSM. The only change: the decoder is frozen.", "good");
  stat(s, rx, y + 3.4, rw, 1.55, "4.1M ≈ 93.7M", "Encoder-only LoRA and decoder-only FT stay within 0.04 Dice of Full FT on both far-OOD sets.", "teal");
  s.addNotes("Clean boxes everywhere. Column by column: in-domain every adapter beats zero-shot, Full FT and encoder-only LoRA top at 0.96. Walk right: on ultrasound and mammography LoRA and both VPTs drop below zero-shot. LoRA gets 0.51 on mammography; zero-shot with no training gets 0.69. Two adapters keep their gains: Full FT at 0.83, and encoder-only LoRA at 0.79, same adapter as LoRA with the decoder frozen. That one change is worth 0.28 Dice. Hold on to that contrast. [4:45]");
}

// =====================================================================
// 5. Boundary error
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Result 1, continued", "Boundary error shows collapse, not a small overlap loss");
  const lw = 6.1;
  para(s, M, y, lw, [{ text: "HD95 with clean boxes, median [IQR] in pixels at 1024². In-domain and close-OOD medians are 0–7 px for every method." }], { size: 11, color: C.muted });
  let yy = table(s, M, y + 0.5, lw, [
    ["Method", "BUSI (far-OOD)", "CBIS-DDSM (far-OOD)"],
    ["Zero-shot", "14 [9, 25]", "12 [7, 17]"],
    ["Decoder-only", "4 [2, 17]", "4 [3, 6]"],
    ["VPT-shallow", { text: "21 [5, 387]", bold: true, color: C.bad }, "14 [8, 37]"],
    ["VPT-deep", { text: "25 [6, 394]", bold: true, color: C.bad }, { text: "18 [10, 359]", bold: true, color: C.bad }],
    ["LoRA", { text: "26 [6, 402]", bold: true, color: C.bad }, { text: "47 [13, 438]", bold: true, color: C.bad }],
    ["Encoder-only LoRA", { text: "3 [1, 11]", bold: true, color: C.good }, "5 [3, 7]"],
    ["Full FT", "4 [1, 13]", { text: "4 [3, 6]", bold: true, color: C.good }],
  ], [2.1, 1.9, lw - 4.0], { size: 13 });
  yy += 0.2;
  stat(s, M, yy, lw, 1.15, "~400 px", "Upper quartile of LoRA and VPT boundary error on far-OOD data: a quarter of cases segment the wrong structure entirely. Dice gap to Full FT ~0.12; boundary gap more than tenfold.", "bad");
  const rx = M + lw + 0.4, rw = W - M - rx;
  image(s, rx, y, rw, path.join(FIG, "camera_ready/hd95_robustness.png"), 1480 / 1184, "HD95 (log scale) vs evaluation jitter, rand100-trained models. LoRA and VPT sit an order of magnitude above the rest on both far-OOD sets at every jitter level.");
  s.addNotes("Dice understates it. HD95 median and IQR. On ultrasound encoder-only LoRA's median is 3 px, Full FT 4. LoRA's median is 26, but the upper quartile is 402 px; on mammography 438. That is a quarter of far-OOD cases where the model segments the wrong structure. Dice gap 0.12, boundary gap tenfold. Adaptation hurts means catastrophic failures on a subset, not a uniform small loss. [5:45]");
}

// =====================================================================
// 6. Result 2: prompt brittleness
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Result 2", "Clean-box training makes adapters brittle to prompt noise");
  const fw = 6.9;
  image(s, M, y, fw, path.join(IMG, "p06_0_2460x1793.png"), 2460 / 1793, "Dice vs evaluation jitter, averaged over the three training-jitter regimes (paper Fig. 2).");
  const rx = M + fw + 0.4, rw = W - M - rx;
  para(s, rx, y, rw, [{ text: "Dice under evaluation jitter, models trained with clean boxes" }], { size: 11, color: C.muted });
  let yy = table(s, rx, y + 0.3, rw, [
    ["Setting", "Zero-shot", "Dec-only", "LoRA", "Enc-LoRA", "Full FT"],
    ["CBIS, 0 px", "0.692", "0.828", "0.509", "0.793", "0.829"],
    ["CBIS, 20 px", { text: "0.748", bold: true, color: C.good }, "0.706", "0.478", "0.753", "0.744"],
    ["CBIS, 100 px", { text: "0.391", bold: true, color: C.good }, "0.303", "0.270", "0.367", "0.336"],
    ["BUSI, 100 px", { text: "0.758", bold: true, color: C.good }, "0.659", "0.619", "0.701", "0.672"],
    ["ISIC, 200 px", { text: "0.767", bold: true, color: C.good }, "0.710", "0.787", { text: "0.645", bold: true, color: C.bad }, "0.712"],
  ], [1.35, (rw - 1.35) / 5, (rw - 1.35) / 5, (rw - 1.35) / 5, (rw - 1.35) / 5, (rw - 1.35) / 5], { size: 11 });
  yy += 0.25;
  bullets(s, rx, yy, rw, [
    [{ text: "Zero-shot " }, GOOD("improves"), { text: " with 20 px of slack and is the best model far-OOD once boxes are 100 px off. MedSAM was pretrained on imperfect boxes." }],
    [{ text: "Clean-box adapters " }, BAD("unlearn that tolerance"), { text: ". Even encoder-only LoRA becomes the most jitter-sensitive model in-domain at 200 px." }],
  ], { size: 13 });
  s.addNotes("Prompt axis, all models trained on clean boxes. Zero-shot row first: 20 px of slack on mammography goes up, 0.69 to 0.75. MedSAM likes a little room. Every adapted model goes down. At 100 px, zero-shot is the best model on both far-OOD datasets. Last row: in-domain at 200 px encoder-only LoRA becomes the most jitter-sensitive model of all, 0.65 against 0.77. Clean-box training unlearns the prompt tolerance MedSAM came with. [7:00]");
}

// =====================================================================
// 7. Result 3: jitter training
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Result 3", "Random 0–100 px jitter training gives the most reliable adapters");
  const lw = 7.3;
  para(s, M, y, lw, [{ text: "Mean Dice gain over zero-shot: trained with clean boxes → trained with random 0–100 px jitter (paper Table 1)" }], { size: 11, color: C.muted });
  let yy = table(s, M, y + 0.3, lw, [
    ["Method", "In-domain", "Far-OOD", "Overall"],
    ["Full FT", { text: "−0.006 → +0.034", bold: true, color: C.good }, "−0.014 → −0.029", { text: "−0.009 → +0.013", bold: true, color: C.good }],
    ["Encoder-only LoRA", { text: "−0.027 → +0.053", bold: true, color: C.good }, "−0.024 → −0.054", { text: "−0.023 → +0.007", bold: true, color: C.good }],
    ["LoRA", "+0.012 → +0.035", { text: "−0.142 → −0.228", bold: true, color: C.bad }, "−0.038 → −0.051"],
    ["Decoder-only", "−0.016 → +0.014", { text: "−0.036 → −0.088", bold: true, color: C.bad }, "−0.023 → −0.022"],
    ["VPT-shallow", "−0.008 → +0.009", { text: "−0.127 → −0.230", bold: true, color: C.bad }, "−0.048 → −0.071"],
    ["VPT-deep", "−0.010 → +0.013", { text: "−0.126 → −0.198", bold: true, color: C.bad }, "−0.049 → −0.058"],
  ], [1.9, 1.8, 1.8, lw - 5.5], { size: 12 });
  yy += 0.2;
  bullets(s, M, yy, lw, [
    [{ text: "Jitter training restores in-domain and close-OOD robustness for every adapter. Fixed 20 px training overfits one noise level." }],
    [{ text: "It costs far-OOD Dice: " }, GOOD("small"), { text: " for Full FT and encoder-only LoRA, " }, BAD("large"), { text: " for LoRA, VPT and decoder-only." }],
    [{ text: "Only " }, B("Full FT and encoder-only LoRA"), { text: " end up above zero-shot overall." }],
  ], { size: 12.5, gap: 0.06 });
  const rx = M + lw + 0.4, rw = W - M - rx;
  stat(s, rx, y, (rw - 0.2) / 2, 1.25, "0.71 → 0.84", "Full FT, ISIC at 200 px jitter, clean → rand100", "good", 22);
  stat(s, rx + (rw + 0.2) / 2, y, (rw - 0.2) / 2, 1.25, "0.51 → 0.21", "LoRA, CBIS-DDSM clean boxes, clean → rand100", "bad", 22);
  image(s, rx, y + 1.45, rw, path.join(FIG, "paper_visuals/degradation_curves.png"), 2980 / 2106, "Solid: clean-trained · dashed: fixed 20 px · dotted: random 0–100 px. Dotted curves are flattest in-domain and lowest on CBIS-DDSM.");
  s.addNotes("Train with jitter. Mean Dice gain over zero-shot, clean-trained then rand100-trained. In-domain everyone improves: Full FT to plus 0.034, encoder-only LoRA to plus 0.053; Full FT at 200 px goes 0.71 to 0.84. Fixed 20 px training does not do this. Far-OOD column: jitter training costs every method, but the size differs. Full FT one and a half points, encoder-only LoRA three, LoRA, decoder-only and VPT eight to ten; LoRA on mammography 0.51 to 0.21. Overall column: only Full FT and encoder-only LoRA end above zero-shot. Those two adapters, and random jitter, are what we recommend. [8:30]");
}

// =====================================================================
// 8. Mechanism: correlations
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Mechanism", "Far-OOD loss tracks decoder drift, not encoder drift");
  const lw = 7.3;
  para(s, M, y, lw, [{ text: "Spearman ρ between linear CKA (adapted vs zero-shot MedSAM) and Dice gain over zero-shot. n = 3,000 pooled observations; FDR-corrected * q < 0.05, *** q < 0.001" }], { size: 11, color: C.muted });
  let yy = table(s, M, y + 0.5, lw, [
    ["CKA feature", "ID (ISIC)", "Close-OOD (PH²)", "Far-OOD (BUSI / CBIS)"],
    ["IoU token layer", "0.10", "−0.02", { text: "0.76 ***", bold: true, color: C.good }],
    ["Output layer", "0.06", "0.00", { text: "0.76 ***", bold: true, color: C.good }],
    ["Decoder layers", "−0.08", "−0.02", { text: "0.73 ***", bold: true, color: C.good }],
    ["Upscaled embedding", "0.20", "0.29 *", { text: "0.71 ***", bold: true, color: C.good }],
    ["Encoder layers", "−0.28 *", "−0.28 *", { text: "0.09", bold: true, color: C.bad }],
  ], [2.2, 1.5, 1.7, lw - 5.4], { size: 13 });
  yy += 0.2;
  bullets(s, M, yy, lw, [
    [B("Decoder-side similarity predicts far-OOD gain; encoder similarity does not"), { text: ", and is weakly negative in-domain." }],
    [{ text: "Decoder CKA on CBIS-DDSM follows the Dice ranking exactly: Full FT 0.95 · decoder-only 0.94 · encoder-only LoRA 0.91 · VPT-shallow 0.83 · VPT-deep 0.80 · LoRA 0.76." }],
    [B("Encoder-only LoRA works"), { text: " because it adapts the encoder to the visual shift while preserving the decoder pathway." }],
  ], { size: 12.5, gap: 0.06 });
  const rx = M + lw + 0.4, rw = W - M - rx;
  image(s, rx, y, rw, path.join(IMG, "fig5_farood_crop.png"), 926 / 622, "Far-OOD Dice gain vs overall CKA, one point per method × training jitter × evaluation jitter (Spearman 0.64). Bottom-left cluster: LoRA and VPT. Top-right, next to zero-shot: Full FT and encoder-only LoRA.");
  card(s, rx, 5.55, rw, 1.15, C.tealSoft);
  para(s, rx + 0.2, 5.68, rw - 0.4, [B("Caveat. "), { text: "Pooled observations share images and backbones, so the effective sample size is well below n = 3,000. Treat the p-values as descriptive; the 0.7 vs 0.1 contrast is the evidence." }], { size: 11.5 });
  s.addNotes("Why does LoRA collapse and encoder-only LoRA survive when they change the same encoder weights? Linear CKA between each adapted model and base MedSAM, component by component, correlated with Dice gain across all methods, seeds and jitter levels. In-domain and close-OOD nothing correlates; drift is free there. Far-OOD the decoder-side features light up: IoU token 0.76, output layer 0.76, decoder layers 0.73, upscaled embedding 0.71. Encoder layers: 0.09. Encoder similarity is even weakly negative in-domain. Decoder CKA on mammography follows the Dice ranking exactly. It is not how much the representation drifts, it is where. Caveat: pooled observations, descriptive p-values. [10:00]");
}

// =====================================================================
// 9. Mechanism: where the drift happens (figure)
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Mechanism, continued", "Where the representation drifts: component-wise CKA to zero-shot MedSAM");
  const fw = 10.4;
  const fh = fw / (3044 / 1096);
  s.addImage({ path: path.join(IMG, "p16_0_3044x1096.png"), x: (W - fw) / 2, y, w: fw, h: fh });
  let yy = y + fh + 0.15;
  const cw = (W - 2 * M - 0.6) / 3;
  const notes = [
    ["Encoder (E00–E11, image embedding)", "LoRA, VPT and encoder-only LoRA all drift, down to 0.4–0.6 at the image embedding. Encoder-only LoRA drifts as much as LoRA and is still robust, so encoder drift is not the problem.", C.tealSoft],
    ["Decoder attention, IoU and mask tokens", "LoRA and VPT collapse to 0.2–0.3 at the final attention queries and the IoU and mask tokens. Decoder-only, Full FT and encoder-only LoRA stay at 0.7–0.9.", C.badSoft],
    ["What the decoder does", "It turns a prompt into a mask. That mapping is the transferable part of MedSAM; adapters that rewrite it throw the far-OOD ability away.", C.goodSoft],
  ];
  notes.forEach((n, i) => {
    const x = M + i * (cw + 0.3);
    card(s, x, yy, cw, 6.9 - yy, n[2]);
    s.addText(n[0], { x: x + 0.2, y: yy + 0.1, w: cw - 0.4, h: 0.32, fontFace: FONT, fontSize: 12.5, bold: true, color: C.deep, isTextBox: true, margin: 0 });
    para(s, x + 0.2, yy + 0.45, cw - 0.4, [{ text: n[1] }], { size: 11 });
  });
  s.addNotes("Same story component by component, left to right through the network. Encoder blocks: LoRA, VPT and encoder-only LoRA all drift, down to 0.4 to 0.6 at the image embedding. Encoder-only LoRA drifts as much as LoRA and is still robust, so encoder drift is not the problem. Decoder: decoder-only, Full FT and encoder-only LoRA stay at 0.7 to 0.9. LoRA and VPT collapse to 0.2 or 0.3 at the final attention queries and the IoU and mask tokens, the components that turn a prompt into a mask. That mapping is the transferable part of MedSAM. Encoder-only LoRA works because it adapts the encoder to the visual shift and leaves that mapping alone. [10:30]");
}

// =====================================================================
// 10. Practical guidance
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Practical guidance", "Which adapter, which training regime");
  const rows = [
    ["Compute available, far-OOD data expected", "Full fine-tuning at LR 1e-5", "Best trade-off in every regime; 93.7M parameters.", C.goodSoft],
    ["Parameter-efficient adaptation", "Encoder-only LoRA", "Best PEFT far-OOD, competitive with Full FT, 4.1M parameters. Keep the decoder frozen.", C.goodSoft],
    ["Only the target domain, clean prompts", "Standard LoRA", "Largest in-domain and close-OOD gains; do not deploy it far-OOD.", C.tealSoft],
    ["Any regime", "Train with random 0–100 px box jitter", "Fixed-jitter training overfits one noise level.", C.tealSoft],
    ["Any regime", "Avoid VPT on MedSAM", "Negligible in-domain gain, large far-OOD loss.", C.badSoft],
    ["Severe prompt noise on far-OOD data", "Zero-shot MedSAM as fallback", "Still hard to beat once boxes are 100 px off.", C.badSoft],
  ];
  const rh = (6.85 - y - 5 * 0.1) / 6;
  rows.forEach((r, i) => {
    const yy = y + i * (rh + 0.1);
    card(s, M, yy, W - 2 * M, rh, r[3]);
    s.addText(r[0], { x: M + 0.25, y: yy, w: 3.9, h: rh, fontFace: FONT, fontSize: 12.5, color: C.muted, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(r[1], { x: M + 4.25, y: yy, w: 3.9, h: rh, fontFace: FONT, fontSize: 15, bold: true, color: C.deep, isTextBox: true, margin: 0, valign: "middle" });
    s.addText(r[2], { x: M + 8.3, y: yy, w: W - 2 * M - 8.55, h: rh, fontFace: FONT, fontSize: 12.5, color: C.ink, isTextBox: true, margin: 0, valign: "middle" });
  });
  s.addNotes("What should a site do? Compute available and far-OOD data expected: full fine-tuning at a low learning rate, best trade-off in every regime. Parameter-efficient: encoder-only LoRA, decoder frozen, best PEFT far-OOD and competitive with Full FT. Only the training domain with clean prompts: standard LoRA still gives the largest in-domain gain. Whatever you pick, train with random 0-100 px jitter. We would not recommend VPT on MedSAM in any regime. Under severe prompt noise on far-OOD data, zero-shot MedSAM is still the model to beat; keep it as a fallback. [11:15]");
}

// =====================================================================
// 11. Limitations
// =====================================================================
{
  const s = newSlide();
  let y = title(s, "Limitations", "Read the ranking, not the absolute Dice");
  const items = [
    ["Asymmetric ladder", "One training domain (dermoscopy) and two far-OOD targets. Modality shift is confounded with lesion-type mismatch; the absolute far-OOD Dice is a robustness indicator, not a clinical estimate."],
    ["Jitter difficulty is dataset-dependent", "200 px is +152% box area on ISIC but +1,328% on CBIS-DDSM. Far-OOD evaluation is harder by construction."],
    ["CKA correlations are descriptive", "Observations share images and backbones across methods, seeds and jitter levels, so the effective sample size is well below n = 3,000."],
    ["Fixed recipe", "5 epochs, one hyperparameter set for all datasets, LoRA rank 8 with α = 16, 10 prompt tokens per insertion point. VPT is an additive perturbation at fixed positions because SAM's windowed attention prevents token prepending."],
  ];
  const cw = (W - 2 * M - 0.3) / 2, ch = (6.85 - y - 0.3) / 2;
  items.forEach((it, i) => {
    const x = M + (i % 2) * (cw + 0.3), yy = y + Math.floor(i / 2) * (ch + 0.3);
    card(s, x, yy, cw, ch);
    numCircle(s, x + 0.25, yy + 0.25, i + 1, 0.38);
    s.addText(it[0], { x: x + 0.75, y: yy + 0.2, w: cw - 1.0, h: 0.48, fontFace: FONT, fontSize: 16, bold: true, color: C.deep, isTextBox: true, margin: 0, valign: "middle" });
    para(s, x + 0.25, yy + 0.85, cw - 0.5, [{ text: it[1] }], { size: 13 });
  });
  s.addNotes("Honest caveats. The ladder is asymmetric: one training domain and two far-OOD targets, so modality shift is confounded with lesion-type mismatch; read the ranking, not the absolute Dice. The same pixel jitter is a much harder task far-OOD. The CKA correlations are descriptive. And the recipe is fixed: five epochs, rank 8, ten prompt tokens, VPT as additive perturbation because of SAM's windowed attention. [11:40]");
}

// =====================================================================
// 12. Take-home
// =====================================================================
{
  const s = newSlide(true);
  let y = title(s, "Take-home", "Three things to remember", true);
  const items = [
    ["Adaptation can hurt.", "LoRA and VPT beat zero-shot in-domain and fall below it on far-OOD modalities, with boundary errors in the hundreds of pixels."],
    ["Prompt noise must be trained for.", "Random 0–100 px jitter gives the most reliable adapters; only Full FT and encoder-only LoRA end up above zero-shot overall."],
    ["The decoder is the locus.", "Preserving decoder and output representations predicts far-OOD robustness (ρ ≈ 0.73–0.76); encoder similarity does not (ρ = 0.09). Encoder-only LoRA is the CKA-informed adapter."],
  ];
  items.forEach((it, i) => {
    const yy = y + 0.1 + i * 1.35;
    s.addShape(pres.ShapeType.ellipse, { x: M, y: yy + 0.05, w: 0.6, h: 0.6, fill: { color: "9FD1D6" }, line: { color: "9FD1D6", width: 0 } });
    s.addText(String(i + 1), { x: M, y: yy + 0.05, w: 0.6, h: 0.6, fontFace: FONT, fontSize: 20, bold: true, color: C.deep, align: "center", valign: "middle", isTextBox: true, margin: 0 });
    s.addText(it[0], { x: M + 0.85, y: yy, w: 9.3, h: 0.45, fontFace: FONT, fontSize: 20, bold: true, color: C.white, isTextBox: true, margin: 0 });
    para(s, M + 0.85, yy + 0.5, 9.3, [{ text: it[1] }], { size: 14, color: "DDEDEF" });
  });
  card(s, W - M - 1.5, y + 0.1, 1.5, 1.5, C.white);
  s.addImage({ path: path.join(IMG, "qr_repo.png"), x: W - M - 1.4, y: y + 0.2, w: 1.3, h: 1.3 });
  s.addText("Code, configs, checkpoints\ngithub.com/ImSounic/medsam-vpt", { x: W - M - 2.8, y: y + 1.7, w: 2.8, h: 0.55, fontFace: FONT, fontSize: 10, color: "9FD1D6", align: "right", isTextBox: true, margin: 0 });
  s.addText("Thank you. Questions?  ·  Poster session 12:30–13:30 today", { x: M, y: H - 0.95, w: 9, h: 0.4, fontFace: FONT, fontSize: 16, color: C.white, isTextBox: true, margin: 0 });
  s.addText("Marko Haralović · Sounic Akkaraju · Carlo Baretta · Vasil Zapryanov · Alexia Briassouli   |   University of Twente · University of Zagreb (FER)", { x: M, y: H - 0.55, w: W - 2 * M, h: 0.3, fontFace: FONT, fontSize: 11, color: "9FD1D6", isTextBox: true, margin: 0 });
  s.addNotes("Three things to take away. Adaptation can hurt: LoRA and VPT beat zero-shot in-domain and fall below it far-OOD, with boundary errors in the hundreds of pixels. Prompt noise must be trained for: random jitter gives the most reliable adapters, and only Full FT and encoder-only LoRA end up above zero-shot overall. The decoder is the locus: preserving decoder and output representations predicts far-OOD robustness, encoder similarity does not, and encoder-only LoRA follows from that. Code and configs are on GitHub. Thank you; I will be at the poster over lunch. [12:00]");
}

// push cards behind their content
const isCard = (o) => o && o.options && o.options.rectRadius === 0.1 && !o.text;
pres.slides.forEach((sl) => {
  const objs = sl._slideObjects;
  sl._slideObjects = [...objs.filter(isCard), ...objs.filter((o) => !isCard(o))];
});

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f, "slides:", slideNo));
