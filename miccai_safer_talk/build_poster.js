// A0 portrait poster for "When Adaptation Hurts" (MICCAI 2026 SAFER).
// Run: node build_poster.js  -> writes ../../../../medsam-vpt path given below.
const pptxgen = require("pptxgenjs");
const path = require("path");

const REPO = "/Users/imsounic/medsam-vpt";
const IMG = path.join(REPO, "miccai_safer_talk/pdf_images");
const FIG = path.join(REPO, "figures");
const OUT = path.join(REPO, "miccai_safer_talk/poster_A0.pptx");

// ---------- palette ----------
const C = {
  ink: "12262E",      // body text
  deep: "0F2F3C",     // header + take-home background
  teal: "0E7C86",     // section headings, table header
  tealSoft: "DDEDEF", // ladder chips
  card: "F2F6F7",     // card fill
  line: "D5DEE2",
  muted: "5B6B72",
  bad: "C8452C",
  good: "1E8E5A",
  white: "FFFFFF",
};
const FONT = "Calibri";

// ---------- geometry (inches) ----------
const W = 33.11, H = 46.81;
const M = 0.8;                 // outer margin
const HEADER_H = 5.0;
const COL_GAP = 0.55;
const COL_W = (W - 2 * M - 2 * COL_GAP) / 3;
const COL_X = [M, M + COL_W + COL_GAP, M + 2 * (COL_W + COL_GAP)];
const TOP = M + HEADER_H + 0.55;
const BOTTOM = H - M;
const PAD = 0.3;              // card inner padding
const GAP = 0.35;              // gap between cards

// type sizes (pt)
const T = { title: 78, sub: 34, aff: 26, h: 34, body: 25, table: 20, cap: 19, small: 19 };

const pres = new pptxgen();
pres.defineLayout({ name: "A0P", width: W, height: H });
pres.layout = "A0P";
pres.author = "Haralović, Akkaraju, Baretta, Zapryanov, Briassouli";
pres.title = "When Adaptation Hurts — MICCAI 2026 SAFER poster";
const s = pres.addSlide();
s.background = { color: C.white };

// ---------- helpers ----------
function card(x, y, w, h, fill = C.card) {
  s.addShape(pres.ShapeType.roundRect, {
    x, y, w, h, fill: { color: fill }, line: { color: fill, width: 0 }, rectRadius: 0.18,
  });
}
function heading(x, y, w, text, color = C.teal) {
  const cpl = Math.floor(w / (T.h / 72 * 0.53));
  const lines = Math.max(1, Math.ceil(text.length / cpl));
  const h = lines * (T.h / 72 * 1.2) + 0.08;
  s.addText(text, {
    x, y, w, h, fontFace: FONT, fontSize: T.h, bold: true, color, isTextBox: true, margin: 0,
    valign: "top",
  });
  return y + h + 0.08;
}
// bullets: array of strings or arrays of runs [{text, options}]. Glyph drawn as its own box
// because pptxgenjs cannot put a bullet on a multi-run paragraph without splitting it.
function bullets(x, y, w, items, opts = {}) {
  const size = opts.size || T.body;
  const lineH = size / 72 * 1.28;
  const ind = size / 72 * 1.1;
  const tw = w - ind;
  const charsPerLine = Math.floor(tw / (size / 72 * 0.47));
  let yy = y;
  items.forEach((it, i) => {
    const runs = Array.isArray(it) ? it : [{ text: it }];
    const txt = runs.map(r => r.text).join("");
    const lines = Math.max(1, Math.ceil(txt.length / charsPerLine));
    const h = lines * lineH + 0.05;
    const glyph = opts.numbered ? `${i + 1}.` : "•";
    s.addText(glyph, { x, y: yy, w: ind, h: lineH + 0.05, fontFace: FONT, fontSize: size, bold: !!opts.numbered, color: opts.glyphColor || opts.color || C.teal, isTextBox: true, margin: 0, valign: "top" });
    s.addText(runs.map(r => ({ text: r.text, options: Object.assign({ fontFace: FONT, fontSize: size, color: opts.color || C.ink }, r.options || {}) })),
      { x: x + ind, y: yy, w: tw, h, isTextBox: true, margin: 0, valign: "top" });
    yy += h + 0.12;
  });
  return yy;
}
function para(x, y, w, runs, opts = {}) {
  const size = opts.size || T.body;
  const lineH = size / 72 * 1.28;
  const charsPerLine = Math.floor(w / (size / 72 * 0.47));
  const txt = runs.map(r => r.text).join("");
  const lines = Math.max(1, Math.ceil(txt.length / charsPerLine));
  const h = lines * lineH + 0.08;
  s.addText(runs.map(r => ({ text: r.text, options: Object.assign({ fontFace: FONT, fontSize: size, color: opts.color || C.ink }, r.options || {}) })),
    { x, y, w, h, isTextBox: true, margin: 0, valign: "top", align: opts.align || "left" });
  return y + h;
}
// rows: array of arrays; each cell string or {text, bold, color, fill}
function table(x, y, w, rows, colW, opts = {}) {
  const size = opts.size || T.table;
  const base = size / 72 * 1.75;
  const rowH = rows.map((r) => {
    let lines = 1;
    r.forEach((c, ci) => {
      const txt = typeof c === "string" ? c : c.text;
      const cpl = Math.floor((colW[ci] - 0.17) / (size / 72 * 0.5));
      lines = Math.max(lines, Math.ceil(txt.length / cpl));
    });
    return base + (lines - 1) * size / 72 * 1.15;
  });
  const data = rows.map((r, ri) => r.map((c, ci) => {
    const cell = typeof c === "string" ? { text: c } : c;
    const isHead = ri === 0;
    return {
      text: cell.text,
      options: {
        fontFace: FONT, fontSize: size, bold: isHead || !!cell.bold,
        color: isHead ? C.white : (cell.color || C.ink),
        fill: { color: isHead ? C.teal : (cell.fill || (ri % 2 === 0 ? C.white : "E9F0F2")) },
        align: ci === 0 ? "left" : "center", valign: "middle",
        margin: [2, 6, 2, 6],
      },
    };
  }));
  s.addTable(data, {
    x, y, w, colW, rowH, border: { type: "solid", pt: 0.75, color: C.line }, fontFace: FONT,
  });
  return y + rowH.reduce((a, b) => a + b, 0);
}
function image(x, y, w, file, aspect, caption) {
  const h = w / aspect;
  s.addImage({ path: file, x, y, w, h });
  let yy = y + h + 0.08;
  if (caption) yy = para(x, yy, w, [{ text: caption }], { size: T.cap, color: C.muted });
  return yy;
}
const B = (t) => ({ text: t, options: { bold: true } });
const BAD = (t) => ({ text: t, options: { bold: true, color: C.bad } });
const GOOD = (t) => ({ text: t, options: { bold: true, color: C.good } });

// ================= HEADER =================
s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: W, h: M + HEADER_H, fill: { color: C.deep }, line: { color: C.deep, width: 0 } });
const qrW = 3.4;
// University logos, stacked in a white card left of the QR code. Drop PNGs into
// miccai_safer_talk/logos/ (fer_zagreb.png, eth_zurich.png, utwente.png); missing files are skipped.
const fs = require("fs");
const LOGOS = ["logos/fer_zagreb.png", "logos/eth_zurich.png", "logos/utwente.png"]
  .map((f) => path.join(REPO, "miccai_safer_talk", f)).filter((f) => fs.existsSync(f));
let logoCardX = W - M - qrW;   // left edge of whatever sits at the header's right
if (LOGOS.length) {
  const sizeOf = (f) => { const b = fs.readFileSync(f); return [b.readUInt32BE(16), b.readUInt32BE(20)]; }; // PNG IHDR
  const gap = 0.2, padX = 0.35;
  const lh = (qrW - gap * (LOGOS.length + 1)) / LOGOS.length;
  const dims = LOGOS.map((f) => { const [pw, ph] = sizeOf(f); const w = Math.min(lh * pw / ph, 4.2); return [w, w * ph / pw]; });
  const lw = Math.max(...dims.map((d) => d[0])) + 2 * padX;
  const lx = W - M - qrW - 0.4 - lw, ly = M + 0.05;
  logoCardX = lx;
  s.addShape(pres.ShapeType.roundRect, { x: lx, y: ly, w: lw, h: qrW, fill: { color: C.white }, line: { color: C.white, width: 0 }, rectRadius: 0.15 });
  let cy = ly + gap;
  LOGOS.forEach((f, i) => {
    const [w, h] = dims[i];
    s.addImage({ path: f, x: lx + (lw - w) / 2, y: cy + (lh - h) / 2, w, h });
    cy += lh + gap;
  });
}
const titleW = logoCardX - M - 0.4;
s.addText("When Adaptation Hurts", {
  x: M, y: M - 0.1, w: titleW, h: 1.5, fontFace: FONT, fontSize: T.title, bold: true, color: C.white, isTextBox: true, margin: 0, valign: "top",
});
s.addText("Connecting Representational Drift to OOD Failures in MedSAM Fine-Tuning", {
  x: M, y: M + 1.45, w: titleW, h: 1.0, fontFace: FONT, fontSize: 40, color: C.white, isTextBox: true, margin: 0, valign: "top",
});
s.addText([
  { text: "Marko Haralović", options: { bold: true } }, { text: "1,2,3", options: { superscript: true } }, { text: "   " },
  { text: "Sounic Akkaraju", options: { bold: true } }, { text: "3", options: { superscript: true } }, { text: "   " },
  { text: "Carlo Baretta", options: { bold: true } }, { text: "3", options: { superscript: true } }, { text: "   " },
  { text: "Vasil Zapryanov", options: { bold: true } }, { text: "3", options: { superscript: true } }, { text: "   " },
  { text: "Alexia Briassouli", options: { bold: true } }, { text: "3", options: { superscript: true } },
], { x: M, y: M + 2.75, w: titleW, h: 0.7, fontFace: FONT, fontSize: T.sub, color: C.white, isTextBox: true, margin: 0, valign: "top" });
s.addText("1 University of Zagreb, Faculty of Electrical Engineering and Computing   ·   2 ETH Zurich   ·   3 University of Twente, Enschede", {
  x: M, y: M + 3.5, w: titleW, h: 0.6, fontFace: FONT, fontSize: T.aff, color: "C9DCE2", isTextBox: true, margin: 0, valign: "top",
});
s.addText("MICCAI 2026 · SAFER Workshop: Stable Adaptation and Faithful Evaluation of Reasoning in Medical Foundation Models · Strasbourg, 27 September 2026", {
  x: M, y: M + 4.2, w: titleW, h: 0.6, fontFace: FONT, fontSize: T.aff, color: "C9DCE2", isTextBox: true, margin: 0, valign: "top",
});
// QR + repo
s.addShape(pres.ShapeType.roundRect, { x: W - M - qrW, y: M + 0.05, w: qrW, h: qrW, fill: { color: C.white }, line: { color: C.white, width: 0 }, rectRadius: 0.15 });
s.addImage({ path: path.join(IMG, "qr_repo.png"), x: W - M - qrW + 0.2, y: M + 0.25, w: qrW - 0.4, h: qrW - 0.4 });
s.addText("Code, configs, checkpoints\ngithub.com/ImSounic/medsam-vpt", {
  x: W - M - qrW - 0.6, y: M + qrW + 0.15, w: qrW + 1.2, h: 1.1, fontFace: FONT, fontSize: 22, color: C.white, align: "center", isTextBox: true, margin: 0, valign: "top",
});

// ================= COLUMN 1 =================
{
  const x = COL_X[0], w = COL_W, iw = w - 2 * PAD, ix = x + PAD;
  let y = TOP;

  // --- Why ---
  let y0 = y; let yy = y + PAD;
  yy = heading(ix, yy, iw, "Three challenges, studied separately");
  yy = bullets(ix, yy, iw, [
    [B("ROI sensitivity. "), { text: "MedSAM needs a bounding box. Clinical boxes are loose or inconsistent, and Dice follows box quality." }],
    [B("Distribution shift. "), { text: "Accuracy falls when scanner, organ or modality changes." }],
    [B("Adaptation cost. "), { text: "Fine-tuning helps in-domain but is expensive; LoRA and VPT are cheap yet sensitive to data and prompt quality." }],
  ]);
  yy += 0.12;
  yy = para(ix, yy, iw, [B("This work "), { text: "evaluates adaptation strategy × prompt noise × domain shift jointly, and links the outcome to which internal representations drift (linear CKA vs zero-shot MedSAM)." }]);
  yy += PAD;
  card(x, y0, w, yy - y0); // draw behind (z-order fixed below)
  y = yy + GAP;

  // --- Setup ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "Setup: adapters and the drift ladder");
  yy = table(ix, yy, iw, [
    ["Method", "Params", "Updated"],
    ["Zero-shot MedSAM", "0", "none (frozen baseline)"],
    ["Decoder-only FT", "4.1M", "mask decoder"],
    ["VPT-shallow", "4.1M", "10 tokens at input + decoder"],
    ["VPT-deep", "4.2M", "10 tokens per block + decoder"],
    ["LoRA (r = 8, α = 16)", "4.4M", "encoder + decoder adapters"],
    ["Encoder-only LoRA", "4.1M", "encoder adapters, decoder frozen"],
    ["Full FT", "93.7M", "image encoder + mask decoder"],
  ], [3.6, 1.4, iw - 5.0]);
  yy += 0.3;
  yy = para(ix, yy, iw, [B("Train "), { text: "on ISIC 2018 dermoscopy (2,594 images, 80/10/10), 5 epochs, AdamW, Dice + CE, prompt encoder frozen, 3 seeds. Full FT: LR 1e-5, batch 1. PEFT: LR 1e-4, batch 4." }]);
  yy += 0.25;
  yy = para(ix, yy, iw, [B("Evaluate along a drift ladder")]);
  yy += 0.1;
  // ladder chips
  const chips = [["ISIC 2018", "in-domain", "1,000 test"], ["PH²", "close-OOD dermoscopy", "200"], ["BUSI", "far-OOD ultrasound", "647"], ["CBIS-DDSM", "far-OOD mammography", "362"]];
  const cw = (iw - 3 * 0.25) / 4, ch = 1.55;
  chips.forEach((c, i) => {
    const cx = ix + i * (cw + 0.25);
    s.addShape(pres.ShapeType.roundRect, { x: cx, y: yy, w: cw, h: ch, fill: { color: i >= 2 ? "FBE4DE" : C.tealSoft }, line: { color: i >= 2 ? "FBE4DE" : C.tealSoft, width: 0 }, rectRadius: 0.12 });
    s.addText([
      { text: c[0], options: { bold: true, fontSize: 22, breakLine: true } },
      { text: c[1], options: { fontSize: 18, breakLine: true } },
      { text: c[2] + " images", options: { fontSize: 18, color: C.muted } },
    ], { x: cx + 0.1, y: yy + 0.08, w: cw - 0.2, h: ch - 0.16, fontFace: FONT, color: C.ink, align: "center", valign: "middle", isTextBox: true, margin: 0 });
  });
  yy += ch + 0.3;
  yy = para(ix, yy, iw, [B("Prompt protocol. "), { text: "Train with clean boxes, fixed 20 px, or random 0–100 px jitter. Evaluate at 0 / 20 / 50 / 100 / 200 px on 1024² images. 200 px = +280% box area on average, +1,328% on small CBIS-DDSM lesions." }]);
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy + GAP;

  // --- Fig 1 ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "What jitter looks like");
  yy = image(ix, yy, iw, path.join(FIG, "paper_visuals/prompt_perturbation_examples.png"), 4023 / 1142,
    "Fig. 1. Tight box (white, dashed), perturbed prompt (red), predicted mask (cyan). Equal pixel jitter is a far larger relative change on small far-OOD lesions.");
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy + GAP;

  // --- Result 2: prompt noise ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "Result 2 · Clean-box adapters are prompt-brittle");
  yy = image(ix, yy, iw, path.join(IMG, "p06_0_2460x1793.png"), 2460 / 1793,
    "Fig. 2. Dice vs evaluation jitter, averaged over the three training-jitter regimes.");
  yy += 0.15;
  yy = bullets(ix, yy, iw, [
    [{ text: "Zero-shot MedSAM " }, GOOD("improves"), { text: " with 20 px of slack (CBIS-DDSM 0.692 → 0.748) and is the best model on both far-OOD sets once boxes are 100 px off (CBIS 0.391 vs ≤ 0.367 for any adapter)." }],
    [{ text: "MedSAM was pretrained on imperfect boxes; clean-box adapters " }, BAD("unlearn that tolerance"), { text: ". Even encoder-only LoRA becomes the most jitter-sensitive model in-domain (ISIC, 200 px: 0.645 vs zero-shot 0.767)." }],
  ]);
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy;
  console.log("col1 end", y.toFixed(2), "of", BOTTOM);
}

// ================= COLUMN 2 =================
{
  const x = COL_X[1], w = COL_W, iw = w - 2 * PAD, ix = x + PAD;
  let y = TOP;

  // --- Result 1 ---
  let y0 = y, yy = y + PAD;
  yy = heading(ix, yy, iw, "Result 1 · Adaptation often hurts far-OOD");
  yy = para(ix, yy, iw, [{ text: "Dice with clean boxes, models trained with clean boxes, mean of 3 seeds" }], { size: T.cap, color: C.muted });
  yy += 0.1;
  const r1 = [
    ["Method", "ISIC (ID)", "PH² (close)", "BUSI (far)", "CBIS (far)"],
    ["Zero-shot", "0.907", "0.905", "0.823", "0.692"],
    ["Decoder-only", "0.949", "0.947", "0.894", "0.828"],
    ["VPT-shallow", "0.945", "0.943", { text: "0.792", color: C.bad, bold: true }, { text: "0.566", color: C.bad, bold: true }],
    ["VPT-deep", "0.947", "0.946", { text: "0.802", color: C.bad, bold: true }, { text: "0.573", color: C.bad, bold: true }],
    ["LoRA", "0.956", "0.957", { text: "0.780", color: C.bad, bold: true }, { text: "0.509", color: C.bad, bold: true }],
    ["Encoder-only LoRA", "0.958", { text: "0.960", bold: true, color: C.good }, { text: "0.908", bold: true, color: C.good }, "0.793"],
    ["Full FT", { text: "0.961", bold: true, color: C.good }, "0.958", "0.901", { text: "0.829", bold: true, color: C.good }],
  ];
  yy = table(ix, yy, iw, r1, [3.0, (iw - 3.0) / 4, (iw - 3.0) / 4, (iw - 3.0) / 4, (iw - 3.0) / 4]);
  yy += 0.25;
  yy = bullets(ix, yy, iw, [
    [{ text: "Every adapter beats zero-shot in-domain. LoRA and both VPT variants " }, BAD("fall below zero-shot"), { text: " on ultrasound and mammography (red)." }],
    [B("Full FT"), { text: " gives the best overall trade-off. " }, B("Encoder-only LoRA"), { text: " is the strongest parameter-efficient method: the same adapter as LoRA with the decoder left alone, +0.28 Dice on mammography." }],
  ]);
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy + GAP;

  // --- HD95 ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "Boundary error: collapse, not overlap loss");
  yy = para(ix, yy, iw, [{ text: "HD95 with clean boxes, median [IQR] in pixels at 1024²; lower is better. In-domain and close-OOD medians are 0–7 px for every method." }], { size: T.cap, color: C.muted });
  yy += 0.1;
  yy = table(ix, yy, iw, [
    ["Method", "BUSI (far-OOD)", "CBIS-DDSM (far-OOD)"],
    ["Zero-shot", "14 [9, 25]", "12 [7, 17]"],
    ["Decoder-only", "4 [2, 17]", "4 [3, 6]"],
    ["VPT-shallow", { text: "21 [5, 387]", color: C.bad, bold: true }, "14 [8, 37]"],
    ["VPT-deep", { text: "25 [6, 394]", color: C.bad, bold: true }, { text: "18 [10, 359]", color: C.bad, bold: true }],
    ["LoRA", { text: "26 [6, 402]", color: C.bad, bold: true }, { text: "47 [13, 438]", color: C.bad, bold: true }],
    ["Encoder-only LoRA", { text: "3 [1, 11]", color: C.good, bold: true }, "5 [3, 7]"],
    ["Full FT", "4 [1, 13]", { text: "4 [3, 6]", color: C.good, bold: true }],
  ], [3.2, (iw - 3.2) / 2, (iw - 3.2) / 2]);
  yy += 0.25;
  yy = bullets(ix, yy, iw, [
    [{ text: "For LoRA and VPT the upper quartile reaches " }, BAD("~400 px"), { text: ": a quarter of far-OOD cases segment the wrong structure entirely. The Dice gap to Full FT is ~0.12; the boundary gap is more than tenfold." }],
  ]);
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy + GAP;

  // --- Result 3: training jitter ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "Result 3 · Train with random 0–100 px jitter");
  yy = para(ix, yy, iw, [{ text: "Mean Dice gain over zero-shot: trained with clean boxes → trained with random 0–100 px jitter" }], { size: T.cap, color: C.muted });
  yy += 0.1;
  yy = table(ix, yy, iw, [
    ["Method", "In-domain", "Far-OOD", "Overall"],
    ["Full FT", { text: "−0.006 → +0.034", bold: true, color: C.good }, "−0.014 → −0.029", { text: "−0.009 → +0.013", bold: true, color: C.good }],
    ["Encoder-only LoRA", { text: "−0.027 → +0.053", bold: true, color: C.good }, "−0.024 → −0.054", { text: "−0.023 → +0.007", bold: true, color: C.good }],
    ["LoRA", "+0.012 → +0.035", { text: "−0.142 → −0.228", bold: true, color: C.bad }, "−0.038 → −0.051"],
    ["Decoder-only", "−0.016 → +0.014", { text: "−0.036 → −0.088", bold: true, color: C.bad }, "−0.023 → −0.022"],
    ["VPT-shallow", "−0.008 → +0.009", { text: "−0.127 → −0.230", bold: true, color: C.bad }, "−0.048 → −0.071"],
    ["VPT-deep", "−0.010 → +0.013", { text: "−0.126 → −0.198", bold: true, color: C.bad }, "−0.049 → −0.058"],
  ], [2.5, (iw - 2.5) / 3, (iw - 2.5) / 3, (iw - 2.5) / 3]);
  yy += 0.25;
  yy = bullets(ix, yy, iw, [
    [{ text: "Jitter training restores in-domain and close-OOD robustness for every adapter (Full FT at 200 px: 0.712 → 0.839; encoder-only LoRA: 0.645 → 0.789). Fixed 20 px training overfits one noise level." }],
    [{ text: "It costs far-OOD Dice. The cost is " }, GOOD("small"), { text: " for Full FT and encoder-only LoRA, " }, BAD("large"), { text: " for LoRA, VPT and decoder-only (LoRA on mammography: 0.509 → 0.214)." }],
    [{ text: "Only Full FT and encoder-only LoRA end up above zero-shot overall." }],
  ]);
  yy += 0.15;
  yy = image(ix, yy, iw, path.join(FIG, "paper_visuals/degradation_curves.png"), 2980 / 2106,
    "Fig. 3. Dice vs evaluation jitter per training regime (solid: clean, dashed: fixed 20 px, dotted: random 0–100 px). Dotted curves are flattest in-domain and lowest on CBIS-DDSM.");
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy;
  console.log("col2 end", y.toFixed(2), "of", BOTTOM);
}

// ================= COLUMN 3 =================
{
  const x = COL_X[2], w = COL_W, iw = w - 2 * PAD, ix = x + PAD;
  let y = TOP;

  // --- Mechanism ---
  let y0 = y, yy = y + PAD;
  yy = heading(ix, yy, iw, "Why does adaptation fail far-OOD?");
  yy = para(ix, yy, iw, [{ text: "Linear CKA scores how similar a layer's activations are before and after adaptation (1 = unchanged). We compute it per component against zero-shot MedSAM on the same images and ask whose drift predicts the far-OOD loss, i.e. which part of the model may change." }]);
  yy += 0.05;
  yy = para(ix, yy, iw, [{ text: "Spearman ρ between linear CKA (adapted vs zero-shot MedSAM) and Dice gain over zero-shot. n = 3,000 pooled observations; FDR-corrected * q<0.05, *** q<0.001" }], { size: T.cap, color: C.muted });
  yy += 0.1;
  yy = table(ix, yy, iw, [
    ["CKA feature", "ID", "Close-OOD", "Far-OOD"],
    ["IoU token layer", "0.10", "−0.02", { text: "0.76 ***", bold: true, color: C.good }],
    ["Output layer", "0.06", "0.00", { text: "0.76 ***", bold: true, color: C.good }],
    ["Decoder layers", "−0.08", "−0.02", { text: "0.73 ***", bold: true, color: C.good }],
    ["Upscaled embedding", "0.20", "0.29 *", { text: "0.71 ***", bold: true, color: C.good }],
    ["Encoder layers", "−0.28 *", "−0.28 *", { text: "0.09", bold: true, color: C.bad }],
  ], [3.6, (iw - 3.6) / 3, (iw - 3.6) / 3, (iw - 3.6) / 3]);
  yy += 0.25;
  yy = bullets(ix, yy, iw, [
    [B("Decoder-side similarity predicts far-OOD gain; encoder similarity does not"), { text: " (and is weakly negative in-domain)." }],
    [{ text: "Decoder CKA on CBIS-DDSM follows the Dice ranking exactly, from Full FT (0.95) down to LoRA (0.76)." }],
    [B("Encoder-only LoRA works"), { text: " because it adapts the encoder to the visual shift while preserving the decoder pathway." }],
  ]);
  yy += 0.15;
  yy = image(ix, yy, iw, path.join(IMG, "p16_0_3044x1096.png"), 3044 / 1096,
    "Fig. 4. Linear CKA to zero-shot MedSAM per component (clean boxes). Encoder blocks drift for LoRA and encoder-only LoRA alike; the decoder attention, IoU and mask tokens collapse only for LoRA and VPT.");
  yy += 0.15;
  const sw = iw * 0.48;
  const sh = sw / (926 / 622);
  s.addImage({ path: path.join(IMG, "fig5_farood_crop.png"), x: ix, y: yy, w: sw, h: sh });
  para(ix + sw + 0.3, yy + 0.1, iw - sw - 0.3, [
    B("Fig. 5. "), { text: "Far-OOD Dice gain vs overall CKA, one point per method × training jitter × evaluation jitter. Spearman 0.64. The lowest-similarity, lowest-gain cluster is LoRA and VPT; Full FT and encoder-only LoRA sit at the top right next to zero-shot." },
  ], { size: T.cap, color: C.muted });
  yy += sh + PAD;
  card(x, y0, w, yy - y0);
  y = yy + GAP;

  // --- Guidance ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "What we would do with MedSAM now");
  yy = bullets(ix, yy, iw, [
    [{ text: "With compute available and far-OOD data expected, full fine-tuning at LR 1e-5 gives the best trade-off in every regime." }],
    [{ text: "On a parameter budget, " }, B("encoder-only LoRA"), { text: " comes closest to Full FT far-OOD with 4.1M trainable parameters." }],
    [{ text: "For a single target domain with clean prompts, standard LoRA still gives the largest in-domain and close-OOD gains." }],
    [{ text: "Whatever the adapter, train with random 0–100 px box jitter; fixed jitter overfits one noise level." }],
    [{ text: "VPT on MedSAM is not worth it. The in-domain gain is negligible and the far-OOD loss is large." }],
    [{ text: "Under severe prompt noise far-OOD, zero-shot MedSAM is still hard to beat and remains the fallback." }],
  ]);
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy + GAP;

  // --- Limitations ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "Limitations");
  yy = bullets(ix, yy, iw, [
    [{ text: "One training domain and two far-OOD targets: modality shift is confounded with lesion-type mismatch. Read the ranking, not the absolute Dice." }],
    [{ text: "The same pixel jitter is a harder task far-OOD (+152% box area on ISIC vs +1,328% on CBIS-DDSM at 200 px)." }],
    [{ text: "CKA observations are not independent; the effective sample size is well below n = 3,000. Fixed recipe: 5 epochs, LoRA rank 8, 10 prompt tokens." }],
  ], { size: T.small });
  yy += PAD;
  card(x, y0, w, yy - y0);
  y = yy + GAP;

  // --- Take-home (dark) ---
  y0 = y; yy = y + PAD;
  yy = heading(ix, yy, iw, "Take-home", C.white);
  yy = bullets(ix, yy, iw, [
    [B("Adaptation can hurt. "), { text: "LoRA and VPT beat zero-shot in-domain and fall below it on far-OOD modalities, with boundary errors in the hundreds of pixels." }],
    [B("Prompt noise must be trained for. "), { text: "Random 0–100 px jitter gives the most reliable adapters; only Full FT and encoder-only LoRA end above zero-shot overall." }],
    [B("The decoder is the locus. "), { text: "Preserving decoder and output representations predicts far-OOD robustness (ρ ≈ 0.73–0.76); encoder similarity does not (ρ = 0.09). Encoder-only LoRA is the CKA-informed adapter." }],
  ], { color: C.white, numbered: true, glyphColor: C.white });
  yy += PAD;
  card(x, y0, w, yy - y0, C.deep);
  y = yy;
  console.log("col3 end", y.toFixed(2), "of", BOTTOM);
}

// Cards were added after their content, so push every roundRect card to the back.
// pptxgenjs has no z-order API; reorder the slide's internal object list instead.
const objs = s._slideObjects;
const isCard = (o) => o && o.options && o.options.rectRadius === 0.18 && !o.text;
const cards = objs.filter(isCard);
const rest = objs.filter((o) => !isCard(o));
// keep header rect (first object) in front of nothing: it is a plain rect, goes first anyway
s._slideObjects = [...cards, ...rest];

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
