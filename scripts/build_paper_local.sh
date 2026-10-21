#!/usr/bin/env bash
# Local page-fit check with tectonic (XeTeX). The camera-ready itself is built with
# pdflatex on Overleaf; axessibility is pdfTeX-only, so it is commented out here.
# Usage: bash scripts/build_paper_local.sh [out_dir]   (builds main and supplement)
set -euo pipefail
cd "$(dirname "$0")/../paper"
OUT="${1:-build}"
mkdir -p "$OUT"
for doc in main supplement; do
  sed 's/^\\usepackage\[accsupp\]{axessibility}/% &/' "$doc.tex" > "_local_$doc.tex"
  tectonic -o "$OUT" "_local_$doc.tex" 2>&1 | grep -iE "^error|undefined" || true
  mv "$OUT/_local_$doc.pdf" "$OUT/$doc.pdf"
  rm -f "_local_$doc.tex"
  echo "[build] $OUT/$doc.pdf: $(pdfinfo "$OUT/$doc.pdf" | awk '/Pages/ {print $2}') pages"
done
ref_page=$(for p in $(seq 1 20); do pdftotext -f "$p" -l "$p" "$OUT/main.pdf" - 2>/dev/null | grep -q '^References$' && { echo "$p"; break; }; done)
echo "[build] References start on page ${ref_page:-?} (body must end by page 8)"
