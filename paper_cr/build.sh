#!/usr/bin/env bash
# Local check with tectonic (XeTeX): axessibility is pdfTeX-only, so it is commented out
# here; Overleaf (pdflatex) builds the file unchanged. Usage: bash build.sh main|submitted
set -euo pipefail
cd "$(dirname "$0")"
doc="${1:-main}"
mkdir -p build
sed 's/^\\usepackage\[accsupp\]{axessibility}/% &/' "$doc.tex" > "_local_$doc.tex"
tectonic -o build "_local_$doc.tex" 2>&1 | grep -iE "^error|undefined" || true
mv "build/_local_$doc.pdf" "build/$doc.pdf"; rm -f "_local_$doc.tex"
ref=$(for p in $(seq 1 20); do pdftotext -f "$p" -l "$p" "build/$doc.pdf" - 2>/dev/null | grep -q '^References$' && { echo "$p"; break; }; done)
echo "[build] build/$doc.pdf: $(pdfinfo "build/$doc.pdf" | awk '/Pages/ {print $2}') pages, References on page ${ref:-?}"
