#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PAPER="$ROOT/paper/iclr2027"
TECTONIC="${TECTONIC:-$HOME/.cache/opd-paper-tools/bin/tectonic}"
export FONTCONFIG_FILE="${FONTCONFIG_FILE:-/etc/fonts/fonts.conf}"
cd "$PAPER"
for lang in en zh; do
    mkdir -p "build/$lang"
    "$TECTONIC" --keep-logs --keep-intermediates --outdir "build/$lang" "main_$lang.tex"
    pdftotext -layout "build/$lang/main_$lang.pdf" "build/$lang/main_$lang.txt"
done
python3 "$ROOT/scripts/check_iclr2027_paper.py" "$PAPER" "$@"
