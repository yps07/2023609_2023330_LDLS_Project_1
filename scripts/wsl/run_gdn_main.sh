#!/usr/bin/env bash
# Run Option 5 GDN operator benchmark (main matrix) -> results/gdn_main.csv
set -u
PROJ="/mnt/d/Padhai/sem 7/ldls/LDLS Project 1"
OUT="$PROJ/results"
mkdir -p "$OUT"
"$HOME/gdn_bench" > "$OUT/gdn_main.csv" 2> "$OUT/gdn_main.err"
echo "rc=$?"
echo "--- stderr ---"; cat "$OUT/gdn_main.err"
echo "--- csv ---"; cat "$OUT/gdn_main.csv"
