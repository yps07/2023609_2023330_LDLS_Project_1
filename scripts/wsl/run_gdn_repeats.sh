#!/usr/bin/env bash
set -u
PROJ="/mnt/d/Padhai/sem 7/ldls/LDLS Project 1"
OUT="$PROJ/results"
for i in 1 2 3; do
  echo "=== run $i ==="
  /root/gdn_bench > "$OUT/gdn_run${i}.csv" 2> "$OUT/gdn_run${i}.err"
  echo "rc=$? rows=$(tail -n +2 $OUT/gdn_run${i}.csv | wc -l)"
done
echo "ALL DONE"
