#!/usr/bin/env bash
# Phase A, Step 2: STREAM memory bandwidth benchmark.
# Downloads stream.c, compiles with OpenMP, runs a small thread sweep.
# Run on an idle machine, plugged in, performance power mode.
set -euo pipefail

PROJ="/mnt/d/Padhai/sem 7/ldls/LDLS Project 1"
WORK="$HOME/stream"
mkdir -p "$WORK" "$PROJ/results"
cd "$WORK"

if [ ! -f stream.c ]; then
  wget -q https://www.cs.virginia.edu/stream/FTP/Code/stream.c
fi

# Array size: 60M elements -> 3 arrays x 480 MB = 1.44 GB, far above the 12 MB LLC.
gcc -O3 -march=native -fopenmp -DSTREAM_ARRAY_SIZE=60000000 -DNTIMES=20 stream.c -o stream

OUT="$PROJ/results/stream_$(date +%Y%m%d_%H%M%S).csv"
echo "threads,copy_MBps,scale_MBps,add_MBps,triad_MBps" | tee "$OUT"
for t in 1 2 4 6 8 12; do
  line=$(OMP_NUM_THREADS=$t ./stream | awk '/^(Copy|Scale|Add|Triad):/{printf "%s%s", sep, $2; sep=","}')
  echo "$t,$line" | tee -a "$OUT"
done
echo "Saved: $OUT"
