#!/usr/bin/env bash
# Phase A, Step 2 (cont.): record hardware details required by the project rules.
set -euo pipefail
PROJ="/mnt/d/Padhai/sem 7/ldls/LDLS Project 1"
OUT="$PROJ/docs/hardware.md"
mkdir -p "$PROJ/docs"

{
echo "# Hardware Record"
echo
echo "Date: $(date -Iseconds)"
echo
echo '## CPU (lscpu)'
echo '```'
lscpu
echo '```'
echo
echo '## Caches (lscpu -C)'
echo '```'
lscpu -C
echo '```'
echo
echo '## SIMD / CPU flags (grep)'
echo '```'
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|avx512.*|fma|f16c|sse4_2)$' | sort | tr '\n' ' '
echo
echo '```'
echo
echo '## Memory'
echo '```'
free -h
echo '```'
echo
echo '## NUMA / topology notes'
echo '```'
lscpu | grep -Ei 'numa|socket|core|thread'
echo '```'
} > "$OUT"

echo "Wrote $OUT"
