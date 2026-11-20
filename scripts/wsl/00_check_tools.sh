#!/usr/bin/env bash
# Phase A, Step 0: check which build tools already exist in WSL.
set -u
for t in git gcc g++ cmake make python3 curl wget; do
  printf '%-8s ' "$t"
  if command -v "$t" >/dev/null 2>&1; then
    "$t" --version 2>/dev/null | head -1
  else
    echo MISSING
  fi
done
