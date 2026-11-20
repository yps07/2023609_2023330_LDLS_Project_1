#!/usr/bin/env bash
set -e
SRC="/mnt/d/Padhai/sem 7/ldls/LDLS Project 1/src/gdn_bench.cpp"
OUT="$HOME/gdn_bench"
INC="$HOME/llama.cpp/ggml/include"
LIB="$HOME/llama.cpp/build/bin"
g++ -O3 -march=native -std=c++17 \
  -I"$INC" \
  "$SRC" -o "$OUT" \
  -L"$LIB" -lggml -lggml-cpu -lggml-base \
  -Wl,-rpath,"$LIB" \
  -fopenmp -lpthread -ldl
echo "built: $OUT"
"$OUT" 2>&1 | head -5
