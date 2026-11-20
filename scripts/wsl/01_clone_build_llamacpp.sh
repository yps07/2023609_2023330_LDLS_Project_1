#!/usr/bin/env bash
# Phase A, Step 1: clone llama.cpp and build it CPU-only.
# Build lives in the WSL native filesystem (~/llama.cpp) for fast I/O and
# reliable mmap; the Windows project dir keeps only repo artifacts.
set -euo pipefail

SRC="$HOME/llama.cpp"
PROJ="/mnt/d/Padhai/sem 7/ldls/LDLS Project 1"

if [ ! -d "$SRC" ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$SRC"
else
  echo "llama.cpp already cloned at $SRC"
fi

cd "$SRC"
echo "=== llama.cpp commit ==="
git rev-parse HEAD | tee "$PROJ/docs/llamacpp_commit.txt"
git log -1 --format='%ci %s' | tee -a "$PROJ/docs/llamacpp_commit.txt"

echo "=== configure ==="
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_NATIVE=ON

echo "=== build ===
"
cmake --build build --config Release -j 12

echo "=== binaries ==="
ls build/bin/

echo "=== version ==="
./build/bin/llama-cli --version
