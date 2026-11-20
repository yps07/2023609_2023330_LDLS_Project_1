# Option 5 (Gated DeltaNet) — Working Plan

Chosen option: **5 — Gated DeltaNet (operator study)** on synthetic tensors with
Qwen3-Next layer shapes (16 query/key heads, 32 value heads, head size 128).
Engine: llama.cpp CPU-only build (already done), plus a small C++ driver
(`src/gdn_bench.cpp`) that times `ggml_gated_delta_net` directly.
No full model (smallest Gated DeltaNet models need ~20 GB).

**Key question:** What does one decode step cost? How does the fixed recurrent
state memory compare with the KV cache of an attention layer at 1k–32k context?

PARAM Rudra shared run (Llama-3.2-1B Q4_K_M, 512 prompt / 128 decode): pending access.

## Phase A — Setup & correctness (Day 1–2)

- [x] WSL2 memory raised to 8 GB (`.wslconfig`), swap 2 GB
- [x] Build llama.cpp CPU-only (`scripts/wsl/01_clone_build_llamacpp.sh`) — commit 51a77c4
- [x] STREAM memory bandwidth (`scripts/wsl/02_stream.sh`) — triad ~44 GB/s plateau
- [x] Hardware record (`scripts/wsl/03_hardware_record.sh`) — `docs/hardware.md`
- [x] Driver compiles and runs (`scripts/wsl/build_gdn.sh`) — `~/gdn_bench`
- [x] Sanity check: decode step ~321 us at 1 thread (matches a plausible cost)

## Phase B — Measurement (Day 3–4)

- [x] Run main matrix (`scripts/wsl/run_gdn_main.sh`) — 25 points, `results/gdn_main.csv`
- [x] Run own experiment (`gdn_bench --exp`) — K=1 vs K=4, permuted vs contiguous; `results/gdn_exp.csv`
- [x] Roofline prediction (`docs/prediction.md`) — bytes/step, predicted vs measured, KV cache comparison
- [x] Plots (`plots/plot_gdn.py` + 6 PNGs)

## Phase C — Profile & explain (Day 5–6)

- [x] Per-operator breakdown: single op dominates (GATED_DELTA_NET); documented in prediction.md and plan
- [x] Profiling method: driver high-res clock + ggml compute; test-backend-ops confirms op runs on CPU backend
- [x] Peak RSS measured: 74.6 MB (`results/gdn_rss.txt`), well under 4.5 GB
- [x] 3-run repeat (`results/gdn_repeats_summary.csv`): 5-8% spread at 1-4 threads, 16% at 8T, 32.5% at 12T
- [x] Correctness check: `--dump` output deterministic across 2 runs (identical min/max/sum/sumsq), all finite
- [x] Roofline: predicted vs measured, gap explained (`docs/prediction.md`)
- [x] Own experiment: K=1 vs K=4 (+21%), contiguous vs permuted; `results/gdn_exp.csv`

## Phase D — Shared run & write-up (Day 7)

- [ ] PARAM Rudra reference run (when access arrives)
- [ ] 4-page report, 10-min talk
- [ ] Required plots: decode step cost vs threads; state vs KV cache (1k–32k);
      predicted vs measured; thread scaling; (operator breakdown / experiment)

## Notes

- No perplexity (no model). Correctness = output matches test-backend-ops GATED_DELTA_NET.
- Memory budget trivially met (~2 MB state + few MB buffers, << 4.5 GB).