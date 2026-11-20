# LDLS Project 1 — Option 5: Gated DeltaNet operator study on CPU

CPU operator benchmarking of **llama.cpp's `ggml_gated_delta_net` operator** on
synthetic tensors with **Qwen3-Next layer shapes** (16 query/key heads, 32 value
heads, head size 128), per the course project spec (`Project 1.pdf`).

**Key question:** What does one decode step cost? How does the fixed recurrent
state memory compare with the KV cache of an attention layer at 1k–32k context?

This is an **operator study** (no full model — the smallest Gated DeltaNet models
need ~20 GB). We time the operator alone on synthetic tensors using a small
driver program built on top of the CPU-only llama.cpp build.

## Hardware

- Laptop CPU: AMD Ryzen 5 5625U (6 cores / 12 threads, AVX2, no AVX-512)
- RAM: 8 GB (Windows 11 host), WSL2 Ubuntu 24.04 capped at 6 GB (12 vCPUs)
- Full details: [docs/hardware.md](docs/hardware.md)

## Layout

| Path | Contents |
|---|---|
| `docs/` | Build notes, hardware record, plan, prediction |
| `scripts/wsl/` | Setup/benchmark scripts (run inside WSL2 Ubuntu) |
| `src/` | C++ driver (`gdn_bench.cpp`) for the operator |
| `results/` | Raw CSV results, logs, checksums |
| `plots/` | Plotting code and generated figures |
| `report/` | 4-page report |
| `slides/` | 10-minute talk |

llama.cpp source/build lives in the WSL native filesystem (`~/llama.cpp`); this
repo records the exact commit (`docs/llamacpp_commit.txt`).

## Rules compliance

- Memory budget: < 4.5 GB (operator uses ~2 MB state + a few MB buffers).
- Thread scaling: 1 thread up to all performance cores (1, 2, 4, 8, 12).
- Every measurement repeated 3 times; mean and spread reported (200 iters per run).
- Profiling, roofline, and one own experiment included.

## Shared reference run (PARAM Rudra)

Run on PARAM Rudra CPU node (Intel Xeon Gold 6240R, 48 cores, 187 GB RAM) via sbatch (4 CPUs, 5-min QoS limit):
Llama-3.2-1B Q4_K_M, 512-token prompt, 128-token decode, 4 threads, 3 reps.
- Prefill: 163.60 ± 3.24 tok/s
- Decode:  29.02 ± 0.19 tok/s
- Peak RSS: ~1.36 GB (well under 4.5 GB budget)
- Raw results: `results/rudra_reference.csv`
- Rudra artifacts: `6_group/p1_rudra/` on paramrudra.cdacdelhi.in (commit 51a77c4, build 12542)

## Reproduce

See `docs/plan.md` and run the numbered scripts in `scripts/wsl/` in order.

## Hardware

- Laptop CPU: AMD Ryzen 5 5625U (6 cores / 12 threads, AVX2, no AVX-512)
- RAM: 8 GB (Windows 11 host), WSL2 Ubuntu 24.04 capped at 6 GB (12 vCPUs)
- Full details: [docs/hardware.md](docs/hardware.md)

## Layout

| Path | Contents |
|---|---|
| `docs/` | Build notes, hardware record, plan |
| `scripts/wsl/` | All setup/benchmark scripts (run inside WSL2 Ubuntu) |
| `results/` | Raw CSV results, logs, checksums |
| `plots/` | Plotting code and generated figures |
| `report/` | 4-page report |
| `slides/` | 10-minute talk |

llama.cpp source/build and the GGUF models live in the WSL native filesystem
(`~/llama.cpp`, `~/models`) for fast I/O and correct mmap behaviour; this repo
records the exact commit (`docs/llamacpp_commit.txt`) and model checksums
(`results/model_sha256.txt`) so everything is reproducible.

## Rules compliance

- Quantization: Q4_K_M for all models; Q8_0 only for mamba-1.4b (<=1.5B params). No F16.
- Memory budget: model + state + buffers < 4.5 GB per run; peak RSS logged every run.
- Context: up to 32k tokens (pure recurrent model).
- Every measurement repeated 3x; mean and spread reported.

## Reproduce

See `docs/plan.md` and run the numbered scripts in `scripts/wsl/` in order.
