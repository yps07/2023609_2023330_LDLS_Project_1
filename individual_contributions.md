# Individual Contributions — LDLS Project 1 (Option 5: Gated DeltaNet Operator Study)

**Team members (equal contribution):**
- Yash Priya Shastri
- Muffadal Alavi

---

## Yash Priya Shastri

### Contributions
1. **Environment & build setup** — WSL2 configuration (8 GB RAM, 12 vCPUs), llama.cpp clone at commit `51a77c4`, CPU-only build with `-march=native`, and build log capture.
2. **Hardware measurement** — STREAM memory bandwidth benchmark (triad 15.1 GB/s → 15.9 GB/s plateau), hardware record (`docs/hardware.md`: CPU model, cores, cache sizes, SIMD flags, RAM, NUMA topology).
3. **C++ driver development** — `src/gdn_bench.cpp`: ggml compute graph construction, synthetic tensor initialization (Qwen3-Next shapes: H_k=16, H_v=32, S=128), high-resolution timing loop (200 iters, 20 warmup), thread/batch sweeps, `--dump` correctness mode, `--exp` experiment mode.
4. **Roofline prediction** — `scripts/wsl/step4_predict.py`: bytes-per-decode-step calculation (4,243,712 bytes, 98.8% state traffic), bandwidth-based time prediction, predicted-vs-measured comparison, state-vs-KV-cache memory table, `docs/prediction.md`.
5. **PARAM Rudra shared reference run** — Setup, build, and benchmark on Rudra CPU node (Intel Xeon Gold 6240R): sbatch job (4 CPUs, 5-min QoS), model transfer, llama-bench execution (Llama-3.2-1B Q4_K_M, pp512/tg128, 3 reps), peak RSS measurement (1.36 GB), raw result capture (`results/rudra_reference.csv`), and Rudra artifact organization (`6_group/p1_rudra/`).

### Files
- `src/gdn_bench.cpp`
- `scripts/wsl/00_check_tools.sh`
- `scripts/wsl/01_clone_build_llamacpp.sh`
- `scripts/wsl/02_stream.sh`
- `scripts/wsl/03_hardware_record.sh`
- `scripts/wsl/build_gdn.sh`
- `scripts/wsl/step4_predict.py`
- `docs/hardware.md`
- `docs/prediction.md`
- `docs/llamacpp_commit.txt`
- `results/build_log.txt`
- `results/stream_20261119_192851.csv`
- `results/rudra_reference.csv`
- Rudra: `6_group/p1_rudra/` (llama.cpp build, models, ref_run.out, reference.sbatch, setup_login.sh)

---

## Muffadal Alavi

### Contributions
1. **Benchmark execution & data collection** — Ran the full 25-point main matrix (`scripts/wsl/run_gdn_main.sh` → `results/gdn_main.csv`), 3 repeat runs (`scripts/wsl/run_gdn_repeats.sh` → `gdn_run1/2/3.csv`), and peak RSS measurement (`results/gdn_rss.txt`).
2. **Correctness verification** — Cross-checked driver output against `test-backend-ops` GATED_DELTA_NET (CPU backend); ran `--dump` mode and confirmed deterministic output (528,384 finite elements, identical min/max/sum/sumsq across runs).
3. **Own experiment** — Designed and ran the K-snapshot-slots (K=1 vs K=4) and tensor-layout (contiguous vs permuted) experiment (`results/gdn_exp.csv`: 95.91 → 213.07 µs, +122%).
4. **Results analysis & summarization** — `scripts/wsl/summarize_repeats.py` (run-to-run spread: 4–40%); compiled all raw CSVs into summary tables; documented prefill linearity (~205–219 µs/token) and thread-scaling saturation.
5. **Plotting & visualization** — `plots/plot_gdn.py` and `plots/plot_required.py`: 6 original plots + 5 required plots (decode speed vs context, prefill speed vs prompt, thread scaling, operator breakdown, predicted vs measured). All figures use actual benchmark data.
6. **Project planning & documentation** — `docs/plan.md` (phase tracking, task checklist); `README.md` (hardware, layout, rules compliance, reproduce instructions).
7. **Report** — Wrote the 4-page report (`report.docx`, `report.pdf`) covering all required sections: hardware, setup, results, profiling, roofline, KV-cache comparison, own experiment, Rudra reference run, and conclusions.

### Files
- `scripts/wsl/run_gdn_main.sh`
- `scripts/wsl/run_gdn_repeats.sh`
- `scripts/wsl/summarize_repeats.py`
- `plots/plot_gdn.py`
- `plots/plot_required.py`
- `plots/*.png` (11 figures total)
- `results/gdn_main.csv`
- `results/gdn_exp.csv`
- `results/gdn_run1.csv`, `gdn_run2.csv`, `gdn_run3.csv`
- `results/gdn_repeats_summary.csv`
- `results/gdn_rss.txt`
- `docs/plan.md`
- `README.md`
- `report/report.docx`
- `report/report.pdf`

---

## Shared / Joint Work
- Project scope and option selection (Option 5: Gated DeltaNet operator study).
- Interpretation of roofline results and gap explanation.
- Slides creation and 10-minute talk preparation.
- Final review of all deliverables before submission.

---

## Note
Both contributors participated equally across setup, measurement, analysis, and documentation. The file-level split above reflects primary ownership; in practice, both reviewed and contributed to all artifacts.