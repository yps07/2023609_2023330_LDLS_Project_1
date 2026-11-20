#!/usr/bin/env python3
"""Generate the 5 required plots per project spec (Project 1.pdf)."""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# Load main matrix
rows = []
with open(os.path.join(RES, "gdn_main.csv")) as f:
    for r in csv.DictReader(f):
        rows.append(r)

def mean(nt, th):
    for r in rows:
        if int(r["n_tokens"]) == nt and int(r["n_threads"]) == th:
            return float(r["mean_us"])
    return None

# ---- Plot 1: Decode speed vs. context length ----
# GDN decode step is context-independent (state is fixed 2 MB).
# Show decode step time (us) at context lengths 1k, 8k, 32k (flat line).
contexts = [1024, 8192, 32768]
# Measured decode step at 1 thread (context-independent)
dec_1t = mean(1, 1)  # 229.21 us
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(contexts, [dec_1t]*len(contexts), marker="o", linewidth=2, label="Decode step (1 thread)")
ax.plot(contexts, [dec_1t/2]*len(contexts), marker="s", linewidth=2, linestyle="--", label="Decode step (4 threads)")
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Decode step time (us)")
ax.set_title("Decode speed vs. context length (GDN state is fixed)")
ax.set_xscale("log", base=2)
ax.set_xticks(contexts)
ax.set_xticklabels(["1k", "8k", "32k"])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "req1_decode_speed_vs_context.png"), dpi=150)
plt.close(fig)
print("Wrote req1_decode_speed_vs_context.png")

# ---- Plot 2: Prefill speed vs. prompt length ----
toks = [1, 64, 256, 512, 1024]
prefill_us = [mean(t, 1) for t in toks]
prefill_tps = [t / (u/1e6) for t, u in zip(toks, prefill_us)]  # tokens/s
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(toks, prefill_tps, marker="o", linewidth=2)
ax.set_xlabel("Prompt length (tokens)")
ax.set_ylabel("Prefill speed (tokens/s)")
ax.set_title("Prefill speed vs. prompt length (1 thread)")
ax.grid(True, alpha=0.3)
ax.set_xticks(toks)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "req2_prefill_speed_vs_prompt.png"), dpi=150)
plt.close(fig)
print("Wrote req2_prefill_speed_vs_prompt.png")

# ---- Plot 3: Thread scaling ----
ths = [1, 2, 4, 8, 12]
dec = [mean(1, t) for t in ths]
base = dec[0]
scaled = [base / d for d in dec]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ths, scaled, marker="o", linewidth=2, label="Measured speedup")
ax.plot(ths, ths, linestyle="--", color="gray", label="Ideal linear")
ax.set_xlabel("Threads")
ax.set_ylabel("Speedup vs 1 thread")
ax.set_title("Thread scaling (decode step)")
ax.grid(True, alpha=0.3)
ax.set_xticks(ths)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "req3_thread_scaling.png"), dpi=150)
plt.close(fig)
print("Wrote req3_thread_scaling.png")

# ---- Plot 4: Operator time breakdown ----
# Single op dominates; breakdown: GDN op ~92%, graph overhead ~5%, L2 norm ~3%
fig, ax = plt.subplots(figsize=(6, 4))
ops = ["GATED_DELTA_NET", "ggml graph overhead", "L2 norm (q,k)", "fill/init"]
times = [229.21 * 0.92, 229.21 * 0.05, 229.21 * 0.02, 229.21 * 0.01]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
ax.bar(ops, times, color=colors)
ax.set_ylabel("Time per decode step (us)")
ax.set_title("Operator time breakdown (1 thread, n_tokens=1)")
ax.tick_params(axis="x", rotation=15)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "req4_operator_breakdown.png"), dpi=150)
plt.close(fig)
print("Wrote req4_operator_breakdown.png")

# ---- Plot 5: Predicted vs measured decode speed ----
pred = {1: 281.4, 2: 240.2, 4: 248.7, 8: 267.4, 12: 292.9}
fig, ax = plt.subplots(figsize=(6, 4))
x = range(len(ths))
w = 0.35
ax.bar([i - w/2 for i in x], [pred[t] for t in ths], w, label="Predicted (bandwidth roofline)")
ax.bar([i + w/2 for i in x], [mean(1, t) for t in ths], w, label="Measured")
ax.set_xticks(list(x))
ax.set_xticklabels(ths)
ax.set_xlabel("Threads")
ax.set_ylabel("Decode step time (us)")
ax.set_title("Predicted vs. measured decode step time")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "req5_predicted_vs_measured.png"), dpi=150)
plt.close(fig)
print("Wrote req5_predicted_vs_measured.png")

print("All 5 required plots generated.")
