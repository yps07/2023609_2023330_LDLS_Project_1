#!/usr/bin/env python3
"""Plotting code for Option 5 (Gated DeltaNet operator study).

Reads results/gdn_main.csv and results/gdn_exp.csv, writes PNGs to plots/.
Run from the repo root:  python plots/plot_gdn.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(ROOT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# ---- load main matrix ----
rows = []
with open(os.path.join(RES, "gdn_main.csv")) as f:
    for r in csv.DictReader(f):
        rows.append(r)

def mean(nt, th):
    for r in rows:
        if int(r["n_tokens"]) == nt and int(r["n_threads"]) == th:
            return float(r["mean_us"])
    return None

# 1. Decode step cost vs threads
ths = [1, 2, 4, 8, 12]
dec = [mean(1, t) for t in ths]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ths, dec, marker="o", linewidth=2)
ax.set_xlabel("Threads")
ax.set_ylabel("Decode step time (us)")
ax.set_title("Gated DeltaNet decode step vs. threads (n_tokens=1)")
ax.grid(True, alpha=0.3)
ax.set_xticks(ths)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "decode_step_vs_threads.png"), dpi=150)
plt.close(fig)

# 2. Per-token cost vs n_tokens (linearity)
toks = [1, 64, 256, 512, 1024]
per_tok = [mean(t, 1) / t for t in toks]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(toks, per_tok, marker="o", linewidth=2)
ax.set_xlabel("n_tokens (batch size)")
ax.set_ylabel("Time per token (us)")
ax.set_title("Gated DeltaNet: per-token cost vs. batch size (1 thread)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "per_token_vs_batch.png"), dpi=150)
plt.close(fig)

# 3. Thread scaling (decode, normalized to 1 thread)
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
fig.savefig(os.path.join(PLOTS, "thread_scaling.png"), dpi=150)
plt.close(fig)

# 4. Predicted vs measured (roofline)
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
fig.savefig(os.path.join(PLOTS, "predicted_vs_measured.png"), dpi=150)
plt.close(fig)

# 5. State vs KV cache
contexts = [1024, 8192, 32768]
state_mb = 2.097152  # MB
# attention KV cache: 2 (K+V) * n_kv_heads * 128 * ctx * bytes
def kv_mb(nh, eb, L):
    return 2 * nh * 128 * L * eb / 1e6
fig, ax = plt.subplots(figsize=(7, 4))
styles = [
    (8, 2, "Attn KV, 8 heads, F16", "tab:blue", "o", "-"),
    (8, 1, "Attn KV, 8 heads, Q8_0", "tab:orange", "s", "-"),
    (4, 2, "Attn KV, 4 heads, F16", "tab:green", "^", "-"),
]
for nh, eb, label, color, marker, ls in styles:
    ax.plot(contexts, [kv_mb(nh, eb, L) for L in contexts],
            marker=marker, linestyle=ls, color=color, linewidth=2, markersize=7, label=label)
ax.axhline(state_mb, linestyle="--", color="black", linewidth=2, label=f"GDN state (fixed, {state_mb:.2f} MB)")
ax.set_xlabel("Context length (tokens)")
ax.set_ylabel("Memory (MB)")
ax.set_title("GDN state vs. attention KV cache")
ax.set_xscale("log", base=2)
ax.set_xticks(contexts)
ax.set_xticklabels([f"{c//1024}k" if c >= 1024 else str(c) for c in contexts])
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "state_vs_kv_cache.png"), dpi=150)
plt.close(fig)

# 6. Own experiment: K and layout
exp = {}
with open(os.path.join(RES, "gdn_exp.csv")) as f:
    for r in csv.DictReader(f):
        exp[r["variant"]] = float(r["mean_us"])
fig, ax = plt.subplots(figsize=(6, 4))
names = list(exp.keys())
vals = [exp[n] for n in names]
ax.bar(names, vals)
ax.set_ylabel("Decode step time (us)")
ax.set_title("Own experiment: K slots and tensor layout (4 threads)")
ax.tick_params(axis="x", rotation=15)
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "experiment_k_layout.png"), dpi=150)
plt.close(fig)

print("Wrote 6 plots to", PLOTS)
for fn in sorted(os.listdir(PLOTS)):
    print(" ", fn)
