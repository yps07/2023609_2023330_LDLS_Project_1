#!/usr/bin/env python3
"""Step 4 (roofline) + KV-cache comparison for Option 5 (Gated DeltaNet)."""
# GDN shapes (Qwen3-Next): H_k=16, H_v=32, S=128, n_tokens=1 (decode), K=1.
# State is F32 (ggml default for this op). Bytes/step = inputs + state(read+write) + outputs.
# Bandwidth from STREAM final idle run.

S, Hk, Hv = 128, 16, 32
K = 1
n_tokens = 1
n_seqs = 1

# bytes per element
F32 = 4

q_bytes  = S * Hk * n_tokens * n_seqs * F32
k_bytes  = S * Hk * n_tokens * n_seqs * F32
v_bytes  = S * Hv * n_tokens * n_seqs * F32
g_bytes  = 1 * Hv * n_tokens * n_seqs * F32
beta_bytes = 1 * Hv * n_tokens * n_seqs * F32
state_bytes = S * S * Hv * n_seqs * F32          # read
out_attn_bytes = S * Hv * n_tokens * n_seqs * F32
out_state_bytes = S * S * Hv * n_seqs * F32 * K  # written (K slots)
# state is also written (new state) -- counted in out_state_bytes

inputs = q_bytes + k_bytes + v_bytes + g_bytes + beta_bytes
state_rw = state_bytes + out_state_bytes  # read + write
outputs = out_attn_bytes
total = inputs + state_rw + outputs

# FLOPs per step (rough): recurrence is O(S^2 * Hv) per token, plus gates
flops = 2 * S * S * Hv * n_tokens * n_seqs  # FMA count x2

BW_1T = 15080.1e6   # bytes/s, 1-thread triad (this machine)
BW_8T = 15868.5e6   # 8-thread triad plateau (this machine)

pred_1t = total / BW_1T * 1e6   # us
pred_8t = total / BW_8T * 1e6

# measured (from gdn_main.csv): decode step (n_tokens=1)
measured = {1: 229.21, 2: 125.30, 4: 76.17, 8: 87.55, 12: 166.84}

lines = []
lines.append("# Step 4 — Roofline prediction (decode step)\n")
lines.append("Option 5 studies a single operator, so the \"decode step\" is one call to\n")
lines.append("`ggml_gated_delta_net` on a single token (n_tokens=1) with Qwen3-Next\n")
lines.append("shapes (H_k=16, H_v=32, S=128, K=1). Predicted **before** measuring.\n")
lines.append("\n## Bytes per decode step\n")
lines.append("| Component | Bytes |\n|---|---:|\n")
lines.append(f"| q input | {q_bytes:,} |\n")
lines.append(f"| k input | {k_bytes:,} |\n")
lines.append(f"| v input | {v_bytes:,} |\n")
lines.append(f"| g input | {g_bytes:,} |\n")
lines.append(f"| beta input | {beta_bytes:,} |\n")
lines.append(f"| state (read) | {state_bytes:,} |\n")
lines.append(f"| state (write, K={K} slot) | {out_state_bytes:,} |\n")
lines.append(f"| attention output | {out_attn_bytes:,} |\n")
lines.append(f"| **Total** | **{total:,}** |\n")
lines.append(f"\nState traffic (read+write) = {state_rw:,} bytes = "
            f"{100*state_rw/total:.1f}% of total. The state dominates; arithmetic\n")
lines.append(f"intensity ≈ {flops/total:.3f} FLOP/byte (memory-bound regime).\n")
lines.append(f"State size (fixed) = {state_bytes:,} bytes ≈ {state_bytes/1024/1024:.2f} MB, "
            f"independent of context (recurrent).\n")

lines.append("\n## Predicted decode-step time = bandwidth ÷ bytes/step\n")
lines.append("| Bandwidth | Predicted (us/step) |\n|---|---:|\n")
lines.append(f"| 1 thread, Triad 15.1 GB/s | {pred_1t:.1f} |\n")
lines.append(f"| 8 threads, Triad 15.9 GB/s | {pred_8t:.1f} |\n")

lines.append("\n## Predicted vs measured (decode step, us/step)\n")
lines.append("| Threads | Predicted (us) | Measured (us) | Ratio (meas/pred) |\n|---:|---:|---:|---:|\n")
for t, m in measured.items():
    bw = BW_1T if t == 1 else (BW_8T if t == 8 else None)
    pred = (total / (BW_1T if t == 1 else BW_8T)) * 1e6 if t in (1, 8) else None
    if pred is None:
        # scale bandwidth linearly-ish for other thread counts using measured STREAM ratios
        pass
    # use a simple bandwidth estimate for 2/4/12 threads from STREAM
    bw_map = {1: 15080.1e6, 2: 17666.7e6, 4: 17064.2e6, 8: 15868.5e6, 12: 14489.4e6}
    pred_t = total / bw_map[t] * 1e6
    ratio = m / pred_t
    lines.append(f"| {t} | {pred_t:.1f} | {m:.1f} | {ratio:.2f}x |\n")

lines.append("\n## Notes on the gap\n")
lines.append("- Measured is 0.31-0.81x of the bandwidth prediction: the state fits better in\n")
lines.append("  cache than the STREAM-based model assumes, and threading helps compute.\n")
lines.append("- The gap vs. a pure bandwidth model comes from: (a) cache effects on the 2 MB\n")
lines.append("  state (exceeds L2 on this CPU), (b) dequantization/normalization overhead,\n")
lines.append("  (c) thread scheduling/synchronization (esp. at 12 threads, where measured\n")
lines.append("      time regresses vs 8 threads), (d) ggml graph overhead.\n")
lines.append("- At 12 threads the op gets *slower* than at 8 (166.8 us vs 87.6 us): the state\n")
lines.append("  (2 MB) and tiny per-token work do not amortize thread spawn/sync cost.\n")
lines.append("- Prefill (n_tokens>1) is ~205-219 us/token regardless of batch size (linear),\n")
lines.append("  confirming the op scales linearly in n_tokens with no super-linear cost.\n")

# KV cache comparison
lines.append("\n## State memory vs. attention KV cache (1k–32k context)\n")
lines.append("Gated DeltaNet state is **fixed** (~2.0 MB) regardless of context. An\n")
lines.append("attention layer's KV cache **grows** with context:\n")
lines.append("KV_cache_bytes = 2 (K+V) × n_kv_heads × head_dim × context × bytes_per_elem.\n")
lines.append("\nGDN state (fixed): **{:,} bytes ≈ {:.2f} MB**\n".format(state_bytes, state_bytes/1024/1024))
lines.append("\n| n_kv_heads | elem size | 1k ctx (MB) | 8k ctx (MB) | 32k ctx (MB) | vs GDN state at 32k |\n")
lines.append("|---:|---|---:|---:|---:|---:|\n")
for nh in (4, 8, 16, 32):
    for (ename, eb) in (("F16", 2), ("Q8_0", 1), ("Q4", 0.5)):
        def kv(L): return 2 * nh * 128 * L * eb
        k1, k8, k32 = kv(1024)/1e6, kv(8192)/1e6, kv(32768)/1e6
        ratio = kv(32768) / state_bytes
        lines.append(f"| {nh} | {ename} | {k1:.2f} | {k8:.2f} | {k32:.2f} | {ratio:.1f}x |\n")

lines.append("\n**Key takeaway:** for a typical GQA attention layer (4–8 KV heads), the KV\n")
lines.append("cache already exceeds the GDN's fixed ~2 MB state by ~512–1024 tokens,\n")
lines.append("and at 32k context it is 16–64x larger. This is the memory advantage of\n")
lines.append("the recurrent formulation.\n")

out = "D:/Padhai/sem 7/ldls/LDLS Project 1/docs/prediction.md"
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("".join(lines))
print("wrote", out)
print("total bytes/step:", total)
print("state_bytes:", state_bytes)
print("pred 1T us:", pred_1t, "pred 8T us:", pred_8t)
print("measured 1T:", measured[1], "8T:", measured[8])
