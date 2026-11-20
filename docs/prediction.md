# Step 4 — Roofline prediction (decode step)
Option 5 studies a single operator, so the "decode step" is one call to
`ggml_gated_delta_net` on a single token (n_tokens=1) with Qwen3-Next
shapes (H_k=16, H_v=32, S=128, K=1). Predicted **before** measuring.

## Bytes per decode step
| Component | Bytes |
|---|---:|
| q input | 8,192 |
| k input | 8,192 |
| v input | 16,384 |
| g input | 128 |
| beta input | 128 |
| state (read) | 2,097,152 |
| state (write, K=1 slot) | 2,097,152 |
| attention output | 16,384 |
| **Total** | **4,243,712** |

State traffic (read+write) = 4,194,304 bytes = 98.8% of total. The state dominates; arithmetic
intensity ≈ 0.247 FLOP/byte (memory-bound regime).
State size (fixed) = 2,097,152 bytes ≈ 2.00 MB, independent of context (recurrent).

## Predicted decode-step time = bandwidth ÷ bytes/step
| Bandwidth | Predicted (us/step) |
|---|---:|
| 1 thread, Triad 15.1 GB/s | 281.4 |
| 8 threads, Triad 15.9 GB/s | 267.4 |

## Predicted vs measured (decode step, us/step)
| Threads | Predicted (us) | Measured (us) | Ratio (meas/pred) |
|---:|---:|---:|---:|
| 1 | 281.4 | 229.2 | 0.81x |
| 2 | 240.2 | 125.3 | 0.52x |
| 4 | 248.7 | 76.2 | 0.31x |
| 8 | 267.4 | 87.5 | 0.33x |
| 12 | 292.9 | 166.8 | 0.57x |

## Notes on the gap
- Measured is 0.31-0.81x of the bandwidth prediction: the state fits better in
  cache than the STREAM-based model assumes, and threading helps compute.
- The gap vs. a pure bandwidth model comes from: (a) cache effects on the 2 MB
  state (exceeds L2 on this CPU), (b) dequantization/normalization overhead,
  (c) thread scheduling/synchronization (esp. at 12 threads, where measured
      time regresses vs 8 threads), (d) ggml graph overhead.
- At 12 threads the op gets *slower* than at 8 (166.8 us vs 87.6 us): the state
  (2 MB) and tiny per-token work do not amortize thread spawn/sync cost.
- Prefill (n_tokens>1) is ~205-219 us/token regardless of batch size (linear),
  confirming the op scales linearly in n_tokens with no super-linear cost.

## State memory vs. attention KV cache (1k–32k context)
Gated DeltaNet state is **fixed** (~2.0 MB) regardless of context. An
attention layer's KV cache **grows** with context:
KV_cache_bytes = 2 (K+V) × n_kv_heads × head_dim × context × bytes_per_elem.

GDN state (fixed): **2,097,152 bytes ≈ 2.00 MB**

| n_kv_heads | elem size | 1k ctx (MB) | 8k ctx (MB) | 32k ctx (MB) | vs GDN state at 32k |
|---:|---|---:|---:|---:|---:|
| 4 | F16 | 2.10 | 16.78 | 67.11 | 32.0x |
| 4 | Q8_0 | 1.05 | 8.39 | 33.55 | 16.0x |
| 4 | Q4 | 0.52 | 4.19 | 16.78 | 8.0x |
| 8 | F16 | 4.19 | 33.55 | 134.22 | 64.0x |
| 8 | Q8_0 | 2.10 | 16.78 | 67.11 | 32.0x |
| 8 | Q4 | 1.05 | 8.39 | 33.55 | 16.0x |
| 16 | F16 | 8.39 | 67.11 | 268.44 | 128.0x |
| 16 | Q8_0 | 4.19 | 33.55 | 134.22 | 64.0x |
| 16 | Q4 | 2.10 | 16.78 | 67.11 | 32.0x |
| 32 | F16 | 16.78 | 134.22 | 536.87 | 256.0x |
| 32 | Q8_0 | 8.39 | 67.11 | 268.44 | 128.0x |
| 32 | Q4 | 4.19 | 33.55 | 134.22 | 64.0x |

**Key takeaway:** for a typical GQA attention layer (4–8 KV heads), the KV
cache already exceeds the GDN's fixed ~2 MB state by ~512–1024 tokens,
and at 32k context it is 16–64x larger. This is the memory advantage of
the recurrent formulation.
