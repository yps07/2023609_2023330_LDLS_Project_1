#!/usr/bin/env python3
"""Compute run-to-run spread across the 3 repeat runs."""
import csv, os, statistics
ROOT = "D:/Padhai/sem 7/ldls/LDLS Project 1"
RES = ROOT + "/results"
runs = []
for i in (1, 2, 3):
    with open(f"{RES}/gdn_run{i}.csv") as f:
        runs.append({(r["n_tokens"], r["n_threads"]): float(r["mean_us"]) for r in csv.DictReader(f)})

keys = sorted(runs[0].keys(), key=lambda k: (int(k[0]), int(k[1])))
out = f"{RES}/gdn_repeats_summary.csv"
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n_tokens", "n_threads", "run1_us", "run2_us", "run3_us", "mean_us", "std_us", "spread_pct"])
    for k in keys:
        vals = [runs[i][k] for i in range(3)]
        m = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        w.writerow([k[0], k[1], f"{vals[0]:.2f}", f"{vals[1]:.2f}", f"{vals[2]:.2f}",
                    f"{m:.2f}", f"{sd:.2f}", f"{100*sd/m:.1f}%"])
print("wrote", out)
# print decode point summary
print("\nDecode step (n_tokens=1) run-to-run:")
for k in keys:
    if k[0] == "1":
        vals = [runs[i][k] for i in range(3)]
        m = statistics.mean(vals); sd = statistics.stdev(vals)
        print(f"  threads={k[1]:>2}: runs={[round(v,1) for v in vals]} mean={m:.1f} us std={sd:.1f} us ({100*sd/m:.1f}%)")
