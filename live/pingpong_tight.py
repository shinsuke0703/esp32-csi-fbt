# -*- coding: utf-8 -*-
"""往復の時間 (t_aux) が最頻値の近くにそろった往復だけで、往復の積の位相がそろうか。"""
import sys, numpy as np
exec(open(r"e:/CSI/live/pingpong_analyze.py", encoding="utf-8").read().split('print(f"そろった往復')[0].replace('print(f"記録', '0 and print(f"記録'))
print("組 | 全部: 回数 R | ±2 µs: 回数 R | ±0.5 µs: 回数 R | 片道 (±2 µs) R")
for (a, b), L in sorted(per_pair.items()):
    if len(L) < 50: continue
    ph = np.array([x["prod"] for x in L]); one = np.array([x["phi_ij"] for x in L]); dt = np.array([x["dt"] for x in L], float)
    ok = np.isfinite(dt); ph, one, dt = ph[ok], one[ok], dt[ok]
    vals, cnt = np.unique(dt, return_counts=True); mode = vals[np.argmax(cnt)]
    R = lambda v: np.abs(np.mean(np.exp(1j * v))) if len(v) else np.nan
    m2 = np.abs(dt - mode) <= 2; m0 = np.abs(dt - mode) <= 0.5
    print(f"({a},{b}) | {len(ph):4d} {R(ph):5.2f} | {m2.sum():4d} {R(ph[m2]):5.2f} | {m0.sum():4d} {R(ph[m0]):5.2f} | {R(one[m2]):5.2f}   (最頻値 {mode:.0f} µs)")
print("R: 1 に近いほどそろう。一様なら 1/√回数 (例: 400 回で 0.05、30 回で 0.18)")
