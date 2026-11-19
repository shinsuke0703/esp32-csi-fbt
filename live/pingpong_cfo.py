# -*- coding: utf-8 -*-
"""往復の積の位相が「周波数のずれ (CFO) × 往復の時間」で説明できるか。組ごとに、位相 ≈ 2π·δf·Δt + c となる δf を探す。
Δt には j が PING を受けてから PONG の送信を頼むまでの時間 (t_aux、µs、j の時計) を使う。"""
import sys, numpy as np
exec(open(r"e:/CSI/live/pingpong_analyze.py", encoding="utf-8").read().split('print(f"そろった往復')[0].replace('print(f"記録', '0 and print(f"記録'))
fs = np.linspace(-150e3, 150e3, 30001)
print("組 | 回数 | そのままの集まり R | δf を補正した集まり R | 最適な δf [kHz] | Δt の範囲 µs (5〜95%)")
for (a, b), L in sorted(per_pair.items()):
    if len(L) < 50: continue
    ph = np.array([x["prod"] for x in L]); dt = np.array([x["dt"] for x in L], float); ok = np.isfinite(dt)
    ph, dt = ph[ok], dt[ok] * 1e-6
    R0 = np.abs(np.mean(np.exp(1j * ph)))
    Rf = np.abs(np.exp(1j * (ph[None] - 2 * np.pi * fs[:, None] * dt[None])).mean(1)); k = int(np.argmax(Rf))
    print(f"({a},{b}) | {len(ph):4d} | {R0:5.2f} | {Rf[k]:5.2f} | {fs[k] / 1e3:7.2f} | {np.percentile(dt, 5) * 1e6:.0f}〜{np.percentile(dt, 95) * 1e6:.0f}")
print(f"\n参考: 完全にばらばら (一様) なら R ≈ 1/√回数 ≈ 0.05、雑音だけなら R ≈ 0.9 以上")
