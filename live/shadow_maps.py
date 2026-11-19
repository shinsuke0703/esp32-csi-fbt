# -*- coding: utf-8 -*-
"""校正の段階 1 の下見: 組ごとに「人がどこにいるとき RSSI が弱まるか」の地図 (上から見た図)。使い方: python shadow_maps.py sessions/<名前>"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo"]
from calib_common import load_session
S = load_session(sys.argv[1])
print(f"傾き {S['tilt']:+.1f} 度、Kinect の高さ {-S['kinect_pos'][1]:.2f} m、組 {len(S['pairs'])}")
fig, axs = plt.subplots(3, 5, figsize=(18, 10)); axs = axs.ravel()
sm = lambda x, n: np.convolve(x, np.ones(n) / n, "same")
for ax, p in zip(axs, S["pairs"]):
    r = sm(S["rssi"][p], 10); att = np.median(r) - r
    b = S["body"][p]
    sc = ax.scatter(b[:, 0], b[:, 2], c=att, s=2, cmap="magma", vmin=-2, vmax=8)
    ax.set_title(f"{p[0]}-{p[1]}: 弱まり上位5% {np.percentile(att, 95):.1f} dB"); ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.4); ax.set_ylim(0.8, 2.8)
    print(f"{p[0]}-{p[1]}: 弱まり 上位 5 % {np.percentile(att, 95):.1f} dB、上位 1 % {np.percentile(att, 99):.1f} dB")
fig.colorbar(sc, ax=axs.tolist(), label="RSSI の弱まり [dB] (中央値から)")
fig.savefig(os.path.join(sys.argv[1], "shadow_maps.png"), dpi=60)
