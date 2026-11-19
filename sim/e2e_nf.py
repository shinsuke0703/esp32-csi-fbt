# -*- coding: utf-8 -*-
"""雑音なしで、照合の方式ごとに腰の水平位置の誤差を比べる (ノード位置は正確)。歩行区間、1 フレームずつ (解き始めなし)。
 A: 位相の照合 (縦の線 0.1〜1.7 m、全体の複素係数は未知)
 B: 遮蔽の地図 (電波トモグラフィ): 直接波の振幅の減り |W| < 1 のリンクの線分の近くに人がいる
 C: A と B の掛け合わせ"""
import sys
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim
S = load_sim(r"E:/CSI/sim_out/mesh3d_60"); E = load_sim(r"E:/CSI/sim_out/mesh3d_empty")
meta = S["meta"]; tones = np.array(meta["tones"]); fc = meta["fc"]; C = 299792458.0; k0 = 2 * np.pi * fc / C
sel = np.abs(tones) <= 26; nodes = np.array(meta["tx"], float); N = 6
pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
rt = lambda H: np.stack([H[:, i * N + j][:, sel] * H[:, j * N + i][:, sel] for i, j in pairs], 1)
P = rt(S["H"]); Pb = rt(E["H"])[0]
Q = np.sum(P * np.conj(Pb), -1) / np.sum(np.abs(Pb) ** 2, -1); W = np.sqrt(Q); eta = W - 1
pel = S["params"][:, :2]; walk = np.arange(8 * 60, 28 * 60, 3)
g = np.arange(-2.6, 2.6001, 0.04); GX, GZ = np.meshgrid(g, g, indexing="ij"); ins = GX ** 2 + GZ ** 2 <= 2.6 ** 2
pts = np.stack([GX[ins], GZ[ins]], 1)
Tm = np.zeros((len(pts), 15), complex); D = np.zeros((len(pts), 15))
for p, (i, j) in enumerate(pairs):
    for h in np.arange(0.1, 1.75, 0.1):
        X = np.c_[pts[:, 0], np.full(len(pts), h), pts[:, 1]]
        L1 = np.linalg.norm(X - nodes[i], axis=1); L2 = np.linalg.norm(X - nodes[j], axis=1)
        Tm[:, p] += np.exp(-1j * k0 * (L1 + L2)) / (L1 * L2)
    a, b = nodes[i][[0, 2]], nodes[j][[0, 2]]; u = b - a; s = np.clip(((pts - a) @ u) / (u @ u), 0, 1)
    D[:, p] = np.linalg.norm(pts - (a + s[:, None] * u), axis=1)          # リンクの線分 (水平に投影) までの距離
nrm = np.sqrt((np.abs(Tm) ** 2).sum(1))
res = {"A": [], "B": [], "C": []}
for t in walk:
    sA = np.abs(Tm.conj() @ eta[t]) ** 2 / nrm ** 2; sA /= sA.max()
    att = np.clip(1 - np.abs(W[t]), 0, None)                              # 遮蔽の強さ
    sB = np.exp(-(D / 0.25) ** 2) @ att; sB = sB / (sB.max() + 1e-12)
    for key, sc in (("A", sA), ("B", sB), ("C", sA * sB)):
        res[key].append(np.linalg.norm(pts[np.argmax(sc)] - pel[t]))
for key, v in res.items():
    v = np.array(v); print(f"{key}: 中央値 {np.median(v) * 100:.1f} cm、10 cm 以内 {np.mean(v < 0.1) * 100:.0f} %、30 cm 以内 {np.mean(v < 0.3) * 100:.0f} %")
print(f"(比較: いつも部屋の中央と答えた場合 中央値 {np.median(np.linalg.norm(pel[walk], axis=1)) * 100:.1f} cm)")
print(f"|W| の範囲: {np.abs(W[walk]).min():.2f}〜{np.abs(W[walk]).max():.2f}")

print("\n組 | 体がリンクの線上 (水平距離 < 15 cm) のときの |W| | 離れている (> 60 cm) ときの |W| | 件数")
tt = np.arange(8 * 60, 28 * 60)
for p, (i, j) in enumerate(pairs):
    a, b = nodes[i][[0, 2]], nodes[j][[0, 2]]; u = b - a; s = np.clip(((pel[tt] - a) @ u) / (u @ u), 0, 1)
    dd = np.linalg.norm(pel[tt] - (a + s[:, None] * u), axis=1); w = np.abs(W[tt, p])
    on, off = dd < 0.15, dd > 0.6
    print(f"{i}-{j} | {w[on].mean() if on.any() else float('nan'):.2f} | {w[off].mean():.2f} | {on.sum()}")
# 時刻のずれの確認: 遮蔽の時系列と、線上にいるかどうかの時系列の相互相関の最大の位置
lags = range(-120, 121, 2); best = []
for p, (i, j) in enumerate(pairs):
    a, b = nodes[i][[0, 2]], nodes[j][[0, 2]]; u = b - a; s = np.clip(((pel - a) @ u) / (u @ u), 0, 1)
    near = (np.linalg.norm(pel - (a + s[:, None] * u), axis=1) < 0.3).astype(float); att = 1 - np.abs(W[:, p])
    cc = [np.corrcoef(near[tt], att[tt + l])[0, 1] for l in lags]; best.append((list(lags)[int(np.nanargmax(cc))], np.nanmax(cc)))
print("相互相関が最大になるずれ [フレーム] と相関:", best)
