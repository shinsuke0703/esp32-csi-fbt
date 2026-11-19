# -*- coding: utf-8 -*-
"""フレーム間の位相の変化 arg(η_t·η_{t−d}*) が、テンプレートの経路長の変化 −k·(L_p(x_t) − L_p(x_{t−d})) と合うか (雑音なし、ノード正確)。"""
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
Q = np.sum(P * np.conj(Pb), -1) / np.sum(np.abs(Pb) ** 2, -1); eta = np.sqrt(Q) - 1
pel = S["params"][:, :2]; tt = np.arange(8 * 60, 28 * 60)
for h in (1.0,):
    L = np.stack([np.linalg.norm(np.c_[pel[:, 0], np.full(len(pel), h), pel[:, 1]] - nodes[i], axis=1) +
                  np.linalg.norm(np.c_[pel[:, 0], np.full(len(pel), h), pel[:, 1]] - nodes[j], axis=1) for i, j in pairs], 1)
    for d in (1, 3, 6):
        obs = np.angle(eta[tt] * np.conj(eta[tt - d])); mod = -k0 * (L[tt] - L[tt - d])
        R = np.abs(np.mean(np.exp(1j * (obs - mod)), 0))
        R0 = np.abs(np.mean(np.exp(1j * obs), 0))
        print(f"差 {d} フレーム: 観測とモデルの一致 R 中央値 {np.median(R):.2f} (範囲 {R.min():.2f}〜{R.max():.2f})、モデルなし (変化 0 と仮定) {np.median(R0):.2f}")
