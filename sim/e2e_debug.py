# -*- coding: utf-8 -*-
"""e2e_pelvis の失敗の切り分け: 雑音なしで、テンプレートの位相が観測 η の位相と時間方向にそろうか (リンクごとの R)。"""
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
print("相反性 (H_ij と H_ji の差の比):", float(np.abs(S["H"][100, 1 * N + 2] - S["H"][100, 2 * N + 1]).mean() / np.abs(S["H"][100, 1 * N + 2]).mean()))
Q = np.sum(P * np.conj(Pb), -1) / np.sum(np.abs(Pb) ** 2, -1); eta = np.sqrt(Q) - 1
Hd = (S["H"][:, :, sel] - E["H"][:1, :, sel])                      # 片道の体の成分 (正解)
eta1 = np.stack([np.sum(Hd[:, i * N + j] * np.conj(E["H"][0, i * N + j, sel]), -1) / np.sum(np.abs(E["H"][0, i * N + j, sel]) ** 2) for i, j in pairs], 1)
pel = S["params"][:, :2]; walk = slice(8 * 60, 28 * 60)
# LoS の位相の符号: 背景の各リンクの位相の傾き (トーン方向) から遅延の符号を見る
kk = tones[sel]; b = E["H"][0, 1 * N + 2, sel]; sl = np.angle(np.sum(b[1:] * np.conj(b[:-1])))
d12 = np.linalg.norm(nodes[1] - nodes[2]); print(f"背景のトーン方向の位相の傾き {sl:+.4f} rad/トーン、exp(−j2πfτ) なら {-2*np.pi*meta['df']*d12/C:+.4f}")
for HS in ([0.9, 1.1, 1.3, 1.5], list(np.arange(0.1, 1.75, 0.1)), list(np.arange(0.5, 1.75, 0.1))):
  for sign in (-1,):
    Rs, Rs1 = [], []
    for p, (i, j) in enumerate(pairs):
        X = np.c_[pel[walk, 0], np.zeros(len(pel[walk])), pel[walk, 1]]
        T_ = 0
        for h in HS:
            X[:, 1] = h; L = np.linalg.norm(X - nodes[i], axis=1) + np.linalg.norm(X - nodes[j], axis=1)
            T_ = T_ + np.exp(sign * 1j * k0 * L) / (np.linalg.norm(X - nodes[i], axis=1) * np.linalg.norm(X - nodes[j], axis=1))
        Rs.append(np.abs(np.mean(np.exp(1j * np.angle(eta[walk, p] * np.conj(T_))))))
        Rs1.append(np.abs(np.mean(np.exp(1j * np.angle(eta1[walk, p] * np.conj(T_))))))
    print(f"高さ {np.round(HS[0],1)}〜{np.round(HS[-1],1)} ({len(HS)} 点) 符号 {sign:+d}: R (η̂ 往復から) 中央値 {np.median(Rs):.2f} 範囲 {min(Rs):.2f}〜{max(Rs):.2f} | R (片道の正解の体の成分) 中央値 {np.median(Rs1):.2f}")
