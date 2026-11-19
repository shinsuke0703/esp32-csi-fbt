# -*- coding: utf-8 -*-
"""ノード間の距離を「行列ペンシル法」で超解像に推定する (ranging_sim.py の改良版)。
使い方: python ranging_mp.py [SNR=25] [経路の数 K=8]
  13 チャネルのトーンを 312.5 kHz 間隔の 1 本の周波数軸 (2402〜2482 MHz) に並べ (重なりは平均)、
  P(f) = Σ_p A_p·exp(−j2πf·T_p) を行列ペンシル法で分解する。往復の積なので T = 2×(片道の遅延)、
  直接波は一番早い成分 (T_min) なので d = c·T_min/2。
"""
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim

SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 25
K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
C = 299792458.0; rng = np.random.default_rng(3); M = 6; chans = range(1, 14); DF = 312.5e3
sims = {ch: load_sim(rf"E:/CSI/sim_out/chan/ch{ch:02d}") for ch in chans}
meta = sims[1]["meta"]; nodes = np.array(meta["tx"], float); tones = np.array(meta["tones"]); tsel = np.abs(tones) <= 26
delta = np.r_[0.03, 0.045 + rng.normal(0, 0.003, M - 1)]
F0 = 2402e6; NG = int(round((2482e6 - F0) / DF)) + 1

def grid_response(a, b):
    acc = np.zeros(NG, complex); cnt = np.zeros(NG)
    for ch in chans:
        s = sims[ch]; f = s["meta"]["fc"] + tones[tsel] * DF; H = s["H"][0, a * M + b][tsel].astype(complex)
        sd = np.sqrt((np.abs(H) ** 2).mean() / 10 ** (SNR / 10) / 2); n = lambda: sd * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
        p = (H + n()) * (H + n()) * np.exp(-2j * np.pi * f * 2 * (delta[a] + delta[b]) / C)
        g = np.round((f - F0) / DF).astype(int); ok = (g >= 0) & (g < NG)
        np.add.at(acc, g[ok], p[ok]); np.add.at(cnt, g[ok], 1)
    have = cnt > 0; acc[have] /= cnt[have]
    return acc, have

def matrix_pencil(x, K, L=None):
    N = len(x); L = L or N // 3
    Y = np.array([x[i:i + L + 1] for i in range(N - L)])
    U, s, Vh = np.linalg.svd(Y, full_matrices=False); V = Vh.conj().T[:, :K]
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    Z = np.vander(z, N, increasing=True).T; amp = np.linalg.lstsq(Z, x, rcond=None)[0]
    return z, amp

pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
dmeas, dtrue = {}, {}
for a, b in pairs:
    x, have = grid_response(a, b)
    # 抜けている所 (DC 付近など) は前後から補間して、等間隔の系列にする
    idx = np.arange(NG); x = np.interp(idx, idx[have], x[have].real) + 1j * np.interp(idx, idx[have], x[have].imag)
    z, amp = matrix_pencil(x, K)
    T = -np.angle(z) / (2 * np.pi * DF)                     # 遅延 [s] (符号付き、±1/(2·DF))
    plaus = (T > 5e-9) & (T < 120e-9) & (np.abs(z) > 0.9) & (np.abs(z) < 1.1)   # 往復 5〜120 ns、減衰しない成分だけ
    if not plaus.any(): plaus = np.ones(len(T), bool)
    strong = plaus & (np.abs(amp) > 0.25 * np.abs(amp[plaus]).max())
    Tmin = T[strong].min()
    dmeas[(a, b)] = C * Tmin / 2; dtrue[(a, b)] = np.linalg.norm(nodes[a] - nodes[b])
err = np.array([dmeas[p] - dtrue[p] - delta[p[0]] - delta[p[1]] for p in pairs])
print(f"SNR {SNR} dB、K={K}、周波数の点 {NG} (2402〜2482 MHz)")
print(f"距離の誤差 (内部遅延を除いた分): 中央値 {np.median(np.abs(err)) * 100:.1f} cm、最大 {np.abs(err).max() * 100:.1f} cm")

D = np.zeros((M, M))
for (a, b), d in dmeas.items(): D[a, b] = D[b, a] = d - 0.09
J = np.eye(M) - 1 / M; Bm = -0.5 * J @ (D ** 2) @ J; w, V = np.linalg.eigh(Bm); i3 = np.argsort(w)[::-1][:3]
X0 = V[:, i3] * np.sqrt(np.maximum(w[i3], 1e-9))
def resid(xv):
    X = xv[:3 * M].reshape(M, 3); bias = np.array([xv[-2]] + [xv[-1]] * (M - 1))
    return np.array([np.linalg.norm(X[a] - X[b]) + bias[a] + bias[b] - dmeas[(a, b)] for a, b in pairs])
r = least_squares(resid, np.r_[X0.ravel(), 0.04, 0.04], loss="soft_l1", f_scale=0.02)
X = r.x[:3 * M].reshape(M, 3)
mx, my = X.mean(0), nodes.mean(0); U, _, Vt = np.linalg.svd((X - mx).T @ (nodes - my)); Xa = (X - mx) @ (U @ Vt) + my
e = np.linalg.norm(Xa - nodes, axis=1)
print(f"位置の誤差: 中央値 {np.median(e) * 100:.1f} cm、最大 {e.max() * 100:.1f} cm   内部遅延の推定 C3 {r.x[-2]*100:.1f} cm / classic {r.x[-1]*100:.1f} cm (正解 3.0 / {delta[1:].mean()*100:.1f})")
