# -*- coding: utf-8 -*-
"""ノード間の距離を、経路を 1 本ずつ取り出す方法 (CLEAN + 変数射影の最小二乗) で推定する。
使い方: python ranging_clean.py [SNR=25] [経路の数=4]
  1. 13 チャネルの往復の積を 1 本の周波数軸に並べ、平方根で H(f) を取り出す (符号は周波数方向にそろえる)。
  2. 一致度 |Σ H(f) exp(+j2πfτ)| が最大の τ を足す → すべての τ を最小二乗で詰める (強さは線形に解く) → 残差から次。
  3. 強い経路 (最大の 0.3 倍以上) のうち一番早いものを直接波とし、d = c·τ。
  4. 15 組の距離 → MDS → 最小二乗 (内部遅延: C3 と classic で別の値)。
"""
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, r"e:/CSI/sim")
exec(open(r"e:/CSI/sim/ranging_mp.py", encoding="utf-8").read().split("def matrix_pencil")[0].replace("K = int(sys.argv[2]) if len(sys.argv) > 2 else 8", "NP_ = int(sys.argv[2]) if len(sys.argv) > 2 else 4"))
f = F0 + np.arange(NG) * DF

def h_from_product(x, have):
    idx = np.arange(NG); x = np.interp(idx, idx[have], x[have].real) + 1j * np.interp(idx, idx[have], x[have].imag)
    h = np.sqrt(x)
    for k in range(1, NG):                       # 周波数方向に符号をそろえる
        if np.real(h[k] * np.conj(h[k - 1])) < 0: h[k] = -h[k]
    return h

def fit_paths(h, n):
    taus = []; grid = np.arange(0.5e-9, 60e-9, 0.05e-9); E = np.exp(2j * np.pi * np.outer(grid, f))
    def amps(t): A = np.exp(-2j * np.pi * np.outer(f, t)); return A, np.linalg.lstsq(A, h, rcond=None)[0]
    res = h.copy()
    for _ in range(n):
        taus.append(grid[np.argmax(np.abs(E @ res))])
        def r(t): A, c = amps(t); d = A @ c - h; return np.r_[d.real, d.imag]
        taus = list(least_squares(r, np.array(taus) * 1e9, x_scale=1.0, diff_step=1e-3,
                                  args=(), kwargs={}).x * 1e-9) if False else list(least_squares(lambda t: r(t * 1e-9), np.array(taus) * 1e9).x * 1e-9)
        A, c = amps(np.array(taus)); res = h - A @ c
    A, c = amps(np.array(taus))
    return np.array(taus), c

pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
dmeas, dtrue = {}, {}
for a, b in pairs:
    x, have = grid_response(a, b); h = h_from_product(x, have)
    taus, c = fit_paths(h, NP_)
    strong = (np.abs(c) > 0.3 * np.abs(c).max()) & (taus > 0)
    dmeas[(a, b)] = C * taus[strong].min(); dtrue[(a, b)] = np.linalg.norm(nodes[a] - nodes[b])
err = np.array([dmeas[p] - dtrue[p] - delta[p[0]] - delta[p[1]] for p in pairs])
print(f"SNR {SNR} dB、経路 {NP_} 本")
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
print(f"位置の誤差: 中央値 {np.median(e) * 100:.1f} cm、最大 {e.max() * 100:.1f} cm   内部遅延の推定 C3 {r.x[-2]*100:.1f} / classic {r.x[-1]*100:.1f} cm (正解 3.0 / {delta[1:].mean()*100:.1f})")
