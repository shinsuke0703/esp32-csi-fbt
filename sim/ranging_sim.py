# -*- coding: utf-8 -*-
"""ノードの位置の自己校正 (シミュレーション): チャネル 1〜13 を切り替えた ACK 往復の位相 → ノード間の距離 → 3 次元の位置。
使い方: python ranging_sim.py [SNR=25] [チャネルごとの位相のずれ rad = 0]
  往復の積: P(f) = (H + n1)(H + n2)·exp(−j2πf·2(δ_a + δ_b)/c)·exp(jψ_ch)
     δ: 機器の内部遅延 (距離に直して 3〜6 cm、未知)。ψ_ch: チャネルを変えたときに往復でも消えない位相のずれ (実機で要確認)。
  距離: 全チャネル・全トーンをまとめて、候補の距離 d で A(d) = |Σ P(f)·exp(+j2πf·2d/c)| が最大になる d (位相をつながなくてよい)。
     マルチパスの影響を減らすため、最大の 0.5 倍を超える一番近い (早い) 山を使う。
  位置: 古典的 MDS で初期値 → 非線形最小二乗 (内部遅延も未知数。classic 5 台は共通、C3 は別)。
"""
import sys
import numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim

SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 25
PSI = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
C = 299792458.0; rng = np.random.default_rng(3)
chans = range(1, 14); M = 6
sims = {ch: load_sim(rf"E:/CSI/sim_out/chan/ch{ch:02d}") for ch in chans}
meta = sims[1]["meta"]; nodes = np.array(meta["tx"], float); tones = np.array(meta["tones"])
tsel = np.abs(tones) <= 26                                  # 20 MHz (LLTF 相当)
delta = rng.uniform(0.03, 0.06, M)                          # 内部遅延 [m 相当]。classic 5 台 (1〜5) は似た値、C3 (0) は別
delta[1:] = 0.045 + rng.normal(0, 0.003, M - 1); delta[0] = 0.03
dgrid = np.arange(0.5, 12, 0.002)

def measure(a, b):
    F, P = [], []
    psi_ch = {ch: rng.uniform(-PSI, PSI) for ch in chans}
    for ch in chans:
        s = sims[ch]; f = s["meta"]["fc"] + tones[tsel] * s["meta"]["df"]
        H = s["H"][0, a * M + b][tsel].astype(complex)
        sd = np.sqrt((np.abs(H) ** 2).mean() / 10 ** (SNR / 10) / 2)
        n = lambda: sd * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
        p = (H + n()) * (H + n()) * np.exp(-2j * np.pi * f * 2 * (delta[a] + delta[b]) / C) * np.exp(1j * psi_ch[ch])
        F.append(f); P.append(p)
    F = np.concatenate(F); P = np.concatenate(P)
    A = np.abs(np.exp(2j * np.pi * np.outer(2 * dgrid / C, F)) @ P)
    # 最大の 0.5 倍を超える山のうち、一番近いもの (直接波は一番早く届く)
    peaks = np.where((A[1:-1] > A[:-2]) & (A[1:-1] >= A[2:]) & (A[1:-1] > 0.5 * A.max()))[0] + 1
    return dgrid[peaks[0]] if len(peaks) else dgrid[np.argmax(A)]

pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
dtrue = {p: np.linalg.norm(nodes[p[0]] - nodes[p[1]]) for p in pairs}
dmeas = {p: measure(*p) for p in pairs}
err = np.array([dmeas[p] - dtrue[p] - (delta[p[0]] + delta[p[1]]) for p in pairs])
print(f"SNR {SNR} dB、チャネル 1〜13 (帯域 {2472 + 10 - (2412 - 10)} MHz)、チャネルごとの位相のずれ ±{PSI} rad")
print(f"距離の誤差 (内部遅延を除いた分): 中央値 {np.median(np.abs(err)) * 100:.1f} cm、最大 {np.abs(err).max() * 100:.1f} cm")
print("  " + "  ".join(f"{a}-{b}: {dtrue[(a, b)]:.2f}m→{dmeas[(a, b)] - delta[a] - delta[b]:.2f}" for a, b in pairs[:8]))

# 古典的 MDS → 最小二乗 (内部遅延: classic 共通 β1、C3 β0)
D = np.zeros((M, M))
for (a, b), d in dmeas.items(): D[a, b] = D[b, a] = d - 0.09                   # 内部遅延の典型値を引いた初期値
J = np.eye(M) - 1 / M; Bm = -0.5 * J @ (D ** 2) @ J; w, V = np.linalg.eigh(Bm); idx = np.argsort(w)[::-1][:3]
X0 = V[:, idx] * np.sqrt(np.maximum(w[idx], 1e-9))
def resid(x):
    X = x[:3 * M].reshape(M, 3); b0, b1 = x[-2:]
    bias = np.array([b0] + [b1] * (M - 1))
    return np.array([np.linalg.norm(X[a] - X[b]) + bias[a] + bias[b] - dmeas[(a, b)] for a, b in pairs])
r = least_squares(resid, np.r_[X0.ravel(), 0.04, 0.04], loss="soft_l1", f_scale=0.05)
X = r.x[:3 * M].reshape(M, 3)
def procrustes(X, Y):   # X を Y に合わせる (回転・平行移動・鏡像)
    mx, my = X.mean(0), Y.mean(0); U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my)); R = U @ Vt
    return (X - mx) @ R + my
Xa = procrustes(X, nodes); e = np.linalg.norm(Xa - nodes, axis=1)
print(f"位置の誤差 (回転・平行移動・鏡像をそろえた後): 中央値 {np.median(e) * 100:.1f} cm、最大 {e.max() * 100:.1f} cm")
print(f"  推定した内部遅延: C3 {r.x[-2] * 100:.1f} cm (正解 {delta[0] * 100:.1f})、classic {r.x[-1] * 100:.1f} cm (正解 {delta[1:].mean() * 100:.1f})")
print("  高さの誤差 [cm]:", np.round((Xa[:, 1] - nodes[:, 1]) * 100, 1).tolist())
