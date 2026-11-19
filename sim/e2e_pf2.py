# -*- coding: utf-8 -*-
"""腰の位置の追跡 (解き始めなし) その 2: 粒子ごとの軌跡に沿って、直近 M フレームの η̂ を位相をそろえて足し合わせる (コヒーレント積分)。
使い方: python e2e_pf2.py [clean|noisy] [ノード誤差 cm "0,5,20,50"] [M=8] [seed=0]
モデル: η_p(t) ≈ A_p·exp(−jk·L_p(x_t)) + 雑音。A_p (リンクごとの複素振幅、体の形で決まる) は未知で、窓の中では一定とみなす。
  粒子 (x_{t−M+1..t}) に対する統計量 Z_p = Σ_s η_p(t−s)·exp(+jk·L_p(x_{t−s}))、対数尤度 ∝ Σ_p |Z_p|² / (M·σ_p²) (A_p を消去した形)。
  窓は 1 フレームずつずれて重なるので、1/M をかけて足す。
状態: 位置 (x, z)、高さ h (0.4〜1.4 m、体の散乱の中心)、速度。
"""
import sys
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from e2e_common import load, observe, C

MODE = sys.argv[1] if len(sys.argv) > 1 else "clean"
SIGS = [float(s) / 100 for s in (sys.argv[2] if len(sys.argv) > 2 else "0").split(",")]
M = int(sys.argv[3]) if len(sys.argv) > 3 else 8
KAPPA = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
D = load(); k0 = 2 * np.pi * D["fc"] / C; pel = D["pel"]; T = len(pel); dt = 1 / D["fps"]
rng = np.random.default_rng(SEED)
eta, w = observe(D, rng=rng, clean=(MODE == "clean"))
SIG_W = 0.02 if MODE == "clean" else 0.15
sig2 = (SIG_W * np.abs(1 + eta)) ** 2 + 0.03 ** 2                  # η̂ の雑音の分散 (+ モデルの誤差の下駄)
NP_ = 10000; R_ROOM = 2.6

def plen(X, nd):   # X [n, 3] (x, h, z) → [n, 15]
    X3 = X[:, [0, 1, 2]]
    return np.stack([np.linalg.norm(X3 - nd[i], axis=1) + np.linalg.norm(X3 - nd[j], axis=1) for i, j in D["pairs"]], 1)

def run(sig):
    nd = D["nodes"] + (rng.normal(0, sig, D["nodes"].shape) if sig > 0 else 0)
    r = R_ROOM * np.sqrt(rng.random(NP_)); th = rng.uniform(0, 2 * np.pi, NP_)
    X = np.c_[r * np.cos(th), rng.uniform(0.6, 1.2, NP_), r * np.sin(th)]; V = rng.normal(0, 0.2, (NP_, 3)) * [1, 0.2, 1]
    ph_hist = np.zeros((M, NP_, 15), complex)                                 # exp(+jk L) の履歴
    est = np.zeros((T, 3)); logw = np.zeros(NP_)
    for t in range(T):
        if t > 0:
            V = V * 0.995 + rng.normal(0, 0.6 * dt, V.shape) * [1, 0.5, 1]
            X = X + V * dt + rng.normal(0, 0.003, X.shape)
            rr = np.hypot(X[:, 0], X[:, 2]); out = rr > R_ROOM
            X[out, 0] *= R_ROOM / rr[out]; X[out, 2] *= R_ROOM / rr[out]; V[out, ::2] *= -0.5
            lo, hi = X[:, 1] < 0.4, X[:, 1] > 1.4; X[lo, 1] = 0.4; X[hi, 1] = 1.4; V[lo | hi, 1] *= -0.5
        ph_hist = np.roll(ph_hist, 1, 0); ph_hist[0] = np.exp(1j * k0 * plen(X, nd))
        if t >= M - 1:
            e = eta[t - M + 1:t + 1][::-1] * w[t - M + 1:t + 1][::-1]       # [M, 15] (s = 0 が最新)
            Z = np.einsum("sp,snp->np", e, ph_hist)                              # [NP, 15]
            s2 = sig2[t - M + 1:t + 1].mean(0)
            amp2 = (np.abs(e) ** 2).mean(0); rel = amp2 / (amp2 + s2)            # リンクの信頼度 (信号 / (信号 + 雑音))
            coh = np.abs(Z) / (np.abs(e).sum(0)[None] + 1e-12)                   # 位相のそろい具合 0〜1
            logw = logw + KAPPA * (coh * rel[None]).sum(1)
            logw -= logw.max(); p = np.exp(logw); p /= p.sum()
            if 1 / np.sum(p ** 2) < NP_ / 2:
                idx = rng.choice(NP_, NP_, p=p); X, V, ph_hist = X[idx], V[idx], ph_hist[:, idx]; logw = np.zeros(NP_); p = np.full(NP_, 1 / NP_)
                X = X + rng.normal(0, 0.005, X.shape); V = V + rng.normal(0, 0.02, V.shape)
                ph_hist[0] = np.exp(1j * k0 * plen(X, nd))   # 最新の位置だけ揺らした分を反映 (古い履歴はそのまま)
        else:
            p = np.full(NP_, 1 / NP_)
        est[t] = p @ X
    return np.linalg.norm(est[:, [0, 2]] - pel, axis=1), est

tt = np.arange(T) / D["fps"]
print(f"[{MODE}, 窓 {M} フレーム] 外れ値で捨てた割合 {100 * (1 - w.mean()):.1f} %")
print("ノード誤差 | 歩行 前半 (8〜18 秒) | 歩行 後半 (18〜28 秒) | 止まって動く区間 (28 秒〜) |   (水平の誤差 中央値 cm / 10 cm 以内 %)")
for sg in SIGS:
    e, est = run(sg)
    seg = lambda a, b: e[(tt >= a) & (tt < b)]
    print(f"{sg * 100:4.0f} cm | " + " | ".join(f"{np.median(seg(a, b)) * 100:6.1f} / {np.mean(seg(a, b) < 0.1) * 100:3.0f} %" for a, b in ((8, 18), (18, 28), (28, 99))), flush=True)
    np.save(f"e2e_pf2_est_{MODE}_{int(sg * 100)}.npy", est)
