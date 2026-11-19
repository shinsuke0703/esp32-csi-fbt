# -*- coding: utf-8 -*-
"""腰の水平位置を、フレーム間の位相の変化 (ドップラー) でパーティクルフィルタ追跡する。解き始めなし (部屋全体に一様にばらまく)。
使い方: python e2e_pf.py [clean|noisy] [ノード誤差 cm のリスト "0,5,20,50"] [seed=0]
観測: Δφ_p(t) = arg(η̂_p(t)·η̂_p(t−1)*)、モデル: −k·(L_p(x_t) − L_p(x_{t−1}))、L_p は高さ 1.0 m の点を通る 2 局間の経路長。
尤度: リンクごとに exp(κ_p·cos(観測 − モデル))、κ_p は |η̂| が小さいリンクほど小さく (雑音に埋もれる) する。
状態: 位置 (x, z) と速度 (vx, vz)。速度はゆっくり変わる (歩行)。止まっている間は位相の変化が 0 で、位置は保たれる。
"""
import sys
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from e2e_common import load, observe, C

MODE = sys.argv[1] if len(sys.argv) > 1 else "clean"
SIGS = [float(s) / 100 for s in (sys.argv[2] if len(sys.argv) > 2 else "0").split(",")]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
D = load(); k0 = 2 * np.pi * D["fc"] / C; pel = D["pel"]; T = len(pel); dt = 1 / D["fps"]
rng = np.random.default_rng(SEED)
eta, w = observe(D, rng=rng, clean=(MODE == "clean"))
w = np.ones_like(w)   # ゲートは使わず、尤度に外れ値の項を入れる
NOC3 = "noc3" in sys.argv[4:]   # C3 (ノード 0) を含む組を使わない (DevKitC 5 台の 10 組だけ)
if NOC3: w[:, [p for p, pr in enumerate(D["pairs"]) if 0 in pr]] = 0
NP_ = 20000; R_ROOM = 2.6; H = 1.0
SIG_W = 0.0 if MODE == "clean" else np.array([0.35 if 0 in pr else 0.16 for pr in D["pairs"]]) / 2 + 0.02; VAR_MODEL = 0.25

def plen(X, nd):   # X [n, 2] → [n, 15]
    X3 = np.c_[X[:, 0], np.full(len(X), H), X[:, 1]]
    return np.stack([np.linalg.norm(X3 - nd[i], axis=1) + np.linalg.norm(X3 - nd[j], axis=1) for i, j in D["pairs"]], 1)

def run(sig):
    nd = D["nodes"] + (rng.normal(0, sig, D["nodes"].shape) if sig > 0 else 0)
    r = R_ROOM * np.sqrt(rng.random(NP_)); th = rng.uniform(0, 2 * np.pi, NP_)
    X = np.c_[r * np.cos(th), r * np.sin(th)]; V = rng.normal(0, 0.3, (NP_, 2))
    Lprev = plen(X, nd); est = np.zeros((T, 2)); logw = np.zeros(NP_)
    for t in range(1, T):
        V = V * 0.995 + rng.normal(0, 0.6 * dt, V.shape)                      # 加速度のゆらぎ
        X = X + V * dt + rng.normal(0, 0.003, X.shape)
        out = np.sum(X ** 2, 1) > R_ROOM ** 2; X[out] *= R_ROOM / np.linalg.norm(X[out], axis=1)[:, None]; V[out] *= -0.5
        L = plen(X, nd)
        obs = np.angle(eta[t] * np.conj(eta[t - 1])); ww = w[t] * w[t - 1]
        a1, a0 = np.abs(eta[t]) + 1e-6, np.abs(eta[t - 1]) + 1e-6
        sw1, sw0 = SIG_W * np.abs(1 + eta[t]), SIG_W * np.abs(1 + eta[t - 1])  # W の雑音 (往復の位相のぶれの半分 × |W|)
        var = sw1 ** 2 / a1 ** 2 + sw0 ** 2 / a0 ** 2 + VAR_MODEL               # 位相の差の分散 = 雑音 + モデルの誤差
        kap = ww / np.clip(var, 0.05, None) * (var < 3.0)                        # 雑音に埋もれたリンクは使わない
        mod = -k0 * (L - Lprev)
        ll = kap[None] * (np.cos(obs[None] - mod) - 1)                            # 0 が最大
        logw = logw + np.log(0.1 + np.exp(ll)).sum(1)                             # 外れ値 (10 % 程度) で尤度が 0 にならないように
        logw -= logw.max(); p = np.exp(logw); p /= p.sum()
        est[t] = p @ X
        if 1 / np.sum(p ** 2) < NP_ / 2:                                         # 有効な粒子が半分を切ったら引き直す
            idx = rng.choice(NP_, NP_, p=p); X, V, L = X[idx], V[idx], L[idx]; logw = np.zeros(NP_)
            X = X + rng.normal(0, 0.01, X.shape); V = V + rng.normal(0, 0.03, V.shape); L = plen(X, nd)
        Lprev = L
    return np.linalg.norm(est - pel, axis=1)

tt = np.arange(T) / D["fps"]
print(f"[{MODE}{' DevKitC 10 組だけ' if NOC3 else ' 15 組'}]")
print("ノード誤差 | 歩行 前半 (8〜18 秒) | 歩行 後半 (18〜28 秒) | 止まって動く区間 (28 秒〜) |   (中央値 cm / 10 cm 以内 %)")
for sg in SIGS:
    e = run(sg)
    seg = lambda a, b: e[(tt >= a) & (tt < b)]
    print(f"{sg * 100:4.0f} cm | " + " | ".join(f"{np.median(seg(a, b)) * 100:6.1f} / {np.mean(seg(a, b) < 0.1) * 100:3.0f} %" for a, b in ((8, 18), (18, 28), (28, 99))), flush=True)
