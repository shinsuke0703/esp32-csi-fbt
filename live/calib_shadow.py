# -*- coding: utf-8 -*-
"""校正の段階 1: 遮蔽 (RSSI の弱まり) から 6 台の 3 次元位置を当てはめる (Kinect で人の位置が分かっている前提)。
使い方: python calib_shadow.py sessions/<名前> [初期値の数=60]
モデル: 人の胴を、腰の位置 (X, Z) を通る高さ 0.1〜1.7 m の縦の柱とする。リンク i-j の弱まりの予測は
   A_p · exp(−(d / w)²)   d: 線分 r_i〜r_j (高さが柱の範囲の部分) と柱の水平の距離の最小値、w = 0.25 m、A_p ≥ 0 はリンクごと (閉じた形で求める)
観測: RSSI (2 方向の和) を 10 パケット平均し、中央値からの弱まり [dB]。0.1 秒ごとに間引く。
ノードの位置は部屋の座標 (Kinect の真下の床が原点、X 左右、Y 高さ、Z 奥行き)。探す範囲: X −3〜3、Y 0〜2.6、Z −1〜5 m。
ばらばらの初期値から何度も解き、残差が小さい解どうしがそろうかを見る。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import least_squares
from calib_common import load_session

ses = sys.argv[1]; NST = int(sys.argv[2]) if len(sys.argv) > 2 else 60
S = load_session(ses); pairs = S["pairs"]; W = 0.25; NSEG = 24
LO = np.array([-3.0, 0.0, -1.0]); HI = np.array([3.0, 2.6, 5.0])
sm = lambda x, n: np.convolve(x, np.ones(n) / n, "same")
obs, bod = [], []
for p in pairs:
    r = sm(S["rssi"][p], 10); att = np.median(r) - r; t = S["t"][p]
    idx = np.searchsorted(t, np.arange(t[0] + 0.5, t[-1] - 0.5, 0.1))
    obs.append(att[idx]); bod.append(S["body"][p][idx][:, [0, 2]])
print(f"組 {len(pairs)}、1 組あたり {len(obs[0])} 時点、腰の範囲 X {np.min([b[:, 0].min() for b in bod]):.2f}〜{np.max([b[:, 0].max() for b in bod]):.2f}、"
      f"Z {np.min([b[:, 1].min() for b in bod]):.2f}〜{np.max([b[:, 1].max() for b in bod]):.2f} m")
u = (np.arange(NSEG) + 0.5) / NSEG

def predict_shape(R):
    out = []
    for p, (i, j) in enumerate(pairs):
        pts = R[i][None] + u[:, None] * (R[j] - R[i])[None]            # 線分の点 [NSEG, 3]
        inb = (pts[:, 1] > 0.1) & (pts[:, 1] < 1.7)
        if not inb.any(): out.append(np.zeros(len(bod[p]))); continue
        dd = np.linalg.norm(bod[p][:, None, :] - pts[inb][None, :, [0, 2]], axis=2).min(1)
        out.append(np.exp(-(dd / W) ** 2))
    return out

def resid(x):
    R = x.reshape(6, 3); sh = predict_shape(R); res = []
    for p in range(len(pairs)):
        s = sh[p]; a = max(0.0, float(s @ obs[p]) / (float(s @ s) + 1e-9))   # A_p (閉じた形、0 以上)
        res.append(obs[p] - a * s)
    return np.concatenate(res)

rng = np.random.default_rng(0); sols = []
for k in range(NST):
    x0 = (LO + rng.random((6, 3)) * (HI - LO)).ravel()
    sol = least_squares(resid, x0, bounds=(np.tile(LO, 6), np.tile(HI, 6)), loss="soft_l1", f_scale=2.0, diff_step=0.02, max_nfev=300)
    sols.append((sol.cost, sol.x.reshape(6, 3)))
    if (k + 1) % 10 == 0: print(f"  {k + 1}/{NST} 回、ここまでの最小の残差 {min(c for c, _ in sols):.0f}", flush=True)
sols.sort(key=lambda s: s[0])
print("\n残差の小さい順 (上位 8 個) と、最良の解との位置の差 (ノードごと、cm):")
best = sols[0][1]
for c, R in sols[:8]:
    print(f"  残差 {c:9.0f} | " + " ".join(f"{np.linalg.norm(R[i] - best[i]) * 100:5.0f}" for i in range(6)))
print("\n最良の解 (部屋の座標 X 左右, Y 高さ, Z 奥行き [m]):")
for i in range(6): print(f"  ノード {i}{' (C3)' if i == 0 else ''}: " + " ".join(f"{v:+.2f}" for v in best[i]))
json.dump(dict(nodes=best.tolist(), cost=sols[0][0], top_costs=[s[0] for s in sols[:8]]), open(os.path.join(ses, "calib_shadow.json"), "w"), indent=1)
