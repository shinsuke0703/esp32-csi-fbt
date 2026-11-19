# -*- coding: utf-8 -*-
"""node_geom_obs.json (RSSI とチャネル間の往復位相) から 6 台の立体配置を推定する。
モデル:
  RSSI の往復の和  S_ij = a_i + a_j − 20·n·log10(d_ij)          a: ノードごとの送受信の定数 (DevKitC 共通 / C3 の 2 つ)、n: 減衰の指数
  位相差 (25 MHz)  Δφ_ij = −4π·Δf·(d_ij + b_i + b_j)/c  (折り返しあり)   b: 機器の遅れ (DevKitC 共通 / C3 の 2 つ)
  チャネル 1→6 と 6→11 の 2 本。残差は折り返してから比べる。初期値を多数変えて、残差最小を選ぶ。
  ばらつきの目安: 観測を間引いて解き直した解 (ブートストラップ) の広がり。
"""
import json
import numpy as np
from scipy.optimize import least_squares

C = 299792458.0; DF = 25e6
obs = json.load(open("node_geom_obs.json"))
pairs = [tuple(map(int, k.split("-"))) for k in obs]
S = np.array([obs[k]["rssi_f"] + obs[k]["rssi_r"] for k in obs])
PH = []
for k in obs:
    ph = {int(c): v for c, v in obs[k]["phase"].items()}
    PH.append([ph[6] - ph[1], ph[11] - ph[6]])
PH = np.array(PH)
SIG_R, SIG_P = 4.0, 0.8          # RSSI [dB]、位相 [rad] のばらつきの想定
dev = lambda i: 1 if i == 0 else 0   # 0 = C3

def unpack(x):
    X = np.zeros((6, 3)); X[1, 0] = x[0]; X[2, :2] = x[1:3]; X[3:] = x[3:12].reshape(3, 3)
    return X, x[12:14], x[14], x[15:17]
def resid(x, w=None):
    X, a, n, b = unpack(x); r = []
    for p, (i, j) in enumerate(pairs):
        dij = np.linalg.norm(X[i] - X[j]) + 1e-3
        r.append((S[p] - (a[dev(i)] + a[dev(j)] - 20 * n * np.log10(dij))) / SIG_R)
        m = -4 * np.pi * DF * (dij + b[dev(i)] + b[dev(j)]) / C
        r += list(np.angle(np.exp(1j * (PH[p] - m))) / SIG_P)
    r.append((n - 2.5) / 0.5)        # 屋内の減衰指数は 2〜3 くらい (弱い事前知識)
    r = np.array(r)
    return r if w is None else r * w

def solve(w=None, nstart=300, seed=0):
    rng = np.random.default_rng(seed); best = None
    for s in range(nstart):
        X0 = rng.uniform(-2.5, 2.5, (6, 3)); X0 -= X0[0]
        x0 = np.r_[abs(X0[1, 0]) + 0.5, X0[2, :2], X0[3:].ravel(), -40, -40, 2.5, rng.uniform(-1, 1), rng.uniform(-1, 1)]
        sol = least_squares(resid, x0, args=(w,), loss="soft_l1", f_scale=1.5)
        if best is None or sol.cost < best.cost: best = sol
    return best

def dists(X): return np.array([np.linalg.norm(X[i] - X[j]) for i, j in pairs])

best = solve()
X, a, n, b = unpack(best.x)
r = resid(best.x)
print(f"残差 (σ 単位): RSSI の RMS {np.sqrt(np.mean(r[0:45:3] ** 2)):.2f}、位相の RMS {np.sqrt(np.mean(np.r_[r[1:45:3], r[2:45:3]] ** 2)):.2f}  "
      f"(1 前後なら想定どおり)  減衰の指数 n = {n:.2f}、機器の遅れ b = {np.round(b, 2).tolist()} m")
# ブートストラップ: 組を復元抽出して解き直し、距離の広がりを見る
D0 = dists(X); Db = []
rng = np.random.default_rng(1)
for t in range(20):
    cnt = np.bincount(rng.integers(0, 15, 15), minlength=15).astype(float)
    w = np.r_[np.repeat(np.sqrt(cnt), 3), 1.0]
    Db.append(dists(unpack(solve(w, nstart=40, seed=10 + t).x)[0]))
Db = np.array(Db); spread = np.percentile(np.abs(Db - D0), 68, axis=0)
print("\n組 | 推定した距離 [m] | ばらつきの目安 (±) [m]")
for p, (i, j) in enumerate(pairs): print(f"{i}-{j} | {D0[p]:.2f} | {spread[p]:.2f}")
# 表示用: 重心を原点に、最も広がった方向を横、最も薄い方向を縦 (高さの候補) にそろえる
Xc = X - X.mean(0); U, sv, Vt = np.linalg.svd(Xc); Xd = Xc @ Vt.T
Xd = Xd[:, [0, 2, 1]]   # x, 高さ(最も薄い方向), z
json.dump(dict(nodes=Xd.tolist(), pairs=[f"{i}-{j}" for i, j in pairs], dist=D0.tolist(), spread=spread.tolist(), n=float(n), b=b.tolist()),
          open("node_fit.json", "w"), indent=1)
print("\n立体の座標 (重心が原点、縦は最も薄い方向):")
for i, p in enumerate(Xd): print(f"ノード {i}: " + " ".join(f"{v:+.2f}" for v in p))
