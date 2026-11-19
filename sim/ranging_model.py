# -*- coding: utf-8 -*-
"""ノードの位置の自己校正 (人は何も持たない): 床の反射の幾何をモデルに入れ、6 台の位置を全リンクの周波数応答に直接当てはめる。
使い方: python ranging_model.py [SNR=25]
  モデル (リンク i-j): H(f) = a_L·exp(−j2πf(d_ij + β_i + β_j)/c) + a_F·exp(−j2πf(d^F_ij + β_i + β_j)/c)
     d_ij = |r_i − r_j|、d^F_ij = |r_i − r̃_j| (r̃_j は床 y = 0 で鏡に映した r_j)、β = 機器の内部遅延 (C3 と classic で別)。
     a_L, a_F (複素数) はリンクごとに線形に解く (変数射影)。H(f) は往復の積の平方根 (全体の符号は a に吸収される)。
  初期値: 一致度の最大 → MDS (ranging_clean.py の 1 本の経路と同じ、誤差 約 60 cm)。床は y = 0 (最も低いノードより下) とする。
"""
import sys
import numpy as np
from scipy.optimize import least_squares
sys.argv = [sys.argv[0], sys.argv[1] if len(sys.argv) > 1 else "25", "1"]
exec(open(r"e:/CSI/sim/ranging_clean.py", encoding="utf-8").read())   # 測距 (1 本の経路) と MDS の結果: X (位置), r.x[-2:] (内部遅延)

pairs = [(a, b) for a in range(M) for b in range(a + 1, M)]
hs = {}
for a, b in pairs:
    x, have = grid_response(a, b); hs[(a, b)] = h_from_product(x, have)
fs = f[::2]; hsub = {p: h[::2] for p, h in hs.items()}   # 計算を軽くするため周波数を半分に間引く
for p in hsub: hsub[p] = hsub[p] / np.sqrt(np.mean(np.abs(hsub[p]) ** 2))   # リンクごとの強さをそろえる

# 初期値: MDS の結果を、床が y = 0 になるよう上下の向きと高さをそろえる (一番低いノードを 0.3 m と仮定)
X0 = X.copy()
# MDS の座標は向きが任意なので、主軸のうち一番広がりが小さい軸を高さとみなす (ノードは水平方向に広く並ぶ)
c0 = X0.mean(0); U, s_, Vt = np.linalg.svd(X0 - c0); Xr = (X0 - c0) @ Vt.T          # 主軸の座標 (広い順)
X0 = np.column_stack([Xr[:, 0], Xr[:, 2] - Xr[:, 2].min() + 0.3, Xr[:, 1]])          # (x, y=高さ, z)

def model_resid(xv, return_fit=False):
    R = xv[:3 * M].reshape(M, 3); bias = np.array([xv[-2]] + [xv[-1]] * (M - 1)); out = []
    for (a, b) in pairs:
        dL = np.linalg.norm(R[a] - R[b]); img = R[b] * [1, -1, 1]; dF = np.linalg.norm(R[a] - img)
        A = np.column_stack([np.exp(-2j * np.pi * fs * (dL + bias[a] + bias[b]) / C), np.exp(-2j * np.pi * fs * (dF + bias[a] + bias[b]) / C)])
        cf = np.linalg.lstsq(A, hsub[(a, b)], rcond=None)[0]; d = A @ cf - hsub[(a, b)]
        out.append(np.r_[d.real, d.imag])
    return np.concatenate(out)

best = None
for flip in (1, -1):   # 上下の取り違え (鏡像) の両方から試す
    X1 = X0.copy()
    if flip < 0: X1[:, 1] = X1[:, 1].max() - X1[:, 1] + 0.3
    rr = least_squares(model_resid, np.r_[X1.ravel(), r.x[-2], r.x[-1]], loss="soft_l1", f_scale=0.3, max_nfev=300)
    if best is None or rr.cost < best.cost: best = rr
Xm = best.x[:3 * M].reshape(M, 3)
# 評価: 床があるので上下は決まる。水平方向の回転・平行移動・鏡像だけそろえる
def align_xz(X, Y):
    mx, my = X[:, [0, 2]].mean(0), Y[:, [0, 2]].mean(0); U, _, Vt = np.linalg.svd((X[:, [0, 2]] - mx).T @ (Y[:, [0, 2]] - my)); Rm = U @ Vt
    Z = X.copy(); Z[:, [0, 2]] = (X[:, [0, 2]] - mx) @ Rm + my; return Z
Xa2 = align_xz(Xm, nodes); e2 = np.linalg.norm(Xa2 - nodes, axis=1)
print(f"\n床の反射を入れたモデルで当てはめ: 位置の誤差 中央値 {np.median(e2) * 100:.1f} cm、最大 {e2.max() * 100:.1f} cm")
print(f"  高さの誤差 [cm]: {np.round((Xa2[:, 1] - nodes[:, 1]) * 100, 1).tolist()}")
print(f"  内部遅延の推定 C3 {best.x[-2] * 100:.1f} / classic {best.x[-1] * 100:.1f} cm (正解 3.0 / {delta[1:].mean() * 100:.1f})")
