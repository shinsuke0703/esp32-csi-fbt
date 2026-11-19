# -*- coding: utf-8 -*-
"""「C3 を持って歩く」校正に、壁 6 面の鏡像による反射を入れて解く (calib_sim.py のデータを使う)。
使い方: python calib_walls.py [SNR=25] [外れ値=0.08] [seed=0]

解く側のモデル (リンク j、キーフレーム k):
  Δd_j(k) = [L_los(p_k, r_j) − L_los(p_0, r_j)] − λ/(4π)·wrap( 2·δ_j(p_k) − 2·δ_j(p_0) )
     δ_j(p) = arg H_j(p) − arg LOS_j(p): 反射による位相のゆがみ。H_j = LOS + Σ_w Γ_w·(鏡像の経路)  (壁 6 面、鏡像法、中心周波数で計算)
  往復の積は H² なので位相は 2 倍。ゆがみは ±π 以内とみなして折り返す (キーフレームごとに独立に計算できる)。
比べる 3 つ: A 直接波だけ / B 壁の位置と反射係数が既知 / C 壁の位置と反射係数も推定 (初期値は ±0.5 m、Γ = −0.4)
座標は部屋に合わせる: 床 y = 0、壁は座標軸に平行。初期値: ノードと設置台は正解 + 30 cm の誤差 (粗い測距 + MDS 相当)、
  C3 の軌跡は初期値のノードからの多辺測量。データを作る側には物体 6 個の散乱があり、解く側には入れない。
"""
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

argv = sys.argv[1:] + [None] * 4
SNR = argv[0] or "25"; OUT = argv[1] or "0.08"; SEED = argv[2] or "0"
SEGLEN = float(argv[3] or 0)   # 区間の長さ [秒] (0 なら区間なし)。区間ごとに距離の変化の基準を自由な未知数にする (半波長のずれを吸収)
sys.argv = ["calib_sim.py", SNR, OUT, SEED]
src = open(r"e:/CSI/sim/calib_sim.py", encoding="utf-8").read().split("# ---- 解く側 2: キーフレームで最小二乗 ----")[0]
exec(src)   # 部屋・軌跡・観測 dd (57 Hz、ループの閉合ずみ) を作る

kf = np.arange(0, len(tt), int(FPS / 10)); K = len(kf)
rng2 = np.random.default_rng(100 + int(SEED))
WALLS_TRUE = np.array([-W / 2, W / 2, -W / 2, W / 2, Hroom])            # x 下限, x 上限, z 下限, z 上限, 天井 (床は 0)
G_TRUE = np.array([-0.5, -0.3, -0.3])                                   # 床, 天井, 壁 (4 面共通)

def images(q, walls):
    """q の 6 面の鏡像と、それぞれの反射係数の番号 (0 床, 1 天井, 2 壁)。"""
    xl, xh, zl, zh, h = walls; out = []
    for axis, plane, gi in ((1, 0.0, 0), (1, h, 1), (0, xl, 2), (0, xh, 2), (2, zl, 2), (2, zh, 2)):
        img = q.copy(); img[..., axis] = 2 * plane - q[..., axis]; out.append((img, gi))
    return out

def distortion(P, r, walls, G):
    """反射による位相のゆがみ δ = arg H − arg LOS (P [N,3]、r [3])。"""
    Ll = np.linalg.norm(P - r, axis=1); H = np.exp(-2j * np.pi * fc * Ll / C) / Ll
    for img, gi in images(r, walls):
        L = np.linalg.norm(P - img, axis=1); H = H + G[gi] * np.exp(-2j * np.pi * fc * L / C) / L
    return np.angle(H * np.exp(2j * np.pi * fc * Ll / C))

def model_dd(r0_, R, P, walls, G, use_walls):
    out = []
    for j in range(5):
        L = np.linalg.norm(P - R[j], axis=1); L0 = np.linalg.norm(r0_ - R[j])
        d = L - L0
        if use_walls:
            dl = distortion(P, R[j], walls, G); d0 = distortion(r0_[None], R[j], walls, G)[0]
            d = d - LAM / (4 * np.pi) * np.angle(np.exp(1j * 2 * (dl - d0)))
        out.append(d)
    return np.array(out)   # [5, K]

def solve(mode):
    """mode: 'A' 直接波だけ / 'B' 壁既知 / 'C' 壁も推定"""
    use_walls = mode != "A"; est_walls = mode == "C"
    r0i = r0 + rng_init.normal(0, 0.3, 3) * [1, 0.3, 1]; Ri = nodes + rng_init.normal(0, 0.3, (5, 3))
    wall0 = WALLS_TRUE + rng_init.normal(0, 0.5, 5) if est_walls else WALLS_TRUE; G0 = np.array([-0.4, -0.4, -0.4]) if est_walls else G_TRUE
    # 軌跡の初期値: 初期値のノードからの多辺測量
    P0 = np.zeros((K, 3)); prev = r0i.copy()
    for k in range(K):
        rk = np.linalg.norm(Ri - r0i, axis=1) + dd[kf[k]]
        prev = least_squares(lambda p: np.linalg.norm(p - Ri, axis=1) - rk, prev).x; P0[k] = prev
    nW = 8 if est_walls else 0
    seg_id = (np.arange(K) * (kf[1] - kf[0]) / FPS // SEGLEN).astype(int) if SEGLEN > 0 else np.zeros(K, int); NS = seg_id.max() + 1
    nseg = 5 * (NS - 1)   # 最初の区間は基準 0 (r0 から測る)
    x0 = np.r_[r0i, Ri.ravel(), P0.ravel(), (np.r_[wall0, G0] if est_walls else []), np.zeros(nseg)]
    def unpack(x):
        r0_ = x[:3]; R = x[3:18].reshape(5, 3); P = x[18:18 + 3 * K].reshape(K, 3)
        if est_walls: walls, G = x[18 + 3 * K:23 + 3 * K], x[23 + 3 * K:26 + 3 * K]
        else: walls, G = WALLS_TRUE, G_TRUE
        return r0_, R, P, walls, G
    def offsets(x):
        o = np.zeros((5, NS)); o[:, 1:] = x[len(x) - nseg:].reshape(5, NS - 1) if nseg else 0; return o[:, seg_id]   # [5, K]
    def resid(x):
        r0_, R, P, walls, G = unpack(x)
        m = model_dd(r0_, R, P, walls, G, use_walls) + offsets(x)
        out = [(m - dd[kf].T).ravel(), 1.0 * (P[0] - r0_), 1.0 * (P[-1] - r0_), 0.02 * np.diff(P, axis=0).ravel()]
        Rall = np.vstack([r0_, R])                                       # 粗い測距 (±30 cm) を弱い制約に
        for a_ in range(6):
            for b_ in range(a_ + 1, 6): out.append(np.atleast_1d(0.1 * (np.linalg.norm(Rall[a_] - Rall[b_]) - Dn[a_, b_])))
        return np.concatenate(out)
    n = len(x0); nr = len(resid(x0)); S = lil_matrix((nr, n), dtype=int)
    for j in range(5):
        for k in range(K):
            row = j * K + k; S[row, 0:3] = 1; S[row, 3 + 3 * j:6 + 3 * j] = 1; S[row, 18 + 3 * k:21 + 3 * k] = 1
            if est_walls: S[row, 18 + 3 * K:18 + 3 * K + 8] = 1
            if nseg and seg_id[k] > 0: S[row, n - nseg + j * (NS - 1) + seg_id[k] - 1] = 1
    o = 5 * K
    for c in range(3): S[o + c, c] = 1; S[o + c, 18 + c] = 1; S[o + 3 + c, c] = 1; S[o + 3 + c, 18 + 3 * (K - 1) + c] = 1
    o += 6
    for k in range(K - 1):
        for c in range(3): S[o + 3 * k + c, 18 + 3 * k + c] = 1; S[o + 3 * k + c, 21 + 3 * k + c] = 1
    S[o + 3 * (K - 1):, 0:18] = 1
    sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=0.03, max_nfev=200)
    r0_, R, P, walls, G = unpack(sol.x)
    e_raw = np.linalg.norm(np.vstack([r0_, R]) - np.vstack([r0, nodes]), axis=1)
    # 全体の回転・平行移動 (と鏡像) をそろえてから比べる (距離の変化の観測は、全体を回しても変わらないため)
    X = np.vstack([r0_, R, P]); Y = np.vstack([r0, nodes, traj[kf]]); mx, my = X.mean(0), Y.mean(0)
    U, _, Vt = np.linalg.svd((X - mx).T @ (Y - my)); Xa = (X - mx) @ (U @ Vt) + my
    e_nodes = np.linalg.norm(Xa[:6] - Y[:6], axis=1); e_traj = np.linalg.norm(Xa[6:] - Y[6:], axis=1)
    e_init = np.linalg.norm(np.vstack([r0i, Ri]) - np.vstack([r0, nodes]), axis=1)
    extra = f"  壁の推定 {np.round(walls, 2).tolist()} (正解 {WALLS_TRUE.tolist()})、Γ {np.round(G, 2).tolist()}" if est_walls else ""
    return e_nodes, e_traj, e_init, extra

# 粗い測距 (6 台の 15 組、±30 cm) — calib_sim.py の初期値と同じ考え方
allp = np.vstack([r0, nodes]); Dn = np.linalg.norm(allp[:, None] - allp[None], axis=-1)
nz = rng2.normal(0, 0.3, (6, 6)); nz = np.triu(nz, 1); Dn = np.abs(Dn + nz + nz.T)
for mode, name in (("A", "A 直接波だけ"), ("B", "B 壁が既知")):
    rng_init = np.random.default_rng(1000 + int(SEED))   # 3 つとも同じ初期値
    en, et, ei, extra = solve(mode)
    print(f"[区間 {SEGLEN:.0f} 秒] {name}: 設置台とノードの位置の誤差 (回転・平行移動をそろえた後) 中央値 {np.median(en) * 100:.1f} cm (最大 {en.max() * 100:.1f})、C3 の軌跡 中央値 {np.median(et) * 100:.1f} cm  [初期値 {np.median(ei) * 100:.1f} cm]{extra}", flush=True)
