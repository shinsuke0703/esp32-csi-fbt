# -*- coding: utf-8 -*-
"""「C3 を持って歩く」校正の最適化を改良したもの: 交互に解く (IRLS) + 複数の初期値。
使い方: python calib_opt.py [SNR=25] [外れ値=0.08] [seed=0] [初期値の数=20]

観測: dd[j, k] = |p_k − r_j| − |r0 − r_j| + c[j, k] + 雑音   (c: 壁の反射による補正、モード B では壁から計算、A では 0)
  1. ノードを固定 → 各キーフレームの p_k を、5 本の観測から一度に解く (3×3 のガウス・ニュートンを全キーフレームでまとめて)
     軌跡のなめらかさは、前後のキーフレームとの差を弱い観測として足す
  2. 軌跡を固定 → 各ノードの r_j を、そのリンクの全キーフレームから解く (3×3、ノードごと)
  外れ値には Huber の重み (IRLS)。r0 (設置台) は原点に固定 (ゲージ)。これを交互に繰り返す。
  初期値を何通りも変えて解き、重み付きの残差が一番小さいものを選ぶ (全体の最小二乗での仕上げは未実装)。
"""
import sys
import numpy as np
from scipy.optimize import least_squares

argv = sys.argv[1:] + [None] * 4
SNR = argv[0] or "25"; OUT = argv[1] or "0.08"; SEED = argv[2] or "0"; NSTART = int(argv[3] or 20)
sys.argv = ["calib_walls.py", SNR, OUT, SEED, "0"]
src = open(r"e:/CSI/sim/calib_walls.py", encoding="utf-8").read().split('for mode, name in (("A", "A 直接波だけ"), ("B", "B 壁が既知")):')[0]
exec(src)   # 部屋・軌跡・観測 dd・壁のモデル (distortion, model_dd)・粗い測距 Dn

DDk = dd[kf].T                                     # [5, K] 観測 (キーフレーム)
HUB = 0.03                                         # Huber の境目 [m]
def huber_w(r): a = np.abs(r); return np.where(a <= HUB, 1.0, HUB / np.maximum(a, 1e-12))

def correction(R, P, use_walls):
    """壁の反射による補正 c[j, k] (model_dd の直接波以外の部分)。r0 は原点。"""
    if not use_walls: return np.zeros((5, K))
    rr0 = np.zeros(3) + r0_world
    c = np.zeros((5, K))
    for j in range(5):
        dl = distortion(P + r0_world, R[j] + r0_world, WALLS_TRUE, G_TRUE); d0 = distortion(rr0[None], R[j] + r0_world, WALLS_TRUE, G_TRUE)[0]
        c[j] = -LAM / (4 * np.pi) * np.angle(np.exp(1j * 2 * (dl - d0)))
    return c

def step_traj(R, P, c, lam_s=0.3, iters=3):
    """ノードを固定して、全キーフレームの p_k をまとめて解く。"""
    for _ in range(iters):
        D = P[None] - R[:, None]; L = np.linalg.norm(D, axis=2) + 1e-9          # [5, K]
        res = L - np.linalg.norm(R, axis=1)[:, None] + c - DDk; w = huber_w(res)
        U = D / L[..., None]                                                    # ∂L/∂p
        A = np.einsum("jk,jka,jkb->kab", w, U, U); b = -np.einsum("jk,jk,jka->ka", w, res, U)
        # なめらかさ: p_k を前後の平均に寄せる (弱く)
        nb = np.zeros_like(P); nb[1:-1] = 0.5 * (P[:-2] + P[2:]); nb[0] = P[1]; nb[-1] = P[-2]
        A = A + lam_s * np.eye(3)[None]; b = b + lam_s * (nb - P)
        P = P + np.linalg.solve(A, b[..., None])[..., 0]
        P[0] = 0; P[-1] = 0                                                     # 最初と最後は設置台
    return P

def step_nodes(R, P, c, iters=3):
    """軌跡を固定して、各ノードの r_j を解く。"""
    for _ in range(iters):
        for j in range(5):
            D = P - R[j]; L = np.linalg.norm(D, axis=1) + 1e-9; n0 = np.linalg.norm(R[j]) + 1e-9
            res = L - n0 + c[j] - DDk[j]; w = huber_w(res)
            g = -D / L[:, None] - R[j] / n0                                     # ∂res/∂r_j
            A = (w[:, None, None] * g[:, :, None] * g[:, None, :]).sum(0) + 1e-6 * np.eye(3)
            R[j] = R[j] - np.linalg.solve(A, (w * res) @ g)
    return R

def cost(R, P, c):
    res = np.linalg.norm(P[None] - R[:, None], axis=2) - np.linalg.norm(R, axis=1)[:, None] + c - DDk
    a = np.abs(res); return float(np.sum(np.where(a <= HUB, 0.5 * a ** 2, HUB * (a - 0.5 * HUB))))

def init_traj(R):
    P = np.zeros((K, 3)); prev = np.zeros(3)
    for k in range(K):
        rk = np.linalg.norm(R, axis=1) + DDk[:, k]
        prev = least_squares(lambda p: np.linalg.norm(p - R, axis=1) - rk, prev).x; P[k] = prev
    return P

def run(mode, R_init, n_alt=40):
    use_walls = mode == "B"
    R = R_init.copy(); P = init_traj(R); c = np.zeros((5, K))
    for it in range(n_alt):
        P = step_traj(R, P, c); R = step_nodes(R, P, c)
        if use_walls and it >= 5: c = correction(R, P, True)   # 形がある程度決まってから反射の補正を入れる
    return R, P, c, cost(R, P, c)

def evaluate(R, P):
    X = np.vstack([np.zeros(3), R, P]) + r0_world; Y = np.vstack([r0, nodes, traj[kf]])
    mx, my = X.mean(0), Y.mean(0); U_, _, Vt = np.linalg.svd((X - mx).T @ (Y - my)); Xa = (X - mx) @ (U_ @ Vt) + my
    return np.linalg.norm(Xa[:6] - Y[:6], axis=1), np.linalg.norm(Xa[6:] - Y[6:], axis=1)

# 初期値の元: 粗い測距 (15 組、±30 cm) → 古典的 MDS (calib_sim.py と同じ)。座標は r0 を原点にする
Jm = np.eye(6) - 1 / 6; Bm = -0.5 * Jm @ (Dn ** 2) @ Jm; wv, Vv = np.linalg.eigh(Bm); i3 = np.argsort(wv)[::-1][:3]
X = Vv[:, i3] * np.sqrt(np.maximum(wv[i3], 1e-9)); R_mds = X[1:] - X[0]
# MDS の座標は向きが任意。モード B (壁のモデル) は部屋の座標が必要なので、評価用に正解の向きへ合わせたものを使う
# (実機では壁の向きの推定が必要 → 後の課題)。A は向きによらない
U_, _, Vt = np.linalg.svd(R_mds.T @ (nodes - r0)); R_mds = R_mds @ (U_ @ Vt)
r0_world = r0.copy()
rng_s = np.random.default_rng(500 + int(SEED))
for mode, name in (("A", "A 直接波だけ"), ("B", "B 壁が既知")):
    best = None; costs = []
    for s in range(NSTART):
        Ri = R_mds + (rng_s.normal(0, 0.3, R_mds.shape) if s > 0 else 0)
        R, P, c, J = run(mode, Ri); costs.append(J)
        if best is None or J < best[3]: best = (R, P, c, J, s)
    R, P, c, J, s = best
    en, et = evaluate(R, P); en0, _ = evaluate(R_mds, init_traj(R_mds))
    print(f"{name}: ノードの誤差 中央値 {np.median(en) * 100:.1f} cm (最大 {en.max() * 100:.1f})、軌跡 中央値 {np.median(et) * 100:.1f} cm"
          f"  [MDS の初期値 {np.median(en0) * 100:.1f} cm、初期値 {NSTART} 通りのうち {s} 番目が最良]", flush=True)
