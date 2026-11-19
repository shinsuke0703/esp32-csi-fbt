# -*- coding: utf-8 -*-
"""「C3 を持って歩く」校正のシミュレーション: 5 台の固定ノードの位置を、手に持って歩く C3 との ACK 往復の位相から推定する。
使い方: python calib_sim.py [SNR=25] [外れ値=0.08] [seed=0]

データを作る側 (現実に近づける):
  部屋 7×7×2.7 m。ノード 0 (C3) の設置台 r0 と、固定ノード 1〜5。C3 を棒の先に付けて r0 から出発し、
  部屋の中を高さを変えながら (0.8〜2.0 m) 約 0.5 m/s で 60 秒歩いて r0 に戻る。
  電波: 直接波 + 床と 4 面の壁での反射 (鏡像) + 物体 6 個での散乱。LLTF 52 本 (ch6)。往復 57 Hz。
  往復の積 P = (H + n1)(H + n2)·exp(jψ_j) (ψ_j は組ごとの一定の位相)。外れ値は位相がばらばらのサンプル。
解く側 (直接波だけのモデル):
  1. 組ごとに往復の積の位相を時間方向につなぐ (外れ値は前後と比べてはじく) → 距離の変化 Δd_j(t) = −λ/(4π)·Δφ
  2. 10 Hz のキーフレームで、|p_k − r_j| − |r0 − r_j| = Δd_j(t_k) を最小二乗で解く (未知数: r_1..r_5、p_k)。
     ゲージ: r0 = 原点、ノード 1 は +x 軸上の高さ 0 の面 (向きを固定)、ノード 2 は z ≥ 0 側 (鏡像を固定)。最初と最後の p は r0。
  初期値: 各ノードまでの距離を粗く (±30 cm の誤差、80 MHz の測距相当) 与え、そこから多辺測量で作る。
"""
import sys
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 25
OUT = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
rng = np.random.default_rng(int(sys.argv[3]) if len(sys.argv) > 3 else 0)
C = 299792458.0; fc = 2.437e9; DF = 312.5e3; LAM = C / fc
tones = np.r_[np.arange(-26, 0), np.arange(1, 27)]; f = fc + tones * DF
FPS = 57.0; DUR = 60.0

# ---- 部屋と配置 (ノードはばらばらの高さで部屋を囲う) ----
W = 7.0; Hroom = 2.7
r0 = np.array([0.0, 1.0, 3.2])                                    # C3 の設置台 (壁際)
nodes = np.array([[3.1, 2.3, 1.5], [3.0, 0.4, -2.0], [-0.5, 2.1, -3.2], [-3.1, 0.6, -1.0], [-3.0, 1.8, 2.0]])
clutter = np.column_stack([rng.uniform(-3, 3, 6), rng.uniform(0.3, 1.5, 6), rng.uniform(-3, 3, 6)])

# ---- C3 を持って歩く軌跡: 設置台 → 各ノードの近く (部屋の中心側に 0.8 m) を順に回る → 設置台。0.4 m/s、高さは 0.9〜1.9 m で変える ----
center = np.array([0.0, 1.4, 0.0])
wps = [r0]
for j, nd in enumerate(nodes):
    v = center - nd; v[1] = 0; v = v / np.linalg.norm(v)
    wp = nd + 0.8 * v; wp[1] = 0.9 if j % 2 == 0 else 1.9; wps.append(wp)
wps.append(r0); wps = np.array(wps)
seg = np.linalg.norm(np.diff(wps, axis=0), axis=1); s_cum = np.r_[0, np.cumsum(seg)]
DUR = s_cum[-1] / 0.4                                              # 0.4 m/s で歩く
tt = np.arange(0, DUR, 1 / FPS)
from scipy.interpolate import CubicSpline
cs = CubicSpline(s_cum / s_cum[-1] * DUR, wps, bc_type="clamped"); traj = cs(tt)
speed = np.linalg.norm(np.diff(traj, axis=0), axis=1) * FPS
print(f"軌跡: {DUR:.0f} 秒、平均 {speed.mean():.2f} m/s、最大 {speed.max():.2f} m/s、高さ {traj[:, 1].min():.2f}〜{traj[:, 1].max():.2f} m")

def channel(p, q):
    """p と q の間の H(f): 直接波 + 床・4 面の壁の鏡像 + 物体での散乱。"""
    H = np.zeros(len(f), complex)
    def add(L, a): H[:] += a / L * np.exp(-2j * np.pi * f * L / C)
    add(np.linalg.norm(p - q), 1.0)
    for axis, plane, g in ((1, 0.0, -0.5), (1, Hroom, -0.3), (0, W / 2, -0.3), (0, -W / 2, -0.3), (2, W / 2, -0.3), (2, -W / 2, -0.3)):
        img = q.copy(); img[axis] = 2 * plane - q[axis]; add(np.linalg.norm(p - img), g)
    for c in clutter:
        L = np.linalg.norm(p - c) + np.linalg.norm(c - q); add(L, 0.15 * 3.0 / L)
    return H

# ---- 観測 (往復の積) ----
phi = np.zeros((len(tt), 5))
for j in range(5):
    psi = rng.uniform(-np.pi, np.pi)
    for k, p in enumerate(traj):
        H = channel(p, nodes[j]); s = np.sqrt((np.abs(H) ** 2).mean() / 10 ** (SNR / 10) / 2)
        P = (H + s * (rng.standard_normal(52) + 1j * rng.standard_normal(52))) * (H + s * (rng.standard_normal(52) + 1j * rng.standard_normal(52)))
        ph = np.angle(P.sum()) + psi
        if rng.random() < OUT: ph = rng.uniform(-np.pi, np.pi)
        phi[k, j] = ph

# ---- 解く側 1: 外れ値をはじいて時間方向につなぐ → 距離の変化 ----
def robust_unwrap(ph, tol=1.0, alpha=0.6, beta=0.15):
    """α-β 追跡 (2 次の PLL) で位相をつなぐ。位相 φ と回る速さ ω を持ち、次の位相を予測して、
    予測から tol rad 以上外れたサンプルは外れ値として捨てる (予測だけ進める)。動いていても正しく追える。"""
    n = len(ph); out = np.zeros(n); phi_ = ph[0]; w = 0.0
    for i in range(n):
        pred = phi_ + w
        inn = np.angle(np.exp(1j * (ph[i] - pred)))
        if abs(inn) < tol: phi_ = pred + alpha * inn; w = w + beta * inn
        else: phi_ = pred
        out[i] = phi_
    return out
dd = np.column_stack([-LAM / (4 * np.pi) * (lambda u: u - u[0])(robust_unwrap(phi[:, j])) for j in range(5)])
LOOPFIX = True
if LOOPFIX: dd = dd - np.outer(np.arange(len(tt)) / (len(tt) - 1), dd[-1])   # ループの閉合
true_dd = np.column_stack([np.linalg.norm(traj - nodes[j], axis=1) - np.linalg.norm(traj[0] - nodes[j]) for j in range(5)])
err_dd = np.abs(dd - true_dd)
print(f"距離の変化の誤差 (位相をつないだ結果 vs 正解): 中央値 {np.median(err_dd) * 100:.1f} cm、最後 (設置台に戻った時) {np.abs(dd[-1] - true_dd[-1]).max() * 100:.1f} cm")

# ---- 解く側 2: キーフレームで最小二乗 ----
kf = np.arange(0, len(tt), int(FPS / 10)); K = len(kf)
d0 = np.linalg.norm(nodes - r0, axis=1)                             # 設置台 - ノードの距離 (真値)
d0_init = d0 + rng.normal(0, 0.3, 5)                                # 粗い測距 (±30 cm)
def to_world(x):
    R = np.zeros((5, 3)); R[0] = [x[0], 0.0, 0.0]; R[1] = [x[1], x[2], 0.0]; R[2:] = x[3:12].reshape(3, 3)   # ノード 1 は +x 軸上、ノード 2 は xy 平面上 (回転を止める)
    P = x[12:].reshape(K, 3); return R, P
def resid(x):
    R, P = to_world(x); out = []
    P = P.copy(); P[0] = 0; P[-1] = 0                               # 最初と最後は設置台 (原点)
    for j in range(5):
        out.append(np.linalg.norm(P - R[j], axis=1) - np.linalg.norm(R[j]) - dd[kf, j])
    out.append(0.02 * np.diff(P, axis=0).ravel())                   # 軌跡のなめらかさ (弱く)
    Rall = np.vstack([np.zeros(3), R])                               # 粗い測距 (±30 cm) を弱い制約として (6 台の 15 組)
    for a_ in range(6):
        for b_ in range(a_ + 1, 6): out.append(np.atleast_1d(0.1 * (np.linalg.norm(Rall[a_] - Rall[b_]) - Dn[a_, b_])))
    return np.concatenate(out)
# 初期値: 6 台 (設置台 r0 とノード 1〜5) の 15 組すべての粗い測距 (±30 cm、80 MHz の測距相当) → 古典的 MDS
allp = np.vstack([r0, nodes]); Dn = np.linalg.norm(allp[:, None] - allp[None], axis=-1)
noise = rng.normal(0, 0.3, (6, 6)); noise = np.triu(noise, 1); Dn = np.abs(Dn + noise + noise.T)
Jm = np.eye(6) - 1 / 6; Bm = -0.5 * Jm @ (Dn ** 2) @ Jm; wv, Vv = np.linalg.eigh(Bm); i3 = np.argsort(wv)[::-1][:3]
X = Vv[:, i3] * np.sqrt(np.maximum(wv[i3], 1e-9)); R0 = X[1:] - X[0]   # r0 を原点に
if np.linalg.det(np.c_[R0[0], R0[1], R0[2]]) * np.linalg.det(np.c_[(nodes - r0)[0], (nodes - r0)[1], (nodes - r0)[2]]) < 0: R0[:, 2] *= -1   # 鏡像は向きの手がかりで決める想定
# ゲージに合わせて回転 (ノード 1 を +x 軸上へ)
def rot_to_x(v):
    a = v / np.linalg.norm(v); b = np.array([1.0, 0, 0]); c = np.cross(a, b); s = np.linalg.norm(c); cc = a @ b
    if s < 1e-9: return np.eye(3)
    vx = np.array([[0, -c[2], c[1]], [c[2], 0, -c[0]], [-c[1], c[0], 0]]); return np.eye(3) + vx + vx @ vx * ((1 - cc) / s ** 2)
Rg = rot_to_x(R0[0]); R0 = R0 @ Rg.T
a2 = np.arctan2(R0[1, 2], R0[1, 1]); cx, sx = np.cos(-a2), np.sin(-a2)   # x 軸の周りに回して、ノード 2 を xy 平面へ
R0 = R0 @ np.array([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
x0 = np.r_[np.linalg.norm(R0[0]), R0[1, 0], R0[1, 1], R0[2:].ravel(), np.zeros(3 * K)]
# 軌跡の初期値: 各キーフレームで、初期ノード位置からの多辺測量 (距離 = 初期距離 + 距離の変化)
P0 = np.zeros((K, 3)); RR, _ = to_world(x0)
for k in range(K):
    rk = np.linalg.norm(RR, axis=1) + dd[kf[k]]
    sol = least_squares(lambda p: np.linalg.norm(p - RR, axis=1) - rk, P0[k - 1] if k else np.zeros(3)); P0[k] = sol.x
x0[12:] = P0.ravel()
S = lil_matrix((5 * K + 3 * (K - 1) + 15, len(x0)), dtype=int)
S[5 * K + 3 * (K - 1):, :12] = 1
for j in range(5):
    cols = [0] if j == 0 else ([1, 2] if j == 1 else list(range(3 + 3 * (j - 2), 6 + 3 * (j - 2))))
    for k in range(K):
        S[j * K + k, cols] = 1; S[j * K + k, 12 + 3 * k: 15 + 3 * k] = 1
for k in range(K - 1):
    for c in range(3): S[5 * K + 3 * k + c, 12 + 3 * k + c] = 1; S[5 * K + 3 * k + c, 15 + 3 * k + c] = 1
sol = least_squares(resid, x0, jac_sparsity=S, loss="soft_l1", f_scale=0.05, max_nfev=300)
R, P = to_world(sol.x)

# 評価: 推定した座標を正解に合わせる (回転・鏡像・平行移動、r0 = 原点)
def align(X, Y):
    U, _, Vt = np.linalg.svd(X.T @ Y); Rm = U @ Vt; return X @ Rm
Rt = nodes - r0; Ra = align(R, Rt)
en = np.linalg.norm(Ra - Rt, axis=1)
Pa = align(np.vstack([R, P]), np.vstack([Rt, traj[kf] - r0]))[5:]
ep = np.linalg.norm(Pa - (traj[kf] - r0), axis=1)
print(f"ノードの位置の誤差: {np.round(en * 100, 1).tolist()} cm (中央値 {np.median(en) * 100:.1f} cm)")
print(f"C3 の軌跡の誤差: 中央値 {np.median(ep) * 100:.1f} cm、最大 {ep.max() * 100:.1f} cm")
print(f"初期値のノード誤差: 中央値 {np.median(np.linalg.norm(align(R0, Rt) - Rt, axis=1)) * 100:.1f} cm")
