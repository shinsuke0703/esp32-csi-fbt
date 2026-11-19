# -*- coding: utf-8 -*-
"""Carbon (Gemini) の理論 (esp32_6node_slam_fbt_optitrack_study.pdf) を実装したもの。
  段階 1 (式 1 の往復): ACK 往復の積 P_ij ≈ c·H_ij² から、背景を引いた体の成分 H_dyn ≈ (P − P_bg) / (2·sqrt(P_bg))。
     (PDF の S_ij = (Y_ij·Y_ji)^{1/2} は平方根の符号があいまいになるので、平方根を使わない形に直している)
  式 2: 体の成分の位相の変化 → 体を経由する 2 局間の経路長の変化 ΔL_ij(t) = −λ/2π · Δarg H_dyn
     (位相は元のレートのままつなぐ。フレーム間の経路長の変化が半波長より小さければ一意に決まる)
  式 3 (段階 B の BA-SLAM): ΔL_ij(t) = (|p_t − r_i| + |p_t − r_j|) − (|p_1 − r_i| + |p_1 − r_j|) を、
     人の位置 p_t (キーフレームごと) とノードの位置 r_i について非線形最小二乗で解く。人は 1 点の散乱体とみなす。
"""
import numpy as np
from scipy.optimize import least_squares

LAM = 299792458 / 2.437e9   # m (ch6)


def hdyn_from_P(P, bg_mask):
    """P [T, K] (往復の積、トーンごと) → H_dyn [T, K]。bg_mask: 人がいない時刻 (背景)。"""
    P = P / np.sqrt(np.abs(P).mean(1, keepdims=True) + 1e-30)          # AGC による振幅の変化を外す
    Pbg = np.median(P[bg_mask].real, 0) + 1j * np.median(P[bg_mask].imag, 0)
    Hbg = np.sqrt(Pbg)
    for k in range(1, len(Hbg)):                                       # 背景の符号を周波数方向にそろえる (1 回だけ)
        if np.real(Hbg[k] * np.conj(Hbg[k - 1])) < 0: Hbg[k] *= -1
    Hbg = np.where(np.abs(Hbg) < 1e-9, 1e-9, Hbg)                      # ゼロ割を避ける
    return (P - Pbg) / (2 * Hbg), Hbg


def delta_path(t, Hd, Hbg, tol=1.0):
    """体の成分の位相の変化 → 経路長の変化 ΔL [m]。前後と大きく違うサンプル (外れ値) は使わない。
    戻り値: 使ったサンプルの時刻, ΔL, 体の成分の強さ (dB)"""
    z = (Hd * np.conj(Hbg / np.abs(Hbg))).sum(1)                       # トーンを合わせた体の成分 (背景の位相をそろえて足す)
    pw = 10 * np.log10((np.abs(Hd) ** 2).mean(1) / (np.abs(Hbg) ** 2).mean() + 1e-30)
    ph = np.angle(z); n = len(ph); keep = np.ones(n, bool)
    zz = np.exp(1j * ph)
    for i in range(n):
        lo, hi = max(0, i - 4), min(n, i + 5)
        keep[i] = abs(np.angle(zz[i] * np.conj(np.mean(zz[lo:hi])))) < tol
    u = np.unwrap(ph[keep])
    return t[keep], -LAM / (2 * np.pi) * (u - u[0]), pw[keep]


def resample(tk, t, x):
    return np.interp(tk, t, x)


def ba_solve(obs, R0, Pxz0, y_person=1.0, fix_nodes=False, w_smooth=0.05, verbose=0):
    """obs: {(i, j): ΔL [T]} (キーフレームの時刻はそろえてある)。R0 [M, 3] ノードの初期値、Pxz0 [T, 2] 人の (x, z) の初期値。
    人の高さは y_person に固定 (胴体を 1 点とみなす)。w_smooth: 人の軌跡のなめらかさの重み [1/m]。
    ゲージ (全体の平行移動・回転) は、ノード 0 の位置を初期値に固定し、ノード 1 の向きを固定することで止める。"""
    M = len(R0); T = len(Pxz0); links = sorted(obs)
    free_nodes = [] if fix_nodes else list(range(1, M))

    def unpack(x):
        R = R0.copy(); o = 0
        for i in free_nodes:
            if i == 1: R[1, 0] = x[o]; R[1, 1] = x[o + 1]; o += 2      # ノード 1 は x と高さだけ (z は初期値に固定 → 回転を止める)
            else: R[i] = x[o:o + 3]; o += 3
        Pxz = x[o:].reshape(T, 2)
        return R, np.column_stack([Pxz[:, 0], np.full(T, y_person), Pxz[:, 1]])

    def resid(x):
        R, P = unpack(x); out = []
        for (i, j) in links:
            L = np.linalg.norm(P - R[i], axis=1) + np.linalg.norm(P - R[j], axis=1)
            out.append((L - L[0]) - obs[(i, j)])
        out.append(w_smooth * np.diff(P[:, [0, 2]], axis=0).ravel() * 10)   # なめらかさ (0.1 m/キーフレームで 0.05 m 相当)
        return np.concatenate(out)

    x0 = []
    for i in free_nodes:
        x0 += [R0[1, 0], R0[1, 1]] if i == 1 else list(R0[i])
    x0 = np.array(x0 + list(Pxz0.ravel()))
    r = least_squares(resid, x0, loss="soft_l1", f_scale=0.03, max_nfev=200, verbose=verbose)
    R, P = unpack(r.x)
    return R, P, r
