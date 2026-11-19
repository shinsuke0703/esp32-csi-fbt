# -*- coding: utf-8 -*-
"""CGSC 定理 1 の確認: 半角の残差 g = Im(e^{−jφ/2}·W) と、従来の折り返し残差 wrap(φ − arg W²) で、
ガウス・ニュートン法の収束できる範囲 (解き始めのずれ) と、深いフェージングの近くでの安定性を比べる。
設定: 6 ノード (calib_sim と同じ配置) の 15 リンク、体の代わりに点の散乱体 1 個 (3 次元の位置が未知)。
  η_p(x) = A_p·exp(−jk(L_p(x) − L_p,ref))   L_p: 2 ノードと散乱体を結ぶ経路長、|η| は 0.3〜0.8 (実測の範囲)
  観測 U_p = W_p²/|W_p|² に位相雑音 0.28 rad。解き始めのずれ r を変えて、正しい位置 (1 cm 以内) に収束した割合を数える。"""
import numpy as np
LAM = 0.1236; k = 2 * np.pi / LAM
nodes = np.array([[0, 1.0, 3.2], [3.1, 2.3, 1.5], [3.0, 0.4, -2.0], [-0.5, 2.1, -3.2], [-3.1, 0.6, -1.0], [-3.0, 1.8, 2.0]])
pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
rng = np.random.default_rng(0)

def eta(x, A, Lref):
    L = np.array([np.linalg.norm(x - nodes[i]) + np.linalg.norm(x - nodes[j]) for i, j in pairs])
    return A * np.exp(-1j * k * (L - Lref))
def resid(x, phi, A, Lref, kind):
    W = 1 + eta(x, A, Lref)
    if kind == "wrap": return np.angle(np.exp(1j * (phi - np.angle(W ** 2))))
    return np.imag(np.exp(-1j * phi / 2) * W) / 0.5          # 半角の残差 (大きさをそろえるため 0.5 で割る)
def gn(x0, phi, A, Lref, kind, iters=30):
    x = x0.copy()
    for _ in range(iters):
        r = resid(x, phi, A, Lref, kind); J = np.zeros((15, 3))
        for c in range(3):
            e = np.zeros(3); e[c] = 1e-4; J[:, c] = (resid(x + e, phi, A, Lref, kind) - r) / 1e-4
        x = x - np.linalg.lstsq(J.T @ J + 1e-6 * np.eye(3), J.T @ r, rcond=None)[0]
    return x

print("解き始めのずれ | 折り返し残差の成功率 | 半角残差の成功率   (各 200 回、|η| 0.3〜0.8、位相雑音 0.28 rad)")
for r0 in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
    ok = {"wrap": 0, "half": 0}
    for t in range(200):
        xt = np.array([0, 1.0, 0]) + rng.uniform(-0.8, 0.8, 3) * [1, 0.5, 1]
        A = rng.uniform(0.3, 0.8, 15); Lref = rng.uniform(0, LAM, 15)
        W = 1 + eta(xt, A, Lref); phi = np.angle(W ** 2) + rng.normal(0, 0.28, 15)
        d = rng.normal(size=3); d /= np.linalg.norm(d); x0 = xt + d * r0 / 100
        for kind in ok: ok[kind] += np.linalg.norm(gn(x0, phi, A, Lref, kind) - xt) < 0.01
    print(f"{r0:4.1f} cm | {ok['wrap'] / 2:5.1f} % | {ok['half'] / 2:5.1f} %")

# 深いフェージングの近く: |η| を 1 に近づけ、W がほぼ 0 になるリンクを含めたときの収束
print("\n深いフェージング (|η| 0.9〜0.99 のリンクを 3 本含む)、解き始めのずれ 1 cm:")
ok = {"wrap": 0, "half": 0}
for t in range(200):
    xt = np.array([0, 1.0, 0]) + rng.uniform(-0.8, 0.8, 3) * [1, 0.5, 1]
    A = rng.uniform(0.3, 0.8, 15); A[:3] = rng.uniform(0.9, 0.99, 3); Lref = rng.uniform(0, LAM, 15)
    L = np.array([np.linalg.norm(xt - nodes[i]) + np.linalg.norm(xt - nodes[j]) for i, j in pairs])
    Lref[:3] = L[:3] - (np.pi + rng.normal(0, 0.15, 3)) / k        # その 3 本は真の位置でほぼ打ち消し合う (W ≈ 0)
    W = 1 + eta(xt, A, Lref); phi = np.angle(W ** 2) + rng.normal(0, 0.28, 15)
    d = rng.normal(size=3); d /= np.linalg.norm(d); x0 = xt + d * 0.01
    for kind in ok: ok[kind] += np.linalg.norm(gn(x0, phi, A, Lref, kind) - xt) < 0.01
print(f"折り返し残差 {ok['wrap'] / 2:.1f} %、半角残差 {ok['half'] / 2:.1f} %")
