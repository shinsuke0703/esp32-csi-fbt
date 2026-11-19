# -*- coding: utf-8 -*-
"""配置の比較: 2D リング (全員が同じ高さ) と 3D メッシュ (高さを散らす)。全員が交代で 1 回ずつ送信 (予算 = 台数)。
使い方: python mesh_compare.py ring2d current mesh3d mesh3d_ceil cube8
"""
import sys, json
import numpy as np

UNIT_M, UNIT_R = 0.05, np.deg2rad(5)
GROUPS = {"体の位置": [0, 1, 2], "向き・背骨": [3, 4, 5], "肩": [7, 8, 10, 11], "肘": [9, 12], "股関節": [13, 14, 16, 17], "膝": [15, 18]}

def load(name):
    base = rf"E:/CSI/sim_out/jac_{name}"; meta = json.load(open(base + ".json", encoding="utf-8"))
    N, L, K, NP = meta["N"], meta["L"], meta["K"], meta["NP"]
    a = np.fromfile(base + ".bin", dtype=np.float64); o = N * NP
    H0 = a[o:o + N * L * K * 2].reshape(N, L, K, 2); o += N * L * K * 2
    J = a[o:].reshape(N, NP, L, K, 2)
    return meta, H0[..., 0] + 1j * H0[..., 1], J[..., 0] + 1j * J[..., 1]

def run(name, snr_db):
    meta, H0, J = load(name); N, L, K, NP = meta["N"], meta["L"], meta["K"], meta["NP"]
    M = len(meta["tx"]); tones = np.array(meta["tones"], float)
    unit = np.array([UNIT_M] * 3 + [UNIT_R] * (NP - 3))
    R = lambda z: np.concatenate([z.real, z.imag], 0)
    links = [t * M + r for t in range(M) for r in range(M) if t != r]   # 全員が 1 回ずつ送信 → 自分以外が受信
    ndof, crb = [], []
    for n in range(N):
        F = np.zeros((NP, NP))
        for l in links:
            h = H0[n, l]; Jr = R(J[n][:, l].T)
            Nm = np.stack([R(1j * h), R(1j * tones * h), R(h)], 1); Q, _ = np.linalg.qr(Nm); Jr = Jr - Q @ (Q.T @ Jr)
            F += Jr.T @ Jr / ((np.abs(h) ** 2).mean() / 10 ** (snr_db / 10) / 2)
        Fs = F * np.outer(unit, unit); ev = np.linalg.eigvalsh(Fs); ndof.append(int((ev >= 1).sum()))
        crb.append(np.sqrt(np.diag(np.linalg.inv(Fs + 1e-6 * np.eye(NP)))) * unit)
    return M, np.array(ndof), np.median(np.array(crb), 0)

names = sys.argv[1:] or ["ring2d", "current", "mesh3d", "mesh3d_ceil", "cube8"]
desc = {"ring2d": "2D リング (全員 1.2 m)", "current": "今の配置 (1.0〜2.4 m)", "mesh3d": "3D メッシュ (0.3/2.4 m 交互)",
        "mesh3d_ceil": "3D メッシュ + 天井の中央", "cube8": "参考: 8 台で立方体の角"}
for snr in (25, 15):
    print(f"\nSNR {snr} dB、全員が交代で 1 回ずつ送信、HT40、60 姿勢の中央値。CRB は 位置 [cm] / 関節角 [°]")
    print(f"  {'配置':28s} 台数 決まる自由度   " + "  ".join(f"{g:>6s}" for g in GROUPS))
    for nm in names:
        M, d, c = run(nm, snr)
        cells = []
        for g, idx in GROUPS.items():
            v = c[idx].max(); cells.append(f"{v * 100:6.1f}" if g == "体の位置" else f"{np.rad2deg(v):6.1f}")
        print(f"  {desc.get(nm, nm):28s} {M:3d}  {int(np.median(d)):2d} [{d.min():2d},{d.max():2d}]   " + "  ".join(cells))
print("\n各グループの値は、その中で一番悪いパラメータの CRB。head_yaw は原理的に決まらないので除外 (自由度の上限 18)。")
