# -*- coding: utf-8 -*-
"""Carbon の理論 (ACK 往復 + 体の成分の位相 + BA-SLAM) を、シミュレータの正解で検証する。
使い方: python carbon_sim.py [SNR=25]
  データ: mesh3d_60 (6 台の 3D メッシュ、60 Hz、正解付き)、背景は mesh3d_empty。歩行の区間 (8〜28 秒、円の上を歩く)。
  往復の積: P = (H + n1)(H + n2) (ACK 往復では共通位相が消えるので、残るのは雑音だけ)。外れ値 5% (位相がばらばら)。
  トーン: LLTF と同じ 52 本 (-26..26、0 を除く) に相当するものを使う。
"""
import sys, json
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim
from carbon_model import hdyn_from_P, delta_path, ba_solve, LAM

SNR = float(sys.argv[1]) if len(sys.argv) > 1 else 25
rng = np.random.default_rng(0)
S = load_sim(r"E:/CSI/sim_out/mesh3d_60"); E = load_sim(r"E:/CSI/sim_out/mesh3d_empty")
meta = S["meta"]; fps = meta["fps"]; tones = np.array(meta["tones"]); M = len(meta["tx"])
nodes = np.array(meta["tx"], float)
tsel = (np.abs(tones) <= 26 * 2)   # シミュレータは HT40 (間隔 312.5 kHz、±58)。LLTF 相当の帯域 (±26 本 × 2 倍の番号) に近い ±52 を使う
tsel &= (tones % 2 == 0)           # 52 本程度に間引く
t0s, t1s = 8.0, 28.0; i0, i1 = int(t0s * fps), int(t1s * fps)
nbg = int(2 * fps)                 # 先頭に無人の 2 秒を足す (背景用)

obs60 = {}; pw_all = {}
for a in range(M):
    for b in range(a + 1, M):
        H = np.concatenate([np.repeat(E["H"][:1, a * M + b][:, tsel], nbg, 0), S["H"][i0:i1, a * M + b][:, tsel]]).astype(complex)
        s = np.sqrt((np.abs(H) ** 2).mean() / 10 ** (SNR / 10) / 2)
        n1 = s * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape)); n2 = s * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
        P = (H + n1) * (H + n2) * np.exp(1j * rng.uniform(0, 2 * np.pi))       # 組ごとの一定の位相 (c_ij)
        bad = rng.random(len(P)) < 0.05; P[bad] *= np.exp(1j * rng.uniform(0, 2 * np.pi, bad.sum()))[:, None]
        t = np.arange(len(P)) / fps - 2.0                                      # 0 = 歩行の区間の始まり
        Hd, Hbg = hdyn_from_P(P, t < 0)
        tt, dL, pw = delta_path(t[t >= 0], Hd[t >= 0], Hbg)
        obs60[(a, b)] = (tt, dL); pw_all[(a, b)] = np.median(pw)

# 正解の経路長の変化 (胴体 = 骨盤と胸の中間を 1 点の散乱体とみなす) と比べる
J = S["joints"][i0:i1].astype(float); torso = (J[:, 0] + J[:, 1]) / 2
tk = np.arange(0, t1s - t0s, 0.1)                                             # キーフレーム 10 Hz
ptrue = np.array([np.interp(tk, np.arange(len(torso)) / fps, torso[:, c]) for c in range(3)]).T
print(f"SNR {SNR} dB、組 {len(obs60)}、キーフレーム {len(tk)} (10 Hz、{t1s - t0s:.0f} 秒の歩行)")
print("組ごと: 体の成分の強さ (dB) / 観測した経路長の変化と、胴体 1 点で計算した正解との差 (RMS cm)")
obs = {}
for (a, b), (tt, dL) in obs60.items():
    y = np.interp(tk, tt, dL); obs[(a, b)] = y
    Lt = np.linalg.norm(ptrue - nodes[a], axis=1) + np.linalg.norm(ptrue - nodes[b], axis=1); Lt -= Lt[0]
    print(f"  ({a},{b}) {pw_all[(a, b)]:6.1f} dB  差 {np.sqrt(np.mean((y - Lt) ** 2)) * 100:6.1f} cm  (変化の幅 正解 {np.ptp(Lt) * 100:5.0f} cm / 観測 {np.ptp(y) * 100:5.0f} cm)")

def report(name, R, P):
    e = np.linalg.norm(P[:, [0, 2]] - ptrue[:, [0, 2]], axis=1)
    en = np.linalg.norm(R - nodes, axis=1)
    print(f"{name}: 人の位置の誤差 (水平) 中央値 {np.median(e) * 100:.1f} cm、最大 {e.max() * 100:.1f} cm / ノードの位置の誤差 中央値 {np.median(en) * 100:.1f} cm")

P0 = np.tile(np.array([[0.0, 0.0]]), (len(tk), 1))
R, P, r = ba_solve(obs, nodes, P0, fix_nodes=True); report("A. ノードの位置が既知 (人の軌跡だけ推定)", R, P)
P0b = np.tile(ptrue[:1, [0, 2]], (len(tk), 1)); R, P, r = ba_solve(obs, nodes, P0b, fix_nodes=True); report("B. ノード既知、人の最初の位置も既知", R, P)
R0 = nodes + rng.normal(0, 0.3, nodes.shape); R0[0] = nodes[0]; R0[1, 2] = nodes[1, 2]
R, P, r = ba_solve(obs, R0, P0b, fix_nodes=False); report("C. ノードの位置も推定 (初期値に 30 cm の誤差)", R, P)
print(f"   (C の初期値のノード誤差 中央値 {np.median(np.linalg.norm(R0 - nodes, axis=1)) * 100:.1f} cm)")
np.savez(r"E:/CSI/sim_out/carbon_sim_result.npz", tk=tk, ptrue=ptrue, R=R, P=P, nodes=nodes)
