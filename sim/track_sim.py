# -*- coding: utf-8 -*-
"""シミュレーションで、CSI メッシュ + 数理モデルによる 60 Hz の全身追跡を検証する。
使い方: python track_sim.py 秒数 [SNR=25] [開始秒=0]
  データ: E:/CSI/sim_out/mesh3d_60 (6 台の 3D メッシュ、60 Hz、正解付き) と mesh3d_empty (無人 = 地図)
  欠点 (ESP32): 位相・傾き・AGC・int8・雑音・抜け 5% をわざと加える
  初期値: 最初のフレームの正解 (既知の姿勢から始める = 目視で確認した姿勢から始めるのに相当)
"""
import sys, time
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim, Esp32Impairment
from csimesh import from_sim
from meshmodel import MeshModel, Tracker, joints, NP

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 2; snr = float(sys.argv[2]) if len(sys.argv) > 2 else 25
t0s = float(sys.argv[3]) if len(sys.argv) > 3 else 0
S = load_sim(r"E:/CSI/sim_out/mesh3d_60"); Emp = load_sim(r"E:/CSI/sim_out/mesh3d_empty")
fps = S["meta"]["fps"]; i0 = int(t0s * fps); n = int(dur * fps)
mesh = from_sim(S, Esp32Impairment(snr_db=snr, loss=0.05, seed=5)).undirected()
M = len(mesh.nodes); B = np.array([Emp["H"][0, a * M + b] for a, b in mesh.edges])   # 地図 = 無人の部屋の CSI
SP = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
model = MeshModel(mesh.nodes, mesh.edges, mesh.tones, B, tone_step=3, spacing=SP)
truth_p = S["params"].astype(float); truth_j = S["joints"]

def run(use_meas):
    trk = Tracker(model, truth_p[i0], sigma_h=10 ** (-snr / 20) * 0.75 + 0.02)
    est = np.zeros((n, NP))
    for k in range(n):
        i = i0 + k
        est[k] = trk.step(mesh.h[i], mesh.valid[i] if use_meas else np.zeros(mesh.E, bool))
    err = np.array([np.linalg.norm(joints(est[k]) - truth_j[i0 + k], axis=1).mean() for k in range(n)])
    return est, err

t = time.time(); est, err = run(True); el = time.time() - t
_, err0 = run(False)
print(f"{mesh.summary()}  SNR {snr} dB")
print(f"区間 {t0s:.0f}〜{t0s + dur:.0f} 秒 ({n} フレーム)、計算 {el:.1f} 秒 ({el / n * 1000:.0f} ms/フレーム)")
print(f"関節位置の誤差 (20 関節の平均): CSI メッシュあり 平均 {err.mean() * 100:.1f} cm (最後 {err[-1] * 100:.1f} cm)  /  CSI なし (予測だけ) 平均 {err0.mean() * 100:.1f} cm (最後 {err0[-1] * 100:.1f} cm)")
segs = [("T ポーズ", 0, 3), ("待機", 3, 8), ("歩行", 8, 28), ("腕", 28, 38), ("スクワット", 38, 46), ("手を伸ばす", 46, 52), ("不規則", 52, 82)]
for name, a, b in segs:
    m = (np.arange(n) / fps + t0s >= a) & (np.arange(n) / fps + t0s < b)
    if m.any(): print(f"  {name:8s} 誤差 {err[m].mean() * 100:5.1f} cm  (CSI なし {err0[m].mean() * 100:5.1f} cm)")
pe = np.abs(est - truth_p[i0:i0 + n]); names = ["位置x", "位置z", "高さ"]
print("  パラメータごとの誤差の平均: " + ", ".join(f"{nm} {pe[:, i].mean() * 100:.1f}cm" for i, nm in enumerate(names)) +
      ", 関節角 " + f"{np.rad2deg(np.delete(pe[:, 3:], 3, axis=1)).mean():.1f}°")
np.savez(rf"E:/CSI/sim_out/track_{int(t0s)}_{int(dur)}_{int(snr)}.npz", est=est, err=err, err0=err0)
