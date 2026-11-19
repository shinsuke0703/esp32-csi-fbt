"""推定器の仕組みの確認: 推定器自身のモデルで作ったデータ (モデルのずれなし) で追跡できるか。使い方: python track_selfcheck.py 秒数 開始秒"""
import sys, time
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim
from csimesh import canon
from meshmodel import MeshModel, Tracker, joints, NP
dur, t0s = float(sys.argv[1]), float(sys.argv[2]); snr = 25
DATA = sys.argv[3] if len(sys.argv) > 3 else r"E:/CSI/sim_out/mesh3d_60"; STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 1
S = load_sim(DATA); Emp = load_sim(r"E:/CSI/sim_out/mesh3d_empty")
M = 6; edges = [(a, b) for a in range(M) for b in range(M) if a < b]
nodes = np.array(S["meta"]["tx"]); tones = np.array(S["meta"]["tones"])
B = np.array([Emp["H"][0, a * M + b] for a, b in edges]).astype(complex)
m = MeshModel(nodes, edges, tones, B, tone_step=3, spacing=0.15)
fps = S["meta"]["fps"]; i0 = int(t0s * fps); idx = list(range(i0, i0 + int(dur * fps), STRIDE)); n = len(idx); rng = np.random.default_rng(0)
truth = S["params"].astype(float)
trk = Tracker(m, truth[i0], sig_pos=0.03 * STRIDE * 60 / fps, sig_ang=np.deg2rad(6) * STRIDE * 60 / fps, sigma_h=10 ** (-snr / 20) * 0.75 + 0.02); err = []
t = time.time()
for k in range(n):
    i = idx[k]; Hm = m.H(truth[i]); s = np.sqrt((np.abs(Hm) ** 2).mean(1, keepdims=True) / 10 ** (snr / 10) / 2)
    Y = Hm + s * (rng.standard_normal(Hm.shape) + 1j * rng.standard_normal(Hm.shape))
    hobs = np.zeros((len(edges), len(tones)), complex); hobs[:, m.tsel] = canon(Y, m.tones)[0]
    p = trk.step(hobs, np.ones(len(edges), bool))
    err.append(np.linalg.norm(joints(p) - S["joints"][i], axis=1).mean())
err = np.array(err)
print(f"[{fps / STRIDE:.0f} Hz で測定] モデルのずれなし、SNR {snr} dB、{t0s:.0f}〜{t0s+dur:.0f} 秒: 関節位置の誤差 平均 {err.mean()*100:.1f} cm、最後 {err[-1]*100:.1f} cm ({(time.time()-t)/n*1000:.0f} ms/フレーム)")
