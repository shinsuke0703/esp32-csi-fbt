# -*- coding: utf-8 -*-
"""同時収録 (rec_both.py) の時刻合わせ: CSI と Kinect の「動きの強さ」の相互相関から、Kinect の時刻の遅れを測って sync.json に保存する。
使い方: python sync_check.py sessions/<名前>
  CSI の動きの強さ: 組ごとに往復の積 (トーンの和) を 10 パケット平均 (パケットごとの位相のぶれを消す) → 0.1 秒の差の大きさ
                    (差の区間の中央の時刻に置く)。Kinect と相関の強い組 (上位 5 組) を平均。
  Kinect の動きの強さ: 20 関節の中心差分の速さの合計。
  両方を 100 Hz に並べ、0.15 秒でならして、ずれを ±0.6 秒で探す (放物線で 10 ms より細かく)。
  遅れ > 0 なら Kinect の時刻が CSI より遅れている → Kinect の時刻から引けばそろう。
2026-11-19 の walk_sync1 で約 +207 ms (相関 0.8)。
"""
import sys, os, json
import numpy as np
import pandas as pd

ses = sys.argv[1]
d = dict(np.load(os.path.join(ses, "csi.npz"))); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
ser = {}
for (s, q) in ping:
    if (s, q) in ack:
        a, b = ping[(s, q)], ack[(s, q)]
        ser.setdefault((s, int(tgt[a])), []).append((d["t_host"][b], (lltf(a) * lltf(b)).sum()))
k = pd.read_csv(os.path.join(ses, "kinect.csv")); k = k[k["state"] == 2].drop_duplicates("frame")
tk = k["t_unix"].values
t0 = max(min(min(x[0] for x in v) for v in ser.values()), tk.min()); t1 = min(max(max(x[0] for x in v) for v in ser.values()), tk.max())
grid = np.arange(t0, t1, 0.01)
rates = {f"{a}-{b}": round(len(v) / (max(x[0] for x in v) - min(x[0] for x in v)), 1) for (a, b), v in sorted(ser.items())}
print(f"重なっている時間 {t1 - t0:.1f} 秒。CSI の組ごとのレート [Hz]:", rates)
sm = lambda x, n: np.convolve(x, np.ones(n) / n, "same"); z = lambda x: (x - x.mean()) / (x.std() + 1e-12)
J = [c[:-2] for c in k.columns if c.endswith("_x")]
X = np.stack([k[[f"{j}_x", f"{j}_y", f"{j}_z"]].values for j in J], 1)
ks = sm(np.interp(grid, tk[1:-1], np.linalg.norm(X[2:] - X[:-2], axis=2).sum(1) / (tk[2:] - tk[:-2])), 15)
def xcorr(a, b, n=60):
    lags = np.arange(-n, n + 1); c = np.array([np.mean(z(a)[max(0, -l):len(a) - max(0, l)] * z(b)[max(0, l):len(b) - max(0, -l)]) for l in lags])
    i = int(np.argmax(c)); off = 0.5 * (c[i - 1] - c[i + 1]) / (c[i - 1] - 2 * c[i] + c[i + 1]) if 0 < i < len(c) - 1 else 0.0
    return c[i], (lags[i] + off) * 10.0
sig = {}
for p, v in ser.items():
    v = sorted(v, key=lambda x: x[0]); t = np.array([x[0] for x in v]); P = np.array([x[1] for x in v])
    if len(t) < 100: continue
    Pm = np.convolve(P, np.ones(10) / 10, "same"); hz = len(t) / (t[-1] - t[0]); lag = max(1, int(round(0.1 * hz)))
    dm = np.abs(Pm[lag:] - Pm[:-lag]) / np.median(np.abs(Pm)); sig[p] = z(np.interp(grid, 0.5 * (t[lag:] + t[:-lag]), dm))
per = sorted(((xcorr(sm(s, 15), ks)[0], p) for p, s in sig.items()), reverse=True)
print("組ごとの相関 (上位):", [(f"{p[0]}-{p[1]}", round(c, 2)) for c, p in per[:8]])
top = [p for c, p in per[:5]]
c, lag_ms = xcorr(sm(np.mean([sig[p] for p in top], 0), 15), ks)
print(f"上位 5 組の平均: 相関 {c:.2f}、Kinect の遅れ {lag_ms:+.0f} ms (正なら Kinect の時刻が CSI より遅れている)")
hip = k[["HipCenter_x", "HipCenter_z"]].values
print(f"Kinect の腰の範囲: 左右 {hip[:, 0].min():.2f}〜{hip[:, 0].max():.2f} m、奥行き {hip[:, 1].min():.2f}〜{hip[:, 1].max():.2f} m")
json.dump(dict(kinect_lag_ms=float(lag_ms), corr=float(c), pairs=[f"{a}-{b}" for a, b in top], rates=rates),
          open(os.path.join(ses, "sync.json"), "w"), indent=1, ensure_ascii=False)
print("→", os.path.join(ses, "sync.json"))
