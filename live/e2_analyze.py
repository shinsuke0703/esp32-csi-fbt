# -*- coding: utf-8 -*-
"""実験 E2 の判定: 1 巡ごとにチャネル 1/6/11 を切り替えても、ACK 往復の積の位相がそろうか。
使い方: python e2_analyze.py e2_static_hop.npz
合格の基準 (Carbon の資料の 6 章): 全 15 組 × 3 チャネルで R ≥ 0.85、かつ巡回のレート ≥ 50 Hz。
チャネルは記録の cwb の欄 (受信したチャネル)。
"""
import sys
import numpy as np

d = dict(np.load(sys.argv[1])); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = [k for k in ping if k in ack]
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
T = d["t_host"].max() - d["t_host"].min()
res = {}; mism = 0
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]
    cp, ca = int(d["cwb"][a]), int(d["cwb"][b])
    if cp != ca: mism += 1; continue
    P = (lltf(a)[valid] * lltf(b)[valid]).sum()
    res.setdefault((s, int(tgt[a]), cp), []).append(np.angle(P))
R = lambda v: float(np.abs(np.mean(np.exp(1j * np.array(v)))))
chans = sorted({k[2] for k in res})
print(f"{T:.0f} 秒、そろった往復 {sum(len(v) for v in res.values())} 回 (PING と ACK のチャネルが違ったもの {mism} 回)、チャネル {chans}")
print("組 | " + " | ".join(f"ch{c}: 回数 Hz R" for c in chans))
allR, rates = [], []
for s in range(5):
    for t in range(s + 1, 6):
        cells = []
        for c in chans:
            v = res.get((s, t, c), []); r = R(v) if len(v) > 5 else float("nan"); hz = len(v) / T
            cells.append(f"{len(v):4d} {hz:4.1f} {r:4.2f}"); allR.append(r); rates.append(hz)
        print(f"{s}-{t} | " + " | ".join(cells))
allR = np.array(allR)
print(f"\nR: 中央値 {np.nanmedian(allR):.2f}、最小 {np.nanmin(allR):.2f}、R ≥ 0.85 は {(allR >= 0.85).sum()}/{len(allR)}")
print(f"1 組あたりのレート (3 チャネル合計): 中央値 {np.median(np.array(rates).reshape(-1, len(chans)).sum(1)):.1f} Hz")
print("判定: " + ("合格" if (np.nanmin(allR) >= 0.85) else "不合格 (R < 0.85 の組とチャネルがある)") + " (レートの基準 50 Hz は巡回のレートで別に判定)")
