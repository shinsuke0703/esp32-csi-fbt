# -*- coding: utf-8 -*-
"""静止の実機データから、ノード間の距離の手がかりを取り出す (校正の前段)。
使い方: python node_geom.py e2_static_hop.npz
  (1) RSSI (両方向の平均、多数のパケットで平均して 1 dB 刻みをならす) → 自由空間の減衰からの距離
  (2) チャネル 1/6/11 の ACK 往復の積 P の位相の差 → 距離 (25 MHz 差で 6 m、50 MHz 差で 3 m の折り返し)
      arg P(ch) = −4π f_ch d / c + 機器の定数。往復なので各ノードの局部発振器の位相は消える。
"""
import sys, json
import numpy as np

C = 299792458.0
FCH = {1: 2412e6, 6: 2437e6, 11: 2462e6}
d = dict(np.load(sys.argv[1])); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = [k for k in ping if k in ack]
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
acc = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]
    cp, ca = int(d["cwb"][a]), int(d["cwb"][b])
    if cp != ca: continue
    t = int(tgt[a]); P = lltf(a)[valid] * lltf(b)[valid]
    e = acc.setdefault((s, t, cp), dict(P=np.zeros(valid.sum(), complex), n=0, r_f=[], r_r=[]))
    e["P"] += P / max(np.abs(P).mean(), 1e-9); e["n"] += 1; e["r_f"].append(int(d["rssi"][a])); e["r_r"].append(int(d["rssi"][b]))
pairs = [(s, t) for s in range(5) for t in range(s + 1, 6)]
out = {}
print("組 | RSSI 平均 (行き/帰り, dBm) | 往復の位相 ch1 ch6 ch11 [rad] | 位相差からの距離 (1-6, 6-11, 1-11) [m, 折り返し前]")
for (s, t) in pairs:
    ph = {}; rs = []
    for c in (1, 6, 11):
        e = acc.get((s, t, c))
        if not e or e["n"] < 20: continue
        ph[c] = float(np.angle(e["P"].sum())); rs.append((np.mean(e["r_f"]), np.mean(e["r_r"])))
    rf = np.mean([x[0] for x in rs]); rr = np.mean([x[1] for x in rs])
    dist = {}
    for c1, c2 in ((1, 6), (6, 11), (1, 11)):
        if c1 in ph and c2 in ph:
            df = FCH[c2] - FCH[c1]; amb = C / (2 * df)
            dist[f"{c1}-{c2}"] = float((-(np.angle(np.exp(1j * (ph[c2] - ph[c1])))) * C / (4 * np.pi * df)) % amb)
    out[f"{s}-{t}"] = dict(rssi_f=rf, rssi_r=rr, phase=ph, dist=dist)
    print(f"{s}-{t} | {rf:6.1f} / {rr:6.1f} | " + " ".join(f"{ph.get(c, float('nan')):+.2f}" for c in (1, 6, 11)) + " | " +
          " ".join(f"{v:4.2f}" for v in dist.values()))
json.dump(out, open("node_geom_obs.json", "w"), indent=1)
