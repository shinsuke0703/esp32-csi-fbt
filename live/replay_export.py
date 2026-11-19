# -*- coding: utf-8 -*-
"""実機の ACK 往復の収録を、Unity で再生するためのファイル (JSON) に書き出す。
使い方: python replay_export.py motion_protocol.npz 出力.json [layout.json]
  各リンクの「体の成分の強さ」|H_dyn|²/|H_bg|² [dB] (背景 = 最初の 1〜18 秒) を 30 Hz で書き出す。
  ノードの位置は layout.json があればそれを使い、なければ仮の配置 (半径 2.5 m の円、高さは交互に 0.5/2.0 m)。
  layout.json の形: {"nodes": [[x, y, z], ... 6 個]}  (単位 m、y が上)
"""
import sys, json, math
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from carbon_model import hdyn_from_P

src, out = sys.argv[1], sys.argv[2]
d = dict(np.load(src)); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = [k for k in ping if k in ack]
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
t0 = d["t_host"].min()
series = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]
    series.setdefault((s, int(tgt[a])), []).append((d["t_host"][b] - t0, lltf(a)[valid] * lltf(b)[valid]))
FPS = 30; T = max(v[-1][0] for v in series.values()); tk = np.arange(0, T, 1 / FPS)
links = sorted(series); vals = []
for k in links:
    L = sorted(series[k], key=lambda x: x[0]); t = np.array([x[0] for x in L]); P = np.array([x[1] for x in L])
    Hd, Hbg = hdyn_from_P(P, (t >= 1) & (t < 18))
    p = 10 * np.log10((np.abs(Hd) ** 2).mean(1) / (np.abs(Hbg) ** 2).mean() + 1e-12)
    p = np.where(np.isfinite(p), p, np.nan)
    # 外れ値に強くするため 0.2 秒の移動中央値をとってから 30 Hz に
    w = 11; pm = np.array([np.nanmedian(p[max(0, i - w // 2): i + w // 2 + 1]) for i in range(len(p))])
    ok = np.isfinite(pm); vals.append(np.interp(tk, t[ok], pm[ok]) if ok.sum() > 10 else np.full(len(tk), -30.0))
vals = np.array(vals)   # [リンク, フレーム]

if len(sys.argv) > 3:
    nodes = json.load(open(sys.argv[3], encoding="utf-8"))["nodes"]; layout = "layout.json"
else:
    nodes = [[round(2.5 * math.sin(2 * math.pi * i / 6), 3), 0.5 if i % 2 == 0 else 2.0, round(2.5 * math.cos(2 * math.pi * i / 6), 3)] for i in range(6)]; layout = "仮の配置"
doc = {
    "source": src, "layout": layout, "fps": FPS, "nodes": [{"x": n[0], "y": n[1], "z": n[2]} for n in nodes],
    "links": [{"a": a, "b": b} for a, b in links],
    "segments": [{"name": "静止 (人は外か端)", "t0": 0, "t1": 20}, {"name": "歩く", "t0": 20, "t1": 50},
                 {"name": "腕の上げ下げ (6〜8 秒周期)", "t0": 50, "t1": 80}, {"name": "中央で静止", "t0": 80, "t1": 100}],
    "frames": [{"t": round(float(t), 3), "v": [round(float(x), 2) for x in vals[:, i]]} for i, t in enumerate(tk)],
}
json.dump(doc, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"{len(tk)} フレーム ({FPS} Hz、{T:.0f} 秒)、リンク {len(links)}、ノードの位置: {layout} → {out}")
print("リンクごとの体の成分の強さ (区間の中央値 dB): 静止 / 歩く / 腕 / 中央で静止")
for i, k in enumerate(links):
    seg = [np.nanmedian(vals[i, (tk >= a) & (tk < b)]) for a, b in ((2, 18), (22, 48), (52, 78), (82, 98))]
    print(f"  {k[0]}-{k[1]}: " + " / ".join(f"{x:6.1f}" for x in seg))
