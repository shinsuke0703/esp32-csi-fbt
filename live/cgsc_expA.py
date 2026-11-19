# -*- coding: utf-8 -*-
"""CGSC の実験 A を録画済みデータで: 人がリンクを横切るとき、Q = P/P_bg の偏角が分岐切断 (±π) に近づくか。
主値の平方根が使える条件 |arg Q| < π (余裕を見て 2.5 rad 未満) を何 % のフレームで満たすか。0〜20 秒 = 無人、20〜60 秒 = 横切る。"""
import sys
import numpy as np
d = dict(np.load(sys.argv[1])); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = [k for k in ping if k in ack]
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
t0 = d["t_host"].min(); ser = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]
    ser.setdefault((s, int(tgt[a])), []).append((d["t_host"][b] - t0, (lltf(a)[valid] * lltf(b)[valid]).sum()))
print("組 | 横切る区間の |arg Q| の 99% 点 [rad] | 2.5 rad 超 (切断の近く) の割合 | |√Q − 1| の 99% 点 (η の大きさ)")
allfr = []
for k, L in sorted(ser.items()):
    L.sort(key=lambda x: x[0]); t = np.array([x[0] for x in L]); P = np.array([x[1] for x in L])
    bg = (t >= 2) & (t < 18); mv = t >= 20
    Pbg = np.median(P[bg].real) + 1j * np.median(P[bg].imag)
    Q = P / Pbg; Q = Q / np.median(np.abs(Q[bg]))
    aq = np.abs(np.angle(Q[mv])); eta = np.abs(np.sqrt(Q[mv]) - 1)
    fr = np.mean(aq > 2.5); allfr.append(fr)
    print(f"{k[0]}-{k[1]} | {np.percentile(aq, 99):.2f} | {fr * 100:5.2f} % | {np.percentile(eta, 99):.2f}")
print(f"\n全組: 切断の近く (|arg Q| > 2.5 rad) のフレーム {np.mean(allfr) * 100:.2f} %、最悪の組 {np.max(allfr) * 100:.2f} %")
