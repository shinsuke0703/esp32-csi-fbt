# -*- coding: utf-8 -*-
"""CGSC の主張のうち、録画済みの実機データで確かめられるもの。
 (1) 定理 2: 単純な平方根 √P / √P_bg と、先に割ってから平方根 √(P/P_bg) で符号が食い違うフレームの割合 (論文: 単純な方 6.13 %)
 (2) 外れ値: 静止データで、往復の位相が直近 9 パケットの円周中央値から 0.8 rad 以上外れる割合 (論文: 1〜14 %)、
     そのゲートで外した後の位相のばらつき
使い方: python cgsc_real_checks.py e1_cross.npz token_60hz_b.npz"""
import sys
import numpy as np

def load(fn):
    d = dict(np.load(fn)); typ, rx, tgt = d["type"], d["rx"], d["target"]
    lltf = lambda i: (lambda r: r[1::2] + 1j * r[0::2])(d["csi"][i].astype(float))
    ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
    ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
    keys = [k for k in ping if k in ack]
    pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
    t0 = d["t_host"].min(); ser = {}
    for (s, q) in keys:
        a, b = ping[(s, q)], ack[(s, q)]
        ser.setdefault((s, int(tgt[a])), []).append((d["t_host"][b] - t0, (lltf(a)[valid] * lltf(b)[valid]).sum()))
    out = {}
    for k, L in ser.items():
        L.sort(key=lambda x: x[0]); out[k] = (np.array([x[0] for x in L]), np.array([x[1] for x in L]))
    return out

print("== (1) 平方根の符号の食い違い (e1_cross、横切る区間 20〜60 秒) ==")
ser = load(sys.argv[1]); flips = []
for k, (t, P) in sorted(ser.items()):
    bg = (t >= 2) & (t < 18); mv = t >= 20
    Pbg = np.median(P[bg].real) + 1j * np.median(P[bg].imag)
    naive = np.sqrt(P[mv]) / np.sqrt(Pbg); cay = np.sqrt(P[mv] / Pbg)
    fr = np.mean(np.real(naive * np.conj(cay)) < 0); flips.append(fr)
    print(f"{k[0]}-{k[1]}: 単純な平方根の符号反転 {fr * 100:5.1f} %  (背景の偏角 {np.angle(Pbg):+.2f} rad)")
print(f"全組の平均 {np.mean(flips) * 100:.1f} %  (割ってから平方根を取れば、定義上 0 %)")

print("\n== (2) 外れ値 (静止データ) ==")
ser = load(sys.argv[2]); rows = []
for k, (t, P) in sorted(ser.items()):
    ph = np.angle(P); n = len(ph); W = 9
    med = np.array([np.angle(np.median(np.cos(ph[max(0, i - W // 2):i + W // 2 + 1])) + 1j * np.median(np.sin(ph[max(0, i - W // 2):i + W // 2 + 1]))) for i in range(n)])
    dev = np.angle(np.exp(1j * (ph - med))); out = np.abs(dev) > 0.8
    R_all = np.abs(np.mean(np.exp(1j * (ph - np.angle(np.mean(np.exp(1j * ph))))))); R_in = np.abs(np.mean(np.exp(1j * ph[~out])))
    rows.append((k, out.mean(), np.std(dev[~out]), R_all, R_in))
    print(f"{k[0]}-{k[1]}: 外れ値 {out.mean() * 100:5.1f} %、ゲート後の位相のばらつき {np.std(dev[~out]):.3f} rad、R {R_all:.3f} → {R_in:.3f}")
o = np.array([r[1] for r in rows]); s = np.array([r[2] for r in rows])
print(f"外れ値: 中央値 {np.median(o) * 100:.1f} %、範囲 {o.min() * 100:.1f}〜{o.max() * 100:.1f} %。ゲート後のばらつき 中央値 {np.median(s):.3f} rad (= 経路 {np.median(s) * 12.36 / (4 * np.pi):.2f} cm)")
