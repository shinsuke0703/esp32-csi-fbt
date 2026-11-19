# -*- coding: utf-8 -*-
"""共通位相のずれが、受信の増幅率 (AGC) の段で説明できるか (静止データ)。
増幅率の目安 g = RSSI − 20·log10(CSI の平均振幅) [dB] (CSI は AGC の後の値なので、RSSI との差が増幅率に対応する)。
各組で、行き (相手が PING を受けた) と帰り (送信役が ACK を受けた) の g を 1 dB 単位で丸めて組み合わせごとにまとめ、
φ の分散のうち「段ごとの平均」で説明できる割合 (決定係数) を求める。高ければ、段ごとの位相を較正すれば消せる。
使い方: python jitter_agc.py usb_fwd_static.npz"""
import sys
import numpy as np

d = dict(np.load(sys.argv[1])); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = sorted(k for k in ping if k in ack)
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
idx = np.arange(64); kidx = np.where(idx < 32, idx, idx - 64)[valid].astype(float)
by_pair = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]; by_pair.setdefault((s, int(tgt[a])), []).append((a, b))
print("組 | φ の std | 増幅率の段の数 (行き, 帰り) | 段で説明できる割合: 行き / 帰り / 両方 | 段の平均の幅 [rad] | 共通成分を除いたあとの std")
for (s, t), L in sorted(by_pair.items()):
    A = np.array([x[0] for x in L]); B = np.array([x[1] for x in L])
    Yf = np.array([lltf(i)[valid] for i in A]); Yb = np.array([lltf(i)[valid] for i in B]); P = Yf * Yb
    ref = np.median(P.real, 0) + 1j * np.median(P.imag, 0); m = np.abs(ref) > 0.05 * np.abs(ref).max()
    P, ref, kk = P[:, m], ref[m], kidx[m]; kc = kk - kk.mean(); adj = np.diff(kk) == 1
    x = P * np.conj(ref) / np.abs(ref) ** 2
    sl = np.angle(np.sum(x[:, 1:] * np.conj(x[:, :-1]) * adj, 1)); x = x * np.exp(-1j * sl[:, None] * kc)
    w = np.abs(ref) ** 2; ph = np.angle((x * w).sum(1) / w.sum()); ph = np.angle(np.exp(1j * (ph - np.angle(np.mean(np.exp(1j * ph))))))
    gf = np.round(d["rssi"][A] - 20 * np.log10(np.abs(Yf[:, m]).mean(1) + 1e-9)).astype(int)
    gb = np.round(d["rssi"][B] - 20 * np.log10(np.abs(Yb[:, m]).mean(1) + 1e-9)).astype(int)
    ok = np.abs(ph) < 0.8; ph, gf, gb = ph[ok], gf[ok], gb[ok]
    def r2(lab):
        tot = ph.var(); u, inv = np.unique(lab, return_inverse=True); means = np.array([ph[inv == i].mean() for i in range(len(u))])
        cnt = np.bincount(inv); keep = cnt[inv] >= 20   # 20 回以上ある段だけ
        fit = means[inv]; return 1 - np.var(ph[keep] - fit[keep]) / np.var(ph[keep]), np.ptp(means[cnt >= 20]) if (cnt >= 20).any() else 0, ph[keep] - fit[keep]
    rf, _, _ = r2(gf); rb, _, _ = r2(gb); rboth, span, res = r2(gf * 1000 + gb)
    print(f"{s}-{t} | {ph.std():.3f} | {len(np.unique(gf))}, {len(np.unique(gb))} | {rf * 100:4.0f} % / {rb * 100:4.0f} % / {rboth * 100:4.0f} % | {span:.2f} | {res.std():.3f}")
