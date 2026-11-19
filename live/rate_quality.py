# -*- coding: utf-8 -*-
"""収録ごとに、往復の位相の質を比べる: 外れ値 (中央値から 0.8 rad 超) の割合、外れ値を除いたばらつき (傾きを外した後)、
1 組あたりの有効レート (外れ値を除いた往復の数 / 秒)、情報量の目安 = 有効レート / ばらつき² (トークン方式に対する比)。
使い方: python rate_quality.py a.npz b.npz ..."""
import sys
import numpy as np
LOCAL = '--local' in sys.argv; sys.argv = [a for a in sys.argv if a != '--local']
def analyze(fn):
    d = dict(np.load(fn)); typ, rx, tgt = d["type"], d["rx"], d["target"]
    lltf = lambda i: (lambda r: r[1::2] + 1j * r[0::2])(d["csi"][i].astype(float))
    ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
    ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
    keys = [k for k in ping if k in ack]
    pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
    idx = np.arange(64); kidx = np.where(idx < 32, idx, idx - 64)[valid].astype(float)
    T = d["t_host"].max() - d["t_host"].min(); ser = {}
    for (s, q) in keys:
        a, b = ping[(s, q)], ack[(s, q)]; ser.setdefault((s, int(tgt[a])), []).append(lltf(a)[valid] * lltf(b)[valid])
    out = {}
    for k, L in sorted(ser.items()):
        P = np.array(L); ref = np.median(P.real, 0) + 1j * np.median(P.imag, 0); x = P * np.conj(ref)
        sl = np.angle(np.sum(x[:, 1:] * np.conj(x[:, :-1]) * (np.diff(kidx) == 1), 1))
        ph = np.angle((x * np.exp(-1j * sl[:, None] * (kidx - kidx.mean()))).sum(1))
        if LOCAL:   # 前後 10 パケットの円周中央値を基準にする (人の動きなどのゆっくりした変化を外れ値に数えない)
            n = len(ph); h = 10; pad = np.pad(ph, h, mode="edge"); win = np.stack([pad[i:i + n] for i in range(2 * h + 1)])
            med = np.angle(np.median(np.cos(win), 0) + 1j * np.median(np.sin(win), 0)); ph = np.angle(np.exp(1j * (ph - med)))
        ok = np.abs(ph) < 0.8
        out[k] = (len(P) / T, 1 - ok.mean(), ph[ok].std(), ok.sum() / T)
    return out
res = {fn: analyze(fn) for fn in sys.argv[1:]}
base = sys.argv[1]
print("組 | " + " | ".join(f"{fn}: Hz 外れ値% std 有効Hz 情報比" for fn in sys.argv[1:]))
ratios = {fn: [] for fn in sys.argv[1:]}
for k in sorted(res[base]):
    cells = []
    for fn in sys.argv[1:]:
        if k not in res[fn]: cells.append("-"); continue
        hz, o, sd, eff = res[fn][k]; b = res[base][k]; r = (eff / sd ** 2) / (b[3] / b[2] ** 2); ratios[fn].append(r)
        cells.append(f"{hz:5.1f} {o * 100:4.1f} {sd:.3f} {eff:5.1f} {r:4.2f}")
    print(f"{k[0]}-{k[1]} | " + " | ".join(cells))
for fn in sys.argv[1:]: print(f"{fn}: 情報量の比 (中央値) {np.median(ratios[fn]):.2f}")
