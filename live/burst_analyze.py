# -*- coding: utf-8 -*-
"""ACK 往復 (バースト) の解析: 往復の積の位相が、静止では一定、動きではなめらかに変わる観測量になっているか。
使い方: python burst_analyze.py 静止.npz [動き.npz]   → 表と図 (burst_phase.png)

往復の積 P_ij(t) = Σ_k Y_ij[k]·Y_ji[k] (Y_ij: s→t の PING を t が受けた CSI、Y_ji: t の ACK を s が受けた CSI、LLTF)。
  共通位相が打ち消せていれば arg P = 2·arg H (+ 一定) で、H は電波の経路だけで決まる。
  なめらかさ: 隣り合うサンプルの位相差 (折り返し後) の大きさ。打ち消せていない片道の位相は、一様にばらつく (平均 |Δ| ≈ π/2)。
"""
import sys
import numpy as np

def load(path):
    d = dict(np.load(path)); typ, rx, tgt = d["type"], d["rx"], d["target"]
    def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
    ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
    ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
    keys = [k for k in ping if k in ack]
    pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); pa = np.mean([np.abs(lltf(ack[k])) ** 2 for k in keys[:3000]], 0)
    valid = (pw > 0.02 * pw.max()) & (pa > 0.02 * pa.max())
    out = {}
    for (s, q) in keys:
        a, b = ping[(s, q)], ack[(s, q)]; t = int(d["target"][a])
        Yij, Yji = lltf(a)[valid], lltf(b)[valid]
        out.setdefault((s, t), []).append((d["t_host"][b], np.angle((Yij * Yji).sum()), np.angle(Yij.sum()), np.abs(Yij).mean(), np.abs(Yji).mean()))
    return {k: np.array(sorted(v)) for k, v in out.items()}

def R(v): return np.abs(np.mean(np.exp(1j * v)))
def step(v): return np.abs(np.angle(np.exp(1j * np.diff(v))))   # 隣のサンプルとの位相差 (折り返し後)

st = load(sys.argv[1]); mo = load(sys.argv[2]) if len(sys.argv) > 2 else None
print("組 s→t | 静止: 回数 Hz | 往復 R | 片道 R | 往復の位相の隣り差 (中央値 rad)" + (" | 動き: 回数 | 往復の隣り差 | 片道の隣り差 | 往復の位相の全変化幅 rad" if mo else ""))
good = []
for k in sorted(st):
    a = st[k]; hz = len(a) / (a[-1, 0] - a[0, 0] + 1e-9)
    line = f"{k[0]}→{k[1]} | {len(a):4d} {hz:4.1f} | {R(a[:, 1]):5.2f} | {R(a[:, 2]):5.2f} | {np.median(step(a[:, 1])):5.2f}"
    if mo and k in mo:
        m = mo[k]; un = np.unwrap(m[:, 1])
        line += f" | {len(m):4d} | {np.median(step(m[:, 1])):5.2f} | {np.median(step(m[:, 2])):5.2f} | {np.percentile(un, 95) - np.percentile(un, 5):6.1f}"
    print(line)
    if R(a[:, 1]) > 0.8: good.append(k)
print(f"\n静止で往復の位相がそろう組 (R > 0.8): {good}")
print("隣り差の目安: 一様にばらつくなら中央値 ≈ 1.57 rad、なめらかなら小さい")

try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    show = good[:6] if good else sorted(st)[:6]
    fig, ax = plt.subplots(len(show), 2 if mo else 1, figsize=(12, 2.2 * len(show)), squeeze=False)
    for r, k in enumerate(show):
        for c, (D, name) in enumerate([(st, "静止"), (mo, "動き")][: 2 if mo else 1]):
            if D is None or k not in D: continue
            a = D[k]; t = a[:, 0] - a[0, 0]
            ax[r, c].plot(t, a[:, 2], ".", ms=2, color="0.7", label="片道 (PING) の位相")
            ax[r, c].plot(t, a[:, 1], ".", ms=2, color="C0", label="往復の積の位相")
            ax[r, c].set_ylim(-3.3, 3.3); ax[r, c].set_title(f"{k[0]}→{k[1]} {name}", fontsize=9)
    ax[0, 0].legend(fontsize=7, loc="upper right")
    plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic", "sans-serif"]
    for a_ in ax.ravel(): a_.title.set_fontfamily(["Yu Gothic", "Meiryo", "sans-serif"])
    fig.tight_layout(); fig.savefig("burst_phase.png", dpi=90); print("図: burst_phase.png")
except Exception as e:
    print("図は作れなかった:", e)
