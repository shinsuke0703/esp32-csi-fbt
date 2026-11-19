# -*- coding: utf-8 -*-
"""実験 E1 の判定 (Carbon の資料 3 本目の 6 章): 人がリンクを横切ったときに
  (i)  サブキャリアの位相のそろい具合 C_p(t) = |Σ_m Z[m]·exp(−j arg Z_bg[m])| / Σ_m |Z[m]| が下がるか (深いフェージングの検出)
       合格: 静止で 0.90 以上 → 横切る瞬間に 0.75 未満
  (ii) RSSI の往復の対数電力比 Λ_p(t) = (ln10/10)·[(RSSI_ij + RSSI_ji) − 背景] の変化が、静止の標準偏差の 6 倍以上出るか
使い方: python e1_analyze.py e1_cross.npz   (0〜20 秒 = 背景、20〜60 秒 = 横切る)
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]

d = dict(np.load(sys.argv[1])); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = [k for k in ping if k in ack]
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
t0 = d["t_host"].min()
ser = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]
    ser.setdefault((s, int(tgt[a])), []).append((d["t_host"][b] - t0, lltf(a)[valid] * lltf(b)[valid], int(d["rssi"][a]) + int(d["rssi"][b])))
rows = {}
for k, L in sorted(ser.items()):
    L.sort(key=lambda x: x[0]); t = np.array([x[0] for x in L]); Z = np.array([x[1] for x in L]); rs = np.array([x[2] for x in L], float)
    bg = (t >= 2) & (t < 18)
    Zbg = np.median(Z[bg].real, 0) + 1j * np.median(Z[bg].imag, 0)
    C = np.abs((Z * np.exp(-1j * np.angle(Zbg))).sum(1)) / np.abs(Z).sum(1)
    Lam = np.log(10) / 10 * (rs - np.median(rs[bg]))
    sm = lambda x: np.convolve(x, np.ones(9) / 9, "same")
    mv = (t >= 20)
    rows[k] = dict(t=t, C=C, Lam=Lam, Cbg=np.median(C[bg]), Cmin=np.percentile(sm(C)[mv], 1), Lstd=np.std(Lam[bg]),
                   Lpk=np.percentile(np.abs(sm(Lam))[mv], 99))
print("組 | C (静止の中央値 → 横切る区間の下位 1%) | RSSI 対数電力比: 静止の std → 横切る区間の上位 1% (比)")
for k, r in rows.items():
    ratio = r["Lpk"] / max(r["Lstd"], 1e-6)
    print(f"{k[0]}-{k[1]} | {r['Cbg']:.2f} → {r['Cmin']:.2f} | {r['Lstd']:.3f} → {r['Lpk']:.3f} ({ratio:4.1f} 倍)")
best = max(rows, key=lambda k: rows[k]["Lpk"] / max(rows[k]["Lstd"], 1e-6))
bc = min(rows, key=lambda k: rows[k]["Cmin"])
print(f"\n一番反応した組: RSSI で {best[0]}-{best[1]}、C で {bc[0]}-{bc[1]}")
r = rows[bc]; ok1 = r["Cbg"] >= 0.90 and r["Cmin"] < 0.75
r2 = rows[best]; ok2 = r2["Lpk"] / max(r2["Lstd"], 1e-6) >= 6.0
print(f"判定 (i) 深いフェージングの検出: {'成立' if ok1 else '不成立'} (C {r['Cbg']:.2f} → {r['Cmin']:.2f})")
print(f"判定 (ii) RSSI で振幅を復元: {'使える' if ok2 else '使えない (RSSI の重みを 0 に)'} ({r2['Lpk'] / max(r2['Lstd'], 1e-6):.1f} 倍)")
fig, ax = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
for k in {best, bc}:
    r = rows[k]; ax[0].plot(r["t"], r["C"], lw=0.6, label=f"{k[0]}-{k[1]}"); ax[1].plot(r["t"], r["Lam"], lw=0.6, label=f"{k[0]}-{k[1]}")
ax[0].set_ylabel("C_p (位相のそろい具合)"); ax[1].set_ylabel("RSSI の対数電力比"); ax[1].set_xlabel("時間 [秒]")
for a in ax: a.axvline(20, color="k", lw=0.6); a.legend(); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig("e1_cross.png", dpi=85); print("図: e1_cross.png")
