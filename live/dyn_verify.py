# -*- coding: utf-8 -*-
"""ACK 往復から H を (符号を除いて) 復元し、背景を引いた「体による成分 H_dyn」で動きを見る。
使い方: python dyn_verify.py motion_protocol.npz
  P_k(t) = Y_ij[k]·Y_ji[k] ≈ c·H_k(t)²  (c は時間によらない定数) → H_k(t) = ±sqrt(P_k(t))。符号は前のサンプルとそろう方を選ぶ。
  背景 H_bg = 最初の静止区間 (0〜18 秒) の平均。H_dyn = H − H_bg。
  - 体による成分の強さ: |H_dyn|² / |H_bg|² (dB)
  - 体の反射経路の長さの変化: H_dyn の位相 (トーン平均) をつないだもの × λ/2π
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
LAM = 299792458 / 2.437e9 * 100
SEG = [("静止 (外/端)", 0, 20), ("歩く", 20, 50), ("腕の上げ下げ", 50, 80), ("中央で静止", 80, 100)]
t0 = d["t_host"].min()

pairs = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]; t = int(tgt[a])
    P = lltf(a)[valid] * lltf(b)[valid]
    pairs.setdefault((s, t), []).append((d["t_host"][b] - t0, P))
res = {}
for k, L in sorted(pairs.items()):
    L.sort(key=lambda x: x[0]); tt = np.array([x[0] for x in L]); P = np.array([x[1] for x in L])
    P = P / np.sqrt(np.abs(P).mean(1, keepdims=True))   # AGC による振幅の変化は外す (形と位相だけ残す)
    # 平方根を使わない: P = H² なので P − P_bg ≈ 2·H_bg·H_dyn (体による成分が背景より小さいとき) → H_dyn ≈ (P − P_bg) / (2·H_bg)
    # サンプルごとの符号のあいまいさがない。H_bg = sqrt(P_bg) の符号は全サンプル共通なので、どちらでも結果の位相の変化は同じ
    bg = (tt >= 1) & (tt < 18)
    Pbg = np.median(P[bg].real, 0) + 1j * np.median(P[bg].imag, 0)   # 外れ値に強い背景
    Hbg = np.sqrt(Pbg)
    kidx = np.where(valid)[0]; ktone = np.where(kidx < 32, kidx, kidx - 64); o = np.argsort(ktone)
    for j in range(1, len(o)):   # 背景の符号を周波数方向にそろえる (1 回だけ)
        if np.real(Hbg[o[j]] * np.conj(Hbg[o[j - 1]])) < 0: Hbg[o[j]] *= -1
    Hd = (P - Pbg) / (2 * Hbg)
    pwr = 10 * np.log10((np.abs(Hd) ** 2).mean(1) / (np.abs(Hbg) ** 2).mean())
    ph = np.unwrap(np.angle((Hd * np.conj(Hbg)).sum(1)))   # 背景に対する体の成分の位相
    res[k] = dict(t=tt, pwr=pwr, path=ph * LAM / (2 * np.pi))

print("区間ごとの体による成分の強さ |H_dyn|²/|H_bg|² [dB] (15 組の中央値 / 最大)")
for name, a_, b_ in SEG:
    v = [np.nanmedian(r["pwr"][(r["t"] >= a_ + 1) & (r["t"] < b_ - 1)]) for r in res.values()]
    print(f"  {name:10s} 中央値 {np.nanmedian(v):6.1f} dB   最小の組 {np.nanmin(v):6.1f} dB   最大の組 {np.nanmax(v):6.1f} dB   (NaN の組 {int(np.isnan(v).sum())})")

fig, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
for k, r in res.items():
    ax[0].plot(r["t"], r["pwr"], lw=0.5, label=f"{k[0]}→{k[1]}")
for k in sorted(res, key=lambda k: -np.median(res[k]["pwr"][res[k]["t"] > 20]))[:4]:
    r = res[k]; ax[1].plot(r["t"], r["path"] - r["path"][0], lw=0.8, label=f"{k[0]}→{k[1]}")
for a_ in ax:
    for name, a0, b0 in SEG: a_.axvline(a0, color="k", lw=0.5); a_.text(a0 + 0.5, 0.95, name, transform=a_.get_xaxis_transform(), fontsize=8)
    a_.grid(alpha=.3)
ax[0].set_ylabel("体による成分の強さ [dB]\n(背景に対する比)"); ax[1].set_ylabel("体の成分の位相から求めた\n経路長の変化 [cm] (強い 4 組)")
ax[0].legend(ncol=8, fontsize=6); ax[1].legend(fontsize=7); ax[1].set_xlabel("時間 [秒]")
fig.tight_layout(); fig.savefig("dyn_verify.png", dpi=85); print("図: dyn_verify.png")
