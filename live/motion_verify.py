# -*- coding: utf-8 -*-
"""時間で区切った動きの収録を検証する: ACK 往復の位相 (経路長) は動きを追えるか。振幅だけの場合と比べる。
使い方: python motion_verify.py motion_protocol.npz
区間: 0-20 秒 静止 (部屋の外か端) / 20-50 歩く / 50-80 中央で腕の上げ下げ (約 4 秒周期) / 80-100 中央で静止
"""
import sys
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Yu Gothic", "Meiryo", "MS Gothic"]
exec(open(r"e:/CSI/live/burst_analyze.py", encoding="utf-8").read().split("st = load(")[0])   # load() を共用

D = load(sys.argv[1]); LAM = 299792458 / 2.437e9 * 100   # cm
SEG = [("静止 (外/端)", 0, 20), ("歩く", 20, 50), ("腕の上げ下げ", 50, 80), ("中央で静止", 80, 100)]
t0 = min(a[0, 0] for a in D.values())

def robust_unwrap(t, ph, w=9, tol=0.8):
    z = np.exp(1j * ph); n = len(ph); keep = np.ones(n, bool)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        keep[i] = abs(np.angle(np.exp(1j * (ph[i] - np.angle(np.mean(z[lo:hi])))))) < tol
    return t[keep], np.unwrap(ph[keep]), keep

def smooth(x, n):
    k = np.ones(n) / n; return np.convolve(x, k, mode="same")

rows = {}
for k, a in sorted(D.items()):
    t = a[:, 0] - t0; tt, u, keep = robust_unwrap(t, a[:, 1])
    dl = -LAM / (4 * np.pi) * u                                    # 経路長の変化 [cm]
    amp = np.log(a[:, 3] * a[:, 4])                                 # 往復の振幅の積 (対数): 位相を使わない場合の比較用
    rows[k] = dict(t=t, tt=tt, dl=dl, amp=amp, keep=keep)

print(f"組 {len(rows)}、1 組あたり {np.median([len(r['t']) / (r['t'][-1] - r['t'][0]) for r in rows.values()]):.1f} Hz")
print("\n区間ごとの値 (15 組の中央値):  [位相] 速い雑音 cm / 動きの大きさ cm (0.25 秒平均した経路長の標準偏差)   [振幅] 動きの大きさ (対数振幅の標準偏差)")
summ = {}
for name, a_, b_ in SEG:
    ph_noise, ph_mot, am_mot = [], [], []
    for k, r in rows.items():
        m = (r["tt"] >= a_) & (r["tt"] < b_); ma = (r["t"] >= a_) & (r["t"] < b_)
        if m.sum() < 30: continue
        x = r["dl"][m]; xs = smooth(x, 14)
        ph_noise.append(np.std(x - xs)); ph_mot.append(np.std(xs[7:-7])); am_mot.append(np.std(smooth(r["amp"][ma], 14)[7:-7]))
    summ[name] = (np.median(ph_noise), np.median(ph_mot), np.median(am_mot))
    print(f"  {name:10s}  位相: 雑音 {summ[name][0]:.2f} / 動き {summ[name][1]:5.2f} cm    振幅: 動き {summ[name][2]:.3f}")
base = summ["静止 (外/端)"]
print("\n動きの見えやすさ (静止 (外/端) に対する比): 位相 / 振幅")
for name in ("歩く", "腕の上げ下げ", "中央で静止"):
    print(f"  {name:10s}  位相 {summ[name][1] / base[1]:5.1f} 倍   振幅 {summ[name][2] / base[2]:5.1f} 倍")

# 腕の上げ下げ (約 4 秒周期) の区間で、0.1〜1 Hz に周期が出るか
print("\n腕の上げ下げの区間: 0.1〜1 Hz で一番強い周期 (組ごと、位相 / 振幅)")
pk = []
for k, r in rows.items():
    m = (r["tt"] >= 52) & (r["tt"] < 78)
    if m.sum() < 200: continue
    tg = np.arange(52, 78, 1 / 50); x = np.interp(tg, r["tt"][m], r["dl"][m]); x -= np.polyval(np.polyfit(tg, x, 1), tg)
    ma = (r["t"] >= 52) & (r["t"] < 78); y = np.interp(tg, r["t"][ma], r["amp"][ma]); y -= np.polyval(np.polyfit(tg, y, 1), tg)
    f = np.fft.rfftfreq(len(tg), 1 / 50); band = (f >= 0.1) & (f <= 1.0)
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2; Y = np.abs(np.fft.rfft(y * np.hanning(len(y)))) ** 2
    fx, fy = f[band][np.argmax(X[band])], f[band][np.argmax(Y[band])]
    sx = X[band].max() / np.median(X[band]); sy = Y[band].max() / np.median(Y[band])
    pk.append((k, fx, sx, fy, sy))
for k, fx, sx, fy, sy in pk:
    print(f"  {k[0]}→{k[1]}: 位相 {fx:.2f} Hz (周期 {1 / fx:4.1f} 秒、ピーク/中央値 {sx:5.0f})   振幅 {fy:.2f} Hz ({sy:5.0f})")

fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for k, r in rows.items():
    ax[0].plot(r["tt"], r["dl"] - np.median(r["dl"][r["tt"] < 20]), lw=0.7, label=f"{k[0]}→{k[1]}")
    ax[1].plot(r["t"], smooth(r["amp"], 14) - np.median(r["amp"][r["t"] < 20]), lw=0.7)
for a_ in ax:
    for name, a0, b0 in SEG: a_.axvspan(a0, b0, alpha=0.06 if SEG.index((name, a0, b0)) % 2 else 0.0, color="k"); a_.text(a0 + 0.5, a_.get_ylim()[1] * 0.9 if a_ is ax[1] else 0, name, fontsize=8)
    a_.grid(alpha=.3)
ax[0].set_ylabel("往復の位相から求めた経路長の変化 [cm]"); ax[1].set_ylabel("往復の振幅 (対数) の変化"); ax[1].set_xlabel("時間 [秒]")
ax[0].legend(ncol=8, fontsize=7, loc="upper left"); fig.tight_layout(); fig.savefig("motion_verify.png", dpi=85); print("\n図: motion_verify.png")
