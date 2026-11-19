# -*- coding: utf-8 -*-
"""仮説 (a) の検証: 往復の位相のぶれ = 2π·Δf·δt (Δf: 2 台の周波数のずれ、δt: ACK を返すタイミングのぶれ)。
PING を 6 Mbps (1 シンボル = 3 バイト = 4 µs) で送り、長さを 3 バイトずつ変えると、PING から ACK までの間隔 T がちょうど 4 µs ずつ伸びる。
往復の位相の平均は −2π·Δf·T だけ回るので、長さに対する回り方から組ごとの Δf が分かる (静止が前提)。
そのうえで、組ごとのぶれ σφ と |Δf| が比例するか、δt = σφ / (2π|Δf|) が組によらずそろうかを見る。
使い方: python cfo_sweep.py [収録する=1]   (収録は usb_sched.py を PL を変えて呼ぶ。6 Mbps)
"""
import sys, subprocess, os
import numpy as np

PLS = [9, 12, 15, 21, 33, 57, 105]     # 4 µs ずつ: 0, 4, 8, 16, 32, 64, 128 µs
DT = {pl: (pl - 9) / 3 * 4e-6 for pl in PLS}
if len(sys.argv) < 2 or sys.argv[1] != "0":
    for pl in PLS:
        r = subprocess.run([sys.executable, "-P", "usb_sched.py", "12", f"cfo_pl{pl}.npz", "fwd", "1", f"PL={pl}", "PR=0x0B"],
                           capture_output=True, text=True, encoding="utf-8", env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        print(pl, [l for l in r.stdout.splitlines() if "秒" in l][-1:])

def pair_phases(fn):
    d = dict(np.load(fn)); typ, rx, tgt = d["type"], d["rx"], d["target"]
    lltf = lambda i: (lambda r: r[1::2] + 1j * r[0::2])(d["csi"][i].astype(float))
    ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
    ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
    out = {}
    for (s, q) in ping:
        if (s, q) in ack:
            a, b = ping[(s, q)], ack[(s, q)]; out.setdefault((s, int(tgt[a])), []).append(lltf(a) * lltf(b))
    return {k: np.array(v) for k, v in out.items()}

runs = {pl: pair_phases(f"cfo_pl{pl}.npz") for pl in PLS}
print("組 | Δf [kHz] (位相の回り方から) | 当てはまりの残差 [rad] | ぶれ σφ [rad] | δt = σφ/(2π|Δf|) [ns]")
rows = []
for k in sorted(runs[PLS[0]]):
    if not all(k in runs[pl] for pl in PLS): continue
    P0 = runs[PLS[0]][k]; ref = np.median(P0.real, 0) + 1j * np.median(P0.imag, 0); m = np.abs(ref) > 0.05 * np.abs(ref).max()
    mean_ph, sig = [], []
    for pl in PLS:
        P = runs[pl][k][:, m]; x = (P * np.conj(ref[m])).sum(1)
        ph = np.angle(x); c = np.angle(np.mean(np.exp(1j * ph))); dev = np.angle(np.exp(1j * (ph - c))); ok = np.abs(dev) < 0.8
        mean_ph.append(np.angle(np.mean(np.exp(1j * ph[ok])))); sig.append(dev[ok].std())
    # 順に折り返しを解きながら直線を当てはめる (T の小さい方から)
    T = np.array([DT[pl] for pl in PLS]); ph = np.array(mean_ph); u = [ph[0]]
    for i in range(1, len(PLS)):
        slope = (u[-1] - u[0]) / T[i - 1] if T[i - 1] > 0 else 0.0
        pred = u[0] + slope * T[i]; u.append(pred + np.angle(np.exp(1j * (ph[i] - pred))))
    u = np.array(u); A = np.c_[T, np.ones_like(T)]; coef, *_ = np.linalg.lstsq(A, u, rcond=None)
    df = -coef[0] / (2 * np.pi); res = np.std(u - A @ coef); s = float(np.median(sig))
    dt_ns = s / (2 * np.pi * abs(df)) * 1e9 if abs(df) > 1 else np.nan
    rows.append((k, df, s)); print(f"{k[0]}-{k[1]} | {df / 1e3:+7.2f} | {res:.3f} | {s:.3f} | {dt_ns:6.1f}")
r = np.array([[abs(x[1]), x[2]] for x in rows])
cc = np.corrcoef(r[:, 0], r[:, 1])[0, 1]
print(f"\n|Δf| と σφ の相関 {cc:+.2f}。比例するなら δt はそろう (上の表の右端)")
