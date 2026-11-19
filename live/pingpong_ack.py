# -*- coding: utf-8 -*-
"""ACK を使った往復の解析。i→j のユニキャスト PING を j が受けた CSI (Y_ij) と、j のハードウェアが SIFS 後に返した ACK を i が受けた CSI (Y_ji)。
ACK はレガシー OFDM なので、両方とも LLTF (先頭 128 バイト、FFT 順) で比べる。
使い方: python pingpong_ack.py pp_ack_static.npz
"""
import sys
import numpy as np

d = dict(np.load(sys.argv[1]))
typ, tx, rx, seq, tgt = d["type"], d["tx"], d["rx"], d["seq"], d["target"]
def lltf(i):
    r = d["csi"][i].astype(float)[:128]; return r[1::2] + 1j * r[0::2]

ping = {(int(tx[i]), int(seq[i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}   # (送信 i, 番号) → j が受けた記録
ack = {(int(rx[i]), int(seq[i])): i for i in np.where(typ == 5)[0]}                    # (PING を送った i, 番号) → i が受けた ACK

# 有効なサブキャリア: PING と ACK の両方で平均電力が十分あるもの
pw_p = np.mean([np.abs(lltf(i)) ** 2 for i in list(ping.values())[:2000]], 0)
pw_a = np.mean([np.abs(lltf(i)) ** 2 for i in list(ack.values())[:2000]], 0)
valid = (pw_p > 0.02 * pw_p.max()) & (pw_a > 0.02 * pw_a.max())
print(f"PING {len(ping)} 件、ACK {len(ack)} 件。ACK の CSI の長さ {sorted(set(d['len'][typ == 5].tolist()))}、sig_mode {sorted(set(d['sig'][typ == 5].tolist()))}")
print(f"共通に使える LLTF のサブキャリア: {valid.sum()} 本 (番号 {np.where(valid)[0].tolist()})")
kidx = np.where(valid)[0]; ktone = np.where(kidx < 32, kidx, kidx - 64)   # FFT 番号 → サブキャリア番号

def R(v): return np.abs(np.mean(np.exp(1j * np.asarray(v)))) if len(v) else np.nan
def slope(Y):
    o = np.argsort(ktone); y = Y[o]; t = ktone[o]; adj = np.diff(t) == 1
    return np.angle((np.conj(y[:-1]) * y[1:])[adj].sum())

res = {}
for (i, s), a in ping.items():
    b = ack.get((i, s))
    if b is None: continue
    j = int(tgt[a]); Yij = lltf(a)[valid]; Yji = lltf(b)[valid]
    res.setdefault((i, j), []).append((np.angle(Yij.sum()), np.angle(Yji.sum()), np.angle((Yij * Yji).sum()), slope(Yij) + slope(Yji),
                                        np.corrcoef(np.abs(Yij), np.abs(Yji))[0, 1], d["t_aux"][b]))
print(f"そろった往復 (PING と ACK): {sum(len(v) for v in res.values())} 回\n")
print("向き i→j | 回数 | 片道 PING R | 片道 ACK R | 往復の積 R | 傾きの和の std | 振幅の形の相関 | 送ってから ACK まで µs (中央値)")
allp = []
for (i, j), L in sorted(res.items()):
    a = np.array(L)
    allp.append(R(a[:, 2]))
    print(f"{i}→{j} | {len(L):4d} | {R(a[:, 0]):5.2f} | {R(a[:, 1]):5.2f} | {R(a[:, 2]):5.2f} | {np.std(a[:, 3]):6.3f} | {np.nanmean(a[:, 4]):5.2f} | {np.median(a[:, 5]):6.0f}")
allp = np.array(allp)
print(f"\n往復の積の R: 中央値 {np.median(allp):.2f}、最小 {allp.min():.2f}、最大 {allp.max():.2f}  ({(allp > 0.8).sum()}/{len(allp)} 組が 0.8 以上)")
print("R: 1 に近いほどそろう。一様なら 1/√回数 ≈ 0.05")

# 傾き (パケット検出のずれ) を各パケットで取り除いてから往復の積をとる
def deslope(Y):
    t = ktone.astype(float); return Y * np.exp(-1j * slope(Y) * t)
print("\n各パケットの傾きを取り除いた場合:")
print("向き i→j | 往復の積 R (傾きそのまま → 取り除く)")
allq = []
for (i, s), a in ping.items():
    pass
res2 = {}
for (i, s), a in ping.items():
    b = ack.get((i, s))
    if b is None: continue
    j = int(tgt[a]); Yij = deslope(lltf(a)[valid]); Yji = deslope(lltf(b)[valid])
    res2.setdefault((i, j), []).append(np.angle((Yij * Yji).sum()))
for (i, j), L in sorted(res2.items()):
    r0 = R(np.array(res[(i, j)])[:, 2]); r1 = R(L); allq.append(r1)
    print(f"{i}→{j} | {r0:5.2f} → {r1:5.2f}")
allq = np.array(allq)
print(f"\n傾きを取り除いた往復の積の R: 中央値 {np.median(allq):.2f}、最小 {allq.min():.2f}、最大 {allq.max():.2f}  ({(allq > 0.8).sum()}/{len(allq)} 組が 0.8 以上)")
