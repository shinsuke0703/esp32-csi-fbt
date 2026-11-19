# -*- coding: utf-8 -*-
"""同じ手番 (数 ms 以内) に同じ送信役が続けて送った往復どうしで、共通位相のずれが相関するかを見る (静止データ、usb_sched の fwd)。
相関が高ければ「送信役の位相がゆっくりずれる」成分 → 同じ手番の組から共通の未知数として推定して消せる。
使い方: python jitter_corr.py usb_fwd_static.npz"""
import sys
import numpy as np

d = dict(np.load(sys.argv[1])); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = sorted(k for k in ping if k in ack)
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
idx = np.arange(64); kidx = np.where(idx < 32, idx, idx - 64)[valid].astype(float)
# 組ごとに共通位相のずれ φ を求める (jitter_anatomy と同じ: 背景の形で正規化、傾きを外し、背景の強さで重み付け)
by_pair = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]; by_pair.setdefault((s, int(tgt[a])), []).append((q, lltf(a)[valid] * lltf(b)[valid]))
phi = {}   # (送信役, seq) → (相手, φ)
for (s, t), L in by_pair.items():
    q = np.array([x[0] for x in L]); P = np.array([x[1] for x in L])
    ref = np.median(P.real, 0) + 1j * np.median(P.imag, 0); m = np.abs(ref) > 0.05 * np.abs(ref).max()
    P, ref, kk = P[:, m], ref[m], kidx[m]; kc = kk - kk.mean(); adj = np.diff(kk) == 1
    x = P * np.conj(ref) / np.abs(ref) ** 2
    sl = np.angle(np.sum(x[:, 1:] * np.conj(x[:, :-1]) * adj, 1)); x = x * np.exp(-1j * sl[:, None] * kc)
    w = np.abs(ref) ** 2; ph = np.angle((x * w).sum(1) / w.sum())
    ph = np.angle(np.exp(1j * (ph - np.angle(np.mean(np.exp(1j * ph))))))
    for qq, pp in zip(q, ph): phi[(s, int(qq))] = (t, pp)
for s in range(5):
    targets = list(range(s + 1, 6)); nt = len(targets)
    if nt < 2: continue
    seqs = sorted(q for (ss, q) in phi if ss == s)
    if not seqs: continue
    # 1 手番 = 連続した seq の nt 個 (fwd の予定では相手の番号順)。先頭の相手 (= s+1) の seq から手番を区切る
    rows = []
    for q in seqs:
        if phi[(s, q)][0] != targets[0]: continue
        row = [phi.get((s, q + i), (None, np.nan)) for i in range(nt)]
        if all(r[0] == targets[i] for i, r in enumerate(row)): rows.append([r[1] for r in row])
    A = np.array(rows); ok = np.all(np.abs(A) < 0.8, 1); A = A[ok]
    C = np.corrcoef(A.T); off = C[np.triu_indices(nt, 1)]
    # 共通成分を除いた後のばらつき: 各手番で、組の平均を引く (送信役の共通のずれを 1 つの未知数として推定するのと同じ)
    resid = A - A.mean(1, keepdims=True)
    print(f"送信役 {s} (相手 {targets}): 手番 {len(A)} 回、組どうしの相関 中央値 {np.median(off):+.2f} (範囲 {off.min():+.2f}〜{off.max():+.2f})、"
          f"位相の std {np.round(A.std(0), 3).tolist()} → 共通成分を除くと {np.round(resid.std(0) * np.sqrt(nt / (nt - 1)), 3).tolist()}")
