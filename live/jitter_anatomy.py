# -*- coding: utf-8 -*-
"""往復の位相のぶれを成分に分ける (静止データ)。
使い方: python jitter_anatomy.py token_60hz_b.npz
各パケットの往復の積 P[k] = Y_ij[k]·Y_ji[k] を、背景 (中央値) の形 ref[k] と比べ、
  x[k] = P[k]·ref[k]* = A·exp(j(φ + s·k)) + トーンごとの雑音
  φ: 共通位相 (ぶれの本体)、s: 傾き (検出の遅れ)、トーンごとの雑音: 共通位相と傾きを外した残り
を求める。トーンごとの雑音から「熱雑音・量子化だけなら共通位相はこれだけぶれるはず」(予測値 = σ_tone / √(有効トーン数)) を出し、
実際の共通位相のぶれと比べる。実際の方が大きければ、トーン全体に共通にかかる何か (発振器・タイミング) が原因。
あわせて、片方向ずつの CSI の振幅 (int8 の大きさ、量子化の粗さ)、RSSI、雑音の床、傾きと共通位相の相関を見る。
"""
import sys
import numpy as np

d = dict(np.load(sys.argv[1])); typ, rx, tgt = d["type"], d["rx"], d["target"]
def lltf(i): r = d["csi"][i].astype(float); return r[1::2] + 1j * r[0::2]
ping = {(int(d["tx"][i]), int(d["seq"][i])): i for i in np.where((typ == 4) & (rx == tgt))[0]}
ack = {(int(rx[i]), int(d["seq"][i])): i for i in np.where(typ == 5)[0]}
keys = [k for k in ping if k in ack]
pw = np.mean([np.abs(lltf(ping[k])) ** 2 for k in keys[:3000]], 0); valid = pw > 0.02 * pw.max()
idx = np.arange(64); kidx = np.where(idx < 32, idx, idx - 64)[valid].astype(float); kc = kidx - kidx.mean()
adj = (np.diff(kidx) == 1)
ser = {}
for (s, q) in keys:
    a, b = ping[(s, q)], ack[(s, q)]
    ser.setdefault((s, int(tgt[a])), []).append((lltf(a)[valid], lltf(b)[valid], d["rssi"][a], d["rssi"][b], d["noise"][a], d["noise"][b], d["t_host"][b]))
print("組 | 片方向の |Y| (行き/帰り) | RSSI (行き/帰り) | 雑音の床 | 共通位相の std 実測 | 熱雑音・量子化からの予測 | 比 | 共通位相と傾きの相関 | 片方向の振幅と共通位相の相関")
rows = []
for k, L in sorted(ser.items()):
    Yf = np.array([x[0] for x in L]); Yb = np.array([x[1] for x in L]); P = Yf * Yb
    ref = np.median(P.real, 0) + 1j * np.median(P.imag, 0)
    m = np.abs(ref) > 0.05 * np.abs(ref).max()                    # この組で使えるトーン (C3 は先頭のトーンが無効なことがある)
    P, ref, kk_ = P[:, m], ref[m], kidx[m]; kc_ = kk_ - kk_.mean(); adj_ = (np.diff(kk_) == 1)
    Yf, Yb = Yf[:, m], Yb[:, m]
    x = P * np.conj(ref) / np.abs(ref) ** 2                      # 背景で正規化 (理想なら 1)
    s = np.angle(np.sum(x[:, 1:] * np.conj(x[:, :-1]) * adj_, 1))
    x2 = x * np.exp(-1j * s[:, None] * kc_)
    w = np.abs(ref) ** 2; c = (x2 * w).sum(1) / w.sum()           # 背景の強さで重み付けした共通成分
    phi = np.angle(c); ok = np.abs(phi - np.angle(np.mean(np.exp(1j * phi)))) < 0.8
    phi = np.angle(np.exp(1j * (phi - np.angle(np.mean(np.exp(1j * phi[ok]))))))
    # トーンごとの雑音: 共通成分を外した残りの位相のばらつき (トーンごと) → 重み付き平均の位相の予測ばらつき
    res = np.angle(x2[ok] / c[ok, None])                          # [n, K]
    var_k = np.var(res, 0)                                        # トーンごとの位相雑音の分散
    pred = np.sqrt(np.sum(w ** 2 * var_k) / w.sum() ** 2)         # 重み付き平均の位相の標準偏差 (独立なら)
    obs = phi[ok].std()
    amp_f = np.abs(Yf).mean(); amp_b = np.abs(Yb).mean()
    cs = np.corrcoef(phi[ok], s[ok])[0, 1]
    ca = np.corrcoef(phi[ok], np.log(np.abs(Yf[ok]).mean(1) * np.abs(Yb[ok]).mean(1)))[0, 1]
    rf = np.mean([x[2] for x in L]); rb = np.mean([x[3] for x in L]); nf = np.mean([x[4] for x in L])
    rows.append((k, obs, pred, amp_f, amp_b, rf, rb))
    print(f"{k[0]}-{k[1]} | {amp_f:5.1f} / {amp_b:5.1f} | {rf:5.1f} / {rb:5.1f} | {nf:5.1f} | {obs:.3f} | {pred:.3f} | {obs / pred:4.1f} | {cs:+.2f} | {ca:+.2f}")
r = np.array([[x[1], x[2]] for x in rows])
print(f"\n共通位相のぶれのうち、トーンごとの雑音 (熱雑音・量子化) で説明できる割合 (分散で): 中央値 {np.median((r[:, 1] / r[:, 0]) ** 2) * 100:.0f} %")
