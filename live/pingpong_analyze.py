# -*- coding: utf-8 -*-
"""往復 (PING/PONG) 実験の解析: ESP32 の共通位相が往復の積で打ち消せるか。
使い方: python pingpong_analyze.py pp.npz

1 回の往復 = ノード i の PING (番号 s、相手 j) を j が受けた CSI Y_ij と、j が返した PONG (同じ番号 s) を i が受けた CSI Y_ji。
  片道の共通位相   φ_ij = arg Σ_k Y_ij[k]          (ESP32 ではパケットごとにほぼ一様にばらつく)
  往復の積の位相   φ_ij + φ_ji = arg Σ_k Y_ij[k]·Y_ji[k]
                    → 「往復で打ち消せる」なら、静止した部屋では毎回ほぼ同じ値になる (= 2 arg H)
  傾き (STO) も同様に、片道と往復の和のばらつきを比べる。
  比較用: 往復の「比」 Y_ij · conj(Y_ji) の位相 (PDF の Ξ。発振器の位相差 2(θi−θj) を表すはず)。
"""
import sys
import numpy as np

d = dict(np.load(sys.argv[1]))   # 最初に全部読み込む (npz のままだと読むたびに展開し直して遅い)
TONES = np.r_[np.arange(-28, 0), np.arange(1, 29)]
CLASSIC = np.r_[np.arange(36, 64), np.arange(1, 29)]   # FFT 順 → 周波数順
C3 = np.r_[np.arange(4, 32), np.arange(33, 61)]        # 周波数順 (-32..31) → 有効な 56 本

def csi(i):
    """記録 i の HT-LTF を周波数順の 56 本に。HT のフレームで長さ 256 (LLTF 128 + HT-LTF 128) を想定。"""
    r = d["csi"][i].astype(float)[128:256]; c = r[1::2] + 1j * r[0::2]
    return c[CLASSIC]   # 往復用ファームでは C3 も FFT 順 (ゼロ: 0, 29..35) だった (2026-11-18 実測)

def circ_std(a):
    R = np.abs(np.mean(np.exp(1j * a))); return np.sqrt(-2 * np.log(max(R, 1e-12)))

def slope(Y):
    adj = np.diff(TONES) == 1
    return np.angle((np.conj(Y[:-1]) * Y[1:])[adj].sum())

typ, tx, rx, seq, tgt = d["type"], d["tx"], d["rx"], d["seq"], d["target"]
ok = (d["sig"] == 1) & (d["len"] >= 256)
print(f"記録 {len(typ)} 件 (HT で長さ 256 以上: {ok.sum()})。受信側のチップ: classic {(d['chip'] == 0).sum()} / C3 {(d['chip'] == 1).sum()}")
ping = {(int(tx[i]), int(seq[i]), int(rx[i])): i for i in np.where((typ == 1) & ok)[0]}
pong = {(int(tgt[i]), int(seq[i]), int(rx[i])): i for i in np.where((typ == 2) & ok)[0]}   # キー: (PING を送った i, 番号, 受信者)
delay = {(int(tgt[i]), int(seq[i])): d["t_aux"][i] for i in np.where(typ == 3)[0]}       # j が PING を受けて PONG を頼むまで [µs]

ids = sorted(set(tx.tolist()) | set(rx.tolist()))
rows = []; per_pair = {}
for (i, s, r), a in ping.items():
    j = int(tgt[a])
    if r != j: continue                              # Y_ij: PING を相手 j が受けたもの
    b = pong.get((i, s, i))                          # Y_ji: 同じ番号の PONG を i が受けたもの
    if b is None: continue
    Yij, Yji = csi(a), csi(b)
    per_pair.setdefault((min(i, j), max(i, j)), []).append(dict(
        i=i, j=j, phi_ij=np.angle(Yij.sum()), phi_ji=np.angle(Yji.sum()), prod=np.angle((Yij * Yji).sum()),
        ratio=np.angle((Yij * np.conj(Yji)).sum()), tau_ij=slope(Yij), tau_ji=slope(Yji),
        amp_corr=np.corrcoef(np.abs(Yij), np.abs(Yji))[0, 1], dt=delay.get((i, s), np.nan), t=d["t_rx"][a]))

print(f"そろった往復: {sum(len(v) for v in per_pair.values())} 回、{len(per_pair)} 組\n")
print("組 (i,j) | 回数 | 片道の位相のばらつき (円周 std, rad) | 往復の積 | 往復の比 | 傾き: 片道 / 往復の和 | 振幅の形の相関 | PONG までの時間 µs")
summ = []
for (a, b), L in sorted(per_pair.items()):
    # 向きをそろえる (PING を送ったのが a の回と b の回で、積は同じ、比は符号が逆)
    sgn = np.array([1 if x["i"] == a else -1 for x in L])
    phi1 = np.array([x["phi_ij"] for x in L]); prod = np.array([x["prod"] for x in L]); ratio = np.array([x["ratio"] for x in L]) * sgn
    t1 = np.array([x["tau_ij"] for x in L]); tsum = np.array([x["tau_ij"] + x["tau_ji"] for x in L])
    ac = np.nanmean([x["amp_corr"] for x in L]); dt = np.nanmedian([x["dt"] for x in L])
    s = (circ_std(phi1), circ_std(prod), circ_std(ratio), circ_std(t1), circ_std(tsum))
    summ.append(s)
    print(f"({a},{b}) | {len(L):4d} | {s[0]:5.2f} | {s[1]:5.2f} | {s[2]:5.2f} | {s[3]:6.3f} / {s[4]:6.3f} | {ac:5.2f} | {dt:6.0f}")
S = np.array(summ)
print(f"\n全組の中央値: 片道の位相 {np.median(S[:,0]):.2f} rad、往復の積 {np.median(S[:,1]):.2f} rad、往復の比 {np.median(S[:,2]):.2f} rad"
      f"  (一様分布なら 1.81 rad 前後、完全に打ち消せれば雑音程度の 0.1 rad 前後)")
print(f"傾き: 片道 {np.median(S[:,3]):.3f}、往復の和 {np.median(S[:,4]):.3f} rad/サブキャリア")

# 3 台の閉路: 同じ巡回の中の往復の比の和が 0 になるか (PDF の閉路の条件)
if len(ids) >= 3:
    from itertools import combinations
    print("\n3 台の閉路 (往復の比の和 arg(Ξij Ξjm Ξmi)、近い時刻の往復を使う):")
    for (p, q, r) in list(combinations(ids, 3))[:6]:
        def series(a, b):
            L = per_pair.get((min(a, b), max(a, b)), []);
            return np.array([x["t"] for x in L]), np.array([x["ratio"] * (1 if x["i"] == a else -1) for x in L])
        (t1, r1), (t2, r2), (t3, r3) = series(p, q), series(q, r), series(r, p)
        if min(len(t1), len(t2), len(t3)) < 5: continue
        # 時刻は受信側の時計なので厳密にはそろわない。同じ巡回 (並び順) でおおよそ対応づける
        n = min(len(r1), len(r2), len(r3)); clo = r1[:n] + r2[:n] + r3[:n]
        print(f"  ({p},{q},{r}): 円周 std {circ_std(clo):.2f} rad (0 に集まれば成り立つ。一様なら 1.81)")
