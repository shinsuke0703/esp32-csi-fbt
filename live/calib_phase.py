# -*- coding: utf-8 -*-
"""校正の段階 2: 往復の位相で 6 台の位置を詰める (Kinect で人の位置が分かっている前提、初期値は段階 1 の calib_shadow.json)。
使い方: python calib_phase.py sessions/<名前> [初期値のファイル=calib_shadow.json] [前半だけ使う割合=1.0]
体の成分: 背景 P_bg = 往復の積 (トーンごと) の時間の中央値 (無人の時間がないので、人がたいてい離れている状態を背景とみなす)。
  W = √(P / P_bg) (割ってから平方根、CGSC 定理 2)、η = W − 1 をトーン方向に背景で重み付けして足す。10 パケット平均して雑音を減らす。
モデル: η_p(t) ≈ A_p(t) · exp(−jk · L_p(x_t))、L_p = |x − r_i| + |x − r_j| (x = 胴の中心、Kinect)。A_p(t) は体の形で決まり、ゆっくり変わる。
一致度: 1 秒ごとの区間で Σ η·exp(+jkL) の大きさ / Σ|η| をとり、全区間・全リンクで平均 (0〜1、ランダムなら 1/√n 程度)。
最大化: 1 台ずつ、今の位置のまわり ±40 cm を 2 cm 格子で探す → 全員について繰り返す (座標ごとの探索)。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from calib_common import load_session

ses = sys.argv[1]; init = sys.argv[2] if len(sys.argv) > 2 else "calib_shadow.json"; frac = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
S = load_session(ses); pairs = S["pairs"]; C = 299792458.0; k0 = 2 * np.pi * 2.437e9 / C
R = np.array(json.load(open(os.path.join(ses, init)))["nodes"])
seg = {}
for p in pairs:
    Pk = S["Pk"][p]; t = S["t"][p]; n = int(len(t) * frac); Pk, t = Pk[:n], t[:n]
    bg = np.median(Pk.real, 0) + 1j * np.median(Pk.imag, 0); w = np.abs(bg) ** 2
    W = np.sqrt((Pk * np.conj(bg)).sum(1) / w.sum()); eta = W - 1
    eta = np.convolve(eta, np.ones(10) / 10, "same")
    x = S["body"][p][:n]
    blk = ((t - t[0]) // 1.0).astype(int)                   # 1 秒ごとの区間
    keep = np.arange(0, n, 3)                                # 20 Hz に間引く
    seg[p] = (eta[keep], x[keep], blk[keep])

def link_score(p, ri, rj):
    eta, x, blk = seg[p]
    L = np.linalg.norm(x - ri, axis=1) + np.linalg.norm(x - rj, axis=1)
    z = eta * np.exp(1j * k0 * L)
    num = np.bincount(blk, weights=z.real) ** 2 + np.bincount(blk, weights=z.imag) ** 2
    den = np.bincount(blk, weights=np.abs(eta))
    ok = den > 0
    return float(np.mean(np.sqrt(num[ok]) / den[ok]))

def total(R): return np.mean([link_score(p, R[p[0]], R[p[1]]) for p in pairs])

print(f"初期値 ({init}) の一致度 {total(R):.3f}")
for p in pairs: print(f"  {p[0]}-{p[1]}: {link_score(p, R[p[0]], R[p[1]]):.3f}")
# ランダムな位置での一致度 (基準)
rng = np.random.default_rng(0)
base = np.mean([total(R + rng.normal(0, 1.0, R.shape)) for _ in range(5)])
print(f"ランダムに 1 m ずらした位置での一致度 (基準) {base:.3f}")
g = np.arange(-0.4, 0.4001, 0.04)
for it in range(4):
    for i in range(6):
        links = [p for p in pairs if i in p]   # ノード i が動くと変わるのは、i を含む 5 リンクだけ → その平均で比べる
        part = lambda ri: np.mean([link_score(p, ri if p[0] == i else R[p[0]], ri if p[1] == i else R[p[1]]) for p in links])
        best = (part(R[i]), R[i].copy())
        # まず粗く (4 cm、±40 cm)、次に細かく (1 cm、±5 cm)
        for step, span in ((0.04, 0.4), (0.01, 0.05)):
            c0 = best[1].copy(); gg = np.arange(-span, span + 1e-9, step)
            for dx in gg:
                for dy in gg:
                    for dz in gg:
                        ri = c0 + np.array([dx, dy, dz])
                        if ri[1] < 0 or ri[1] > 2.6: continue
                        s = part(ri)
                        if s > best[0]: best = (s, ri.copy())
        R[i] = best[1]
    print(f"{it + 1} 周目: 一致度 {total(R):.3f}", flush=True)
for p in pairs: print(f"  {p[0]}-{p[1]}: {link_score(p, R[p[0]], R[p[1]]):.3f}")
print("詰めた位置 (X 左右, Y 高さ, Z 奥行き [m]):")
for i in range(6): print(f"  ノード {i}: " + " ".join(f"{v:+.2f}" for v in R[i]))
json.dump(dict(nodes=R.tolist(), score=total(R), init=init, frac=frac), open(os.path.join(ses, f"calib_phase_{int(frac * 100)}.json"), "w"), indent=1)
