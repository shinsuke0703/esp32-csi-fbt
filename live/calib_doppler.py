# -*- coding: utf-8 -*-
"""校正: フレーム間の位相の変化 (ドップラー) で 6 台の位置を、初期値なしで部屋全体から探す (Kinect で人の位置が分かっている前提)。
使い方: python calib_doppler.py sessions/<名前> [前半だけ使う割合=1.0] [初期値の数=4]
体の成分 η = √(P/P_bg) − 1 (P_bg は時間の中央値)。隣の往復との積 z_t = η_t·η_{t−1}* は、体の形で決まる位相のずれ (ゆっくり変わる) が消え、
   z_t ≈ |η|²·exp(−jk·(L(x_t) − L(x_{t−1})))  (L = |x − r_i| + |x − r_j|、x = 胴の中心)
一致度: 1 秒ごとの区間で |Σ z_t·exp(+jk·ΔL_t)| / Σ|z_t| を平均 (0〜1)。ΔL は「人の速さ × 2 台への方向の和」で決まり、ノードの位置に対して
   ゆるやかにしか変わらないので、絶対の位相 (半波長ごとに偽の答え) より探しやすい。
探し方: 1 台ずつ、部屋全体を 15 cm 格子 → 4 cm → 1 cm で探す (他の台は固定)、を全員について繰り返す。初期値をいくつか変えて、答えがそろうかを見る。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from calib_common import load_session

ses = sys.argv[1]; frac = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; NST = int(sys.argv[3]) if len(sys.argv) > 3 else 4
S = load_session(ses); pairs = S["pairs"]; C = 299792458.0; k0 = 2 * np.pi * 2.437e9 / C
LO = np.array([-3.0, 0.0, -1.0]); HI = np.array([3.0, 2.6, 5.0])
dat = {}
for p in pairs:
    Pk = S["Pk"][p]; t = S["t"][p]; n = int(len(t) * frac); Pk, t = Pk[:n], t[:n]
    bg = np.median(Pk.real, 0) + 1j * np.median(Pk.imag, 0); w = np.abs(bg) ** 2
    eta = np.sqrt((Pk * np.conj(bg)).sum(1) / w.sum()) - 1
    dt = np.diff(t); ok = dt < 0.03                                   # 抜けのない隣どうしだけ
    z = (eta[1:] * np.conj(eta[:-1]))[ok]
    x1 = S["body"][p][:n][1:][ok]; x0 = S["body"][p][:n][:-1][ok]
    blk = ((t[1:][ok] - t[0]) // 1.0).astype(int)
    dat[p] = (z, x0, x1, blk)

def link_score(p, ri, rj):
    z, x0, x1, blk = dat[p]
    dL = (np.linalg.norm(x1 - ri, axis=1) + np.linalg.norm(x1 - rj, axis=1)) - (np.linalg.norm(x0 - ri, axis=1) + np.linalg.norm(x0 - rj, axis=1))
    y = z * np.exp(1j * k0 * dL)
    num = np.sqrt(np.bincount(blk, weights=y.real) ** 2 + np.bincount(blk, weights=y.imag) ** 2); den = np.bincount(blk, weights=np.abs(z))
    ok = den > 0
    return float(np.mean(num[ok] / den[ok]))

def total(R): return float(np.mean([link_score(p, R[p[0]], R[p[1]]) for p in pairs]))

def grid(c, span, step):
    g = [np.arange(max(LO[a], c[a] - span[a]), min(HI[a], c[a] + span[a]) + 1e-9, step) for a in range(3)]
    G = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3); return G

def solve(R):
    for it in range(5):
        moved = 0.0
        for i in range(6):
            links = [p for p in pairs if i in p]
            part = lambda ri: np.mean([link_score(p, ri if p[0] == i else R[p[0]], ri if p[1] == i else R[p[1]]) for p in links])
            stages = [((HI - LO) / 2, 0.15)] if it == 0 else [(np.array([0.6, 0.6, 0.6]), 0.1)]
            stages += [(np.array([0.2, 0.2, 0.2]), 0.04), (np.array([0.05, 0.05, 0.05]), 0.01)]
            c = (LO + HI) / 2 if it == 0 else R[i].copy(); best = (part(R[i]), R[i].copy())
            for span, step in stages:
                for ri in grid(best[1] if span[0] < 1 else c, span, step):
                    s = part(ri)
                    if s > best[0]: best = (s, ri.copy())
            moved = max(moved, np.linalg.norm(best[1] - R[i])); R[i] = best[1]
        print(f"    {it + 1} 周目: 一致度 {total(R):.3f}、一番動いた台 {moved * 100:.0f} cm", flush=True)
        if moved < 0.02: break
    return R

rng = np.random.default_rng(0)
print(f"組 {len(pairs)}、使う時間 {frac * 100:.0f} %。ランダムな配置での一致度 (基準): {np.mean([total(LO + rng.random((6, 3)) * (HI - LO)) for _ in range(5)]):.3f}")
sols = []
for s in range(NST):
    R0 = LO + rng.random((6, 3)) * (HI - LO)
    print(f"  初期値 {s + 1}:", flush=True)
    R = solve(R0.copy()); sols.append((total(R), R.copy()))
sols.sort(key=lambda x: -x[0]); best = sols[0][1]
print("\n一致度の高い順と、最良の解との差 (ノードごと、cm):")
for sc, R in sols: print(f"  一致度 {sc:.3f} | " + " ".join(f"{np.linalg.norm(R[i] - best[i]) * 100:5.0f}" for i in range(6)))
print("\n最良の解 (X 左右, Y 高さ, Z 奥行き [m]、Kinect の真下の床が原点):")
for i in range(6): print(f"  ノード {i}{' (C3)' if i == 0 else ''}: " + " ".join(f"{v:+.2f}" for v in best[i]))
print("リンクごとの一致度:", {f"{p[0]}-{p[1]}": round(link_score(p, best[p[0]], best[p[1]]), 3) for p in pairs})
json.dump(dict(nodes=best.tolist(), score=sols[0][0], all=[(sc, R.tolist()) for sc, R in sols], frac=frac),
          open(os.path.join(ses, f"calib_doppler_{int(frac * 100)}.json"), "w"), indent=1)
