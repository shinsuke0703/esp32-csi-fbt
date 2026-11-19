# -*- coding: utf-8 -*-
"""実機の条件にそろえた、腰の水平位置の推定 (CGSC 型の全範囲探索) のベンチマーク。
使い方: python e2e_pelvis.py [外れ値=0.05] [seed=0]

データを作る側 (Unity CLI シミュレータ mesh3d_60): カプセル 15 個の体 (body_rho 0.5、遮蔽あり)、物 6 個、床の反射。円の上を歩く 8〜28 秒 + 中央での動き。
実機の条件 (2026-11-18 の実測):
  - ACK 往復の積 P_ij[k] = H_ij[k]·H_ji[k]、LLTF 相当の 50 トーン (|k| ≤ 26、312.5 kHz 間隔)、1 チャネル、60 Hz
  - 位相の雑音: 往復の積の和で σ ≈ 0.28 rad になるよう、トーンごとの雑音 (リンクの強さで SNR が変わる) + パケットごとの位相のぶれ
  - パケットごとの傾き (検出の遅れ δ_i + δ_j): 標準偏差 0.006 rad/トーン
  - 外れ値 (位相が一様にばらばら): 既定 5 %
  - AGC: 振幅はパケットごとに未知の倍率。RSSI は 1 dB 刻み → 振幅は RSSI から (量子化の誤差込み)
  - 背景 P_bg は無人のときに 240 パケット平均
解く側 (別のモデル): 胴を高さ 0.9〜1.5 m の縦の点列とみなした散乱 η_p(x) ∝ Σ_h exp(−jk(L_p(x,h) − d_p))·d_p/(L1·L2)。
  床の反射・手足・物は入れない。ノードの位置は 正解 + 誤差 σ (0〜50 cm)。全体の複素係数は未知 (照合で消す)。
推定: η̂_p = √(P/P_bg) − 1 (割ってから平方根、CGSC 定理 2) → 照合のスコア S(x) = |Σ_p T_p(x)*·η̂_p|² / Σ|T_p|² を
  4 cm 格子で全範囲 (半径 2.6 m) → (a) フレームごとの最大 (解き始めなし) / (b) 追跡 (前のフレームから 1 フレーム 4 cm 以内に動く前提の前向き累積)。
"""
import sys
import numpy as np
from scipy.ndimage import maximum_filter
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim

OUTL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05; SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = np.random.default_rng(SEED)
S = load_sim(r"E:/CSI/sim_out/mesh3d_60"); E = load_sim(r"E:/CSI/sim_out/mesh3d_empty")
meta = S["meta"]; tones = np.array(meta["tones"]); fc = meta["fc"]; C = 299792458.0; k0 = 2 * np.pi * fc / C
sel = np.abs(tones) <= 26; kk = tones[sel].astype(float); fk = fc + kk * meta["df"]; kvec = 2 * np.pi * fk / C
nodes = np.array(meta["tx"], float); N = 6
pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
def roundtrip(Hall):   # [T, 36, K] → [T, 15, K']
    return np.stack([Hall[:, i * N + j][:, sel] * Hall[:, j * N + i][:, sel] for i, j in pairs], 1)
Ptrue = roundtrip(S["H"]); Pemp = roundtrip(E["H"])
T = Ptrue.shape[0]; K = sel.sum()
pel = S["params"][:, :2]                                         # 正解の腰 (x, z)

# ---- 実機の欠点を加える ----
ref = np.sqrt(np.mean(np.abs(Pemp[0]) ** 2, 1))                    # リンクの強さ
snr_db = 22 + 20 * np.log10(ref / ref.max())                       # 一番強いリンクでトーンあたり 22 dB (往復の積)
def impair(P, n):
    P = P[:n] if n is not None else P
    Tn = P.shape[0]
    noise = (rng.normal(size=P.shape) + 1j * rng.normal(size=P.shape)) / np.sqrt(2) * (np.abs(P).mean(-1, keepdims=True) * 10 ** (-snr_db[None, :, None] / 20))
    jitter = rng.normal(0, 0.22, (Tn, 15, 1))                        # 往復のぶれ (CFO × 応答の時間のぶれ)
    slope = rng.normal(0, 0.006, (Tn, 15, 1)) * kk                   # 検出の遅れ
    Y = (P + noise) * np.exp(1j * (jitter + slope))
    out = rng.random((Tn, 15)) < OUTL
    Y[out] *= np.exp(1j * rng.uniform(-np.pi, np.pi, (out.sum(), 1)))
    # AGC: 振幅は捨てて、RSSI (1 dB 刻み、2 方向) から戻す
    pw_db = 10 * np.log10(np.mean(np.abs(P + noise) ** 2, -1))       # 真の受信電力 (往復なので 2 方向の和)
    rssi_db = np.round(pw_db / 2 + rng.uniform(-0.5, 0.5, pw_db.shape)) * 2   # 片方向ずつ 1 dB 刻み (近似)
    Y = Y / np.sqrt(np.mean(np.abs(Y) ** 2, -1, keepdims=True)) * 10 ** (rssi_db[..., None] / 20)
    return Y
Ybg = impair(np.repeat(Pemp[:1], 240, 0), None)
def remove_slope(Y, ref):   # 背景に対する傾きをパケットごとに推定して外す
    x = Y * np.conj(ref)
    s = np.angle(np.sum(x[..., 1:] * np.conj(x[..., :-1]) * (np.diff(kk) == 1), -1))
    return Y * np.exp(-1j * s[..., None] * (kk - kk.mean()))
bg0 = Ybg[0]
Ybg = remove_slope(Ybg, bg0)
# 背景: 円周中央値で外れ値を除いてから平均
ph = np.angle(np.sum(Ybg * np.conj(bg0), -1)); medph = np.angle(np.median(np.cos(ph), 0) + 1j * np.median(np.sin(ph), 0))
ok = np.abs(np.angle(np.exp(1j * (ph - medph)))) < 0.8
Pbg = np.array([Ybg[ok[:, p], p].mean(0) for p in range(15)])      # [15, K]
Y = remove_slope(impair(Ptrue, None), Pbg)

# ---- η̂ = √(P/P_bg) − 1 (トーン方向に射影してから) ----
Q = np.sum(Y * np.conj(Pbg), -1) / np.sum(np.abs(Pbg) ** 2, -1)   # [T, 15]
eta = np.sqrt(Q) - 1
# 外れ値のゲート: 直近 9 フレームの円周中央値から 0.8 rad 以上外れた往復は捨てる (重み 0)
phq = np.angle(Q); W9 = 9; w = np.ones_like(phq)
for p in range(15):
    for t in range(T):
        a = phq[max(0, t - W9 // 2):t + W9 // 2 + 1, p]; m = np.angle(np.median(np.cos(a)) + 1j * np.median(np.sin(a)))
        if abs(np.angle(np.exp(1j * (phq[t, p] - m)))) > 0.8: w[t, p] = 0

# ---- 解く側のテンプレート ----
g = np.arange(-2.6, 2.6001, 0.04); GX, GZ = np.meshgrid(g, g, indexing="ij"); inside = GX ** 2 + GZ ** 2 <= 2.6 ** 2
pts = np.stack([GX[inside], GZ[inside]], 1)                         # [G, 2]
HS = np.array([0.9, 1.1, 1.3, 1.5])
def templates(nd):
    Tm = np.zeros((len(pts), 15), complex)
    for p, (i, j) in enumerate(pairs):
        dp = np.linalg.norm(nd[i] - nd[j])
        for h in HS:
            X = np.c_[pts[:, 0], np.full(len(pts), h), pts[:, 1]]
            L1 = np.linalg.norm(X - nd[i], axis=1); L2 = np.linalg.norm(X - nd[j], axis=1)
            Tm[:, p] += np.exp(-1j * k0 * (L1 + L2 - dp)) * dp / (L1 * L2)
    return Tm

def run(sig_node):
    nd = nodes + rng.normal(0, sig_node, nodes.shape) if sig_node > 0 else nodes.copy()
    Tm = templates(nd); nrm = np.sqrt(np.sum(np.abs(Tm) ** 2, 1))
    walk = np.where((np.arange(T) >= 8 * 60) & (np.arange(T) < 28 * 60))[0]
    e_cold, e_trk = [], []; V = None
    for t in range(T):
        sc = np.abs((Tm.conj() * (w[t] * eta[t])[None]).sum(1)) ** 2 / nrm ** 2
        sc = sc / (sc.max() + 1e-30)
        img = np.full(GX.shape, 0.0); img[inside] = sc
        V = img if V is None else 0.9 * maximum_filter(V, size=3) + img   # 1 フレーム 4 cm まで動ける
        V[~inside] = 0
        if t in set(walk) or True:
            e_cold.append(np.linalg.norm(pts[np.argmax(sc)] - pel[t]))
            iv = np.unravel_index(np.argmax(V), V.shape); e_trk.append(np.linalg.norm([GX[iv] - pel[t, 0], GZ[iv] - pel[t, 1]]))
    e_cold, e_trk = np.array(e_cold), np.array(e_trk)
    wk = (np.arange(T) >= 8 * 60) & (np.arange(T) < 28 * 60)
    return e_cold, e_trk, wk

print(f"外れ値 {OUTL * 100:.0f} %、外れ値ゲートで捨てた割合 {100 * (1 - w.mean()):.1f} %、|η̂| の 99% 点 {np.percentile(np.abs(eta), 99):.2f}")
print("ノードの位置の誤差 | 歩行中: フレームごと (中央値 / 10 cm 以内の割合) | 追跡 (中央値 / 10 cm 以内) | 全区間の追跡 中央値")
for sg in (0.0, 0.02, 0.05, 0.10, 0.20, 0.50):
    ec, et, wk = run(sg)
    print(f"{sg * 100:4.0f} cm | {np.median(ec[wk]) * 100:6.1f} cm / {np.mean(ec[wk] < 0.1) * 100:4.0f} % | "
          f"{np.median(et[wk]) * 100:6.1f} cm / {np.mean(et[wk] < 0.1) * 100:4.0f} % | {np.median(et) * 100:6.1f} cm", flush=True)
