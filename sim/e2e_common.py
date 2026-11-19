# -*- coding: utf-8 -*-
"""腰の位置のベンチマークの共通部分: シミュレータの読み込み、実機の欠点の付加、η̂ と外れ値の重み。
実機の条件 (2026-11-18 の実測): ACK 往復の積、LLTF 相当の 50 トーン、1 チャネル、60 Hz、往復の位相のぶれ 0.22 rad + トーンの雑音
(和で 0.28 rad 前後)、パケットごとの傾き 0.006 rad/トーン、外れ値 (既定 5 %)、AGC (振幅は RSSI 1 dB 刻みから)。"""
import sys
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim

C = 299792458.0


def load(base=r"E:/CSI/sim_out/mesh3d_60", empty=r"E:/CSI/sim_out/mesh3d_empty", body_gain=2.4):
    """body_gain: 体の成分 (片道の H − 無人の H) を何倍するか。既定 2.4 = 実機の強さ (約 −9.3 dB) に合わせる
    (シミュレータのままだと歩行中 −16.9 dB で、実機より 7.6 dB 弱い)。"""
    S = load_sim(base); E = load_sim(empty)
    if body_gain != 1.0: S["H"] = E["H"][:1] + body_gain * (S["H"] - E["H"][:1])
    meta = S["meta"]; tones = np.array(meta["tones"]); sel = np.abs(tones) <= 26
    nodes = np.array(meta["tx"], float); N = len(nodes)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    rt = lambda H: np.stack([H[:, i * N + j][:, sel] * H[:, j * N + i][:, sel] for i, j in pairs], 1)
    return dict(P=rt(S["H"]), Pemp=rt(E["H"]), kk=tones[sel].astype(float), fc=meta["fc"], nodes=nodes, pairs=pairs,
                pel=S["params"][:, :2], params=S["params"], joints=S["joints"], fps=meta["fps"])


def observe(D, outl=0.05, rng=None, clean=False, snr_top=22.0, jit=None, slope_sd=0.006):
    """η̂ [T, 15] と外れ値の重み w [T, 15] を返す。clean=True なら欠点なし。"""
    rng = rng or np.random.default_rng(0)
    P, Pemp, kk = D["P"], D["Pemp"], D["kk"]; npair = P.shape[1]
    if clean:
        Pb = Pemp[0]; Q = np.sum(P * np.conj(Pb), -1) / np.sum(np.abs(Pb) ** 2, -1)
        return np.sqrt(Q) - 1, np.ones(Q.shape)
    if jit is None:   # 実機で傾きを外した後の往復の位相のぶれ (token_60hz_b): C3 (ノード 0) を含む組 0.35 rad、DevKitC 同士 0.16 rad
        jit = np.array([0.35 if 0 in pr else 0.16 for pr in D["pairs"]])[None, :, None]
    ref = np.sqrt(np.mean(np.abs(Pemp[0]) ** 2, 1)); snr_db = snr_top + 20 * np.log10(ref / ref.max())

    def impair(Pin):
        Tn = Pin.shape[0]
        noise = (rng.normal(size=Pin.shape) + 1j * rng.normal(size=Pin.shape)) / np.sqrt(2) * (np.abs(Pin).mean(-1, keepdims=True) * 10 ** (-snr_db[None, :, None] / 20))
        Y = (Pin + noise) * np.exp(1j * (rng.normal(0, 1, (Tn, npair, 1)) * jit + rng.normal(0, slope_sd, (Tn, npair, 1)) * kk))
        out = rng.random((Tn, npair)) < outl
        Y[out] *= np.exp(1j * rng.uniform(-np.pi, np.pi, (out.sum(), 1)))
        pw_db = 10 * np.log10(np.mean(np.abs(Pin + noise) ** 2, -1))
        rssi_db = np.round(pw_db / 2 + rng.uniform(-0.5, 0.5, pw_db.shape)) * 2
        return Y / np.sqrt(np.mean(np.abs(Y) ** 2, -1, keepdims=True)) * 10 ** (rssi_db[..., None] / 20)

    def remove_slope(Y, ref_):
        x = Y * np.conj(ref_)
        s = np.angle(np.sum(x[..., 1:] * np.conj(x[..., :-1]) * (np.diff(kk) == 1), -1))
        return Y * np.exp(-1j * s[..., None] * (kk - kk.mean()))

    Ybg = impair(np.repeat(Pemp[:1], 240, 0)); bg0 = Ybg[0]; Ybg = remove_slope(Ybg, bg0)
    ph = np.angle(np.sum(Ybg * np.conj(bg0), -1)); medph = np.angle(np.median(np.cos(ph), 0) + 1j * np.median(np.sin(ph), 0))
    ok = np.abs(np.angle(np.exp(1j * (ph - medph)))) < 0.8
    Pbg = np.array([Ybg[ok[:, p], p].mean(0) for p in range(npair)])
    Y = remove_slope(impair(P), Pbg)
    Q = np.sum(Y * np.conj(Pbg), -1) / np.sum(np.abs(Pbg) ** 2, -1)
    # 外れ値のゲート: 直近 9 フレームの円周中央値から 0.8 rad 以上外れたら重み 0
    phq = np.angle(Q); T = len(Q); w = np.ones_like(phq); h = 4
    pad = np.pad(phq, ((h, h), (0, 0)), mode="edge")
    win = np.stack([pad[i:i + T] for i in range(2 * h + 1)], 0)
    med = np.angle(np.median(np.cos(win), 0) + 1j * np.median(np.sin(win), 0))
    w[np.abs(np.angle(np.exp(1j * (phq - med)))) > 0.8] = 0
    return np.sqrt(Q) - 1, w
