# -*- coding: utf-8 -*-
"""CGSC 定理 3 の確認: 無人の部屋の振幅スペクトル |1 + M(f)| だけから、反射による位相 arg(1 + M(f_c)) を戻せるか。
部屋 7×7×2.7 m (calib_walls と同じ)、壁 6 面の 1 次の鏡像 (Γ 床 −0.5、天井 −0.3、壁 −0.3) + 2 次の鏡像 (積)。15 リンク。
測れる帯域: ch1〜13 を掃引 → 2402〜2482 MHz (80 MHz、312.5 kHz 刻み)。方法は論文の式 (7): 対数振幅のケプストラムの因果部分。
比べるもの: 補正なし (0 とみなす) / 帯域が理想的に広い場合 (2.0〜2.9 GHz、確認用) / 80 MHz / 80 MHz + チャネルごとの振幅の誤差 0.5 dB"""
import numpy as np
C = 299792458.0; W, H = 7.0, 2.7
nodes = np.array([[0, 1.0, 3.2], [3.1, 2.3, 1.5], [3.0, 0.4, -2.0], [-0.5, 2.1, -3.2], [-3.1, 0.6, -1.0], [-3.0, 1.8, 2.0]])
pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
planes = [(1, 0.0, -0.5), (1, H, -0.3), (0, -W / 2, -0.3), (0, W / 2, -0.3), (2, -W / 2, -0.3), (2, W / 2, -0.3)]
def img(p, pl): q = p.copy(); q[pl[0]] = 2 * pl[1] - p[pl[0]]; return q
def paths(a, b):
    d0 = np.linalg.norm(a - b); out = []
    for i, p1 in enumerate(planes):
        a1 = img(a, p1); out.append((p1[2], np.linalg.norm(a1 - b)))
        for j, p2 in enumerate(planes):
            if j != i: out.append((p1[2] * p2[2], np.linalg.norm(img(a1, p2) - b)))
    return d0, out
def M(f, d0, out):  # 直接波で割った反射の和
    return sum(g * (d0 / d) * np.exp(-2j * np.pi * f * (d - d0) / C) for g, d in out)
def kk_phase(f, logamp, fc, tmax_m=120.0):
    """対数振幅 → 実ケプストラム → 因果部分を 2 倍 → 位相。帯域の外は端の値で延長しないよう、窓をかけずにそのまま (論文どおり)。"""
    n = len(f); df = f[1] - f[0]; c = np.fft.ifft(logamp); tau = np.fft.fftfreq(n, df)
    causal = np.zeros(n, complex); pos = (tau > 0) & (tau * C < tmax_m)
    causal[pos] = 2 * c[pos]; causal[0] = c[0]
    lnG = np.fft.fft(causal)            # ln(1+M) の推定 (実部 = 対数振幅、虚部 = 位相)
    return np.interp(fc, f, lnG.imag)
fc = 2437e6; rng = np.random.default_rng(0)
band80 = np.arange(2402e6, 2482e6, 312.5e3); wide = np.arange(2.0e9, 2.9e9, 312.5e3)
rows = []
for (i, j) in pairs:
    d0, out = paths(nodes[i], nodes[j])
    true = np.angle(1 + M(fc, d0, out)); maxM = np.abs(M(band80, d0, out)).max()
    e_wide = np.angle(np.exp(1j * (kk_phase(wide, np.log(np.abs(1 + M(wide, d0, out))), fc) - true)))
    la = np.log(np.abs(1 + M(band80, d0, out)))
    e80 = np.angle(np.exp(1j * (kk_phase(band80, la, fc) - true)))
    ch = ((band80 - 2402e6) // 5e6).astype(int).clip(0, 12)          # どのチャネルで測ったか (5 MHz ごと)
    errs = []
    for t in range(20):
        off = rng.normal(0, 0.5, 13) * np.log(10) / 20                 # チャネルごとの振幅の誤差 0.5 dB
        errs.append(np.angle(np.exp(1j * (kk_phase(band80, la + off[ch], fc) - true))))
    rows.append((true, e_wide, e80, np.sqrt(np.mean(np.square(errs))), maxM))
    print(f"{i}-{j}: 真の位相 {np.degrees(true):+6.1f}°  |M| 最大 {maxM:.2f} | 誤差: 広帯域 {np.degrees(e_wide):+6.1f}°  80 MHz {np.degrees(e80):+6.1f}°  80 MHz+0.5 dB {np.degrees(rows[-1][3]):5.1f}° (RMS)")
r = np.array(rows); rms = lambda v: np.degrees(np.sqrt(np.mean(np.square(v))))
print(f"\nRMS: 補正なし {rms(r[:, 0]):.1f}° / 広帯域 {rms(r[:, 1]):.1f}° / 80 MHz {rms(r[:, 2]):.1f}° / 80 MHz + 振幅誤差 {rms(r[:, 3]):.1f}°")
print("(位相は往復の積で 2 倍になる。ここでの値は片道の arg(1+M))")
