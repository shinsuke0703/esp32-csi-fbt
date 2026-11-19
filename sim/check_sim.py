# -*- coding: utf-8 -*-
"""シミュレータの出力の確認と、最初の数理的な見積もり (有効な自由度)。
使い方: python check_sim.py E:/CSI/sim_out/default E:/CSI/sim_out/empty
"""
import sys
import numpy as np
sys.path.insert(0, r"e:/CSI/sim")
from simio import load_sim, Esp32Impairment

def canon(H, tones):
    """外乱 (共通位相・位相の傾き・利得) を外した「形」。csi_opq.canon と同じ考え方で、傾きは実際のサブキャリア番号で扱う。
    H [..., K] → (h [..., K], (φ0, τ, log a))"""
    k = np.asarray(tones, float)
    dk = np.diff(k)                                                   # DC をまたぐところは 4 本分の間隔
    tau = np.angle((np.conj(H[..., :-1]) * H[..., 1:]).sum(-1)) if np.all(dk == 1) else \
          np.angle((np.conj(H[..., :-1]) * H[..., 1:] * (dk == 1)).sum(-1))
    H1 = H * np.exp(-1j * tau[..., None] * k)
    phi = np.angle(H1.sum(-1))
    a = np.sqrt((np.abs(H) ** 2).mean(-1)) + 1e-30
    return H1 * np.exp(-1j * phi)[..., None] / a[..., None], (phi, tau, np.log(a))

def nmse_db(a, b): return 10 * np.log10((np.abs(a - b) ** 2).sum() / (np.abs(a) ** 2).sum())

S = load_sim(sys.argv[1]); E = load_sim(sys.argv[2]) if len(sys.argv) > 2 else None
H, tones, meta = S["H"], S["meta"]["tones"], S["meta"]
T, L, K = H.shape
print(f"人あり: T={T} ({T / meta['fps']:.0f} 秒)  リンク {L}  サブキャリア {K}  物体 {len(meta['clutter'])} 個")

# 1. 背景は時間で変わらないか
if E is not None:
    He = E["H"]; print(f"1. 背景 (人なし) の時間変動: 最大 |H(t) - H(0)| / |H| = {np.abs(He - He[:1]).max() / np.abs(He).mean():.2e}  (0 なら正しい)")
    Hbg = He[0]
    # 2. 人による変化の大きさ (背景に対する電力比)
    d = H - Hbg[None]
    r = 10 * np.log10((np.abs(d) ** 2).mean(axis=(0, 2)) / (np.abs(Hbg) ** 2).mean(axis=1))
    print("2. 人による変化 / 背景 (電力比 dB、リンクごと):", np.round(r, 1).tolist())
A = np.abs(H)
print("   振幅の時間変動 std/mean (リンクごと):", np.round((A.std(0) / A.mean(0)).mean(1), 3).tolist(), " ← 実機 (体に付けたストロボ) は 0.12〜0.29")

# 3. 欠点を加えても形が戻るか
imp = Esp32Impairment(snr_db=25, loss=0.0, seed=1)
X, ok, info = imp.apply(H, tones)
h0, _ = canon(H, tones); h1, (phi, tau, la) = canon(X.astype(np.complex128), tones)
print(f"3. 欠点 (SNR 25 dB, int8, 位相・傾き・AGC) を加えた後の「形」の誤差: {nmse_db(h0, h1):.1f} dB"
      f"  (傾きの推定誤差 std {np.std(tau - (info['tau'] + np.angle(np.exp(1j*0)))):.4f} は背景の傾きを含む)")
for snr in (10, 20, 30, 40):
    X2, _, _ = Esp32Impairment(snr_db=snr, loss=0.0, seed=2).apply(H, tones); h2, _ = canon(X2.astype(np.complex128), tones)
    print(f"   SNR {snr:2d} dB → 形の誤差 {nmse_db(h0, h2):6.1f} dB")

# 4. 有効な自由度: 形の時間変動の主成分のうち、雑音より大きいもの
c2r = lambda h: np.concatenate([h.real, h.imag], -1)
def dof(Hclean, snr, thr=3.0):
    Xn, _, _ = Esp32Impairment(snr_db=snr, loss=0.0, seed=3).apply(Hclean, tones)
    out = []
    for l in range(L):
        hs, _ = canon(Hclean[:, l].astype(np.complex128), tones); hn, _ = canon(Xn[:, l].astype(np.complex128), tones)
        Z = c2r(hs); sig = np.linalg.svd(Z - Z.mean(0), compute_uv=False) ** 2 / len(Z)
        noise = (np.abs(c2r(hn) - Z) ** 2).mean()                    # 1 次元あたりの雑音の分散
        out.append((int((sig > thr * noise).sum()), sig, noise))
    return out
print("4. 有効な自由度 (形の主成分のうち、分散が雑音の 3 倍を超える数。1 リンクあたり、実数で数える)")
for snr in (15, 25, 35):
    r = dof(H, snr); print(f"   SNR {snr} dB: リンクごと {[x[0] for x in r]}  合計 {sum(x[0] for x in r)}")
r = dof(H, 25)
s = r[0][1]; print("   リンク 0 の主成分の分散 (上位 12、雑音 =", f"{r[0][2]:.2e}):", np.array2string(s[:12], precision=2))
# 全リンクをまとめたときの自由度 (リンク同士の重複を除いた数)
Zall = np.concatenate([c2r(canon(H[:, l].astype(np.complex128), tones)[0]) for l in range(L)], 1)
sv = np.linalg.svd(Zall - Zall.mean(0), compute_uv=False) ** 2 / len(Zall)
noise = np.mean([x[2] for x in r])
print(f"   5 リンクまとめて: 雑音の 3 倍を超える主成分 {int((sv > 3 * noise).sum())} 個 (SNR 25 dB)。体の姿勢パラメータは 19 個")
