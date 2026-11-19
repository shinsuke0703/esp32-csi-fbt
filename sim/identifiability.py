# -*- coding: utf-8 -*-
"""識別性の解析: 1 フレームの CSI から、全身の姿勢パラメータ 19 個のうち何がどれだけ決まるか。
使い方: python identifiability.py E:/CSI/sim_out/jac_2tx

モデル: 観測 y = H(p) · exp(j(φ0 + τk)) · a + 雑音 (リンク・パケットごとに φ0, τ, a は未知 = ESP32 の外乱)
  1. リンクごとに、外乱の方向 (dH/dφ0 = jH, dH/dτ = jkH, dH/d log a = H) を射影で取り除いたヤコビアン Jp を作る
     (外乱を「未知のまま」推定した場合のフィッシャー情報量 = シューア補元と同じ)
  2. 雑音: 複素ガウス、分散 σ² = (そのリンクの平均電力) / SNR。F = Σ_l Jp^T Jp / (σ²/2)
  3. 単位をそろえる: 位置は 5 cm、角度は 5° を 1 とする。F の固有値が 1 以上の方向は「5 cm / 5° より良く決まる」
  4. クラメール・ラオ下限 (CRB): 各パラメータの標準偏差の下限 = sqrt((F^-1)_ii)
"""
import sys, json
import numpy as np

base = sys.argv[1] if len(sys.argv) > 1 else r"E:/CSI/sim_out/jac_2tx"
meta = json.load(open(base + ".json", encoding="utf-8"))
N, L, K, NP = meta["N"], meta["L"], meta["K"], meta["NP"]
a = np.fromfile(base + ".bin", dtype=np.float64)
P = a[:N * NP].reshape(N, NP); o = N * NP
H0 = a[o:o + N * L * K * 2].reshape(N, L, K, 2); o += N * L * K * 2
J = a[o:].reshape(N, NP, L, K, 2)
H0 = H0[..., 0] + 1j * H0[..., 1]; J = J[..., 0] + 1j * J[..., 1]
tones = np.array(meta["tones"], float); names = meta["param_names"]
nrx = len(meta["rx"])
UNIT = np.array([0.05] * 3 + [np.deg2rad(5)] * (NP - 3))   # 1 単位 = 5 cm / 5°

def fisher(n, links, tsel, snr_db, nuisance=True):
    F = np.zeros((NP, NP))
    for l in links:
        h = H0[n, l][tsel]; k = tones[tsel]; Jl = J[n][:, l][:, tsel].T   # [K', NP]
        R = lambda z: np.concatenate([z.real, z.imag], 0)
        Jr = R(Jl)
        if nuisance:
            Nm = np.stack([R(1j * h), R(1j * k * h), R(h)], 1)
            Q, _ = np.linalg.qr(Nm); Jr = Jr - Q @ (Q.T @ Jr)
        s2 = (np.abs(h) ** 2).mean() / 10 ** (snr_db / 10)
        F += Jr.T @ Jr / (s2 / 2)
    return F

def analyze(links, tsel, snr_db, nuisance=True):
    ndof, crb = [], []
    for n in range(N):
        F = fisher(n, links, tsel, snr_db, nuisance)
        Fs = F * np.outer(UNIT, UNIT)                             # 単位をそろえた F
        ev = np.linalg.eigvalsh(Fs); ndof.append(int((ev >= 1).sum()))
        Ci = np.linalg.inv(Fs + 1e-6 * np.eye(NP))                # 決まらない方向は事前分布 (±1000 単位) で止める
        crb.append(np.sqrt(np.diag(Ci)) * UNIT)
    return np.array(ndof), np.array(crb)

ht40 = np.ones(K, bool); ht20 = np.abs(tones) <= 28
one = list(range(nrx))                 # ストロボ 1 台 (tx0) × 受信 5 台
two = list(range(L))                   # ストロボ 2 台 × 受信 5 台
print(f"姿勢 {N} 個 (動きの全区間から等間隔)、リンク最大 {L}、サブキャリア {K}。1 単位 = 5 cm / 5°。")
print("「決まる自由度」= 単位をそろえたフィッシャー情報量の固有値が 1 以上の数 (19 個中)。値は姿勢ごとの中央値 [最小, 最大]\n")

def row(label, links, tsel=ht40, snr=25, nuis=True):
    d, c = analyze(links, tsel, snr, nuis)
    med = np.median(c, 0)
    pos = med[:3].max() * 100; ang = np.rad2deg(np.median(med[3:]))
    print(f"  {label:34s} 決まる自由度 {int(np.median(d)):2d} [{d.min():2d},{d.max():2d}]   CRB 中央値: 位置 最悪 {pos:7.1f} cm / 関節角 中央 {ang:7.1f}°")
    return d, c

print("A. 受信機の数 (ストロボ 1 台、HT40、SNR 25 dB)")
for n_rx in (1, 2, 3, 5): row(f"受信 {n_rx} 台 ({n_rx} リンク)", one[:n_rx])
print("B. ストロボを 2 台に (Lighthouse の基地局を増やす)")
row("ストロボ 2 × 受信 5 (10 リンク)", two)
print("C. 帯域 (ストロボ 1 × 受信 5)")
row("HT20 (|k| ≤ 28)", one, ht20); row("HT40 (114 本)", one, ht40)
print("D. SNR (ストロボ 1 × 受信 5、HT40)")
for s in (15, 25, 35, 45): row(f"SNR {s} dB", one, ht40, s)
print("E. ESP32 の外乱の影響 (ストロボ 1 × 受信 5、HT40、SNR 25 dB)")
row("外乱が既知 (理想の受信機)", one, ht40, 25, False); row("外乱が未知 (ESP32)", one, ht40, 25, True)

print("\nF. パラメータごとの CRB (ストロボ 2 × 受信 5、HT40、SNR 25 dB、姿勢の中央値)")
d, c = analyze(two, ht40, 25)
med = np.median(c, 0)
for i in range(NP):
    v = med[i] * 100 if i < 3 else np.rad2deg(med[i]); u = "cm" if i < 3 else "°"
    flag = "○" if (med[i] <= UNIT[i]) else ("△" if med[i] <= 4 * UNIT[i] else "×")
    print(f"  {flag} {names[i]:12s} {v:9.1f} {u}")
print("  ○: 5 cm / 5° 以内、△: 20 cm / 20° 以内、×: それより悪い")

# G. どの方向が決まらないか: 一番決まらない固有ベクトルの中身 (ストロボ 2 × 受信 5)
n0 = N // 2; Fs = fisher(n0, two, ht40, 25) * np.outer(UNIT, UNIT); ev, V = np.linalg.eigh(Fs)
print(f"\nG. 姿勢 {n0} で一番決まりにくい方向 (固有値 {ev[0]:.2e}) の主な成分:",
      ", ".join(f"{names[i]} {V[i, 0]:+.2f}" for i in np.argsort(-np.abs(V[:, 0]))[:4]))
print(f"   固有値 (小さい順、単位をそろえた F): {np.array2string(ev, precision=1, max_line_width=200)}")
